using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Interface for using Hindsight Experience Replay (HER) - Marcin Andrychowicz et at. 2018 <br></br> <br></br>
    /// 1. Define the <see cref="Agent.spaceSize"/> as double (x2) it's normal size on bake. <br></br>
    /// 2. Call in <see cref="Agent.OnEpisodeBegin"/> -> <see cref="HindsightExperienceReplay.SetGoal(Agent)"/>. <br></br>
    /// 3. Call in <see cref="Agent.CollectObservations(StateVector)"/> -> <see cref="HindsightExperienceReplay.AddGoal(Agent, StateVector)"/> <br></br>
    /// </summary>
    public class HindsightExperienceReplay : MonoBehaviour
    {
        /// <summary>
        /// Strategy for sampling subgoals.
        /// </summary>
        public static ReplayStrategy ReplayStrategy = ReplayStrategy.Future;
        /// <summary>
        /// Denoted as <b>k</b> in the paper.
        /// </summary>
        public static int HindsightGoalsPerEpisode = 4;
        /// <summary>
        /// if true then r(g) = ||g - s|| - ||g - s'|| <br></br>
        /// else r(g) = - ||g - s'||
        ///     
        /// </summary>
        public static bool ShapedRewardFunction = true;

        private static Lazy<Dictionary<Agent, Tensor>> initialGoalOfAgents = new Lazy<Dictionary<Agent, Tensor>>();

        /// <summary>
        /// Samples a sub-goal for the current episode. Must be called in <see cref="Agent.OnEpisodeBegin()"/>.
        /// </summary>
        /// <param name="agent"></param>
        public static void SetGoal(Agent agent)
        {
            if (DeepUnityTrainer.Instance is not IOffPolicy)
                throw new Exception("HER works only for Off Policy algorithms");

            var buffer = DeepUnityTrainer.Instance.train_data;
            Tensor goal = null;

            if(buffer.Count == 0)          
                goal = Tensor.Zeros(agent.model.observationSize);
            
            else           
                goal = Utils.Random.Sample(buffer.frames.Select(x => x.goal));

            
            if (!initialGoalOfAgents.Value.ContainsKey(agent))
            {
                initialGoalOfAgents.Value.Add(agent, goal);
            }

            agent.IsUsingHER = true;
        }
        /// <summary>
        /// <b>Concatenates the state with the goal (sampled at the beggining of the episode).</b><br></br>
        /// Usage Example: <br></br>
        /// <em>CollectObservations</em>(<see cref="StateVector"/> <paramref name="stateVector"/>)<br></br>
        /// { <br></br>
        ///     <em><paramref name="stateVector"/>.AddObservations(...);</em><br></br>
        ///     <em>AddGoal(<paramref name="stateVector"/>);</em><br></br>
        /// } <br></br>
        /// </summary>
        /// <param name="stateVector"></param>
        public static void AddGoal(Agent agent, StateVector stateVector)
        {
            stateVector.AddObservation(initialGoalOfAgents.Value[agent].ToArray());
        }

        /// <summary>
        /// Method called in backend only. Samples a number of subgoals and add them as timesteps in the train buffer. 
        /// </summary>
        /// <param name="agent"></param>
        public static TimestepTuple[] SampleSubgoals(Agent agent)
        {
            ConcurrentBag<TimestepTuple> her = new();
            switch (ReplayStrategy)
            {
                case ReplayStrategy.Future:
                    Parallel.For(0, agent.Memory.Count, t =>
                    {
                        var future_possible_goals = agent.Memory.frames.Skip(t);
                        for (int k = 0; k < HindsightGoalsPerEpisode; k++)
                        {
                            TimestepTuple her_ts = agent.Memory.frames[t].Clone() as TimestepTuple;
                            Tensor future_goal = Utils.Random.Sample(future_possible_goals).nextState;
                            her_ts.reward = GoalReward(her_ts.state, her_ts.nextState, her_ts.goal);
                            her.Add(her_ts);
                        }
                    });
                    break;

                   
                case ReplayStrategy.Final:
                    Tensor final_goal = agent.Memory.frames[agent.Memory.Count - 1].nextState;
                    for (int t = 0; t < agent.Memory.Count; t++)
                    {                
                        TimestepTuple her_ts = agent.Memory.frames[t].Clone() as TimestepTuple;
                        her_ts.goal = final_goal;
                        her_ts.reward = GoalReward(her_ts.state, her_ts.nextState, her_ts.goal);
                        her.Add(her_ts);                 
                    }
                    break;


                default:
                    throw new NotImplementedException("Unhandled Replay Strategy for HER.");
            }


            return her.ToArray();
        }

        private static Tensor GoalReward(Tensor state, Tensor nextState, Tensor goal)
        {
            const NormType norm = NormType.EuclideanL2; // or manhattan
            if (ShapedRewardFunction)
                return (goal - state).Norm(norm) - (goal - nextState).Norm(norm);
            else
                return -(goal - nextState).Norm(norm);
        }
    }
}



