using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;

namespace DeepUnity
{
    /// <summary>
    /// Personal experience buffer of a single agent.
    /// </summary>
    public class MemoryBuffer
    {
        public int Count { get => frames.Count; }
        public List<TimestepBuffer> frames { get; private set; }
       
        public MemoryBuffer()
        {
            frames = new();
        }

        public void Add(TimestepBuffer timestep)
        {
            frames.Add(timestep);
        }
        public void Clear()
        {
            frames.Clear();
        }

        /// <summary>
        /// This method computes the generalized advantage estimation, value function targets and q function targets. 
        /// </summary>
        /// <param name="gamma"></param>
        /// <param name="lambda"></param>
        /// <param name="horizon"></param>
        /// <param name="crticNetwork"></param>
        public void GAE(in float gamma, in float lambda, in int horizon, NeuralNetwork crticNetwork)
        {
            int T = Count;
            Tensor[] all_states_plus_lastNextState = new Tensor[T + 1];
            for (int i = 0; i < T; i++)
                all_states_plus_lastNextState[i] = frames[i].state;
            all_states_plus_lastNextState[T] = frames[T - 1].nextState;

            // Vw_s has length of T + 1
            Tensor Vw_s = crticNetwork.Predict(Tensor.Cat(null, all_states_plus_lastNextState)).Reshape(T + 1);
           
            // Generalized Advantage Estimation
            for (int timestep = 0; timestep < T; timestep++)
            {
                float discount = 1f;
                Tensor Ahat_t = Tensor.Constant(0);
            
                for (int t = timestep; t < T; t++) // In GAE for last step advantage is 0.. kind of lose of information but it works :)
                {
                    Tensor r_t = frames[t].reward;
                    float V_st = Vw_s[t];
                    float V_next_st = frames[t].done[0] == 1 ? 0 : Vw_s[t + 1];  // if the state is terminal, next value is set to 0.
            
                    Tensor delta_t = r_t + gamma * V_next_st - V_st;
                    Ahat_t += discount * delta_t;
                    discount *= gamma * lambda;
            
                    if (frames[t].done[0] == 1)
                        break;

                    if (t - timestep == horizon)
                        break;
                }
            
                Tensor Vt_target = Ahat_t + Vw_s[timestep];
                Tensor Qt_target = frames[timestep].reward + gamma * Vw_s[timestep + 1]; // a.k.a Qhat

                frames[timestep].q_target = Qt_target;
                frames[timestep].value_target = Vt_target;
                frames[timestep].advantage = Ahat_t;
            }     

        }
        
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Trajectory ({Count})");
            sb.AppendLine("{");
            for (int i = 0; i < Count; i++)
            {
                sb.Append($" \t| {i.ToString("000")}");
                sb.Append($" | t: {(frames[i].index).ToString("000")}");
                sb.Append($" | s[t]: [{frames[i].state.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | s'[t]: [{frames[i].nextState.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | a_cont[t]: [{frames[i].action_continuous?.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | a_disc[t]: [{frames[i].action_discrete?.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | r[t]: {frames[i].reward[0].ToString("0.000")}");
                
                if (frames[i].value_target != null)
                    sb.Append($" | V[t]: {frames[i].value_target[0].ToString("0.000")}");

                if (frames[i].advantage != null)
                    sb.Append($" | A[t]: {frames[i].advantage[0].ToString("0.000")}");

                if (frames[i].q_target != null)
                    sb.Append($" | Q[t]: {frames[i].q_target[0].ToString("0.000")}");
                                     
                sb.Append("\n");

                if (frames[i].done[0] == 1)
                    sb.Append("\n");
            }
            sb.AppendLine("}");
            return sb.ToString();
        }
    }
}
