using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    internal abstract class DeepUnityTrainer : MonoBehaviour
    {
        public static DeepUnityTrainer Instance;


        /// <summary>
        /// Current experiences collected by the agents.
        /// </summary>
        public static int MemoriesCount { get => Instance.parallelAgents.Sum(x => x.Memory.Count); }
        
        /// <summary>
        /// The total number or Fixed Updates since the start of the game. (ReadOnly) <br>  </br>
        /// Consider this is not equivalent to the global step, because agents can take decisions at different intervals of frames.
        /// </summary>
        public int FixedFrameCount { get; private set; } = 0;


        public event EventHandler OnTrainingSessionEnd;

        [ViewOnly] public List<Agent> parallelAgents;
        [ViewOnly] public Hyperparameters hp;
        [ViewOnly] public AgentBehaviour model;
        [ViewOnly] public ExperienceBuffer train_data;
        [ViewOnly] public Stopwatch updateBenchmarkClock;
        [ViewOnly] public int currentSteps = 0;
        [ViewOnly] public int updateIterations;
        [ViewOnly] public float actorLoss;
        [ViewOnly] public float criticLoss;
        [ViewOnly] public float entropy;


        public readonly DateTime timeWhenTheTrainingStarted = DateTime.Now;
        [Min(1)] public int autosave = 15; protected float autosaveSecondsElapsed = 0f; // editor only autosave
        [ViewOnly] public bool ended = false;

        [SerializeField, ViewOnly] private float avgDeltaTime = 0.02f;
        const float avgDeltaTimeMomentum = 0.96f;
        string _learningText = "Learning";
        string _runtimeStatsText = " - ";
        GUIStyle _learningTextStyle;
        GUIStyle _runtimeStatsStyle;

        private AudioClip trainingSound;

      

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;

                // Display "Learning..." head up on screen
                _learningTextStyle = new GUIStyle();
                _learningTextStyle.fontSize = 50;
                _learningTextStyle.fontStyle = FontStyle.Bold;
                _learningTextStyle.wordWrap = true;
                _learningTextStyle.normal.textColor = Color.white;
                _runtimeStatsStyle = new GUIStyle(_learningTextStyle);
                _runtimeStatsStyle.fontSize = 20;

                // Play some music in background
                // trainingSound = Resources.Load<AudioClip>("Audio/TrainingSound1"); // note that that audio was removed from there
                // var audiosource = transform.AddComponent<AudioSource>();
                // audiosource.loop = true;
                // audiosource.playOnAwake = false;
                // audiosource.volume = 0.1f;
                // audiosource.clip = trainingSound;
                // audiosource.Play();


                StartCoroutine("DrawDotsToLearningText");
#if UNITY_EDITOR // It seems like that when i build the application the quallity is dropped and this is the reason. SO LET IT LIKE THIS YOU MF.
                QualitySettings.SetQualityLevel(0, true);
#endif
            }
        }
        /// <summary>
        /// Autosave(), EndTrainingTrigger(), TimeScaleAdj()
        /// </summary>
        protected void FixedUpdate()
        {
            OnBeforeFixedUpdate();

            // Check if max steps reached
            if (currentSteps >= hp.maxSteps)
            {
                OnTrainingSessionEnd?.Invoke(this, EventArgs.Empty);
                EndTrainingSession($"Max Steps reached ({hp.maxSteps})");
            }
            // Autosaves the ac
            if (autosaveSecondsElapsed >= autosave * 60f)
            {
                autosaveSecondsElapsed = 0f;
                model.Save();
            }
            autosaveSecondsElapsed += Time.fixedDeltaTime;

            if (hp.timescaleAdjustment == TimescaleAdjustmentType.Dynamic)
            {

                const float timeScaleAdjustmentRate = 1e-3f; ///1e-4..

                float currentFrameRate = 1f / avgDeltaTime;
                float frameRateDifference = model.targetFPS * 0.125f - currentFrameRate;// i ve seen that on 12% is ok, almost the same with fixed static sigma value i will set
                hp.timescale = hp.timescale - frameRateDifference * timeScaleAdjustmentRate;
                hp.timescale = Mathf.Clamp(hp.timescale, 1f, 30f);
            }

            Time.timeScale = hp.timescale;
            FixedFrameCount++;
        }
        private void Update()
        {
            avgDeltaTime = avgDeltaTime * avgDeltaTimeMomentum + Time.deltaTime * (1f - avgDeltaTimeMomentum);

            if (hp.trainer == TrainerType.PPO)
                _runtimeStatsText = $"[Trainer: {hp.trainer} | No. agents {parallelAgents.Count} | Timescale: {Time.timeScale.ToString("0.0")} | Buffer: {MemoriesCount}/{hp.bufferSize} ({(MemoriesCount * 100f / hp.bufferSize).ToString("0.00")}%)]";
            else if (hp.trainer == TrainerType.SAC || hp.trainer == TrainerType.TD3 || hp.trainer == TrainerType.DDPG)
                _runtimeStatsText = $"[Trainer: {hp.trainer} | No. agents {parallelAgents.Count} | Timescale: {Time.timeScale.ToString("0.0")} | Buffer: {train_data.Count}/{hp.replayBufferSize} ({(train_data.Count * 100f / hp.replayBufferSize).ToString("0.00")}%)]";
            else
                throw new NotImplementedException("Unhandled trainer type");
        }
        public void OnGUI()
        {
            GUI.Label(new Rect(10, Screen.height - 65, 400, 60), _learningText, _learningTextStyle);
            GUI.Label(new Rect(10, 10, 800, 60), _runtimeStatsText, _runtimeStatsStyle);
        }

        protected abstract void Initialize();
        protected abstract void OnBeforeFixedUpdate();
        IEnumerator DrawDotsToLearningText()
        {
            while (true)
            {
                yield return new WaitForSecondsRealtime(0.5f);
                _learningText += ".";
                if (_learningText.Length > 11)
                    _learningText = "Learning";
            }
        }
        public static void Subscribe(Agent agent, TrainerType trainer)
        {
            if (Instance == null)
            {
#if UNITY_EDITOR
                UnityEditor.EditorApplication.playModeStateChanged += Autosave;
#endif
                GameObject go = new GameObject($"[DeepUnity] Trainer - {trainer}");

                switch (trainer)
                {
                    case TrainerType.PPO:
                        Instance = go.AddComponent<PPOTrainer>();
                        break;
                    case TrainerType.SAC:
                        Instance = go.AddComponent<SACTrainer>();
                        break;
                    case TrainerType.TD3:
                        Instance = go.AddComponent<TD3Trainer>();
                        break;
                    case TrainerType.DDPG:
                        Instance = go.AddComponent<DDPGTrainer>();
                        break;
                    default: throw new ArgumentException("Unhandled trainer type");
                }


                Instance.parallelAgents = new();
                Instance.hp = agent.model.config;
                Instance.train_data = new ExperienceBuffer(Instance.hp.trainer switch
                {
                    TrainerType.PPO => Instance.hp.bufferSize,
                    TrainerType.SAC => Instance.hp.replayBufferSize,
                    TrainerType.TD3 => Instance.hp.replayBufferSize,
                    TrainerType.DDPG => Instance.hp.replayBufferSize,
                    _ => throw new NotSupportedException("Unhandled Trainer Type in initializing the train_data buffer")
                });
                Instance.model = agent.model;
                Instance.Initialize();
            }

            // Assign common attributes to all agents (based on the last agent that subscribes - this one is actually the first in the Hierarchy)
            Instance.parallelAgents.ForEach(x =>
            {
                x.DecisionRequester.decisionPeriod = agent.DecisionRequester.decisionPeriod;
                x.DecisionRequester.maxStep = agent.DecisionRequester.maxStep;
                x.DecisionRequester.takeActionsBetweenDecisions = agent.DecisionRequester.takeActionsBetweenDecisions;
            });

            Instance.parallelAgents.Add(agent);
        }
#if UNITY_EDITOR
        private static void Autosave(UnityEditor.PlayModeStateChange state) => Instance.model.Save();
#endif
        private void OnApplicationQuit() => Instance.model.Save();

        protected static void EndTrainingSession(string reason)
        {
            if (!Instance.ended)
            {
                ConsoleMessage.Info("Training Session Ended! " + reason);
                Instance.model.Save();
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#endif
            }

            Instance.ended = true;
        }


        /// <summary>
        ///  This part is dedicated for Parallel inference mode on learning. The states of all agents are concatenated and passed once
        ///  through the model for faster speed and reason to use GPU on inference. This does't work for inference when the agents are not
        ///  learning.
        /// </summary>

        int lastCallFixedUpdateFrame = -1;
        Dictionary<Agent, (Tensor, Tensor)> agentsContinuousActionsProbs = new();
        Dictionary<Agent, (Tensor, Tensor)> agentsDiscreteActionsProbs = new();
        public void ParallelInference(Agent ag, int lcf)
        {
            if(lastCallFixedUpdateFrame == lcf)
            {
               
                if (model.IsUsingContinuousActions)
                {
                    var valuesc = agentsContinuousActionsProbs[ag];
                    ag.Timestep.action_continuous = valuesc.Item1;
                    ag.Timestep.prob_continuous = valuesc.Item2;
                }

               
                if (model.IsUsingDiscreteActions)
                {
                    var valuesd = agentsDiscreteActionsProbs[ag];
                    ag.Timestep.action_discrete = valuesd.Item1;
                    ag.Timestep.prob_discrete = valuesd.Item2;
                }
                             
            }
            else
            {
                lastCallFixedUpdateFrame = lcf;
                // PARALLEL OBSERVATION PROCESS ----------------------------
                for (int i = 0; i < parallelAgents.Count; i++)
                {
                    if (parallelAgents[i].LastState == null)
                        parallelAgents[i].Timestep.state = parallelAgents[i].GetState();
                    else
                        parallelAgents[i].Timestep.state = parallelAgents[i].LastState;
                }
                // PARALLEL OBSERVATION PROCESS ----------------------------


                // PARALLEL ACTION PROCESS ---------------------------------
                var allStates = parallelAgents.Where(x => x.behaviourType == BehaviourType.Learn).Select(x => x.Timestep.state).ToArray();
                Tensor stateBatch = Tensor.Concat(null, allStates);
                
                if(model.IsUsingContinuousActions)
                {
                    Tensor cactionBatch;
                    Tensor cprobBatch;
                    model.ContinuousEval(stateBatch, out cactionBatch, out cprobBatch);
                    Tensor[] continuousActionsBatch = Tensor.Split(cactionBatch, 0, 1);
                    Tensor[] continuousProbsBatch = Tensor.Split(cprobBatch, 0, 1);

                    int index = 0;
                    foreach (var agentx in parallelAgents)
                    {
                        agentsContinuousActionsProbs[agentx] = (continuousActionsBatch[index].Squeeze(0), continuousProbsBatch[index].Squeeze(0));
                        index++;
                    }

                    var valuesc = agentsContinuousActionsProbs[ag];
                    ag.Timestep.action_continuous = valuesc.Item1;
                    ag.Timestep.prob_continuous = valuesc.Item2;
                }


                if(model.IsUsingDiscreteActions)
                {
                    Tensor dactionBatch;
                    Tensor dprobBatch;
                    model.DiscreteEval(stateBatch, out dactionBatch, out dprobBatch);
                    Tensor[] discreteActionsBatch = Tensor.Split(dactionBatch, 0, 1);
                    Tensor[] discreteProbsBatch = Tensor.Split(dprobBatch, 0, 1);

                    int index = 0;
                    foreach (var agentx in parallelAgents)
                    {
                        agentsDiscreteActionsProbs[agentx] = (discreteActionsBatch[index].Squeeze(0), discreteProbsBatch[index].Squeeze(0));
                        index++;
                    }

                    var valuesc = agentsDiscreteActionsProbs[ag];
                    ag.Timestep.action_discrete = valuesc.Item1;
                    ag.Timestep.prob_discrete = valuesc.Item2;
                }
                // PARALLEL ACTION PROCESS ---------------------------------
            }
        }
    }
}

