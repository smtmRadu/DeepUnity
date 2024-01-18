using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public abstract class DeepUnityTrainer : MonoBehaviour
    {
        public static DeepUnityTrainer Instance;
        

        /// <summary>
        /// Current experiences collected by the agents.
        /// </summary>
        public static int MemoriesCount { get => Instance.parallelAgents.Sum(x => x.Memory.Count); }

        public event EventHandler OnTrainingSessionEnd;

        [ReadOnly] public List<Agent> parallelAgents;
        [ReadOnly] public Hyperparameters hp;
        [ReadOnly] public AgentBehaviour model;
        [ReadOnly] public ExperienceBuffer train_data;
        [ReadOnly] public Stopwatch updateClock;
        [ReadOnly] public int currentSteps = 0;
        [ReadOnly] public int updateIterations;
        [ReadOnly] public float actorLoss;
        [ReadOnly] public float criticLoss;
        [ReadOnly] public float entropy;
        [ReadOnly] public float learningRate;


        public readonly DateTime timeWhenTheTrainingStarted = DateTime.Now;
        [Min(1)]   public int autosave = 15; protected float autosaveSecondsElapsed = 0f;
        [ReadOnly] public bool ended = false;

        [SerializeField] private float avgDeltaTime = 0.02f;
        const float avgDeltaTimeMomentum = 0.96f;
        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;
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
        }
        private void Update()
        {
            avgDeltaTime = avgDeltaTime * avgDeltaTimeMomentum + Time.deltaTime * (1f - avgDeltaTimeMomentum);
        }


        protected abstract void Initialize();
        protected abstract void OnBeforeFixedUpdate();
        
        public static void Subscribe(Agent agent, TrainerType trainer)
        {
            if(Instance == null)
            {
                EditorApplication.playModeStateChanged += Autosave;
                GameObject go = new GameObject($"[DeepUnity] Trainer - {trainer}");        
                
                switch(trainer)
                {
                    case TrainerType.PPO:
                        Instance = go.AddComponent<PPOTrainer>();
                        break;
                    case TrainerType.SAC:
                        Instance = go.AddComponent<SACTrainer>();
                        break;
                     // case TrainerType.GAIL:
                     //     Instance = go.AddComponent<GAILTrainer>();
                     //   break;
                    default: throw new ArgumentException("Unhandled trainer type");
                }

                
                Instance.parallelAgents = new();           
                Instance.hp = agent.model.config;
                Instance.train_data = new ExperienceBuffer(Instance.hp.bufferSize);
                Instance.model = agent.model;
                Instance.model.InitOptimisers(Instance.hp, trainer);
                Instance.model.InitSchedulers(Instance.hp, trainer);
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
        private static void Autosave(PlayModeStateChange state) => Instance.model.Save();
        protected static void EndTrainingSession(string reason)
        {
            if (!Instance.ended)
            {
                ConsoleMessage.Info("Training Session Ended! " + reason);
                Instance.model.Save();
                EditorApplication.isPlaying = false;
            }

            Instance.ended = true;
        }
    }
}

