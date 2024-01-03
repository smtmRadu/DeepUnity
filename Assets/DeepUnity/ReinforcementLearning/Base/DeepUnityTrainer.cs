using System;
using System.Collections.Generic;
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
        public static int TrainingBufferCount { get => Instance.train_data.Count; }

        public event EventHandler OnTrainingSessionEnd;

        [ReadOnly, SerializeField] protected List<Agent> parallelAgents;
        [ReadOnly, SerializeField] protected Hyperparameters hp;
        [ReadOnly, SerializeField] protected TrainingStatistics track;
        [ReadOnly, SerializeField] protected AgentBehaviour model;
        [ReadOnly, SerializeField] protected ExperienceBuffer train_data;


        [SerializeField, Min(1)]   protected int autosave = 15; protected float autosaveSecondsElapsed = 0f;
        [SerializeField, ReadOnly] protected int currentSteps = 0;
                                   protected bool ended = false;

        protected readonly DateTime timeWhenTheTrainingStarted = DateTime.Now;

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
        protected virtual void FixedUpdate()
        {
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

            // Updating the training statistics
            if (track != null)
            {
                TimeSpan timeElapsed = DateTime.Now - timeWhenTheTrainingStarted;
                track.trainingSessionTime =
                    $"{(int)timeElapsed.TotalHours} hrs : {(int)timeElapsed.TotalMinutes % 60} min : {(int)timeElapsed.TotalSeconds % 60} sec";


                track.inferenceSecondsElapsed += Time.fixedDeltaTime;
                track.inferenceTime =
                    $"{(int)(Math.Ceiling(track.inferenceSecondsElapsed * parallelAgents.Count) / 3600)} hrs : {(int)(Math.Ceiling(track.inferenceSecondsElapsed * parallelAgents.Count) % 3600 / 60)} min : {(int)(Math.Ceiling(track.inferenceSecondsElapsed * parallelAgents.Count) % 60)} sec";
                // track.inferenceTimePerAgent =
                //     $"{(int)(Math.Ceiling(track.inferenceSecondsElapsed) / 3600)} hrs : {(int)(Math.Ceiling(track.inferenceSecondsElapsed) % 3600 / 60)} min : {(int)(Math.Ceiling(track.inferenceSecondsElapsed) % 60)} sec";
            }


           
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

        protected virtual void Initialize() { }
        public static void Subscribe(Agent agent, TrainerType trainer)
        {
            if(Instance == null)
            {
                EditorApplication.playModeStateChanged += Autosave1;
                EditorApplication.pauseStateChanged += Autosave2;
                GameObject go = new GameObject($"[DeepUnity] Trainer - {trainer}");        
                
                switch(trainer)
                {
                    case TrainerType.PPO:
                        Instance = go.AddComponent<PPOTrainer>();
                        break;
                    case TrainerType.SAC:
                        Instance = go.AddComponent<SACTrainer>();
                        break;
                    case TrainerType.GAIL:
                        Instance = go.AddComponent<GAILTrainer>();
                        break;
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
                x.PerformanceTrack = agent.PerformanceTrack;
                x.DecisionRequester.decisionPeriod = agent.DecisionRequester.decisionPeriod;
                x.DecisionRequester.maxStep = agent.DecisionRequester.maxStep;
                x.DecisionRequester.takeActionsBetweenDecisions = agent.DecisionRequester.takeActionsBetweenDecisions;
            });

            if (agent.PerformanceTrack != null)
            {
                Instance.track = agent.PerformanceTrack;
            }

            Instance.parallelAgents.Add(agent);
        }
        private static void Autosave1(PlayModeStateChange state)
        {
            Instance.model.Save();
            if (state == PlayModeStateChange.ExitingPlayMode && Instance.track != null)
            {
                Instance.track.startedAt = Instance.timeWhenTheTrainingStarted.ToLongTimeString() + ", " + Instance.timeWhenTheTrainingStarted.ToLongDateString();
                Instance.track.finishedAt = DateTime.Now.ToLongTimeString() + ", " + DateTime.Now.ToLongDateString();

                

                if (Instance.track.iterations > 0)
                {
                    string pth = Instance.track.ExportAsSVG(Instance.model.behaviourName, Instance.hp, Instance.model, Instance.parallelAgents[0].DecisionRequester);
                    UnityEngine.Debug.Log($"<color=#57f542>Training Session log saved at <b><i>{pth}</i></b>.</color>");
                    AssetDatabase.Refresh();
                }
            }
        }
        private static void Autosave2(PauseState state) => Instance?.model.Save();
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

