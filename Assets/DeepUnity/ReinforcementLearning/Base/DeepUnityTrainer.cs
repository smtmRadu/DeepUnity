using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Unity.VisualScripting;
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

        public event EventHandler OnTrainingSessionEnd;

        [ViewOnly] public List<Agent> parallelAgents;
        [ViewOnly] public Hyperparameters hp;
        [ViewOnly] public AgentBehaviour model;
        [ViewOnly] public ExperienceBuffer train_data;
        [ViewOnly] public Stopwatch updateClock;
        [ViewOnly] public int currentSteps = 0;
        [ViewOnly] public int updateIterations;
        [ViewOnly] public float actorLoss;
        [ViewOnly] public float criticLoss;
        [ViewOnly] public float entropy;
        [ViewOnly] public float learningRate;


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
                // trainingSound = Resources.Load<AudioClip>("Audio/TrainingSound1");
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
        }
        private void Update()
        {
            avgDeltaTime = avgDeltaTime * avgDeltaTimeMomentum + Time.deltaTime * (1f - avgDeltaTimeMomentum);
            _runtimeStatsText = $"[Timescale: {Time.timeScale.ToString("0.0")} | No. agents {parallelAgents.Count}]";
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
    }
}

