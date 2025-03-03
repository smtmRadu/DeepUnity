﻿using System.Text;
using UnityEngine;
using System.IO;
using System.Linq;
using UnityEditor;
using System;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Note: TrainingStatistics Track ram usage increases over time. Planning long training sessions may require more RAM memory.
    /// </summary>
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Training Statistics"), ExecuteInEditMode] // last is for to be called before agent script
    public class TrainingStatistics : MonoBehaviour
    {
        private static bool MainSaved = false;
        private TrainingStatistics Instance;

        [ViewOnly, Tooltip("When the simulation started. (real-time)")]
        public string startedAt = " - ";
        [ViewOnly, Tooltip("When the simulation ended. (real-time)")]
        public string finishedAt = " - ";
        [ViewOnly, Tooltip("How much time passed since the start of the training simulation. (real-time)")]
        public string timeOnSession = " - ";   
        [ViewOnly, Tooltip("Total time spent on policy update. (real-time)")]
        public string timeOnPolicyUpdate = " - ";
        [ViewOnly, Tooltip("How much time took the last policy update iteration. (real-time)")]
        public string timeOnLastPolicyUpdateIter = " - ";
        [ViewOnly, Tooltip("Cumulated time on inference in total. (~real-time)")]
        public string cumulatedInferenceTime = " - ";

        [Space(20)]
        [ViewOnly, Tooltip("Total number of episodes runned by all parallel agents.")]
        public int episodeCount = 0;
        [ViewOnly, Tooltip("Total numbers of steps runned by all parallel agents.")]
        public int stepCount = 0;
        [ViewOnly, Tooltip("How many policy updates were made.")]
        public int iterations = 0;
        [ViewOnly, Tooltip("Parallel agents learning. If this is not equal to your environments, some of them are not having the behaviour to learn.")]
        public int parallelAgents = 0;

        [Space(20)]
        [Header("Environment")]
        [Tooltip("Cumulated reward on each episode.")]
        public PerformanceGraph cumulativeReward = new PerformanceGraph();
        [Tooltip("Steps taken in each episode.")]
        public PerformanceGraph episodeLength = new PerformanceGraph();

        [Header("Losses")]
        [Tooltip("Mean loss of policy function on each epoch")]
        public PerformanceGraph actorLoss = new PerformanceGraph();
        [Tooltip("Mean MSE of Value (for PPO) or Q (for SAC) function on each epoch. Also used for the discriminator loss in Heuristic Training.")]
        public PerformanceGraph criticLoss = new PerformanceGraph();

        [Header("Policy")]
        [Tooltip("The mean standard deviation of the policy for continuous actions, or -probs * log probs for discrete actions.")]
        public PerformanceGraph entropy = new PerformanceGraph();


        private float policyUpdateSecondsElapsed = 0f;
        private float inferenceSecondsElapsed = 0f; // Updated via deltaTime
        private bool collectAgentStuff = false;
        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else if (GetComponent<Agent>().behaviourType == BehaviourType.Learn)
            {
                Instance = this;
#if UNITY_EDITOR
                EditorApplication.playModeStateChanged += Instance.ExportOnEnd;
#endif
            }
        }

        private void FixedUpdate()
        {
            // Training statistics is disabled completely on build
#if !UNITY_EDITOR
            return;
#endif
            if (DeepUnityTrainer.Instance == null)
                return;

            if (!collectAgentStuff)
            {
                try
                {
                    var ags = DeepUnityTrainer.Instance.parallelAgents;
                    foreach (var item in ags)
                    {
                        item.OnEpisodeEnd += UpdateAgentStuff;
                    }
                    collectAgentStuff = true;
                }
                catch { }
            }


            stepCount = DeepUnityTrainer.Instance.currentSteps;
            parallelAgents = DeepUnityTrainer.Instance.parallelAgents.Count;
            inferenceSecondsElapsed += Time.fixedDeltaTime;
            cumulatedInferenceTime = $"{(int)(Math.Ceiling(inferenceSecondsElapsed * DeepUnityTrainer.Instance.parallelAgents.Count) / 3600)} hrs : {(int)(Math.Ceiling(inferenceSecondsElapsed * DeepUnityTrainer.Instance.parallelAgents.Count) % 3600 / 60)} min : {(int)(Math.Ceiling(inferenceSecondsElapsed * DeepUnityTrainer.Instance.parallelAgents.Count) % 60)} sec";


            if (DeepUnityTrainer.Instance.updateIterations > iterations)
                UpdateTrainerStuff();
        }
        private void UpdateAgentStuff(object sender, EventArgs e)
        {
            Agent ag = (Agent)sender;
            episodeCount++;
            episodeLength.Append(ag.EpisodeStepCount);
            cumulativeReward.Append(ag.EpisodeCumulativeReward + ag.Timestep.reward[0]); // Why this? because OnEpisodeEnd is called before the last reward to be added.
        }
        private void UpdateTrainerStuff()
        {
            iterations++;

            TimeSpan timeElapsed = DateTime.Now - DeepUnityTrainer.Instance.timeWhenTheTrainingStarted;
            timeOnSession = $"{(int)timeElapsed.TotalHours} hrs : {(int)timeElapsed.TotalMinutes % 60} min : {(timeElapsed.TotalSeconds % 60).ToString("0.00")} sec";
            policyUpdateSecondsElapsed += (float)DeepUnityTrainer.Instance.updateBenchmarkClock.Elapsed.TotalSeconds;
            timeOnPolicyUpdate = $"{(int)(Math.Ceiling(policyUpdateSecondsElapsed) / 3600)} hrs : {(int)(Math.Ceiling(policyUpdateSecondsElapsed) % 3600 / 60)} min : {(Math.Ceiling(policyUpdateSecondsElapsed) % 60).ToString("0.00")} sec";
            timeOnLastPolicyUpdateIter = $"{(int)DeepUnityTrainer.Instance.updateBenchmarkClock.Elapsed.TotalHours} hrs : {(int)DeepUnityTrainer.Instance.updateBenchmarkClock.Elapsed.TotalMinutes % 60} min : {(DeepUnityTrainer.Instance.updateBenchmarkClock.Elapsed.TotalSeconds % 60).ToString("0.00")} sec";

            actorLoss.Append(DeepUnityTrainer.Instance.actorLoss);
            criticLoss.Append(DeepUnityTrainer.Instance.criticLoss);
            entropy.Append(DeepUnityTrainer.Instance.entropy);
        }






        /// <summary>
        /// Returns the path of the file.
        /// </summary>
        /// <param name="behaviourName"></param>
        /// <returns></returns>
        internal string ExportAsSVG(string behaviourName, Hyperparameters hp, AgentBehaviour behaviour, DecisionRequester decisionRequester)
        {
            string path = "";
#if UNITY_EDITOR
            string behaviourAssetPath = AssetDatabase.GetAssetPath(behaviour);

            string extra = new string(startedAt.Select(x =>
            {
                if (char.IsLetterOrDigit(x))
                    return x;
                else if (char.IsWhiteSpace(x))
                    return '_';
                else if (x == ',')
                    return '_';
                else
                    return '-';
            }).ToArray());


            string directoryPath = Path.Combine(Path.GetDirectoryName(behaviourAssetPath), "Logs");
            string name = $"[{behaviourName}]_{extra}";
            path = Path.Combine(directoryPath, $"{name}.svg");


            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }


            File.Create(path).Dispose();
            File.WriteAllText(path, GenerateSVG(behaviourName, hp, behaviour, decisionRequester));
#endif
            return path;
        }
        private string GenerateSVG(string behaviourName, Hyperparameters hp, AgentBehaviour ab, DecisionRequester dr)
        {
            StringBuilder svgBuilder = new StringBuilder();

            svgBuilder.AppendLine(@"<svg width=""2000"" height=""2500"" xmlns=""http://www.w3.org/2000/svg"">");
            int y = 20;

            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""26"" fill=""black""> [<tspan font-weight=""bold"" fill=""blue"">{behaviourName}</tspan>] Training Session</text>");
            y += 30;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Statistics]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Started at: " + startedAt + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Finished at: " + finishedAt + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Training Session Time: " + timeOnSession + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time: {cumulatedInferenceTime}        [Device: {ab.inferenceDevice}]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Update Time: {timeOnPolicyUpdate}        [Per iteration: {timeOnLastPolicyUpdateIter}]        [Device: {ab.trainingDevice}]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Episode Count: {episodeCount}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Steps: {stepCount}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Iterations: " + iterations + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Parallel Agents: " + parallelAgents + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Timescale: " + Time.timeScale + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Trainer: " + hp.trainer + @"</text>");

            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Hyperparameters]</text>");
            y += 20;
            if (hp.trainer == TrainerType.PPO)
                svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Buffer Size: {hp.bufferSize}        [Batch Size: {hp.batchSize}       (x{hp.bufferSize / hp.batchSize})]</text>");
            else if (hp.trainer == TrainerType.SAC || hp.trainer == TrainerType.TD3 || hp.trainer == TrainerType.DDPG)
                svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Replay Buffer Size: {hp.replayBufferSize}        [Batch Size: {hp.minibatchSize}       (x{hp.replayBufferSize / hp.minibatchSize})]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Num Epoch: {hp.numEpoch}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Time Horizon: {hp.horizon}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Gradient Cliping by Norm: {hp.maxNorm}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Learning Rate Schedule: {hp.LRSchedule}    {""}</text>");

            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Behaviour]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Space Size: {ab.observationSize}</text>");
            y += 20;
            string std_value = ab.stochasticity == Stochasticity.FixedStandardDeviation ? $"[Value: {ab.standardDeviationValue}]" : "";
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Continuous Actions: {ab.continuousDim} [Standard Deviation: {ab.stochasticity}]     {std_value}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Discrete Actions: {ab.discreteDim}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">TargetFPS: {ab.targetFPS} (physics update rate)</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Observations Normalization: {ab.normalize}</text>");
            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Decision Requester]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Max Step: {dr.maxStep}</text>");
            y += 20;
            string decPer = dr.decisionPeriod == 1 ? "1" : $"{dr.decisionPeriod}      [Actions Between Decisions: {dr.takeActionsBetweenDecisions}]";
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Decision Period: {decPer}</text>");


            /// Generate graphs
            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Graphs]</text>");
            y += 20;
            DrawGraph(svgBuilder, cumulativeReward.Keys, ref y, 50, 200, 500, $"Episode Reward ({episodeCount})");
            y += 20;
            DrawGraph(svgBuilder, episodeLength.Keys, ref y, 50, 200, 500, $"Episode Length ({episodeCount})");
            y += 20;
            DrawGraph(svgBuilder, actorLoss.Keys, ref y, 50, 200, 500, "Actor Loss");
            y += 20;
            DrawGraph(svgBuilder, criticLoss.Keys, ref y, 50, 200, 500, "Critic Loss");
            y += 20;
            DrawGraph(svgBuilder, entropy.Keys, ref y, 50, 200, 500, "Entropy");

            // Generate smoothed out graphs
            const int beforeGraphs = 5;
            y -= 120;
            y -= 200 * beforeGraphs;
            y -= 50 * beforeGraphs;
            cumulativeReward.Smooth(0.05f);
            episodeLength.Smooth(0.05f);

            y += 20;
            DrawGraph(svgBuilder, cumulativeReward.Keys, ref y, 750, 200, 500, "Simplified");
            y += 20;
            DrawGraph(svgBuilder, episodeLength.Keys, ref y, 750, 200, 500, "Simplified");



            svgBuilder.AppendLine(@"</svg>");

            return svgBuilder.ToString();
        }
        private static void DrawGraph(StringBuilder svgBuilder, Keyframe[] keyFrames, ref int yOffset, float xOffset, int height, int width, string title)
        {

            if (keyFrames == null || keyFrames.Length == 0 || yOffset < 0 || height <= 0 || width <= 0)
            {
                return;
            }


            svgBuilder.AppendLine($@"<text x=""{xOffset}"" y=""{yOffset}"" font-family=""Arial"" font-size=""12"" fill=""black"">" + title + @"</text>");
            yOffset += 20;

            float maxY = keyFrames.Max(x => x.value);
            float minY = keyFrames.Min(x => x.value);

            svgBuilder.AppendLine($@"<text x=""{xOffset}"" y=""{yOffset - 5}"" font-family=""Arial"" font-size=""10"" fill=""black"">{maxY}</text>");
            svgBuilder.AppendLine($@"<text x=""{xOffset}"" y=""{yOffset + height + 10}"" font-family=""Arial"" font-size=""10"" fill=""black"">{minY}</text>");


            StringBuilder svgGraph = new StringBuilder();
            for (int i = 0; i < keyFrames.Length; i++)
            {
                Keyframe keyframe = keyFrames[i];
                float x = keyframe.time;
                float y = keyframe.value;

                float svgX = x * width + xOffset;
                float svgY = yOffset + height - height * (y - minY) / (maxY - minY + Utils.EPSILON);

                if (i == 0)
                {
                    svgGraph.Append($"M{svgX},{svgY} ");
                }
                else
                {
                    svgGraph.Append($"L{svgX},{svgY} ");
                }
            }
            svgBuilder.AppendLine($@"<path d=""{svgGraph}"" stroke=""blue"" fill=""none"" />");


            yOffset += height + 30;
        }
#if UNITY_EDITOR
        private void ExportOnEnd(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingPlayMode && DeepUnityTrainer.Instance != null) // this check is not mandatory (i inserted it to show less null ref exc when something happens when the assets of behavior are missing)
            {
                startedAt = DeepUnityTrainer.Instance.timeWhenTheTrainingStarted.ToLongTimeString() + ", " + DeepUnityTrainer.Instance.timeWhenTheTrainingStarted.ToLongDateString();
                finishedAt = DateTime.Now.ToLongTimeString() + ", " + DateTime.Now.ToLongDateString();

                if (iterations > 0 && !MainSaved)
                {
                    string pth = ExportAsSVG(DeepUnityTrainer.Instance.model.behaviourName, DeepUnityTrainer.Instance.hp, DeepUnityTrainer.Instance.model, DeepUnityTrainer.Instance.parallelAgents[0].DecisionRequester);
                    Debug.Log($"<color=#57f542>Training Session log saved at <b><i>{pth}</i></b>.</color>");
                    AssetDatabase.Refresh();
                    MainSaved = true;
                }
            }

        }
#endif
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(TrainingStatistics)), CanEditMultipleObjects]
    class CustomAgentPerformanceTrackerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            string[] dontDrawMe = new string[] { "m_Script" };

            /*if(EditorApplication.isPlaying)
            {
                float sessionProgress = script.stepCount / ((float)PPOTrainer.SessionMaxSteps) * 100f;
                StringBuilder sb = new StringBuilder();
                sb.Append("Progress [");
                sb.Append(script.stepCount);
                sb.Append(" / ");
                sb.Append(PPOTrainer.SessionMaxSteps);
                sb.Append($"] \n[");
                for (float i = 1.25f; i <= 100f; i += 1.25f)
                {
                    if (i == 47.5f)
                        sb.Append($"{sessionProgress.ToString("00.0")}%");
                    else if (i > 47.5f && i <= 53.75f)
                        continue;
                    else if (i <= sessionProgress)
                        sb.Append("▮");
                    else
                        sb.Append("▯");
                }
                sb.Append("]");
                EditorGUILayout.HelpBox(sb.ToString(), MessageType.None);
                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            }
            */

            DrawPropertiesExcluding(serializedObject, dontDrawMe);

            // Depending on the version, Performance Graph may require or not increasingly needed RAM memory.
            if (EditorApplication.isPlaying)
            {
                EditorGUILayout.HelpBox("Training Statistics may require considerable free RAM for overnight training sessions.", MessageType.Info);
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
#endif
}


