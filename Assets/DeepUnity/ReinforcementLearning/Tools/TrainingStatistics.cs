using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;
using System.IO;
using System.Linq;
using Unity.VisualScripting;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Training Statistics")]
    public class TrainingStatistics : MonoBehaviour
    {
        [ReadOnly, Tooltip("When the simulation started.")]
        public string startedAt = " - ";
        [ReadOnly, Tooltip("When the simulation ended.")]
        public string finishedAt = " - ";
        [ReadOnly, Tooltip("How much time passed in real time since the start of the training simulation.")] 
        public string realTrainingTime = " - ";

        [HideInInspector] public float policyUpdateSecondsElapsed = 0f;
        [HideInInspector] public float inferenceSecondsElapsed = 0f; // Updated via deltaTime

        [ReadOnly, Tooltip("Total time spent on inference in total.")]
        public string inferenceTime = " - ";
        [ReadOnly, Tooltip("Total time spent on inference per agent.")]
        public string inferenceTimePerAgent = " - ";

        [ReadOnly, Tooltip("Total time spent on policy update.")]
        public string policyUpdateTime = " - ";
        [ReadOnly, Tooltip("How much time takes a policy update iteration.")]
        public string policyUpdateTimePerIteration = " - ";
        
        

        [Space(20)] 
        [ReadOnly, Tooltip("inference time per agent / (inference time per agent + policy update time)")]
        public string inferenceTimeRatio = "- / -";
        [ReadOnly, Tooltip("policy update time / (inference time per agent + policy update time)")]
        public string policyUpdateTimeRatio = "- / -";

        [Space(20)]
        [ReadOnly, Tooltip("Total numbers of steps runned by all parallel agents.")] 
        public int totalSteps = 0;
        [ReadOnly, Tooltip("How many policy updates were made.")] 
        public int iterations = 0;
        [ReadOnly, Tooltip("Parallel agents learning. If this is not equal to your environments, some of them are not having the behaviour to learn.")] 
        public int parallelAgents = 0;

        


        [Space(20)]
        [Tooltip("Cumulated reward on each episode.")] public PerformanceGraph episodeReward = new PerformanceGraph();
        [Tooltip("Steps required in each episode.")] public PerformanceGraph episodeLength = new PerformanceGraph();
        [Header("Losses")]
        [Tooltip("Mean loss of policy function on each epoch")] public PerformanceGraph policyLoss = new PerformanceGraph();
        [Tooltip("Mean MSE of value function on each epoch")]public PerformanceGraph valueLoss = new PerformanceGraph();
        [Header("Policy")]
        public PerformanceGraph learningRate = new PerformanceGraph();
        public PerformanceGraph epsilon = new PerformanceGraph();

        /// <summary>
        /// Returns the path of the file.
        /// </summary>
        /// <param name="behaviourName"></param>
        /// <returns></returns>
        public string ExportAsSVG(string behaviourName, Hyperparameters hp, AgentBehaviour behaviour)
        {
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

            string name = $"[{behaviourName}]TrainingSession_{extra}";

            string path = $"Assets/{behaviourName}/Logs/{name}.svg";

            if (!Directory.Exists($"Assets/{behaviourName}"))
            {
                ConsoleMessage.Warning($"{behaviourName} behaviour folder has been moved from Assets folder, or it's name was changed! The training session log was saved in Assets/Logs in consequence.");

                if (!Directory.Exists($"Assets/Logs"))
                    Directory.CreateDirectory($"Assets/Logs");

                path = $"Assets/Logs/{name}.svg";
            }
            else if (!Directory.Exists($"Assets/{behaviourName}/Logs"))
            {
                Directory.CreateDirectory($"Assets/{behaviourName}/Logs");
            }

            File.Create(path).Dispose();
            File.WriteAllText(path, GenerateSVG(behaviourName, hp, behaviour));
            return path;
        }
        private string GenerateSVG(string behaviourName, Hyperparameters hp, AgentBehaviour ab)
        {
            StringBuilder svgBuilder = new StringBuilder();

            svgBuilder.AppendLine(@"<svg width=""1000"" height=""2500"" xmlns=""http://www.w3.org/2000/svg"">");
            int y = 20;
           
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""26"" fill=""black""> [{behaviourName}] Training Session</text>");
            y += 30;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Statistics]</text>");     
            y += 20;                           
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Started at: " + startedAt + @"</text>");
            y += 20;                          
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Finished at: " + finishedAt + @"</text>");
            y += 20;                          
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Real Training Time: " + realTrainingTime + @"</text>");
            y += 20;                          
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time: {inferenceTime}    [Per agent: {inferenceTimePerAgent}]    [Device: {ab.inferenceDevice}]</text>");       
            y += 20;                                   
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Policy Optimization Time: {policyUpdateTime}    [Per iteration: {policyUpdateTimePerIteration}]    [Device: {ab.trainingDevice}]</text>");   
            y += 20;                                  
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time Per Agent / Real Training Time [ratio]: " + inferenceTimeRatio + @"</text>");
            y += 20;                                 
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Policy Update Time / Real Training Time [ratio]: " + policyUpdateTimeRatio + @"</text>");
            y += 20;                          
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Steps: {totalSteps}    [Per agent: {totalSteps / parallelAgents}]</text>");
            y += 20;                                
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Iterations: " + iterations + @"</text>");
            y += 20;                                  
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Parallel Agents: " + parallelAgents + @"</text>");
            
            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Hyperparameters]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Buffer Size: {hp.bufferSize}    [Batch Size: {hp.batchSize}]    [No. Batches: {hp.bufferSize/hp.batchSize}]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Num Epoch: {hp.numEpoch}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Time Horizon: {hp.horizon}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Gradient Clipiing Normalization: {hp.gradClipNorm}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Advantages Normalization: {hp.normalizeAdvantages}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Training Data shuffle: {hp.shuffleTrainingData}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Learning Rate Schedule: {hp.learningRateSchedule}    [Step Size: {hp.schedulerStepSize}]    [Decay: {hp.schedulerDecay}]</text>");

            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Behaviour]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Space Size: {ab.observationSize}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Continuous Dim: {ab.continuousDim}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Discrete Branches: {ab.discreteBranches.Length}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">TargetFPS: {ab.targetFPS} (physics update rate)</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Observations Normalization: {ab.normalizeObservations}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Standard Deviation: {ab.standardDeviation}    [(Fixed) Standard Deviation Value: {ab.standardDeviationValue}]    [(Trainable) Standard Deviation Scale: {ab.standardDeviationScale}]</text>");

            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Graphs]</text>");
            y += 20;
            DrawGraph(svgBuilder, episodeReward.Keys, ref y, 50, 200, 500, "Episode Reward");
            DrawGraph(svgBuilder, episodeLength.Keys, ref y, 50, 200, 500, "Episode Length");
            DrawGraph(svgBuilder, policyLoss.Keys, ref y, 50, 200, 500, "Policy Loss");
            DrawGraph(svgBuilder, valueLoss.Keys, ref y, 50, 200, 500, "Value Loss");
            DrawGraph(svgBuilder, learningRate.Keys, ref y, 50, 200, 500, "Learning Rate");
            DrawGraph(svgBuilder, epsilon.Keys, ref y, 50, 200, 500, "Epsilon");

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
                float svgY = yOffset + height -  height * (y - minY) / (maxY - minY + Utils.EPSILON);

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
    }









    [CustomEditor(typeof(TrainingStatistics)), CanEditMultipleObjects]
    class CustomAgentPerformanceTrackerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            List<string> dontDrawMe = new List<string>() { "m_Script" };


            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());

          
            serializedObject.ApplyModifiedProperties();
        }
    }

}


