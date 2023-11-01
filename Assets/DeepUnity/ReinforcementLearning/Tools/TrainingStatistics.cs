using System.Text;
using UnityEngine;
using System.IO;
using System.Linq;

namespace DeepUnity
{
    /// <summary>
    /// Note: TrainingStatistics Track ram usage increases over time. Planning long training sessions may require more RAM memory.
    /// </summary>
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Training Statistics")]
    public class TrainingStatistics : MonoBehaviour
    {
        [ReadOnly, Tooltip("When the simulation started.")]
        public string startedAt = " - ";
        [ReadOnly, Tooltip("When the simulation ended.")]
        public string finishedAt = " - ";
        [ReadOnly, Tooltip("How much time passed in real time since the start of the training simulation.")] 
        public string trainingSessionTime = " - ";

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
        [ReadOnly, Tooltip("inference time / training session time")]
        public string inferenceTimeRatio = "- / -";
        [ReadOnly, Tooltip("policy update time / training session time")]
        public string policyUpdateTimeRatio = "- / -";

        [Space(20)]
        [ReadOnly, Tooltip("Total number of episodes runned by all parallel agents.")]
        public int episodeCount = 0;
        [ReadOnly, Tooltip("Total numbers of steps runned by all parallel agents.")] 
        public int stepCount = 0;
        [ReadOnly, Tooltip("How many policy updates were made.")] 
        public int iterations = 0;
        [ReadOnly, Tooltip("Parallel agents learning. If this is not equal to your environments, some of them are not having the behaviour to learn.")] 
        public int parallelAgents = 0;

        


        [Space(20)]
        [Header("Environment")]
        [Tooltip("Cumulated reward on each episode.")] 
        public PerformanceGraph cumulativeReward = new PerformanceGraph();
        [Tooltip("Steps required in each episode.")] 
        public PerformanceGraph episodeLength = new PerformanceGraph();

        [Header("Losses")]
        [Tooltip("Mean loss of policy function on each epoch")] 
        public PerformanceGraph policyLoss = new PerformanceGraph();
        [Tooltip("Mean MSE of value function on each epoch. Also used for the discriminator loss in Heuristic Training.")]
        public PerformanceGraph valueLoss = new PerformanceGraph();

        [Header("Policy")]
        [Tooltip("The entropy H of the policy")]
        public PerformanceGraph entropy = new PerformanceGraph();
        [Tooltip("Learning rate decay on each epoch.")]
        public PerformanceGraph learningRate = new PerformanceGraph();
        

        /// <summary>
        /// Returns the path of the file.
        /// </summary>
        /// <param name="behaviourName"></param>
        /// <returns></returns>
        public string ExportAsSVG(string behaviourName, Hyperparameters hp, AgentBehaviour behaviour, DecisionRequester decisionRequester)
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
                ConsoleMessage.Info($"{behaviourName} behaviour folder has been moved from Assets folder, or it's name was changed! The training session log was saved in Assets/Logs in consequence.");

                if (!Directory.Exists($"Assets/Logs"))
                    Directory.CreateDirectory($"Assets/Logs");

                path = $"Assets/Logs/{name}.svg";
            }
            else if (!Directory.Exists($"Assets/{behaviourName}/Logs"))
            {
                Directory.CreateDirectory($"Assets/{behaviourName}/Logs");
            }

            File.Create(path).Dispose();
            File.WriteAllText(path, GenerateSVG(behaviourName, hp, behaviour, decisionRequester));
            return path;
        }
        private string GenerateSVG(string behaviourName, Hyperparameters hp, AgentBehaviour ab, DecisionRequester dr)
        {
            StringBuilder svgBuilder = new StringBuilder();

            svgBuilder.AppendLine(@"<svg width=""2000"" height=""2500"" xmlns=""http://www.w3.org/2000/svg"">");
            int y = 20;
           
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""26"" fill=""black""> [{behaviourName}] Training Session</text>");
            y += 30;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Statistics]</text>");     
            y += 20;                           
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Started at: " + startedAt + @"</text>");
            y += 20;                          
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Finished at: " + finishedAt + @"</text>");
            y += 20;                          
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Training Session Time: " + trainingSessionTime + @"</text>");
            y += 20;                          
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time: {inferenceTime}       [Per agent: {inferenceTimePerAgent}]        [Device: {ab.inferenceDevice}]</text>");       
            y += 20;                                   
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Policy Update Time: {policyUpdateTime}        [Per iteration: {policyUpdateTimePerIteration}]        [Device: {ab.trainingDevice}]</text>");   
            y += 20;                                  
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time / Training Session Time: " + inferenceTimeRatio + @"</text>");
            y += 20;                                 
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Policy Update Time / Training Session Time: " + policyUpdateTimeRatio + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Episode Count: {episodeCount}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Steps: {stepCount}    [Per agent: {stepCount / parallelAgents}]     [Per Episode (mean): {stepCount / episodeCount}]</text>");
            y += 20;                                
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Iterations: " + iterations + @"</text>");
            y += 20;                                  
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Parallel Agents: " + parallelAgents + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Timescale: " + Time.timeScale + @"</text>");

            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Hyperparameters]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Buffer Size: {hp.bufferSize}        [Batch Size: {hp.batchSize}       (x{hp.bufferSize / hp.batchSize})]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Num Epoch: {hp.numEpoch}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Time Horizon: {hp.horizon}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Gradient Cliping Normalization: {hp.gradClipNorm}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Advantages Normalization: {hp.normalizeAdvantages}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Training Data Shuffle: {hp.shuffleTrainingData}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Learning Rate Schedule: {hp.LRSchedule}    {""}</text>");

            y += 50;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black"">[Behaviour]</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Space Size: {ab.observationSize}</text>");
            y += 20;
            string std_value = ab.standardDeviation == StandardDeviationType.Fixed ?  $"[Value: {ab.standardDeviationValue}]" : "";
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Continuous Actions: {ab.continuousDim} [Standard Deviation: {ab.standardDeviation}]     {std_value}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Discrete Actions: {ab.discreteDim}</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">TargetFPS: {ab.targetFPS} (physics update rate)</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""50"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Observations Normalization: {ab.normalizeObservations}</text>");
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
            DrawGraph(svgBuilder, policyLoss.Keys, ref y, 50, 200, 500, "Policy Loss");
            y += 20;
            DrawGraph(svgBuilder, valueLoss.Keys, ref y, 50, 200, 500, "Value Loss");
            y += 20;
            DrawGraph(svgBuilder, learningRate.Keys, ref y, 50, 200, 500, "Learning Rate");
            y += 20;
            DrawGraph(svgBuilder, entropy.Keys, ref y, 50, 200, 500, "Entropy");

            // Generate smoothed out graphs
            y -= 120;
            y -= 200 * 6;
            y -= 50 * 6;
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
}


