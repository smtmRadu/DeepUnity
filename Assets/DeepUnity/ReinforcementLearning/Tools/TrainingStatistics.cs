using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;
using System.IO;
using System.Linq;

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
        public string ExportAsSVG(string behaviourName)
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

            string name = $"TrainingSession{extra}";

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
            File.WriteAllText(path, GenerateSVG(behaviourName));
            return path;
        }
        private string GenerateSVG(string behaviourName)
        {
            StringBuilder svgBuilder = new StringBuilder();

            svgBuilder.AppendLine(@"<svg width=""1000"" height=""3000"" xmlns=""http://www.w3.org/2000/svg"">");
            int y = 20;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""16"" fill=""black""> [{behaviourName}] Training Session</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Started at: " + startedAt + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Finished at: " + finishedAt + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Real Training Time: " + realTrainingTime + @"</text>");
            y += 20;
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time: " + inferenceTime + @"</text>");
            y += 20;                                   
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time Per Agent: " + inferenceTimePerAgent + @"</text>");
            y += 20;                                  
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Policy Optimization Time: " + policyUpdateTime + @"</text>");
            y += 20;                                   
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Policy Optimization Time / Iteration: " + policyUpdateTimePerIteration + @"</text>");
            y += 20;                                   
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Inference Time Per Agent / Real Training Time (ratio): " + inferenceTimeRatio + @"</text>");
            y += 20;                                   
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Policy Update Time / Real Training Time (ratio): " + policyUpdateTimeRatio + @"</text>");
            y += 20;                                  
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Iterations: " + iterations + @"</text>");
            y += 20;                                   
            svgBuilder.AppendLine($@"<text x=""10"" y=""{y}"" font-family=""Arial"" font-size=""12"" fill=""black"">Parallel Agents: " + parallelAgents + @"</text>");
            y += 40;

            float graphYPosition = y;

            DrawGraph(svgBuilder, episodeReward.Keys, ref graphYPosition, 200, 200, "Episode Reward");
            DrawGraph(svgBuilder, episodeLength.Keys, ref graphYPosition, 200, 200, "Episode Length");
            DrawGraph(svgBuilder, policyLoss.Keys, ref graphYPosition, 200, 200, "Policy Loss");
            DrawGraph(svgBuilder, valueLoss.Keys, ref graphYPosition, 200, 200, "Value Loss");
            DrawGraph(svgBuilder, learningRate.Keys, ref graphYPosition, 200, 200, "Learning Rate");
            DrawGraph(svgBuilder, epsilon.Keys, ref graphYPosition, 200, 200, "Epsilon");

            svgBuilder.AppendLine(@"</svg>");

            return svgBuilder.ToString();
        }
        private static void DrawGraph(StringBuilder svgBuilder, Keyframe[] keyFrames, ref float yOffset, int height, int width, string title)
        {

            if (keyFrames == null || keyFrames.Length == 0 || yOffset < 0 || height <= 0 || width <= 0)
            {
                return;
            }

                   
            svgBuilder.AppendLine($@"<text x=""10"" y=""{yOffset}"" font-family=""Arial"" font-size=""12"" fill=""black"">" + title + @"</text>");
            yOffset += 20;

            float maxY = keyFrames.Max(x => x.value);
            float minY = keyFrames.Min(x => x.value);

            // float maxX = keyFrames.Max(x => x.time);
            // float minX = keyFrames.Min(x => x.time);
            // float dif = maxY - minY != 0 ? maxY - minY : Utils.EPSILON;
            // float zeroline = yOffset + height * maxY / dif ;
            // svgBuilder.AppendLine($@"<path d=""M10,{zeroline} L{width + 10},{zeroline}"" stroke=""gray"" />");
            // svgBuilder.AppendLine($@"<text x=""5"" y=""{zeroline + 5}"" font-family=""Arial"" font-size=""10"" fill=""black"">0</text>");


            svgBuilder.AppendLine($@"<text x=""5"" y=""{yOffset - 5}"" font-family=""Arial"" font-size=""10"" fill=""black"">{maxY}</text>");
            svgBuilder.AppendLine($@"<text x=""5"" y=""{yOffset + height+10}"" font-family=""Arial"" font-size=""10"" fill=""black"">{minY}</text>");


            StringBuilder svgGraph = new StringBuilder();
            for (int i = 0; i < keyFrames.Length; i++)
            {
                Keyframe keyframe = keyFrames[i];
                float x = keyframe.time;
                float y = keyframe.value;

                float svgX = x * width + 5;
                svgX *= 2;
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


            yOffset += height + 30f;
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


