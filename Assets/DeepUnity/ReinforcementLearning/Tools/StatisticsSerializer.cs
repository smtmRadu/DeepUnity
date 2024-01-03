using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, RequireComponent(typeof(Agent))]
    public class StatisticsSerializer : MonoBehaviour
    {
        private Agent ag;
        public LinkedList<(float, float)> cumulativeReward = new();
        public LinkedList<(float, float)> episodeLength = new();

        bool subscribed = false;

        private void FixedUpdate()
        {
            if (!subscribed && DeepUnityTrainer.Instance)
            {
                ag = GetComponent<Agent>();
                ag.OnEpisodeEnd += Collect;


                DeepUnityTrainer.Instance.OnTrainingSessionEnd += GetRewardCoordintes;
                DeepUnityTrainer.Instance.OnTrainingSessionEnd += GetEpisodeCoordintes;
                subscribed = true;
            }
        }
        public void Collect(object Sender, EventArgs e)
        {
            cumulativeReward.AddLast((ag.PerformanceTrack.stepCount, ag.EpsiodeCumulativeReward));
            episodeLength.AddLast((ag.PerformanceTrack.stepCount, ag.EpisodeStepCount));
        }
        public void GetRewardCoordintes(object Sender, EventArgs e)
        {
            StringBuilder sb = new StringBuilder();

            foreach (var item in cumulativeReward)
            {
                sb.Append($"({item.Item1}, {item.Item2})\n");
            }

            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string filePath = Path.Combine(desktopPath, "cumulative_reward.txt");

            using (StreamWriter sw = new StreamWriter(filePath))
            {
                sw.Write(sb.ToString());
            }
        }

        public void GetEpisodeCoordintes(object Sender, EventArgs e)
        {
            StringBuilder sb = new StringBuilder();

            foreach (var item in episodeLength)
            {
                sb.Append($"({item.Item1}, {item.Item2})\n");
            }


            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string filePath = Path.Combine(desktopPath, "episode_length.txt");

            using (StreamWriter sw = new StreamWriter(filePath))
            {
                sw.Write(sb.ToString());
            }
        }
    }

}


