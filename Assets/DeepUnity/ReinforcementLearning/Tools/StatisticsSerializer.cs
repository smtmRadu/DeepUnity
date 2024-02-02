using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, RequireComponent(typeof(Agent))]
    public class StatisticsSerializer : MonoBehaviour
    {
        private StatisticsSerializer Instance;

        public void Awake()
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

        private void Start()
        {
            DeepUnityTrainer.Instance.OnTrainingSessionEnd += GetRewardCoordintes;
            DeepUnityTrainer.Instance.OnTrainingSessionEnd += GetEpisodeCoordintes;
        }

        public void GetRewardCoordintes(object Sender, EventArgs e)
        {
            StringBuilder sb = new StringBuilder();

            foreach (var item in GetComponent<TrainingStatistics>().cumulativeReward.Keys)
            {
                sb.Append($"({item.time}, {item.value})\n");
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

            foreach (var item in GetComponent<TrainingStatistics>().episodeLength.Keys)
            {
                sb.Append($"({item.time}, {item.value})\n");
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


