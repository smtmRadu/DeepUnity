using DeepUnity;
using DeepUnity.Models;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class CNNTesting : MonoBehaviour
    {
        [SerializeField] Sequential net;
        [SerializeField] private int batch_size = 64;
        public PerformanceGraph performanceGraph;

        List<(Tensor, Tensor)> test;
        int totalCorrect = 0;
        int totalSamples = 0;
        public void Start()
        {
            performanceGraph = new();
            Datasets.MNIST(null, out _, out test, DatasetSettings.LoadTestOnly);
            net.Device = Device.GPU;
            Utils.Shuffle(test);
            print($"Parameters: {net.Parameters().Sum(x => x.param.Count())}");
        }

        private void Update()
        
        {
            if (test.Count == 0)
                return;

            int count = Mathf.Min(batch_size, test.Count);
            Tensor inputs = Tensor.Concat(null, test.GetRange(0, count).Select(x => x.Item1).ToArray());
            Tensor targets = Tensor.Concat(null, test.GetRange(0, count).Select(x => x.Item2).ToArray());
            var output = net.Predict(inputs);
            var acc = Metrics.Accuracy(output, targets);
            totalCorrect += Mathf.RoundToInt(acc * count);
            totalSamples += count;
            performanceGraph.Append(acc);
            test.RemoveRange(0, count);

            if (test.Count == 0)
                UnityEngine.Debug.Log($"Test finished. Final accuracy: {(float)totalCorrect / totalSamples * 100f:F2}% ({totalCorrect}/{totalSamples})");
        }

    }
}



