using DeepUnity;
using DeepUnity.Models;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class CNNTesting : MonoBehaviour
    {
        [SerializeField] Sequential net;
        public PerformanceGraph performanceGraph;
        
        List<(Tensor, Tensor)> test;
        public void Start()
        {
            performanceGraph = new();
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out _, out test, DatasetSettings.LoadTestOnly);
            net.Device = Device.GPU;
            Utils.Shuffle(test);
            print($"Parameters: {net.Parameters().Sum(x => x.theta.Count())}");
        }

        private void Update()
        
        {
            if (test.Count == 0)
                return;

            const int batch_size = 64;
            Tensor inputs = Tensor.Concat(null, test.GetRange(0, Mathf.Min(batch_size, test.Count)).Select(x => x.Item1).ToArray());
            Tensor targets = Tensor.Concat(null, test.GetRange(0, Mathf.Min(batch_size, test.Count)).Select(x => x.Item2).ToArray());
            var output = net.Predict(inputs);
            var acc =  Metrics.Accuracy(output, targets);
            performanceGraph.Append(acc);
            test.RemoveRange(0, Mathf.Min(batch_size, test.Count));
        }

    }
}



