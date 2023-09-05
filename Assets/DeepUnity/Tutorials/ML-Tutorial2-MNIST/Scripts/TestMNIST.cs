using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


namespace DeepUnityTutorials
{
    public class TestMNIST : MonoBehaviour
    {
        [SerializeField] private NeuralNetwork network;
        [SerializeField] Device inferenceDevice = Device.CPU;

        List<(Tensor, Tensor)> test = new();
        public string completed = "0/0";
        public string accuracy = "0%";

        private int[] right = new int[10];
        private int[] wrong = new int[10];
        public List<float> accuracyPerDigit = new List<float>()
        {
            0f,
            0f,
            0f,
            0f,
            0f,
            0f,
            0f,
            0f,
            0f,
            0f,
        };
        int sample_index = 0;
        public void Start()
        {
            foreach (var item in network.Parameters())
            {
                item.device = inferenceDevice;
            }
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out _, out test, DatasetSettings.LoadTestOnly);
            Debug.Log("MNIST Dataset test loaded.");

            if (network == null)
            {
                Debug.Log("Please load a network to test.");
            }
            print($"Total test samples {test.Count}.");
            print("Network used: " + network.Summary());
            Utils.Shuffle(test);
        }

        public void Update()
        {
            if (sample_index == test.Count)
                return; // case test finished

            (Tensor, Tensor) sample = test[sample_index++];

            var input = sample.Item1;
            var label = sample.Item2;

            int digit = (int)Tensor.ArgMax(label, -1)[0];
            var output = network.Predict(input);
            float acc = Metrics.Accuracy(output, label);
           
            if(acc == 0)
            {
                wrong[digit]++;
            }
            else
            {
                right[digit]++;
            }
            accuracyPerDigit[digit] = right[digit] / ((float)right[digit] + wrong[digit]) * 100f;

            accuracy = $"{accuracyPerDigit.Average()}%";


            completed = $"{sample_index}/{test.Count}";

           

        }

    }

}


