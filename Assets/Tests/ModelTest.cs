using DeepUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class ModelTest : MonoBehaviour
    {

        public NeuralNetwork net;

        public int samples = 100;

        private Tensor[] inputs;
        private Tensor[] targets;

        private int epoch = 0;
        public void Start()
        {
            if(net == null)
            {
                Dense fc1 = new Dense(1, 100, WeightInit.HE, Device.CPU);
                Dense fc2 = new Dense(100, 100, WeightInit.HE, Device.CPU);
                Dense fc3 = new Dense(100, 1, WeightInit.HE, Device.CPU);
                net = new NeuralNetwork(
                 fc1,
                 new ReLU(),
                 fc2,
                 new ReLU(),
                 fc3
                 );
                net.Compile(new Adam(), "somenet");
            }

            inputs = Enumerable.Range(0, samples).ToList().
                          Select(x => Tensor.Constant(Utils.Random.Gaussian(0f, 0.5f, out _))).ToArray();

            targets = inputs.Select(x => Tensor.Constant(MathF.Cos(x[0]))).ToArray();

        }



        public void Update()
        {
            List<float> accs = new List<float>();

            for (int i = 0; i < samples; i++)
            {
                var prediction = net.Forward(inputs[i]);
                var loss = Loss.MSE(prediction, targets[i]);

                net.Backward(loss);


                // print("fc1g\n" + fc1.gWeights);
                // print("fc21g\n" + fc2.gWeights);
                // print("fc3g\n" + fc3.gWeights);
                net.Step();

                // Compute accuracy
                float acc = Metrics.Accuracy(prediction, targets[i]);
                accs.Add(acc);
            }

            Debug.Log($"Epoch {++epoch} | Accuracy {accs.Average() * 100}%");
        }
    }
}

