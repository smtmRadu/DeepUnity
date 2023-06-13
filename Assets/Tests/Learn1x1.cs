using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

namespace kbRadu
{
    public class Learn1x1 : MonoBehaviour
    {
        public NeuralNetwork net;
        public int no_samples;
        public int batch_size;

        public NDArray[] input_batches;
        public NDArray[] output_batches;
        public Device device;
        public int hid_size = 64;

        private int epoch = 1;
        public void Start()
        {
            
            Settings.Device = device;

            if(net == null)
            {
                net = new NeuralNetwork(
                    new Dense(1, hid_size),
                    new BatchNorm(hid_size),
                    new ReLU(),
                    new Dropout(),
                    new Dense(hid_size, hid_size),
                    new ReLU(),
                    new Dense(hid_size, 1)) ;
                net.Compile(new Adam(), "test_network");
            }

            // create the training data
            var inputs = NDArray.Random(1, no_samples);
            var outputs = NDArray.Cos(inputs);

            input_batches = NDArray.Split(inputs, 1, batch_size);
            output_batches = NDArray.Split(outputs, 1, batch_size);

        }

        public void Update()
        {
            List<float> epoch_errs = new List<float>();
            for (int i = 0; i < input_batches.Length; i++)
            {
                var inp_batch = input_batches[i];
                var out_batch = output_batches[i];

                var prediction = net.Forward(inp_batch);
                var loss = Loss.MSE(prediction, out_batch);

                net.ZeroGrad();
                net.Backward(loss);
                net.Step();

                var acc = Metrics.Accuracy(prediction, out_batch);
                epoch_errs.Add(acc);
            }

            print($"Epoch {epoch} | Accuracy {epoch_errs.Average() * 100f}%");
            epoch++;

            if (epoch % 10 == 0)
                net.Save();
        }

    }
}

