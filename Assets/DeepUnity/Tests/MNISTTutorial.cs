using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class MNISTTutorial : MonoBehaviour
    {
        [SerializeField] Sequential network;
        Optimizer optim;

        List<(Tensor, Tensor)> train = new();
        List<(Tensor, Tensor)> test = new();

        [SerializeField] private int batch_size = 32;

        int epochIndex = 1;
        int batch_index = 0;
        List<(Tensor, Tensor)[]> train_batches;
        public void Start()
        {
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop\\", out train, out test);
            Debug.Log("MNIST Dataset loaded.");

            if (network == null)
            {
                network = new Sequential(
                    new Conv2D((1,28,28), 5, 3),
                    new Sigmoid(),
                    new Flatten(-3, -1),
                    new Dense(5 * 26 * 26, 100, device: Device.GPU),
                    new Sigmoid(),
                    new Dense(100, 10),
                    new Softmax());

                // network = new Sequential(
                //     new Flatten(-3, -1),
                //     new ReLU(),
                //     new Dense(784, 200),
                //     new ReLU(),
                //     new Dense(200, 10),
                //     new Softmax());
            }

            optim = new Adam(network.Parameters);

            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batch_size);
            print($"Total train samples {train.Count}.");
            print($"Total train batches {train_batches.Count}.");
           
        }

        public void Update()
        {            
            if(batch_index == train_batches.Count - 1)
            {
                batch_index = 0;
                print($"Epoch {epochIndex++}");
                network.Save("MNIST_Model");
                Utils.Shuffle(train);
            }

           
            (Tensor, Tensor)[] train_batch = train_batches[batch_index];

            Tensor input = Tensor.Cat(null, train_batch.Select(x => x.Item1).ToArray());
            Tensor target = Tensor.Cat(null, train_batch.Select(x => x.Item2).ToArray());

            Tensor prediction = network.Forward(input);
            Tensor loss = Loss.MSEDerivative(prediction, target);

            optim.ZeroGrad();
            network.Backward(loss);
            optim.Step();

            float train_acc = Metrics.Accuracy(prediction, target);

            Debug.Log($"Batch {batch_index++} | Accuracy {train_acc * 100}%");


            // Tensor valid_input = Tensor.Concat(null, test.Select(x => x.Item1).ToArray());
            // Tensor valid_target = Tensor.Concat(null, test.Select(x => x.Item2).ToArray());
            // float  valid_acc = Metrics.Accuracy(network.Predict(valid_input), valid_target);

        }
    }
}

