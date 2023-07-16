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
        public void Start()
        {
            Datasets.MNIST(out train, out test);
            Debug.Log("MNIST Dataset loaded.");

            if (network == null)
            {
                network = new Sequential(
                    new Conv2D(new int[]{ 28, 28 }, 5, 3),
                    new Sigmoid(),
                    new Flatten(-3, -1),
                    new Dense(5 * 26 * 26, 100),
                    new Sigmoid(),
                    new Dense(100, 10),
                    new SoftMax());
            }

            optim = new Adam(network.Parameters);

           
        }

        public void Update()
        {
            Utils.Shuffle(train);

            List<(Tensor,Tensor)[]> train_batches = Utils.Split(train, batch_size);

            List<float> epoch_train_accuracies = new List<float>();

            for (int i = 0; i < train_batches.Count; i++)
            {
                (Tensor, Tensor)[] train_batch = train_batches[i];

                Tensor input = Tensor.Concat(null, train_batch.Select(x => x.Item1).ToArray());
                Tensor target = Tensor.Concat(null, train_batch.Select(x => x.Item2).ToArray());


                Tensor prediction = network.Forward(input);
                Tensor loss = Loss.MSE(prediction, target);

                optim.ZeroGrad();
                network.Backward(loss);
                optim.Step();

                float train_acc = Metrics.Accuracy(prediction, target);
                epoch_train_accuracies.Add(train_acc);
            }

            network.Save("MNIST_Model");


            Tensor valid_input = Tensor.Concat(null, test.Select(x => x.Item1).ToArray());
            Tensor valid_target = Tensor.Concat(null, test.Select(x => x.Item2).ToArray());
            float valid_acc = Metrics.Accuracy(network.Predict(valid_input), valid_target);
            print($"Epoch {Time.frameCount} | Train Accuracy: {epoch_train_accuracies.Average() * 100f}% | Validation Accuracy: {valid_acc * 100f}%");
        }
    }
}

