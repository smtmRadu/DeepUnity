using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


namespace DeepUnityTutorials
{
    // https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
    public class TrainingCNN : MonoBehaviour
    {
        [Header("Learning MNIST digit recognition.")]
        [SerializeField] NeuralNetwork network;
        [SerializeField] private int batch_size = 64;
        [SerializeField] private PerformanceGraph accuracyGraph = new PerformanceGraph();
        [SerializeField] private PerformanceGraph lossGraph = new PerformanceGraph();

        Optimizer optim;

        List<(Tensor, Tensor)> train = new();
        List<(Tensor, Tensor)> test = new();



        int epochIndex = 1;
        int batch_index = 0;
        List<(Tensor, Tensor)[]> train_batches;
        public void Start()
        {
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out test, DatasetSettings.LoadTrainOnly);
            Debug.Log("MNIST Dataset loaded.");

            if (network == null)
            {
                // network = new Sequential(
                //      new Conv2D((1, 28, 28), 5, 3, Device.GPU),                  
                //      new ReLU(),
                //      new MaxPool2D(2),                               
                //      new Conv2D((5, 13, 13), 10, 3, Device.GPU),                
                //      new ReLU(),
                //      new MaxPool2D(2),                               
                //      new Flatten(-3, -1),                            
                //      new Dense(250, 128, device: Device.GPU),
                //      new Dropout(0.2f),
                //      new Dense(128, 10),
                //      new Softmax()
                //      );

                network = new NeuralNetwork(
                    new Conv2D((1, 28, 28), 32, (3, 3), gamma_init: InitType.HE_Uniform, device: Device.GPU),
                    new ReLU(),
                    new MaxPool2D(2),
                    new Flatten(),
                    new BatchNorm(5408),
                    new Dense(5408, 100, gamma_init: InitType.HE_Uniform, device: Device.GPU),
                    new ReLU(),
                    new BatchNorm(100),
                    new Dense(100, 10),
                    new Softmax()
                    ).CreateAsset("Tutorial3");

                // network = new Sequential(
                //     new Flatten(),
                //     new Dense(784, 64, device: Device.GPU),
                //     new ReLU(),
                //     new Dense(64, 10),
                //     new Softmax());
            }

            optim = new SGD(network.Parameters(), lr: 0.1f, momentum: 0.9f);

            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batch_size);
            print($"Total train samples {train.Count}.");
            print($"Total train batches {train_batches.Count}.");
            print("Network used: " + network.Summary());
        }

        public void Update()
        {
            if (batch_index == train_batches.Count - 1)
            {
                batch_index = 0;
                print($"Epoch {epochIndex++}");
                Utils.Shuffle(train);
            }

            (Tensor, Tensor)[] train_batch = train_batches[batch_index];

            Tensor input = Tensor.Cat(null, train_batch.Select(x => x.Item1).ToArray());
            Tensor target = Tensor.Cat(null, train_batch.Select(x => x.Item2).ToArray());

            Tensor prediction = network.Forward(input);
            Loss loss = Loss.CrossEntropy(prediction, target);

            optim.ZeroGrad();
            network.Backward(loss.Derivative);
            optim.Step();

            float train_acc = Metrics.Accuracy(prediction, target);
            lossGraph.Append(loss.Item);
            accuracyGraph.Append(train_acc);
            Debug.Log($"Epoch {epochIndex} | Batch {batch_index++}/{train_batches.Count} | Accuracy {train_acc * 100}%");
        }
    }


}

