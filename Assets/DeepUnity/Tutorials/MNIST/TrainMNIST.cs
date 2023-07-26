using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TrainMNIST : MonoBehaviour
{
    [SerializeField] Sequential network;
    [SerializeField] private int batch_size = 64;

    Optimizer optim;
    List<(Tensor, Tensor)> train = new();
    List<(Tensor, Tensor)[]> train_batches;
    int epochIndex = 1;
    int batch_index = 0;
    

    public void Start()
    {
        Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
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

            network = new Sequential(
                new Conv2D((1, 28, 28), 1, 3, Device.GPU),
                new ReLU(),
                new Flatten(),
                new Dense(1 * 26 * 26, 64, device: Device.GPU),
                new ReLU(),
                new Dense(64, 10),
                new Softmax()
                );

            // network = new Sequential(
            //     new Flatten(),
            //     new Dense(784, 64, device: Device.GPU),
            //     new ReLU(),
            //     new Dense(64, 10),
            //     new Softmax());
        }

        optim = new Adam(network.Parameters(), 0.01f);

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
            network.Save("MNIST_Model");
            Utils.Shuffle(train);
        }


        (Tensor, Tensor)[] train_batch = train_batches[batch_index];

        Tensor input = Tensor.Cat(null, train_batch.Select(x => x.Item1).ToArray());
        Tensor target = Tensor.Cat(null, train_batch.Select(x => x.Item2).ToArray());

        Tensor prediction = network.Forward(input);
        Tensor dLossDoutput = Loss.CrossEntropyDerivative(prediction, target);

        optim.ZeroGrad();
        network.Backward(dLossDoutput);
        optim.Step();


        float train_acc = Metrics.Accuracy(prediction, target);
        Debug.Log($"Epoch: {epochIndex} | Batch: {batch_index++}/{train_batches.Count} | Train Accuracy: {train_acc * 100}%");
    }
}


