using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TrainMNIST : MonoBehaviour
{
    [SerializeField] Sequential network;
    [SerializeField] new string name = "MNIST_MODEL";
    [SerializeField] private float start_lr = 0.0002f;
    [SerializeField] private int batch_size = 64;

    Optimizer optim;
    StepLR scheduler;
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
            network = new Sequential(
                 new Conv2D((1, 28, 28), 5, 3, Device.GPU),                  
                 new ReLU(),
                 new MaxPool2D(2),                               
                 new Conv2D((5, 13, 13), 10, 3, Device.GPU),                
                 new ReLU(),
                 new MaxPool2D(2),                               
                 new Flatten(-3, -1),                            
                 new Dense(250, 128, device: Device.GPU),
                 new Dropout(0.2f),
                 new Dense(128, 10),
                 new Softmax()
                 );


            // network = new Sequential(
            //     new Conv2D((1, 28, 28), 5, 3, Device.GPU),
            //     new Sigmoid(),
            //     new Flatten(),
            //     new Dense(5 * 26 * 26, 100, device: Device.GPU),
            //     new Sigmoid(),
            //     new Dense(100, 10),
            //     new Softmax()
            //     );

            // network = new Sequential(
            //     new Flatten(),
            //     new Dense(784, 64, device: Device.GPU),
            //     new ReLU(),
            //     new Dense(64, 10),
            //     new Softmax());

            Debug.Log("Network created.");
        }

        optim = new Adamax(network.Parameters, lr: start_lr);
        scheduler = new StepLR(optim, 1, 0.5f);

        Utils.Shuffle(train);
        train_batches = Utils.Split(train, batch_size);
        print($"Total train samples {train.Count}.");
        print($"Total train batches {train_batches.Count}.");
        print("Network used: " + network.Summary());
    }

    public void Update()
    {
        if(batch_index % 100 == 0)
        {
            network.Save(name);
        }
        if (batch_index == train_batches.Count - 1)
        {
            batch_index = 0;
            
            network.Save(name);
            Utils.Shuffle(train);
            scheduler.Step();

            print($"Epoch {epochIndex++} | LR: {scheduler.CurrentLR}");
        }


        (Tensor, Tensor)[] train_batch = train_batches[batch_index];

        Tensor input = Tensor.Cat(null, train_batch.Select(x => x.Item1).ToArray());
        Tensor target = Tensor.Cat(null, train_batch.Select(x => x.Item2).ToArray());

        Tensor prediction = network.Forward(input);
        Loss loss = Loss.CrossEntropy(prediction, target);

        optim.ZeroGrad();
        network.Backward(loss);
        optim.Step();


        float train_acc = Metrics.Accuracy(prediction, target);
        Debug.Log($"Epoch: {epochIndex} | Batch: {batch_index++}/{train_batches.Count} | Train Accuracy: {train_acc * 100}%");
    }
}


