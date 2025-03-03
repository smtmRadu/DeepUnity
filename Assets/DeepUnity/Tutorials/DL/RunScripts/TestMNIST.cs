using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using DeepUnity.Modules;
using DeepUnity.Models;

namespace DeepUnity.Tutorials
{
    public class TrainMNIST : MonoBehaviour
    {
        [SerializeField] Sequential network;
        [SerializeField] new string name = "MNIST_MODEL";
        [SerializeField] private float lr = 0.0002f;
        [SerializeField] private float weightDecay = 0.001f;
        [SerializeField] private int schedulerStepSize = 1;
        [SerializeField] private float schedulerDecay = 0.99f;
        [SerializeField] private int batch_size = 64;
        [SerializeField] private bool augment_data = false;
        [SerializeField] private PerformanceGraph accuracyGraph;
        [SerializeField] private PerformanceGraph lossGraph;
        Optimizer optim;
        StepAnnealing scheduler;
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
                     new Conv2D(1, 5, 3, device: Device.GPU),
                     new ReLU(),
                     new MaxPool2D(2),
                     new Conv2D(5, 10, 3, device: Device.GPU),
                     new ReLU(),
                     new MaxPool2D(2),
                     new Flatten(-3, -1),
                     new Dense(250, 128, device: Device.GPU),
                     new Dropout(0.2f),
                     new Dense(128, 10),
                     new Softmax()
                     ).CreateAsset(name);

                // network = new Sequential(
                //     new Flatten(),
                //     new Dense(784, 10, init: InitType.Glorot_Uniform, device: Device.GPU),
                //     new Softmax()
                //     ).Compile(name);

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
                //     new Dense(784, 100, init: InitType.HE_Normal, device: Device.GPU),
                //     new ReLU(),
                //     new Dense(100, 10, InitType.HE_Normal, device: Device.GPU),
                //     new Softmax()).Compile(name);

                Debug.Log("Network created.");
            }

            network.Device = Device.GPU;
            optim = new Adam(network.Parameters(),  lr: lr, weight_decay: weightDecay);
            scheduler = new StepAnnealing(optim, schedulerStepSize, schedulerDecay);
            accuracyGraph = new PerformanceGraph();
            lossGraph = new PerformanceGraph();
            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batch_size);
            print($"Total train samples {train.Count}.");
            print($"Total train batches {train_batches.Count}.");
            print("Network used: " + network.Summary());
        }

        public void Update()
        {
            if (batch_index % 50 == 0)
                network.Save();

            // Case when epoch finished
            if (batch_index == train_batches.Count - 1)
            {
                batch_index = 0;

                network.Save();
                Utils.Shuffle(train);
                scheduler.Step();

                print($"Epoch {epochIndex++} | LR: {scheduler.CurrentLR}%");
            }


            (Tensor, Tensor)[] train_batch = train_batches[batch_index];

            Tensor input = Tensor.Concat(null, train_batch.Select(x =>
            {
                Tensor img = x.Item1;
                if (augment_data)
                    img = AugmentImage(img);
                return img;
            }).ToArray());

            Tensor target = Tensor.Concat(null, train_batch.Select(x => x.Item2).ToArray());

            Tensor prediction = network.Forward(input);
            Loss loss = Loss.CE(prediction, target);

            optim.ZeroGrad();
            network.Backward(loss.Grad);
            optim.ClipGradNorm(0.5f);
            optim.Step();

            float acc = Metrics.Accuracy(prediction, target);
            accuracyGraph.Append(acc);
            lossGraph.Append(loss.Item);

            Debug.Log($"Epoch: {epochIndex} | Batch: {batch_index++}/{train_batches.Count} | Acc: {acc * 100f}% | Loss: {loss.Item}");
        }


        public Tensor AugmentImage(Tensor image)
        {
            Tensor tex = Utils.Vision.Zoom(image, Utils.Random.Range(0.7f, 1.4f));
            tex = Utils.Vision.Rotate(tex, Utils.Random.Range(-60f, 60f));
            tex = Utils.Vision.Offset(tex, Utils.Random.Range(-5f, 5f), Utils.Random.Range(-5f, 5f));
            tex = Utils.Vision.Noise(tex, Utils.Random.Range(0.05f, 0.15f), Utils.Random.Range(0.20f, 0.30f));
            return tex;
        }
    }

}

