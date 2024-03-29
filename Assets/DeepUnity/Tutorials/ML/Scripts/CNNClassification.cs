using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using DeepUnity.Models;


namespace DeepUnityTutorials
{
    // https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
    public class CNNClassification : MonoBehaviour
    {
        [SerializeField] Sequential network;
        [SerializeField] new string name = "MNIST_MODEL";
        [SerializeField] private float lr = 0.0001f;
        [SerializeField] private int epochs = 100;
        [SerializeField] private float weightDecay = 0.001f;
        [SerializeField] private int batch_size = 64;
        [SerializeField] private bool augment_data = false;
        [SerializeField] private float augment_strength = 1f;
        [SerializeField] private PerformanceGraph accuracyGraph;
        [SerializeField] private PerformanceGraph lossGraph;
        Optimizer optim;
        LRScheduler scheduler;
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
                var skip = new SkipConnectionFork();
                network = new Sequential(
                     new Conv2D(1, 6, 3),  
                     new MaxPool2D(2),
                     new Conv2D(6, 12, 3),
                     new MaxPool2D(2),

                     new Flatten(-3, -1),
                     new LayerNorm(),
                     new PReLU(),

                     new LazyDense(512),
                     new LayerNorm(),
                     new PReLU(),
                     new Dropout(0.2f),

                     skip,
                     new LazyDense(512),
                     new LayerNorm(),
                     new PReLU(),
                     new SkipConnectionJoin(skip),

                     new LazyDense(10),
                     new Softmax()
                     ).CreateAsset(name);

                Debug.Log("Network created.");
            }

            print(network.Predict(Tensor.Random01(1, 28, 28)));

            network.Device = Device.GPU;
            optim = new Adam(network.Parameters(), lr: lr, weightDecay: weightDecay, amsgrad: true);
            scheduler = new LinearLR(optim, epochs: epochs);
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
            network.Backward(loss.Gradient);
            // optim.ClipGradNorm(0.5f);
            optim.Step();

            float acc = Metrics.Accuracy(prediction, target);
            accuracyGraph.Append(acc);
            lossGraph.Append(loss.Item);

            Debug.Log($"Epoch: {epochIndex} | Batch: {batch_index++}/{train_batches.Count} | Acc: {acc * 100f}% | Loss: {loss.Item}");
        }


        public Tensor AugmentImage(Tensor image)
        {
            Tensor tex = Utils.ImageProcessing.Zoom(image, Utils.Random.Range(0.7f * augment_strength, 1.4f * augment_strength));
            tex = Utils.ImageProcessing.Rotate(tex, Utils.Random.Range(-60f * augment_strength, 60f * augment_strength));
            tex = Utils.ImageProcessing.Offset(tex, Utils.Random.Range(-5f * augment_strength, 5f * augment_strength), Utils.Random.Range(-5f * augment_strength, 5f * augment_strength));
            tex = Utils.ImageProcessing.Noise(tex, Utils.Random.Range(0.05f * augment_strength, 0.15f * augment_strength), Utils.Random.Range(0.20f * augment_strength, 0.30f * augment_strength));
            return tex;
        }
    }


}

