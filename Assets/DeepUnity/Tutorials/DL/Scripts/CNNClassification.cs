using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using DeepUnity.Models;
using UnityEngine.UIElements;


namespace DeepUnity.Tutorials
{
    // https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
    public class CNNClassification : MonoBehaviour
    {
        [SerializeField] Sequential network;
        [SerializeField] new string name = "MNIST_MODEL";
        [SerializeField] private float lr = 0.0001f;
        [SerializeField] private int epochs = 100;
        [SerializeField] private float weightDecay = 0.01f;
        [SerializeField] private int batch_size = 64;
        [SerializeField] private bool augment_data = false;
        [SerializeField] private float maxNorm = 0.5f;
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
            if (network == null)
            {
                network = new Sequential(
                          new ZeroPad2D(1),
                          new Conv2D(1, 32, 3),
                          new BatchNorm2D(32),
                          new LeakyReLU(in_place: true),
                          new MaxPool2D(2),
                          
                          new ZeroPad2D(1),
                          new Conv2D(32, 64, 3),
                          new BatchNorm2D(64),
                          new LeakyReLU(in_place: true),
                          new MaxPool2D(2),
                          
                          new ZeroPad2D(1),
                          new Conv2D(64, 128, 3),
                          new BatchNorm2D(128),
                          new LeakyReLU(in_place: true),
                          new MaxPool2D(2),
                          
                          new Flatten(),
                          new LazyDense(256),
                          new RMSNorm(),
                          new LeakyReLU(in_place: true),
                          new Dense(256, 10),
                          new Softmax()).CreateAsset(name); 
            }

            network.Device = Device.GPU;

            print(network.Predict(Tensor.Random01(1, 28, 28)));

            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
            Debug.Log("MNIST Dataset loaded.");

           
            optim = new AdamW(network.Parameters(), lr: lr, weight_decay: weightDecay, amsgrad: true);
            scheduler = new LinearLR(optim, total_iters: epochs);
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

                print($"Epoch {epochIndex++} | LR: {scheduler.CurrentLR}");
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
            optim.ClipGradNorm(maxNorm);
            optim.Step();

            float acc = Metrics.Accuracy(prediction, target);
            accuracyGraph.Append(acc);
            lossGraph.Append(loss.Item);

            Debug.Log($"Epoch: {epochIndex} | Batch: {batch_index++}/{train_batches.Count} | Acc: {acc * 100f}% | Loss: {loss.Item} | Lr: {scheduler.CurrentLR}");
        }


        public Tensor AugmentImage(Tensor image)
        {
            Tensor tex = Utils.Vision.Zoom(image, Utils.Random.Range(0.8f, 1.2f));
            tex = Utils.Vision.Rotate(tex, Utils.Random.Range(-10, 10));
            tex = Utils.Vision.Offset(tex, Utils.Random.Range(-3, 3), Utils.Random.Range(-3, 3));
            tex = Utils.Vision.Noise(tex, Utils.Random.Range(0.01f, 0.05f), Utils.Random.Range(0.01f, 0.1f));
            return tex;
        }
    }


}

