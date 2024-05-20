using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using DeepUnity.Models;


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
        [SerializeField] private float augment_strength = 1f;
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

                         new Conv2D(1, 4, 3),
                         new ReLU(),
                         new MaxPool2D(2),
                         
                         new Conv2D(4, 8, 3),
                         new ReLU(),
                         new MaxPool2D(2),
                        

                         // Block ===============================================================
                         new Reshape(new int[] {8, 5, 5}, new int[] { 8, 25 }),
                         new Permute(-1, -2),
                         new ResidualConnection.Fork(),
                         new Attention(8, 8),
                         new ResidualConnection.Join(),
                         new Reshape(new int[] {25, 8}, new int[] {200}),
                         new RMSNorm(),

                         new ResidualConnection.Fork(),
                         new Dense(200, 200),
                         new RMSNorm(),
                         new PReLU(),
                         // new Dropout(0.1f),
                         new ResidualConnection.Join(),
                         
                         // Block ================================================================

                         new Dense(200, 200),                
                         new RMSNorm(),
                         new PReLU(),                        
                         new Dense(200, 10),

                         new Softmax()).CreateAsset(name);
            }

            network.Device = Device.GPU;

            print(network.Predict(Tensor.Random01(1, 28, 28)));


            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
            Debug.Log("MNIST Dataset loaded.");

           
            optim = new AdamW(network.Parameters(), lr: lr, weightDecay: weightDecay, amsgrad: true);
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
            optim.ClipGradNorm(maxNorm);
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

