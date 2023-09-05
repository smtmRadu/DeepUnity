using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnityTutorials
{
    /// <summary>
    /// Training GANs
    /// https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
    /// </summary>
    public class TrainingGAN : MonoBehaviour
    {
        
        [SerializeField] private List<RawImage> displays;
        [SerializeField] private NeuralNetwork discriminator;
        [SerializeField] private NeuralNetwork generator;

        [Button("SaveNetworks")]
        [SerializeField] private int batch_size;

        public PerformanceGraph G_graph = new PerformanceGraph();
        public PerformanceGraph D_graph = new PerformanceGraph();

        Optimizer d_optim;
        Optimizer g_optim;

        Tensor[] dataset;

        private int batch_index = 0;
        private void Start()
        {
            if (discriminator == null)
            {
                discriminator = new NeuralNetwork(
                    new Flatten(),
                    new Dense(784, 144, device: Device.GPU),
                    new LeakyReLU(),
                    new Dropout(0.3f),
           
                    new Dense(144, 128, device: Device.GPU),
                    new LeakyReLU(),
                    new Dropout(0.3f),

                    new Dense(128, 1, device: Device.CPU),
                    new Sigmoid()
                    ).CreateAsset("discriminator");
            }
            if (generator == null)
            {
                generator = new NeuralNetwork(
                    new Dense(100, 128, device: Device.GPU),
                    new LeakyReLU(),

                    new Dense(128, 144, device: Device.GPU),
                    new LeakyReLU(),

                    new Dense(144, 784, device: Device.GPU),
                    new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 }),
                    new Tanh()
                    ).CreateAsset("generator");
            }

            d_optim = new Adam(discriminator.Parameters(), 0.0002f);
            g_optim = new Adam(generator.Parameters(), 0.0002f);

            List<(Tensor, Tensor)> data;
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out data, out _, DatasetSettings.LoadTrainOnly);

            while(data.Count % batch_size != 0)
            {
                data.RemoveAt(0);
            }
            dataset = data.Select(x => x.Item1).ToArray();


            print($"Loaded {dataset.Length} training images.");
        }
        private void Update()
        {
            if(batch_index >= dataset.Length)
            {
                batch_index = 0;
                SaveNetworks();
               
            }

            // Train Discriminator
            var real_data = Tensor.Cat(null, Utils.GetRange(dataset, batch_index, batch_size));
            var fake_data = generator.Predict(GeneratorSeed(batch_size));
            var d_error = TrainDiscriminator(real_data, fake_data);
            D_graph.Append(d_error);


            // Train Generator
            var g_error = TrainGenerator();
            G_graph.Append(g_error);

            DisplayGeneratorProgress();

            batch_index += batch_size;
        }
        private float TrainDiscriminator(Tensor real_data, Tensor generated_data)
        {
            d_optim.ZeroGrad();
            var prediction_real = discriminator.Forward(real_data);
            var loss_real = Loss.BinaryCrossEntropy(prediction_real, RealTarget(batch_size));
            discriminator.Backward(loss_real.Derivative);

            var prediction_fake = discriminator.Forward(generated_data);
            var loss_fake = Loss.BinaryCrossEntropy(prediction_fake, FakeTarget(batch_size));
            discriminator.Backward(loss_fake.Derivative);

            d_optim.Step();
            return loss_fake.Item + loss_real.Item;
        }
        private float TrainGenerator()
        {
            // If discrimnator says the fake data is real, the loss of generator will be small, and viceversa
            g_optim.ZeroGrad();

            var Gz = generator.Forward(GeneratorSeed(batch_size));
            var DGz = discriminator.Forward(Gz);
            var loss = Loss.BinaryCrossEntropy(DGz, RealTarget(batch_size)); // (batch_size, 1)
            var generatorLossDerivative = discriminator.Backward(loss.Derivative);
            generator.Backward(generatorLossDerivative);
            g_optim.Step();

            return loss.Item;
        }

        private Tensor RealTarget(int batch_size)
        {
            return Tensor.Ones(batch_size, 1);
        }
        private Tensor FakeTarget(int batch_size)
        {
            return Tensor.Zeros(batch_size, 1);
        }
        private Tensor GeneratorSeed(int batch_size)
        {
            return Tensor.RandomNormal(batch_size, 100);
        }

        private void DisplayGeneratorProgress()
        {
            if (displays.Count == 0)
                return;

            var paramst = generator.Parameters();
            
            foreach (var item in paramst)
            {
                item.device = Device.CPU;
            }

            foreach (var dis in displays)
            {
                if (dis == null)
                    continue;
                var sample = generator.Predict(GeneratorSeed(1)).Squeeze(0);
                Texture2D display = dis.texture as Texture2D;
                display.SetPixels(Utils.TensorToColorArray(sample));
                display.Apply();
            }

            foreach (var item in paramst)
            {
                item.device = Device.GPU;
            }

        }
       
        public void SaveNetworks()
        {
            generator.Save();
            discriminator.Save();
            print("Networks saved");
        }
    }



}

