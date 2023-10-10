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

        
        const int latent_dim = 128;

        [Button("SaveNetworks")]
        [SerializeField] private int batch_size = 64;
        [SerializeField] private WhatToDo perform = WhatToDo.Train;

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
                    new Dense(784, 1024, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                    new ReLU(),
                    new Dropout(0.3f),
                
                    new Dense(1024, 512, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                    new ReLU(),
                    new Dropout(0.3f),

                    new Dense(512, 256, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                    new ReLU(),
                    new Dropout(0.3f),

                    new Dense(256, 1, device: Device.CPU),
                    new Sigmoid()
                    ).CreateAsset("discriminator");
            }
            if (generator == null)
            {
                generator = new NeuralNetwork(
                    new Dense(latent_dim, 256, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                    new ReLU(),
                
                    new Dense(256, 512, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                    new ReLU(),

                    new Dense(512, 1024, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                    new ReLU(),

                    new Dense(1024, 784, device: Device.GPU),
                    new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 }),
                    new Tanh()
                    ).CreateAsset("generator");
            }

            d_optim = new Adam(discriminator.Parameters(), 0.0002f);
            g_optim = new Adam(generator.Parameters(), 0.0002f);

            List<(Tensor, Tensor)> data;
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out data, out _, DatasetSettings.LoadTrainOnly);

            Utils.Shuffle(data);
            while(data.Count % batch_size != 0)
            {
                data.RemoveAt(0);
            }
            dataset = data.Select(x => x.Item1).ToArray();

            print($"Loaded {dataset.Length} training images.");

            foreach (var item in displays)
            {
                item.texture = new Texture2D(28, 28);
            }
        }
        private void Update()
        {
            if(perform == WhatToDo.Train)
            {
                if (batch_index >= dataset.Length)
                {
                    batch_index = 0;
                    SaveNetworks();
                    Utils.Shuffle(dataset);

                }

                if (batch_index % 64 == 0)
                    SaveNetworks();

                // Train Discriminator
                var real_data = Tensor.Cat(null, Utils.GetRange(dataset, batch_index, batch_size));
                var fake_data = generator.Predict(GeneratorInput(batch_size, latent_dim));
                var d_error = TrainDiscriminator(real_data, fake_data);
                D_graph.Append(d_error);


                // Train Generator
                var g_error = TrainGenerator();
                G_graph.Append(g_error);
                batch_index += batch_size;

            }
            else
                DisplayGeneratorProgress();

            
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

            var Gz = generator.Forward(GeneratorInput(batch_size, latent_dim));
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
        private Tensor GeneratorInput(int batch_size, int latent_dim)
        {
            return Tensor.RandomNormal(batch_size, latent_dim);
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

                if (!dis.enabled)
                    continue;

                var sample = generator.Predict(GeneratorInput(1, latent_dim)).Squeeze(0);
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

        private enum WhatToDo
        {
            SeeGeneratedImages,
            Train
        }
       
    }



}

