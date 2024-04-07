using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using DeepUnity.Models;

namespace DeepUnityTutorials
{
    /// <summary>
    /// Training GANs
    /// https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
    /// </summary>
    public class GANGeneration : MonoBehaviour
    {
        [SerializeField] private GameObject canvas;
        private List<RawImage> displays;
        [SerializeField] private Sequential discriminator;
        [SerializeField] private Sequential generator;
        
        [Button("SaveNetworks")]
        [SerializeField] private int batch_size = 64;
        [SerializeField] private float lr = 2e-4f;
        [SerializeField] private WhatToDo perform = WhatToDo.Train;
        [SerializeField] private bool writeLoss = true;

        public PerformanceGraph G_graph = new PerformanceGraph();
        public PerformanceGraph D_graph = new PerformanceGraph();

        Optimizer d_optim;
        Optimizer g_optim;

        Tensor[] dataset;

        private int batch_index = 0;

        const int latent_dim = 100;
        const int size = 1024; // 1024 original
        const float dropout = 0.3f; // 0.3f original
        private void Start()
        {          
            if (discriminator == null)
            {
                discriminator = new Sequential(
                    new Flatten(),

                    new Dense(784, size),
                    new Tanh(),
                    new Dropout(dropout),
                
                    new Dense(size, size/4),
                    new Tanh(),
                    new Dropout(dropout),

                    new Dense(size / 4, 1),
                    new Sigmoid()
                    ).CreateAsset("discriminator");
            }
            if (generator == null)
            {
                generator = new Sequential(
                    new Dense(latent_dim, size / 4),
                    new Tanh(),

                    new Dense(size / 4, size / 2),
                    new Tanh(),

                    new Dense(size / 2, size),
                    new Tanh(),

                    new Dense(size, 784),
                    new Tanh(),

                    new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 })
                    ).CreateAsset("generator");
            }

            generator.Device = Device.GPU;
            discriminator.Device = Device.GPU;
            d_optim = new Adam(discriminator.Parameters(), lr);
            g_optim = new Adam(generator.Parameters(), lr);

            List<(Tensor, Tensor)> data;
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out data, out _, DatasetSettings.LoadTrainOnly);

            Utils.Shuffle(data);
            while(data.Count % batch_size != 0)
            {
                data.RemoveAt(0);
            }
            dataset = data.Select(x => x.Item1).ToArray();

            print($"Loaded {dataset.Length} training images.");

          
            displays = new();
            for (int i = 0; i < canvas.transform.childCount; i++)
            {
                displays.Add(canvas.transform.GetChild(i).GetComponent<RawImage>());
            }
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

                if (batch_index % (batch_size * 50) == 0)
                    SaveNetworks();

                // Train Discriminator
                var real_data = Tensor.Concat(null, Utils.GetRange(dataset, batch_index, batch_size).ToArray());
                var fake_data = generator.Predict(GeneratorInput(batch_size, latent_dim));
                var d_error = TrainDiscriminator(real_data, fake_data);
                if(writeLoss) D_graph.Append(d_error);


                // Train Generator
                var g_error = TrainGenerator();
                if (writeLoss) G_graph.Append(g_error);
                batch_index += batch_size;

            }
            else
                DisplayGeneratorProgress();

            
        }
        private float TrainDiscriminator(Tensor real_data, Tensor generated_data)
        {
            d_optim.ZeroGrad();
            var prediction_real = discriminator.Forward(real_data);
            var loss_real = Loss.BCE(prediction_real, RealTarget(batch_size));
            discriminator.Backward(loss_real.Gradient);

            var prediction_fake = discriminator.Forward(generated_data);
            var loss_fake = Loss.BCE(prediction_fake, FakeTarget(batch_size));
            discriminator.Backward(loss_fake.Gradient);
            d_optim.Step();
            return loss_fake.Item + loss_real.Item;
        }
        private float TrainGenerator()
        {
            g_optim.ZeroGrad();

            var Gz = generator.Forward(GeneratorInput(batch_size, latent_dim));
            var DGz = discriminator.Forward(Gz);
            var loss = Loss.BCE(DGz, RealTarget(batch_size)); // (batch_size, 1)
            generator.Backward(discriminator.Backward(loss.Gradient));
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

            generator.Device = Device.CPU;

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

            generator.Device = Device.GPU;
        }

        public void SaveNetworks()
        {
            generator.Save();
            discriminator.Save();
            ConsoleMessage.Info("Networks saved");
        }

        public enum WhatToDo
        {
            SeeGeneratedImages,
            Train
        }
       
    }



}

