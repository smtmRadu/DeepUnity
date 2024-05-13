using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using DeepUnity.Models;
using Unity.Properties;
using System;

namespace DeepUnity.Tutorials
{
    /// <summary>
    /// Training GANs
    /// https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
    /// </summary>
    public class GANGeneration : MonoBehaviour
    {
        [SerializeField] private GameObject canvas;
        private List<RawImage> displays;
        [SerializeField] private Sequential D;
        [SerializeField] private Sequential G;
        
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
            InitType wInit = InitType.Xavier_Uniform;
            InitType bInit = InitType.Zeros;
            if (D == null)
            {
                D = new Sequential(
                    new Flatten(),

                    new Dense(784, size, weight_init:wInit, bias_init:bInit),
                    new Tanh(),
                    new Dropout(dropout),
                
                    new Dense(size, size/4, weight_init: wInit, bias_init: bInit),
                     new Tanh(),
                    new Dropout(dropout),

                    new Dense(size / 4, 1, weight_init: wInit, bias_init: bInit),
                    new Sigmoid()
                    ).CreateAsset("discriminator");
            }
            if (G == null)
            {
                G = new Sequential(
                    new Dense(latent_dim, size / 4, weight_init: wInit, bias_init: bInit),
                    new Tanh(),

                    new Dense(size / 4, size / 2, weight_init: wInit, bias_init: bInit),
                    new Tanh(),

                    new Dense(size / 2, size, weight_init: wInit, bias_init: bInit),
                     new Tanh(),

                    new Dense(size, 784, weight_init: wInit, bias_init: bInit),
                     new Tanh(),

                    new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 })
                    ).CreateAsset("generator");
            }

            G.Device = Device.GPU;
            D.Device = Device.GPU;
            d_optim = new Adam(D.Parameters(), lr);
            g_optim = new Adam(G.Parameters(), lr);

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
                var x = Tensor.Concat(null, Utils.GetRange(dataset, batch_index, batch_size).ToArray());
                var d_error = TrainDiscriminator(x);
                if(writeLoss) D_graph.Append(d_error);


                // Train Generator
                var g_error = TrainGenerator();
                if (writeLoss) G_graph.Append(g_error);


                batch_index += batch_size;

            }
            else
                DisplayGeneratorProgress();

            
        }
        private float TrainDiscriminator(Tensor x) // works
        {
            // Loss = -log(D(x)) - log(1 - D(G(z)))
            // Gradient ascent

            var z = Tensor.RandomNormal(batch_size, latent_dim);
            var Gz = G.Predict(z);

            d_optim.ZeroGrad();

            var Dx = D.Forward(x);
            var loss = -Dx.Log();
            D.Backward(-Dx.Reciprocal());

            var DGz = D.Forward(Gz);
            var loss2 = -(-DGz + 1f).Log();
            D.Backward((-DGz + 1f).Reciprocal());

            d_optim.Step();

            return loss.Average() + loss2.Average();
        }
        private float TrainGenerator()
        {
            // Loss = log(1 - D(G(z)))
            // Gradient descent
            g_optim.ZeroGrad();

            var z = Tensor.RandomNormal(batch_size, latent_dim);
            var Gz = G.Forward(z);
            var DGz = D.Forward(Gz);
            var loss = (-DGz + 1f).Log();
            var dLdG = D.Backward(-(-DGz + 1f).Reciprocal());
            G.Backward(dLdG);
            g_optim.Step();

            return loss.Average();
        }

        private void DisplayGeneratorProgress()
        {
            if (displays.Count == 0)
                return;

            G.Device = Device.CPU;

            foreach (var dis in displays)
            {
                if (dis == null)
                    continue;

                if (!dis.enabled)
                    continue;

                var z = Tensor.RandomNormal(1, latent_dim);
                var sample = G.Predict(z).Squeeze(0);
                Texture2D display = dis.texture as Texture2D;
                display.SetPixels(Utils.TensorToColorArray(sample));
                display.Apply();
            }

            G.Device = Device.GPU;
        }

        public void SaveNetworks()
        {
            G.Save();
            D.Save();
            ConsoleMessage.Info("Networks saved");
        }

        public enum WhatToDo
        {
            SeeGeneratedImages,
            Train
        }
       
    }



}

