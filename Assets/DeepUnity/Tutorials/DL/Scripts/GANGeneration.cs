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
using UnityEditor.PackageManager.UI;

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
        [SerializeField] private float maxNorm = 1F;
        [SerializeField] private WhatToDo perform = WhatToDo.Train;
        public bool WriteLoss = true;

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
            InitType wInit = InitType.Kaiming_Uniform;
            InitType bInit = InitType.Zeros;
            if (D == null)
            {
                D = new Sequential(
                    new Flatten(),

                    new Dense(784, size, weight_init:wInit, bias_init:bInit),
                    new RMSNorm(),
                    new ReLU(true),           
                    new Dropout(dropout, true),
                
                    new Dense(size, size/2, weight_init: wInit, bias_init: bInit),
                    new RMSNorm(),
                     new ReLU(true),
                    new Dropout(dropout, true),

                    new Dense(size / 2, size/4, weight_init: wInit, bias_init: bInit),
                    new RMSNorm(),
                     new ReLU(true),
                    new Dropout(dropout, true),

                    new Dense(size/4, 1, weight_init:wInit, bias_init:bInit),
                    new Sigmoid(true)
                    ).CreateAsset("discriminator");
            }
            if (G == null)
            {
                G = new Sequential(
                    new Dense(latent_dim, size / 4, weight_init: wInit, bias_init: bInit),
                    new RMSNorm(),
                     new ReLU(true),

                    new Dense(size / 4, size / 2, weight_init: wInit, bias_init: bInit),
                    new RMSNorm(),
                    new ReLU(true),

                    new Dense(size / 2, size, weight_init: wInit, bias_init: bInit),
                    new RMSNorm(),
                     new ReLU(true),

                    new Dense(size, 784, weight_init: wInit, bias_init: bInit),
                    new Sigmoid(true),

                    new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 })
                    ).CreateAsset("generator");
            }

            G.Device = Device.GPU;
            D.Device = Device.GPU;
            d_optim = new AdamW(D.Parameters(), lr, amsgrad:true);
            g_optim = new AdamW(G.Parameters(), lr, amsgrad:true);

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

                var x = Tensor.Concat(null, Utils.GetRange(dataset, batch_index, batch_size).ToArray()); // real

                // GAN LOSS
                // // Train Discriminator------------------------------------------------------------------
                var z = Tensor.RandomNormal(batch_size, latent_dim);
                var Gz = G.Predict(z); // fake       
                d_optim.ZeroGrad();
                
                var Dx = D.Forward(x); // pred-real
                Loss loss = Loss.BCE(Dx, Tensor.Ones(Dx.Shape));
                D.Backward(loss.Gradient);
                
                var DGz = D.Forward(Gz); // pred-fake
                Loss loss2 = Loss.BCE(DGz, Tensor.Zeros(DGz.Shape));
                D.Backward(loss2.Gradient);

                d_optim.ClipGradNorm(maxNorm);
                d_optim.Step();
                if(WriteLoss) D_graph.Append(loss.Item + loss2.Item);
                // -----------------------------------------------------------------------------------------------------
                
                // Train Generator ------------------------------------------------------------------
                z = Tensor.RandomNormal(batch_size, latent_dim);
                Gz = G.Forward(z);
                
                D.RequiresGrad = false;
                g_optim.ZeroGrad();
                
                DGz = D.Forward(Gz); // pred-fake 
                loss = Loss.BCE(DGz, Tensor.Ones(DGz.Shape)); //(maximize realism)
                G.Backward(D.Backward(loss.Gradient));
                g_optim.ClipGradNorm(maxNorm);
                g_optim.Step();
                D.RequiresGrad = true;
                if (WriteLoss)  G_graph.Append(loss.Item);
                // // -----------------------------------------------------------------------------------------------------














                batch_index += batch_size;

            }
            else
                DisplayGeneratorProgress();

            
        }
        private void DisplayGeneratorProgress()
        {
            if (displays.Count == 0)
                return;

            var z = Tensor.RandomNormal(displays.Count, latent_dim);
            var samples = G.Predict(z).Split(0, 1);

            for (int i = 0; i < displays.Count; i++)
            {
                Texture2D display = displays[i].texture as Texture2D;
                display.SetPixels(Utils.TensorToColorArray(samples[i].Squeeze(0)));
                display.Apply();
            }
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

