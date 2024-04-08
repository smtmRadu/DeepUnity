using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using DeepUnity.Models;

namespace DeepUnity.Tutorials
{
    public class VAEDenoising : MonoBehaviour
    {
        [Button("SaveNetwork")]
        public WhatToDo perform = WhatToDo.Train;
        public float lr = 1e-3f;
        public int batchSize = 32;
        public PerformanceGraph graph = new PerformanceGraph();
        public GameObject canvas;
        private List<RawImage> displays;

        [SerializeField] VariationalAutoencoder vae;

        Optimizer optim;

        List<(Tensor, Tensor)> train = new();
        List<(Tensor, Tensor)[]> train_batches;

        int batch_index = 0;

        public float noise_prob = 0.1f;
        public float noise_size = 0.1f;

        private void Start()
        {
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batchSize);

            if (vae == null)
            {
                vae = new VariationalAutoencoder(
                    new IModule[] {
                            new Flatten(),
                            new Dense(784, 512, device: Device.GPU),
                            new ReLU(),
                            new Dense(512, 256, device: Device.GPU),
                            new ReLU(),
                            new Dense(256, 10, device: Device.CPU),
                            new ReLU() },
                            10,
                    new IModule[] {
                            new Dense(10, 256, device: Device.CPU),
                            new ReLU(),
                            new Dense(256, 512, device: Device.GPU),
                             new ReLU(),
                            new Dense(512, 784, device: Device.GPU),
                            new Sigmoid(),
                            new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 }) }
                           ).CreateAsset("vae_denoiser");
            }


            Parameter[] parameters = vae.Parameters();

            optim = new Adam(parameters, lr);

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
            if (perform == WhatToDo.Train)
            {
                if (batch_index % 50 == 0)
                {
                    vae.Save();
                }

                // Case when epoch finished
                if (batch_index == train_batches.Count - 1)
                {
                    batch_index = 0;
                    Utils.Shuffle(train);
                }

                float loss_value = 0f;

                var batch = train_batches[batch_index];

                Tensor input = Tensor.Concat(null, batch.Select(x => x.Item1).ToArray());
                input = Utils.ImageProcessing.Noise(input, Utils.Random.Range(0, noise_prob), Utils.Random.Range(0, noise_size));

                Tensor decoded = vae.Forward(input);
                Loss bce = Loss.BCE(decoded, input);
                loss_value += bce.Item;
                vae.Backward(bce.Gradient);
                optim.ClipGradNorm(1f);
                optim.Step();

                // print($"Batch: {batch_index} | Loss: {loss_value}");
                graph.Append(loss_value);

                batch_index++;
            }
            else
            {
                if (displays.Count == 0)
                    return;

                for (int i = 0; i < displays.Count / 2; i++)
                {
                    var sample = Utils.Random.Sample(train).Item1;
                    sample = Utils.ImageProcessing.Noise(sample, Utils.Random.Range(0, noise_prob), Utils.Random.Range(0, noise_size));
                    var tex1 = displays[i].texture as Texture2D;
                    tex1.SetPixels(Utils.TensorToColorArray(sample));
                    tex1.Apply();

                    
                    var recon_sample = vae.Predict(sample);
                    var tex2 = displays[i + displays.Count / 2].texture as Texture2D;
                    tex2.SetPixels(Utils.TensorToColorArray(recon_sample));
                    tex2.Apply();
                }

            }
        }


        public void SaveNetwork() => vae.Save();

        public enum WhatToDo
        {
            SeeGeneratedImages,
            Train
        }
    }



}

