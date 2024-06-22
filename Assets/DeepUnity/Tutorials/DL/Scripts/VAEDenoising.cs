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
        public int epochs = 10;
        public PerformanceGraph graph = new PerformanceGraph();
        public GameObject canvas;
        private List<RawImage> displays;

        [SerializeField] VariationalAutoencoder vae;

        Optimizer optim;
        LRScheduler scheduler;

        List<(Tensor, Tensor)> train = new();
        List<(Tensor, Tensor)[]> train_batches;

        int batch_index = 0;
        
        public float kld_weight = 0.01f;
        public float noise_prob = 0.1f;
        public float noise_size = 0.1f;

        private void Start()
        {
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batchSize);

            var w_init = InitType.Kaiming_Uniform;
            var b_init = InitType.Kaiming_Uniform;
            if (vae == null)
            {
                vae = new VariationalAutoencoder(
                    new IModule[] {
                            new Flatten(),
                            new Dense(784, 512, true, w_init,b_init),
                            new RMSNorm(),
                             new LeakyReLU(in_place:true),
                            new Dense(512, 256, true, w_init,b_init),
                            new RMSNorm(),
                             new LeakyReLU(in_place:true),
                            new Dense(256, 128, true, w_init,b_init),
                             new RMSNorm(),
                             new LeakyReLU(in_place:true),
                            new Dense(128, 8,true, w_init,b_init),
                            new ReLU() },
                            8,
                    new IModule[] {
                            new Dense(8, 128, true, w_init,b_init),
                             new RMSNorm(),
                             new LeakyReLU(in_place:true),
                            new Dense(128, 256, true, w_init,b_init),
                             new RMSNorm(),
                             new LeakyReLU(in_place:true),
                            new Dense(256, 512, true, w_init,b_init),
                             new RMSNorm(),
                            new LeakyReLU(in_place:true),
                            new Dense(512, 784,true, w_init,b_init),
                            new Sigmoid(true),
                            new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 }) }
                           , kld_weight:kld_weight).CreateAsset("vae_denoiser");
            }

            vae.Device = Device.GPU;
            optim = new Adam(vae.Parameters(), lr, amsgrad:true);
            scheduler = new LinearLR(optim, 1, 0.2f, epochs);



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
                    train_batches = Utils.Split(train, batchSize);
                    scheduler.Step();
                }

                float loss_value = 0f;

                var batch = train_batches[batch_index];

                Tensor image = Tensor.Concat(null, batch.Select(x => x.Item1).ToArray());
                Tensor noise_input = Utils.Vision.Noise(image, Utils.Random.Range(0, noise_prob), Utils.Random.Range(0, noise_size));

                Tensor decoded = vae.Forward(noise_input);
                Loss bce = Loss.BCE(decoded, image);
                loss_value += bce.Item;
                vae.Backward(bce.Gradient);
                optim.ClipGradNorm(1);
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
                    sample = Utils.Vision.Noise(sample, Utils.Random.Range(0, noise_prob), Utils.Random.Range(0, noise_size));
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

