using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using DeepUnity.Models;
using Unity.VisualScripting;

namespace DeepUnity.Tutorials
{
    // https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95
    public class VAEReconstruction : MonoBehaviour
    {
        [Button("SaveNetwork")]
        public WhatToDo perform = WhatToDo.Train;
        public float lr = 1e-3f;
        public int batchSize = 32;
        public PerformanceGraph graph = new PerformanceGraph();
        public GameObject canvas;
        private List<RawImage> displays;

        public ModelType modelType = ModelType.VAE;
        [SerializeField] VariationalAutoencoder vae;
        [SerializeField] Sequential ae;
        private List<RawImage> inputPreviews = new();
        private List<RawImage> outputPreviews = new();
        public float gradClipNorm = 1f;
        public int latent_space = 8;
        public float kld_w = 1f;
        Optimizer optim;

        List<(Tensor, Tensor)> train = new();
        List<(Tensor, Tensor)[]> train_batches;

        int batch_index = 0;


        private void Start()
        {
            Datasets.MNIST(null, out train, out _, DatasetSettings.LoadTrainOnly);
            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batchSize);

            var wInit = InitType.Xavier_Uniform;
            var bInit = InitType.Zeros;

            if (modelType == ModelType.VAE && vae == null)
            {
                vae = new VariationalAutoencoder(
                    encoder: new IModule[]
                    {
                         new Flatten(),
                         new Dense(784, 256, weight_init: wInit, bias_init: bInit),
                         new Tanh(),
                         new Dense(256, latent_space, weight_init: wInit, bias_init: bInit),
                         new Tanh(),
                    },
                    latent_space: latent_space,
                    decoder: new IModule[]
                    {
                        new Dense(latent_space, 256, weight_init: wInit, bias_init: bInit),
                        new Tanh(),
                        new Dense(256, 784, weight_init: wInit, bias_init: bInit),
                        new Sigmoid(),
                        new Reshape(new int[] {784}, new int[] {1, 28, 28})
                    },
                    kld_weight: kld_w).CreateAsset("vae");
            }

            if (modelType == ModelType.ConvVAE && vae == null)
            {
                vae = new VariationalAutoencoder(
                    encoder: new IModule[]
                    {
                         // (1, 28, 28) -> (16, 26, 26)
                         new Conv2D(1, 16, 3, weight_init: wInit, bias_init: bInit),
                         new SiLU(),
                         // (16, 26, 26) -> (8, 24, 24)
                         new Conv2D(16, 8, 3, weight_init: wInit, bias_init: bInit),
                         new SiLU(),
                         // (8, 24, 24) -> (2, 22, 22)
                         new Conv2D(8, 2, 3, weight_init: wInit, bias_init: bInit),
                         new SiLU(),
                         // (2, 22, 22) -> (968)
                         new Flatten(),
                         new Dense(968, latent_space, weight_init: wInit, bias_init: bInit),
                         new SiLU(),
                    },
                    latent_space: latent_space,
                    decoder: new IModule[]
                    {
                        new Dense(latent_space, 968, weight_init: wInit, bias_init: bInit),
                        new SiLU(),
                        // (968) -> (2, 22, 22)
                        new Reshape(new int[] {968}, new int[] {2, 22, 22}),
                        // (2, 22, 22) -> (8, 24, 24)
                        new ConvTranspose2D(2, 8, 3, weight_init: wInit, bias_init: bInit),
                        new SiLU(),
                        // (8, 24, 24) -> (16, 26, 26)
                        new ConvTranspose2D(8, 16, 3, weight_init: wInit, bias_init: bInit),
                        new SiLU(),
                        // (16, 26, 26) -> (1, 28, 28)
                        new ConvTranspose2D(16, 1, 3, weight_init: wInit, bias_init: bInit),
                        new Sigmoid(),
                    },
                    kld_weight: kld_w).CreateAsset("conv_vae");
            }

            if (modelType == ModelType.AE && ae == null)
            {
                ae = new Sequential(
                    // Encoder
                    new Flatten(),
                    new Dense(784, 256, weight_init: wInit, bias_init: bInit),
                    new Tanh(),
                    new Dense(256, latent_space, weight_init: wInit, bias_init: bInit),
                    new Tanh(),
                    // Decoder
                    new Dense(latent_space, 256, weight_init: wInit, bias_init: bInit),
                    new Tanh(),
                    new Dense(256, 784, weight_init: wInit, bias_init: bInit),
                    new Sigmoid(),
                    new Reshape(new int[] {784}, new int[] {1, 28, 28})
                ).CreateAsset("ae");
            }

            if (modelType == ModelType.ConvAE && ae == null)
            {
                ae = new Sequential(
                    // Encoder
                    // (1, 28, 28) -> (16, 26, 26)
                    new Conv2D(1, 16, 3, weight_init: wInit, bias_init: bInit),
                    new SiLU(),
                    // (16, 26, 26) -> (8, 24, 24)
                    new Conv2D(16, 8, 3, weight_init: wInit, bias_init: bInit),
                    new SiLU(),
                    // (8, 24, 24) -> (2, 22, 22)
                    new Conv2D(8, 2, 3, weight_init: wInit, bias_init: bInit),
                    new SiLU(),
                    // (2, 22, 22) -> (968)
                    new Flatten(),
                    new Dense(968, latent_space, weight_init: wInit, bias_init: bInit),
                    new SiLU(),
                    // Decoder
                    new Dense(latent_space, 968, weight_init: wInit, bias_init: bInit),
                    new SiLU(),
                    // (968) -> (2, 22, 22)
                    new Reshape(new int[] {968}, new int[] {2, 22, 22}),
                    // (2, 22, 22) -> (8, 24, 24)
                    new ConvTranspose2D(2, 8, 3, weight_init: wInit, bias_init: bInit),
                    new SiLU(),
                    // (8, 24, 24) -> (16, 26, 26)
                    new ConvTranspose2D(8, 16, 3, weight_init: wInit, bias_init: bInit),
                    new SiLU(),
                    // (16, 26, 26) -> (1, 28, 28)
                    new ConvTranspose2D(16, 1, 3, weight_init: wInit, bias_init: bInit),
                    new Sigmoid()
                ).CreateAsset("conv_ae");
            }

            bool isVAE = modelType == ModelType.VAE || modelType == ModelType.ConvVAE;
            if (isVAE)
            {
                optim = new Adam(vae.Parameters(), lr);
                vae.Device = Device.GPU;
            }
            else
            {
                optim = new Adam(ae.Parameters(), lr);
                ae.Device = Device.GPU;
            }

            displays = new();
            for (int i = 0; i < canvas.transform.childCount; i++)
            {
                displays.Add(canvas.transform.GetChild(i).GetComponent<RawImage>());
            }
            foreach (var item in displays)
            {
                item.texture = new Texture2D(28, 28);
            }

            // Find INPUT and OUTPUT RawImages from children
            foreach (Transform child in transform)
            {
                var raw = child.GetComponent<RawImage>();
                if (raw == null) continue;
                if (child.name.StartsWith("INPUT"))
                {
                    raw.texture = new Texture2D(28, 28);
                    inputPreviews.Add(raw);
                }
                else if (child.name.StartsWith("OUTPUT"))
                {
                    raw.texture = new Texture2D(28, 28);
                    outputPreviews.Add(raw);
                }
            }
        }


        private void Update()
        {
            if (perform == WhatToDo.Train)
            {
                bool isVAE = modelType == ModelType.VAE || modelType == ModelType.ConvVAE;
                if (batch_index % 50 == 0)
                {
                    if (isVAE) vae.Save();
                    else ae.Save();
                }

                // Case when epoch finished
                if (batch_index == train_batches.Count - 1)
                {
                    batch_index = 0;
                    Utils.Shuffle(train);
                    train_batches = Utils.Split(train, batchSize);
                }

          

                var batch = train_batches[batch_index];
                Tensor input = Tensor.Concat(null, batch.Select(x => x.Item1).ToArray());
                Tensor decoded;
                if (isVAE)
                {
                    decoded = vae.Forward(input);
                    Loss loss = Loss.BCE(decoded, input);
                    optim.ZeroGrad();
                    vae.Backward(loss.Grad);
                    optim.ClipGradNorm(gradClipNorm);
                    optim.Step();
                    graph.Append(loss.Item);
                }
                else
                {
                    decoded = ae.Forward(input);
                    Loss loss = Loss.BCE(decoded, input);
                    optim.ZeroGrad();
                    ae.Backward(loss.Grad);
                    optim.ClipGradNorm(gradClipNorm);
                    optim.Step();
                    graph.Append(loss.Item);
                }

                // Update live previews with first N samples from batch
                int previewCount = Mathf.Min(inputPreviews.Count, outputPreviews.Count, batch.Length);
                for (int p = 0; p < previewCount; p++)
                {
                    var inTex = inputPreviews[p].texture as Texture2D;
                    inTex.SetPixels(Utils.TensorToColorArray(batch[p].Item1));
                    inTex.Apply();

                    Tensor sample = Tensor.Zeros(1, 28, 28);
                    for (int h = 0; h < 28; h++)
                        for (int w = 0; w < 28; w++)
                            sample[0, h, w] = decoded[p, 0, h, w];

                    var outTex = outputPreviews[p].texture as Texture2D;
                    outTex.SetPixels(Utils.TensorToColorArray(sample));
                    outTex.Apply();
                }

                batch_index++;
            }
            else
            {
                if (displays.Count == 0)
                    return;

                for (int i = 0; i < displays.Count/2; i++)
                {
                    var sample = Utils.Random.Sample(train).Item1;
                    var tex1 = displays[i].texture as Texture2D;
                    tex1.SetPixels(Utils.TensorToColorArray(sample));
                    tex1.Apply();


                    bool isVAE_ = modelType == ModelType.VAE || modelType == ModelType.ConvVAE;
                    var recon_sample = isVAE_ ? vae.Predict(sample) : ae.Predict(sample);
                    var tex2 = displays[i + displays.Count / 2].texture as Texture2D;
                    tex2.SetPixels(Utils.TensorToColorArray(recon_sample));
                    tex2.Apply();
                }
            
            }
        }

        public void SaveNetwork() { if (modelType == ModelType.VAE || modelType == ModelType.ConvVAE) vae.Save(); else ae.Save(); }
        public enum ModelType { VAE, AE, ConvVAE, ConvAE }
        public enum WhatToDo
        {
            SeeGeneratedImages,
            Train
        }
    }



}
