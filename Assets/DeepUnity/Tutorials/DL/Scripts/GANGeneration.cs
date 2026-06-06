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
        [SerializeField] private Architecture architecture = Architecture.MLP;
        public bool WriteLoss = true;

        public PerformanceGraph G_graph = new PerformanceGraph();
        public PerformanceGraph D_graph = new PerformanceGraph();

        Optimizer d_optim;
        Optimizer g_optim;

        Tensor[] dataset;
        private List<RawImage> generatedPreviews = new();
        private Tensor lastGeneratedBatch; // most recent G(z) from training, reused for previews

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
                D = architecture == Architecture.MLP
                    ? BuildMLPDiscriminator(wInit, bInit)
                    : BuildConvDiscriminator(wInit, bInit);
                D.CreateAsset("discriminator");
            }
            if (G == null)
            {
                G = architecture == Architecture.MLP
                    ? BuildMLPGenerator(wInit, bInit)
                    : BuildConvGenerator();
                G.CreateAsset("generator");
            }

            G.Device = Device.GPU;
            D.Device = Device.GPU;
            d_optim = new AdamW(D.Parameters(), lr, amsgrad: true);
            g_optim = new AdamW(G.Parameters(), lr, amsgrad: true);

            if (perform == WhatToDo.Train)
            {
                List<(Tensor, Tensor)> data;
                Datasets.MNIST(null, out data, out _, DatasetSettings.LoadTrainOnly);

                Utils.Shuffle(data);
                while (data.Count % batch_size != 0)
                {
                    data.RemoveAt(0);
                }
                dataset = data.Select(x => x.Item1).ToArray();

                print($"Loaded {dataset.Length} training images.");
            }

            // The Canvas GameObject ships disabled in the scene — force it on so the
            // RawImages actually render. Otherwise the textures get written but nothing
            // is drawn.
            if (canvas != null && !canvas.activeSelf) canvas.SetActive(true);

            displays = new();
            for (int i = 0; i < canvas.transform.childCount; i++)
            {
                displays.Add(canvas.transform.GetChild(i).GetComponent<RawImage>());
            }
            foreach (var item in displays)
            {
                item.texture = new Texture2D(28, 28);
            }

            // Live-preview slots on this GameObject's children (named "GENERATED*").
            // Mirrors the VAEReconstruction "INPUT*"/"OUTPUT*" pattern.
            foreach (Transform child in transform)
            {
                var raw = child.GetComponent<RawImage>();
                if (raw == null) continue;
                if (child.name.StartsWith("GENERATED"))
                {
                    raw.texture = new Texture2D(28, 28);
                    generatedPreviews.Add(raw);
                }
            }
        }

        private Sequential BuildMLPDiscriminator(InitType wInit, InitType bInit)
        {
            return new Sequential(
                new Flatten(),

                new Dense(784, size, weight_init: wInit, bias_init: bInit),
                new RMSNorm(size),
                new ReLU(true),
                new Dropout(dropout, true),

                new Dense(size, size / 2, weight_init: wInit, bias_init: bInit),
                new RMSNorm(size / 2),
                new ReLU(true),
                new Dropout(dropout, true),

                new Dense(size / 2, size / 4, weight_init: wInit, bias_init: bInit),
                new RMSNorm(size / 4),
                new ReLU(true),
                new Dropout(dropout, true),

                new Dense(size / 4, 1, weight_init: wInit, bias_init: bInit),
                new Sigmoid(true)
            );
        }

        private Sequential BuildMLPGenerator(InitType wInit, InitType bInit)
        {
            return new Sequential(
                new Dense(latent_dim, size / 4, weight_init: wInit, bias_init: bInit),
                new RMSNorm(size / 4),
                new ReLU(true),

                new Dense(size / 4, size / 2, weight_init: wInit, bias_init: bInit),
                new RMSNorm(size / 2),
                new ReLU(true),

                new Dense(size / 2, size, weight_init: wInit, bias_init: bInit),
                new RMSNorm(size),
                new ReLU(true),

                new Dense(size, 784, weight_init: wInit, bias_init: bInit),
                new Sigmoid(true),

                new Reshape(new int[] { 784 }, new int[] { 1, 28, 28 })
            );
        }

        private Sequential BuildConvDiscriminator(InitType wInit, InitType bInit)
        {
            // (B, 1, 28, 28) -> (B, 1)
            return new Sequential(
                new ZeroPad2D(1),
                new Conv2D(1, 8, 3, weight_init: wInit, bias_init: bInit),
                new BatchNorm2D(8),
                new LeakyReLU(0.2f, true),
                new MaxPool2D(2),                       // (8, 14, 14)
                new Dropout(dropout, true),

                new ZeroPad2D(1),
                new Conv2D(8, 16, 3, weight_init: wInit, bias_init: bInit),
                new BatchNorm2D(16),
                new LeakyReLU(0.2f, true),
                new MaxPool2D(2),                       // (16, 7, 7)
                new Dropout(dropout, true),

                new ZeroPad2D(1),
                new Conv2D(16, 32, 3, weight_init: wInit, bias_init: bInit),
                new BatchNorm2D(32),
                new LeakyReLU(0.2f, true),              // (32, 7, 7)
                new Dropout(dropout, true),

                new Flatten(),
                new Dense(32 * 7 * 7, 1, weight_init: wInit, bias_init: bInit),
                new Sigmoid(true)
            );
        }

        private Sequential BuildConvGenerator()
        {
            // (B, 100) -> (B, 1, 28, 28)
            // ConvTranspose2D: H_out = H_in + kernel - 1 (no stride support)
            // 7 -> 14 -> 21 -> 28 with three k=8 layers.
            // Xavier_Uniform on the deconv stack — Kaiming amplifies the central
            // pixels of full-convolution outputs (each center cell sums k*k contributions
            // vs. fewer at the borders), which manifests as a "black/white dot" through
            // the final sigmoid at init. Xavier's smaller variance keeps init output
            // ~uniform mid-gray.
            InitType gInit = InitType.Xavier_Uniform;
            InitType gBias = InitType.Zeros;
            return new Sequential(
                new Dense(latent_dim, 32 * 7 * 7, weight_init: gInit, bias_init: gBias),
                new Reshape(new int[] { 32 * 7 * 7 }, new int[] { 32, 7, 7 }),
                new BatchNorm2D(32),
                new ReLU(true),

                new ConvTranspose2D(32, 16, 8, weight_init: gInit, bias_init: gBias),   // (16, 14, 14)
                new BatchNorm2D(16),
                new ReLU(true),

                new ConvTranspose2D(16, 8, 8, weight_init: gInit, bias_init: gBias),    // (8, 21, 21)
                new BatchNorm2D(8),
                new ReLU(true),

                new ConvTranspose2D(8, 1, 8, weight_init: gInit, bias_init: gBias),     // (1, 28, 28)
                new Sigmoid(true)
            );
        }

        private void Update()
        {
            if (perform == WhatToDo.Train)
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

                Train(x);

                UpdateGeneratedPreviews(); // reuses the latest Gz, no extra forward pass

                batch_index += batch_size;
            }
            else
                DisplayGeneratorProgress();
        }

        private void Train(Tensor x)
        {
            // // Train Discriminator -----------------------------------------------------------------
            var z = Tensor.RandomNormal(batch_size, latent_dim);
            var Gz = G.Predict(z); // fake
            lastGeneratedBatch = Gz;
            d_optim.ZeroGrad();

            var Dx = D.Forward(x); // pred-real
            Loss loss = Loss.BCE(Dx, Tensor.Ones(Dx.Shape));
            D.Backward(loss.Grad);

            var DGz = D.Forward(Gz); // pred-fake
            Loss loss2 = Loss.BCE(DGz, Tensor.Zeros(DGz.Shape));
            D.Backward(loss2.Grad);

            d_optim.ClipGradNorm(maxNorm);
            d_optim.Step();
            if (WriteLoss) D_graph.Append(loss.Item + loss2.Item);
            // --------------------------------------------------------------------------------------

            // Train Generator ----------------------------------------------------------------------
            z = Tensor.RandomNormal(batch_size, latent_dim);
            Gz = G.Forward(z);
            lastGeneratedBatch = Gz;

            D.RequiresGrad = false;
            g_optim.ZeroGrad();

            DGz = D.Forward(Gz);
            loss = Loss.BCE(DGz, Tensor.Ones(DGz.Shape)); // (maximize realism)
            G.Backward(D.Backward(loss.Grad));
            g_optim.ClipGradNorm(maxNorm);
            g_optim.Step();
            D.RequiresGrad = true;
            if (WriteLoss) G_graph.Append(loss.Item);
            // --------------------------------------------------------------------------------------
        }

        private bool _previewLogged = false;
        /// <summary>
        /// Pushes the most recent <c>G(z)</c> tensor (already produced during training)
        /// into every preview RawImage we know about — both the <see cref="canvas"/>
        /// children and any "GENERATED*" children of this GameObject. No extra forward pass.
        /// </summary>
        private void UpdateGeneratedPreviews()
        {
            if (lastGeneratedBatch == null) return;

            // Reuses the EXACT same pattern as DisplayGeneratorProgress (which works).
            Tensor[] samples = lastGeneratedBatch.Split(0, 1);

            if (!_previewLogged)
            {
                _previewLogged = true;
                Debug.Log($"[GANGeneration] previews: displays={displays?.Count ?? 0}, generated={generatedPreviews.Count}, " +
                          $"Gz.shape=({string.Join(",", lastGeneratedBatch.Shape)}), samples={samples.Length}, " +
                          $"first sample [0,14,14]={samples[0][0, 14, 14]:F4}");
            }

            WriteSamplesTo(displays, samples);
            WriteSamplesTo(generatedPreviews, samples);
        }

        private void WriteSamplesTo(List<RawImage> targets, Tensor[] samples)
        {
            if (targets == null || targets.Count == 0) return;

            int previewCount = Mathf.Min(targets.Count, samples.Length);
            for (int p = 0; p < previewCount; p++)
            {
                if (targets[p] == null) continue;
                Texture2D tex = targets[p].texture as Texture2D;
                if (tex == null)
                {
                    tex = new Texture2D(28, 28);
                    targets[p].texture = tex;
                }
                tex.SetPixels(Utils.TensorToColorArray(samples[p].Squeeze(0)));
                tex.Apply();
            }
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

        public enum Architecture
        {
            MLP,
            Conv
        }
    }
}
