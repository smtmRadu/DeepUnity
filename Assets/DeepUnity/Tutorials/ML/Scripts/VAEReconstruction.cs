using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Layers;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using DeepUnity.Models;

namespace DeepUnityTutorials
{
    // https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95
    public class VAEReconstruction : MonoBehaviour
    {
        [Button("SaveNetworks")]
        public WhatToDo perform = WhatToDo.Train;
        public float lr = 1e-3f;
        public int batchSize = 32;
        public PerformanceGraph graph = new PerformanceGraph();
        public GameObject canvas;
        private List<RawImage> displays;

        [SerializeField] Sequential encoder;
        [SerializeField] Sequential decoder;
        [SerializeField] Sequential mu;
        [SerializeField] Sequential logvar;

        Optimizer optim;

        List<(Tensor, Tensor)> train = new();
        List<(Tensor, Tensor)[]> train_batches;

        int batch_index = 0;


        private void Start()
        {
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batchSize);

            if(encoder == null)
            {
                encoder = new Sequential(
                new Flatten(),
                new Dense(784, 256, device: Device.GPU),
                new ReLU(),
                new Dense(256, 8),
                new ReLU()).CreateAsset("encoder");

                mu = new Sequential(
                    new Dense(8, 8)).CreateAsset("mu");

                logvar = new Sequential(
                    new Dense(8, 8)).CreateAsset("log_var");

                decoder = new Sequential(
                    new Dense(8, 256),
                    new ReLU(),
                    new Dense(256, 784, device: Device.GPU),
                    new Sigmoid(),
                    new Reshape(new int[] {784}, new int[] {1, 28, 28})).CreateAsset("decoder");
            }
            

            Parameter[] parameters = encoder.Parameters();
            parameters = parameters.Concat(mu.Parameters()).ToArray();
            parameters = parameters.Concat(logvar.Parameters()).ToArray();
            parameters = parameters.Concat(decoder.Parameters()).ToArray();

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
                    SaveNetworks();
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

                Tensor encoded, mean, log_variance, ksi;
                Tensor decoded = Forward(input, out encoded, out mean, out log_variance, out ksi);

                // Backpropagate the MSE loss -> binary_cross_entropy(reconstructured_image, image)
                Loss bce = Loss.BCE(decoded, input);
                loss_value += bce.Item;

                Tensor dBCEdDecoded = bce.Gradient;

                // Backprop MSE  through decoder (z = mu + sigma * ksi)
                Tensor dBCE_dz = decoder.Backward(dBCEdDecoded); // derivative of the loss with respect to z = mu * sigma * std;

                // We sum the gradient from mu and sigma for encoder..
                // Backprop MSE  through mu // dZ/dMu = 1
                Tensor dBCE_dMu = dBCE_dz * 1;
                Tensor dBCE_dEncoder = mu.Backward(dBCE_dMu);

                // Backprop MSE  through sigma  // dZ/dMu = ksi
                Tensor dBCE_dLogVar = dBCE_dz * ksi;
                dBCE_dEncoder += logvar.Backward(dBCE_dLogVar);

                // Backprop MSE  through encoder
                encoder.Backward(dBCE_dEncoder);


                const float kld_weight = 1f;
                Tensor kld = kld_weight * -0.5f * (1f + log_variance - mean.Pow(2f) - log_variance.Exp());
                loss_value += kld.ToArray().Average() * kld_weight;

                // Compute gradients for mu
                Tensor dKLD_dMu = mean; // dKLD / dMu = mean
                Tensor dMu_dEncoded = mu.Backward(kld_weight * dKLD_dMu);
                // Compute gradients for sigma  dKLD / dSigma = 1/2 * (exp(log_var) - 1)
                Tensor dKLD_dLogVar = 0.5f * (log_variance.Exp() - 1f);
                Tensor dLogVar_dEncoded = logvar.Backward(dKLD_dLogVar * kld_weight);

                var dZ_dEnc = dMu_dEncoded + dLogVar_dEncoded;

                // Compute gradients for encoder
                encoder.Backward(dZ_dEnc);
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

                for (int i = 0; i < displays.Count/2; i++)
                {
                    var sample = Utils.Random.Sample(train).Item1;
                    var tex1 = displays[i].texture as Texture2D;
                    tex1.SetPixels(Utils.TensorToColorArray(sample));
                    tex1.Apply();


                    var recon_sample = Forward(sample, out _, out _, out _, out _);
                    var tex2 = displays[i + displays.Count / 2].texture as Texture2D;
                    tex2.SetPixels(Utils.TensorToColorArray(recon_sample));
                    tex2.Apply();
                }
            
            }
        }
        private Tensor Reparametrize(Tensor mu, Tensor log_var, out Tensor ksi)
        {
            var std = Tensor.Exp(0.5f * log_var);
            ksi = Tensor.RandomNormal(log_var.Shape);
            return mu + std * ksi;
        }

        private Tensor Forward(Tensor input, out Tensor encoded, out Tensor mu_v, out Tensor logvar_v, out Tensor ksi)
        {
            encoded = encoder.Forward(input);

            mu_v = mu.Forward(encoded);
            logvar_v = logvar.Forward(encoded);


            var z = Reparametrize(mu_v, logvar_v, out ksi);

            var decoded = decoder.Forward(z);
            return decoded;
        }


        public void SaveNetworks()
        {
            encoder.Save();
            decoder.Save();
            mu.Save();
            logvar.Save();
        }

        public enum WhatToDo
        {
            SeeGeneratedImages,
            Train
        }
    }



}
