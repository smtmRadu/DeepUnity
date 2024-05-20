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
       
        [SerializeField] VariationalAutoencoder vae;
        public float gradClipNorm = 1f;
        public int latent_space = 8;
        public float kld_w = 1f;
        Optimizer optim;

        List<(Tensor, Tensor)> train = new();
        List<(Tensor, Tensor)[]> train_batches;

        int batch_index = 0;


        private void Start()
        {
            Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
            Utils.Shuffle(train);
            train_batches = Utils.Split(train, batchSize);

            if(vae == null)
            {
                var wInit = InitType.Xavier_Uniform;
                var bInit = InitType.Zeros;
                vae = new VariationalAutoencoder(
                    encoder: new IModule[]
                    {
                         new Flatten(),
                         new Dense(784, 256, device: Device.GPU, weight_init:wInit , bias_init: bInit),
                         new Tanh(),
                         new Dense(256, latent_space , weight_init:wInit , bias_init: bInit),
                         new Tanh(),
                    },
                    latent_space: latent_space,
                    decoder: new IModule[]
                    {
                        new Dense(latent_space , 256, weight_init:wInit , bias_init: bInit),
                        new Tanh(),               
                        new Dense(256, 784, device: Device.GPU, weight_init: wInit , bias_init:bInit),
                        new Sigmoid(),
                        new Reshape(new int[] {784}, new int[] {1, 28, 28})
                    },
                    kld_weight: kld_w).CreateAsset("vae");
            }
            
            optim = new Adam(vae.Parameters(), lr);



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
                    SaveNetwork();
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
                var decoded = vae.Forward(input);
                Loss loss = Loss.BCE(decoded, input);

                optim.ZeroGrad();
                vae.Backward(loss.Gradient);
                optim.ClipGradNorm(gradClipNorm);
                optim.Step();

                graph.Append(loss.Item);

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
