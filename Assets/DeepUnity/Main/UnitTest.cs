﻿using UnityEngine;
using DeepUnity.Models;
using System.Collections.Generic;
using UnityEngine.UI;
using DeepUnity.Modules;
using DeepUnity.Optimizers;
using DeepUnity.Activations;

namespace DeepUnity.Tutorials
{
    public class UnitTest : MonoBehaviour
    {
        public Device device = Device.CPU;
        public Sequential net;
        public GameObject canvas;
        public Optimizer optim;

        private List<RawImage> displays;
        public PerformanceGraph performanceGraph = new PerformanceGraph();  

        Tensor input = Tensor.Random01(128, 1, 28, 28);
        Tensor target = Tensor.RandomNormal(128, 2);


        private void Start()
        {
            Attention att = new Attention(50, 50, true, device: device);
            AttentionV2 att2 = new AttentionV2(50, 50, true, device: device);

            Tensor input = Tensor.Random01(64, 64, 50);

            Benckmark.Start();
            for (int i = 0; i < 10; i++)
            {
                att.Forward(input);
                att.Backward(input);

            }      
            Benckmark.Stop();
            // Tensor input = Tensor.Random01(10);
            // 
            // Dense dense = new Dense(10, 10);
            // print(dense.Forward(input));
            // print(dense.Forward(input.Unsqueeze(0).Expand(0, 2)));
            // print(dense.Forward(input.Unsqueeze(0).Expand(0, 2).Unsqueeze(0).Expand(0, 2)));
            // 
            // print(dense.Backward(input));
            // print(dense.Backward(input.Unsqueeze(0).Expand(0, 2)));
            // print(dense.Backward(input.Unsqueeze(0).Expand(0, 2).Unsqueeze(0).Expand(0, 2)));
        }
        // private void Start()
        // {
        //     net = new Sequential(
        //         new Conv2D(1, 3, 3),
        //         new BatchNorm2D(3),
        //         new MaxPool2D(2),
        //         new Conv2D(3, 6, 3),
        //         new BatchNorm2D(6),
        //         new MaxPool2D(2),
        //         new Flatten(),
        //         new LazyDense(64),
        //         new ReLU(),
        //         new Dense(64, 2));
        // 
        //     net.Predict(Tensor.Random01(1, 28, 28));
        //     optim = new Adam(net.Parameters());
        // }
        // 
        // 
        // private void Update()
        // {
        //     Tensor[] batches_i = input.Split(0, 8);
        //     Tensor[] batches_t = target.Split(0, 8);
        //     float mean_loss = 0f;
        //     for (int i = 0; i < batches_i.Length; i++)
        //     {
        //         var output = net.Forward(batches_i[i]);
        //         Loss mse = Loss.MSE(output, batches_t[i]);
        //         optim.ZeroGrad();
        //         net.Backward(mse.Gradient);
        //         optim.Step();
        //         mean_loss += mse.Item;
        //        
        //     }
        //     performanceGraph.Append(mean_loss / batches_i.Length);
        //     print($"Loss {mean_loss / batches_i.Length}");
        // }

    }

}

