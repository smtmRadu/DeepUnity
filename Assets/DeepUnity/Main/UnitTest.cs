using UnityEngine;
using DeepUnity.Models;
using System.Collections.Generic;
using UnityEngine.UI;
using DeepUnity.Optimizers;
using DeepUnity.Modules;

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


        private MultiheadAttention mha;
        private Tensor x, y;
        private void Start()
        {
            x = Tensor.Zeros(1, 1);
            print(x.ToString());
            // Tensor.FusedAdamW(null, null, null, null, null,1,(1,1),(1,1),1,1, true, true);
            // ConvTranspose2D conv = new ConvTranspose2D(2, 3, 3);
            // 
            // Tensor input = Tensor.Random01(3, 2, 8, 8);
            // Tensor target = Tensor.Random01(3, 3, 10, 10);
            // 
            // var optim = new Adam(conv.Parameters());
            // for (int i = 0; i < 100; i++)
            // {
            // 
            //     Loss mse = Loss.MSE(conv.Forward(input), target);
            //     print(mse.Item);
            //     optim.ZeroGrad();
            //     conv.Backward(mse.Gradient);
            //     optim.Step();
            // }

            // ConvTranspose2D convt = new ConvTranspose2D(2, 3, 3);
            // Tensor input = Tensor.Random01(3, 2, 8, 8);
            // 
            // print(convt.Predict(input));
            // convt.Device = Device.GPU;
            // print(convt.Predict(input));

        }

        public void Update()
        {
            var yhat = mha.Forward(x);
            var loss = Loss.MSE(yhat, y);
            optim.ZeroGrad();
            mha.Backward(loss.Grad);
            optim.Step();
            print(loss.Item);
        }
            // private void Start()
            // {
            //     Conv2D conv = new Conv2D(2, 3, 3);
            // 
            //     Tensor target = Tensor.Random01(3, 3, 10, 10);
            //     Tensor input = Tensor.Random01(3, 2, 12, 12);
            // 
            //     var optim = new Adam(conv.Parameters());
            //     for (int i = 0; i < 100; i++)
            //     {
            // 
            //         Loss mse = Loss.MSE(conv.Forward(input), target);
            //         print(mse.Item);
            //         optim.ZeroGrad();
            //         conv.Backward(mse.Gradient);
            //         optim.Step();
            //     }
            // 
            // }
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


