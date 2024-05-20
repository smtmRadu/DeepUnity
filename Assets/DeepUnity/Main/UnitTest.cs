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

        private void Start()
        {

            
            // Tensor a = Tensor.Random01(2, 3, 4);
            // 
            // print(a);
            // Flatten flt = new Flatten();
            // var output = flt.Forward(a);
            // print(output);
            // print(flt.Backward(output));
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


