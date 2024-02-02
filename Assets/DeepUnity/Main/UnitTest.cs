using DeepUnity;
using System;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class UnitTest : MonoBehaviour
    {

        public NeuralNetwork net;
        Optimizer optim;
        [SerializeField] PerformanceGraph graph = new PerformanceGraph();

        Tensor input = Tensor.RandomNormal(64, 6, 10);
        Tensor target = Tensor.Random01(64, 1);

        private void Start()
        {
            Sparsemax sm = new Sparsemax();

            Tensor input = Tensor.RandomNormal(10);
            var output = sm.Forward(input);
            print(output);
            print(sm.Backward(output / 2));

            // Tensor input = Tensor.Arange(0, 9, 1);
            // Tensor targ = Tensor.Zeros(9); targ[7] = 1;
            // Softmax softmax = new Softmax();    
            // var output = softmax.Forward(input);
            // print(output);
            // 
            // Loss g = Loss.MSE(output, targ);
            // print(softmax.Backward(g.Gradient));
            // 
            // return;

            // net = new NeuralNetwork(
            //     new Dense(10, 64),
            //     new Sigmoid(),
            //     new Dense(64, 64),
            //        new Sigmoid(),
            //     new Dense(64, 2),
            //     new Softmax());
            // 
            // optim = new SGD(net.Parameters(), 0.0001f, 0f);
            // 
            // List<(Tensor, Tensor)> data;
            // Datasets.BinaryClassification(out data);
            // 
            // for (int k = 0; k < 100; k++)
            // {
            //     for (int i = 0; i < data.Count; i++)
            //     {
            //         var outp = net.Forward(data[i].Item1);
            //         optim.ZeroGrad();
            //         Loss mse = Loss.MSE(outp, data[i].Item2);
            //         print(mse.Item);
            //         net.Backward(mse.Gradient);
            //         optim.Step();
            // 
            //         graph.Append(mse.Item);
            //     }
            // }
            // 
            // print(net.Forward(Tensor.Constant(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1f })));
        }
    }

}


