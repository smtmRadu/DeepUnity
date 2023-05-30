using DeepUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Test1 : MonoBehaviour
{
    public ComputeShader matmulCS;
    public void Start()
    {
        // var t1 = Tensor<float>.Normal(2, 2);
        // var t2 = Tensor<float>.Normal(2, 2);
        // var dif = t1.Zip(t2, (x, y) => x - y);
        // 
        // Debug.Log(t1);
        // Debug.Log(t2);
        // Debug.Log(dif);
        //MatmulBenchmark();
        //ForwardTest();
        //SimpleBackwardTest();
        LearningTest();
    }
    public void MatmulBenchmark()
    {
        int size = 256;

        var t1 = Tensor<float>.Normal(size, size);
        //Debug.Log("T1: " + t1.ToString());

        var t2 = Tensor<float>.Normal(size, size);
        //Debug.Log("T2: " + t2.ToString());

        //float[] tarr = t1.ToArray();
        //Debug.Log("Arr: " + string.Join(", ", tarr));

        int runs = 100;

        var start = DateTime.Now;

        // for (int i = 0; i < runs; i++)
        // {
        //     Tensor<float>.MatMul(t1, t2);
        // }
        // 
        // Debug.Log("Matmul on CPU: " + (DateTime.Now - start));
        // 
        // 
        // start = DateTime.Now;

        for (int i = 0; i < runs; i++)
        {
            Tensor<float>.MatMul(t1, t2, matmulCS);
        }

        Debug.Log("Matmul on GPU: " + (DateTime.Now - start));


        // var mul = Tensor<float>.MatMul(t1, t2);
        // Debug.Log("Mul: \n" + mul.ToString());
        // 
        // var mulCS = Tensor<float>.MatMul(t1, t2, matmulCS);
        // Debug.Log("MulCS: \n" + mulCS.ToString());

    }
    public void ForwardTest()
    {
        var net = new NeuralNetwork(
            new Dense(1, 5),
            new TanH(),
            new Dense(5, 5),
            new ReLU(),
            new Dense(5, 1),
            new Linear()
            );

        net.Compile(new Adam(), "somenet");

        for (int i = 0; i < 100; i++)
        {
            var input = Tensor<float>.Random(1);
            var output = net.Forward(input);

            Debug.Log(input);
            Debug.Log(output);
        }
    }
    public void Backward()
    {
        var net = new NeuralNetwork(
            new Dense(1, 5, WeightInit.Ones, Device.CPU),
            new Dense(5, 1, WeightInit.Ones, Device.CPU)
            );

        net.Compile(new Adam(), "somenet");


        var value = Tensor<float>.Constant(1f);
        var outs = net.Forward(value);
        var back = net.Backward(value);

        Debug.Log(outs.ToString());
        Debug.Log(back.ToString());
    }
    public void LearningTest()
    {
        var net = new NeuralNetwork(
            new Dense(1, 5),
            new ReLU(),
            new Dense(5, 5),
            new ReLU(),
            new Dense(5, 1)
            );

        net.Compile(new Adam(), "somenet");

        int datasize = 10;
        int epochs = 10;

        // Generate inputs
        Tensor<float>[] inputs = Enumerable.Range(0, datasize).ToList().Select(x => Tensor<float>.Constant(Utils.Random.Gaussian(0f, 3f, out _))).ToArray();
        Tensor<float>[] targets = inputs.Select(x => Tensor<float>.Constant(Mathf.Cos(x[0]))).ToArray();

        for (int ep = 0; ep < epochs; ep++)
        {
            for (int i = 0; i < datasize; i++)
            {
                var prediction = net.Forward(inputs[i]);
                var loss = Loss.MSE(prediction, targets[i]);

                var lastloss = net.Backward(loss);

                net.Step();
            }

            // List<float> accs = new List<float>();
            // // Check accuracy
            // for (int i = 0; i < datasize; i++)
            // {
            //     var prediction = net.Forward(inputs[i]);
            // 
            //     Debug.Log(inputs[i].ToString());
            //     Debug.Log(prediction.ToString());
            //     Debug.Log(targets[i].ToString());
            // 
            //     float acc = Metrics.Accuracy(prediction, targets[i]);
            //     accs.Add(acc);
            // }
            // Debug.Log($"Epoch {ep + 1} | Accuracy {accs.Average() * 100}%");
        }
    }
    public void ConvolutionTest()
    {

    }
}
