using DeepUnity;
using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;

public class Test1 : MonoBehaviour
{
    public int Runs = 10;
    public int MatrixSize = 64;
    public ComputeShader matmulCS;
    public void Start()
    {
        //MatmulBenchmark();
        //TensorTest();
        //ForwardTest();
        //BackwardTest();
         LearningTest();
    }
    public void MatmulBenchmark()
    {
        var t1 = Tensor.Normal(MatrixSize, MatrixSize);
        // Debug.Log("T1: " + t1.ToString());

        var t2 = Tensor.Normal(MatrixSize, MatrixSize);
        // Debug.Log("T2: " + t2.ToString());


        DateTime start;
        TimeSpan end;

         //   start = DateTime.Now;
         //   
         //   for (int i = 0; i < Runs; i++)
         //   {
         //       Tensor.MatMul(t1, t2);
         //   }
         //   
         //   end = DateTime.Now - start;
         //   
         //   Debug.Log("Matmul on CPU: " + end);
        
        
        start = DateTime.Now;

        for (int i = 0; i < Runs; i++)
        {
            Tensor.MatMul(t1, t2, matmulCS);
        }

        end = DateTime.Now - start;

        Debug.Log("Matmul on GPU: " + end);
    }
    public void TensorTest()
    {
        var t1 = Tensor.Fill(3, 2, 2);
        print(t1);

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
            var input = Tensor.Random(1);
            var output = net.Forward(input);

            Debug.Log(input);
            Debug.Log(output);
        }
    }
    public void BackwardTest()
    {
        var net = new NeuralNetwork(
            new Dense(1, 5, WeightInit.Ones, Device.CPU),
            new Dense(5, 1, WeightInit.Ones, Device.CPU)
            );

        net.Compile(new Adam(), "somenet");


        var value = Tensor.Constant(1f);
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

        net.Compile(new SGD(), "somenet");

        int datasize = 10;
        int epochs = 100;

        // Generate inputs
        Tensor[] inputs = Enumerable.Range(0, datasize).ToList().
                          Select(x => Tensor.Constant(Utils.Random.Gaussian(0f, 3f, out _))).ToArray();
        Tensor[] targets = inputs.Select(x => Tensor.Constant(Mathf.Cos(x[0]))).ToArray();

        for (int ep = 0; ep < epochs; ep++)
        {
            List<float> accs = new List<float>();

            for (int i = 0; i < datasize; i++)
            {
                var prediction = net.Forward(inputs[i]);
                var loss = Loss.MSE(prediction, targets[i]);

                net.Backward(loss);
                net.Step();

                // Compute accuracy
                float acc = Metrics.Accuracy(prediction, targets[i]);
                accs.Add(acc);
            }

            Debug.Log($"Epoch {ep + 1} | Accuracy {accs.Average() * 100}%");
        }
    }
    public void ConvolutionTest()
    {

    }
}
