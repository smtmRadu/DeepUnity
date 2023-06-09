using DeepUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


public class UnitTests : MonoBehaviour
{
    public NeuralNetwork network;
    public int Runs = 10;
    public Device device;
    public int MatrixSize = 64;
    public InitType init;

    public void Start()
    {
        //MatmulBenchmark();
        //BenchmarkFoward();
        //BenchmarkBackward();
        //BenchmarkStep();

        // var t = Tensor.Random(10,10);
        // print(t);
        // t.ForEach(x => x / 2f);
        // print(t);
        //MatMulCompare();
        //MatmulBenchmark();
        //TensorTest();
        //ForwardTest();
        //BackwardTest();
        //SaveTest();

        if(!network)
        {
            network = new NeuralNetwork(
                 new Dense(1, MatrixSize, init, device),
                 new ReLU(),
                 new Dense(MatrixSize, MatrixSize, init, device),
                 new ReLU(),
                 new Dense(MatrixSize, 1, init, device),
                 new Linear());

            network.Compile(new Adam(), "none");
        }
 

        inputs = Tensor.Normal(1, 100);
        outputs = inputs.Select(x => MathF.Cos(x));

        slicedinputs = Tensor.Slice(inputs, 1);
        slicedoutputs = Tensor.Slice(outputs, 1);
    }


    int t = 1;

    Tensor inputs;
    Tensor outputs;

    Tensor[] slicedinputs;
    Tensor[] slicedoutputs;



    public void Update()
    {
        t++;
        var errors = new List<float>();

        for (int i = 0; i < slicedinputs.Length; i++)
        {

            var pred = network.Forward(slicedinputs[i]);


            var loss = Loss.MSE(pred, slicedoutputs[i]);

            network.ZeroGrad();
            network.Backward(loss);
            network.Step();


            errors.Add(Metrics.Accuracy(pred, slicedoutputs[i]));

        }

        print($"Epoch {t + 1} | Accuracy {errors.Average() * 100}%");
    }
    public void BenchmarkFoward()
    { 
        NeuralNetwork net = new NeuralNetwork(
            new Dense(10, MatrixSize, device: device),
            new ReLU(),
            new Dense(MatrixSize, MatrixSize, device: device),
            new ReLU(), 
            new Dense(MatrixSize, 10, device: device));

        Tensor input = Tensor.Random(10);

        var start = DateTime.Now;

        for (int i = 0; i < Runs; i++)
        {
            net.Forward(input);
        }

        var end = DateTime.Now - start;

        Debug.Log($"Forward {Runs} runs in {end}");
    }
    public void BenchmarkBackward()
    {
        NeuralNetwork net = new NeuralNetwork(
            new Dense(10, MatrixSize, device: device),
            new ReLU(),
            new Dense(MatrixSize, MatrixSize, device: device),
            new ReLU(),
            new Dense(MatrixSize, 10, device: device));

        Tensor loss = Tensor.Random(10);
        
        var start = DateTime.Now;

        for (int i = 0; i < Runs; i++)
        {
            net.Forward(loss);
            net.Backward(loss);
        }

        var end = DateTime.Now - start;

        Debug.Log($"Backward {Runs} runs in {end}");
    }
    public void BenchmarkStep()
    {
        NeuralNetwork net = new NeuralNetwork(
           new Dense(10, MatrixSize, device: device),
           new ReLU(),
           new Dense(MatrixSize, MatrixSize, device: device),
           new ReLU(),
           new Dense(MatrixSize, 10, device: device));

        net.Compile(new Adam(), "non");

        Tensor loss = Tensor.Zeros(10);

        var start = DateTime.Now;

        for (int i = 0; i < Runs; i++)
        {
            net.Forward(loss);
            net.Backward(loss);
            net.Step();
        }

        var end = DateTime.Now - start;

        Debug.Log($"Step {Runs} runs in {end}");
    }
    public void MatMulCompare()
    {
        int tests = 1000;
        int succes = 0;

        for (int test = 0; test < tests; test++)
        {
            var w1 = UnityEngine.Random.Range(1, 6);
            var mid = UnityEngine.Random.Range(1, 6);
            var h2 = UnityEngine.Random.Range(1, 6);

            var t1 = Tensor.Random(w1, mid);
            var t2 = Tensor.Random(mid, h2);

            var m1 = Tensor.MatMul(t1, t2, Device.CPU);
            var m2 = Tensor.MatMul(t1, t2, Device.GPU);

            if(m1.Equals(m2))
                succes++;
            else
            {
                print("T1: \n" + m1);
                print("T2: \n" + m2);
            }
        }

        Debug.Log((float)succes / tests * 100f + "%");
    }
    public void MatmulBenchmark()
    {
        var t1 = Tensor.Normal(MatrixSize, MatrixSize);
        // Debug.Log("T1: " + t1.ToString());

        var t2 = Tensor.Normal(MatrixSize, MatrixSize);
        // Debug.Log("T2: " + t2.ToString());


        DateTime start;
        TimeSpan end;

        start = DateTime.Now;

        for (int i = 0; i < Runs; i++)
        {
            Tensor.MatMul(t1, t2, device);
        }

        end = DateTime.Now - start;

        Debug.Log($"Matmul on {device}: " + end);
    }
    public void TensorTest()
    {
        var t1 = Tensor.Fill(3, 2, 2);
        print(t1);

    }
    public void ForwardTest()
    {
        var net = new NeuralNetwork(
            new Dense(1, 5, InitType.Uniform),
            new Dense(5, 5, InitType.Uniform),
            new Dense(5, 1, InitType.Uniform),
            new Linear()
            );

        net.Compile(new Adam(), "somenet");

        for (int i = 0; i < 100; i++)
        {
            var input = Tensor.Constant(1);
            var output = net.Forward(input);

            Debug.Log(input);
            Debug.Log(output);
        }
    }
    public void BackwardTest()
    {
        var net = new NeuralNetwork(
            new Dense(1, 5, InitType.Uniform, Device.CPU),
            new Dense(5, 5, InitType.Uniform, Device.CPU),
            new Dense(5, 1, InitType.Uniform, Device.CPU)
            );

        net.Compile(new Adam(), "somenet");


        var input = Tensor.Constant(1f);
        var output = net.Forward(input);
        var back = net.Backward(output);

        Debug.Log(input);
        Debug.Log(output);
        Debug.Log(back);
    }
    public void SaveTest()
    {
        // if (blablabla == null)
        // {
        //    blablabla = new NeuralNetwork(
        //    new Dense(1, 5),
        //    new ReLU(),
        //    new Dense(5, 5),
        //    new ReLU(),
        //    new Dense(5, 1)
        //    );
        // 
        //    blablabla.Compile(new Adam(), "strs");
        // 
        //    blablabla.Save();
        // }


        // var ford = blablabla.Forward(Tensor.Constant(1));
        // print(string.Join(", ", ford.Shape));
        // print(ford);
        // 
        // var c1 = Tensor.Constant(1);
        // print(string.Join(", ", c1.Shape));
        // print(c1);

    }
   
}
