using DeepUnity;
using System;
using UnityEngine;


public class UnitTests : MonoBehaviour
{
    public NeuralNetwork blablabla;
    public int Runs = 10;
    public int MatrixSize = 64;
    public ComputeShader matmulCS;
    public void Start()
    {
        //MatMulCompare();
        //MatmulBenchmark();
        //TensorTest();
        //ForwardTest();
        //BackwardTest();
        //SaveTest();
    }
    public void MatMulCompare()
    {
        int tests = 10000;
        int succes = 0;

        for (int test = 0; test < tests; test++)
        {
            var w1 = UnityEngine.Random.Range(1, 4);
            var mid = UnityEngine.Random.Range(1, 4);
            var h2 = UnityEngine.Random.Range(1, 4);

            var t1 = Tensor.Random(w1, mid);
            var t2 = Tensor.Random(mid, h2);

            var m1 = Tensor.MatMul(t1, t2);
            var m2 = Tensor.MatMul(t1, t2, matmulCS);

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
            new Dense(1, 5, WeightInit.Ones),
            new Dense(5, 5, WeightInit.Ones),
            new Dense(5, 1, WeightInit.Ones),
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
            new Dense(1, 5, WeightInit.Ones, Device.CPU),
            new Dense(5, 5, WeightInit.Ones, Device.CPU),
            new Dense(5, 1, WeightInit.Ones, Device.CPU)
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