using UnityEngine;
using DeepUnity;
using System;
using System.Collections.Generic;

namespace kbRadu
{
    public class TensorManualTests : MonoBehaviour
    {
        public Device matmulDevice;
        public Vector2Int MatShape = new Vector2Int(64, 64);
        public int Runs = 100;

        private void Start()
        {
            //Test_TensorOperaitons();
            //Test_MatMul();
            //Benchmark_Matmul_time();
            //Benchmark_TensorGPUTime();
            //Matmul_Tensor_vs_TensorGPU();
            //Matmul_Tensor_cpu_vs_gpu();
        }

        void Test_MatMulUnbalanced()
        {
            Tensor x = Tensor.Random01(1, 4);
            Tensor y = Tensor.Random01(4);
            print(x);
            print(y);
            print("x * y" + Tensor.MatMul(x, y));

            x = Tensor.Random01(3);
            y = Tensor.Random01(3, 1);
            print(x);
            print(y);
            print("x * y" + Tensor.MatMul(x, y));

            x = Tensor.Random01(1);
            y = Tensor.Random01(1, 4);
            print(x);
            print(y);
            print("x * y" + Tensor.MatMul(x, y));

            x = Tensor.Random01(4, 1);
            y = Tensor.Random01(1);
            print(x);
            print(y);
            print("x * y" + Tensor.MatMul(x, y));
        }
        void Benchmark_Matmul_time()
        {
            if(matmulDevice == Device.CPU)
            {
                var t1 = Tensor.Random01(MatShape.x, MatShape.y);
                var t2 = Tensor.Random01(MatShape.x, MatShape.y);

                Timer.Start();
                for (int i = 0; i < Runs; i++)
                {
                    Tensor.MatMul(t1, t2);
                }
                Timer.Stop();
            }
            else
            {
                var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
                var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);

                Timer.Start();
                for (int i = 0; i < Runs; i++)
                {
                    TensorGPU.MatMul(t1, t2);
                }
                Timer.Stop();
            }

        }
        void Benchmark_TensorGPUTime()
        {
            var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
            var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);
           
            
            Timer.Start();
            for (int i = 0; i < Runs; i++)
            {
                TensorGPU.MatMul(t1, t2);
            }
            Timer.Stop();

        }
        void StdTest()
        {
            RunningStandardizer rn = new RunningStandardizer(10);

            Tensor data = Tensor.RandomRange((0, 360), 1024, 10);

            Tensor[] batches = Tensor.Split(data, 0, 32);

            foreach (var batch in batches)
            {
                rn.Standardise(batch);
            }

            print(rn.Standardise(Tensor.Random01(10)));
        }

        void Test_TensorOperaitons()
        {
            Tensor x = Tensor.Random01(10, 10);
            Tensor y = Tensor.Random01(10, 10);

            print(x);
            print(Tensor.Split(x, 0, 3)[0]);
            print(Tensor.Mean(x, 1));
            print(Tensor.Mean(x, 1));

            print(Tensor.Mean(x, 0));
            print(Tensor.Mean(x, 0));
        }

    }
}

