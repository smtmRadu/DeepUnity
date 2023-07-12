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
            //Test_MatMul();
            Benchmark_TensorTime();
            //Benchmark_TensorGPUTime();
            //Matmul_Tensor_vs_TensorGPU();
            //Matmul_Tensor_cpu_vs_gpu();
        }

        void Benchmark_TensorTime()
        {
            var t1 = Tensor.Random01(MatShape.x, MatShape.y);
            var t2 = Tensor.Random01(MatShape.x, MatShape.y);

            Timer.Start();
            for (int i = 0; i < Runs; i++)
            {
                if (matmulDevice == Device.CPU)
                    Tensor.MatMul(t1, t2);
                else
                    Tensor.MatMulGPU(t1, t2);
            }
            Timer.Stop();

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

            Tensor[] batches = Tensor.Split(data, Dim.height, 32);

            foreach (var batch in batches)
            {
                rn.Standardise(batch);
            }

            print(rn.Standardise(Tensor.Random01(10)));
        }
        void Test_MatMul()
        {
            int success = 0;
            for (int i = 0; i < Runs; i++)
            {
                int batch = (int)Utils.Random.Range(1, 2);
                int channels = (int)Utils.Random.Range(1, 2);
                int a = (int)Utils.Random.Range(1, 5);
                int b = (int)Utils.Random.Range(1, 5);
                int c = (int)Utils.Random.Range(1, 5);

                Tensor x = Tensor.Random01(batch, 1, a, b);              
                Tensor y = Tensor.Random01(channels, b, c);
                TensorGPU xgpu = TensorGPU.Identity(x);
                TensorGPU ygpu = TensorGPU.Identity(y);

                // Tensor -> MatMul and MatMulGPU
                var tensor_cpu = Tensor.MatMul(x, y);
                var tensor_gpu = Tensor.MatMulGPU(x, y);
                
                // TensorGPU -> MatMUl
                var tensorGPU_gpu = TensorGPU.MatMul(xgpu, ygpu);

                if (tensor_cpu.Equals(tensor_gpu) && tensor_cpu.Equals(tensorGPU_gpu))
                    success++;
            }

            print($"Accurracy: {(float)success / (float)Runs * 100}%");
        }

     
    }
}

