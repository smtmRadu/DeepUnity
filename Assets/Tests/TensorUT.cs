using UnityEngine;
using DeepUnity;
using System;

namespace kbRadu
{
    public class TensorUT : MonoBehaviour
    {
        public Device Device;
        public Vector2Int MatShape = new Vector2Int(64, 64);
        public int Runs = 100;

        private void Start()
        {
            var t = Tensor.Random01(MatShape.x, MatShape.y);
            var t2 = Tensor.Random01(MatShape.y, MatShape.x);

            Timer.Start();
            for (int i = 0; i < Runs; i++)
            {
                Tensor.MatMul(t, t2);
            }
           
            Timer.Stop();
        }

        void TensorGPUTime()
        {
            var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
            var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);

            CamSensor ge;
           
            
            Timer.Start();
            for (int i = 0; i < Runs; i++)
            {
                TensorGPU.MatMul(t1, t2);
            }
            Timer.Stop();

        }
        void TensorCPUTime()
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
        void StdTest()
        {
            RunningStandardizer rn = new RunningStandardizer(10);

            Tensor data = Tensor.RandomRange((0, 360), 1024, 10);

            Tensor[] batches = Tensor.Split(data, TDim.height, 32);

            foreach (var batch in batches)
            {
                rn.Standardise(batch);
            }

            print(rn.Standardise(Tensor.Random01(10)));
        }
        void TestCorrelation()
        {
            Tensor input = Tensor.Random01(32, 3, 28, 28);

            Conv2D c = new Conv2D(3, 64, 3);

            DeepUnity.Timer.Start();
            Tensor y = c.Predict(input);
            DeepUnity.Timer.Stop();

            print(input);
            print(y);

        }
        void MatMulTest()
        {

            int goods = 0;
            for (int i = 0; i < Runs; i++)
            {
                int a = (int)Utils.Random.Range(1, 5);
                int b = (int)Utils.Random.Range(1, 5);
                int c = (int)Utils.Random.Range(1, 5);

                Tensor x = Tensor.Random01(a, b);
                Tensor y = Tensor.Random01(b, c);

                DeepUnityMeta.device = Device.CPU;
                var cpu = Tensor.MatMul(x, y);

                DeepUnityMeta.device = Device.GPU;
                var gpu = Tensor.MatMul(x, y);

                if (cpu.Equals(gpu))
                    goods++;
            }

            print($"Accurracy: {(float)goods/(float)Runs * 100}%");
        }

        void MatMulBenchmark()
        {
            DeepUnityMeta.device = Device;
            Tensor x = Tensor.Random01(MatShape.x, MatShape.y);
            Tensor y = Tensor.Random01(MatShape.y, MatShape.x);

            DateTime start;
            TimeSpan end;

            start = DateTime.Now;
            for (int i = 0; i < Runs; i++)
            {
                Tensor.MatMul(x, y);
            }
            end = DateTime.Now - start;
            print($"{Runs} runs on {DeepUnityMeta.device} on {x.Shape} * {y.Shape} in: {end}");

        }
    }
}
