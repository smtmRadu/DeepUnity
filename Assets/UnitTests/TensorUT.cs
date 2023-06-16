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
            MatMulBenchmark();
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

                Settings.Device = Device.CPU;
                var cpu = Tensor.MatMul(x, y);

                Settings.Device = Device.GPU;
                var gpu = Tensor.MatMul(x, y);

                if (cpu.Equals(gpu))
                    goods++;
            }

            print($"Accurracy: {(float)goods/(float)Runs * 100}%");
        }

        void MatMulBenchmark()
        {
            Settings.Device = Device;
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
            print($"{Runs} runs on {Settings.Device} on {x.ShapeToString} * {y.ShapeToString} in: {end}");

        }
    }
}

