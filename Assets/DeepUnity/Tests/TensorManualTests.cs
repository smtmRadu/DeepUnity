using UnityEngine;
using DeepUnity;
using Unity.VisualScripting;

namespace kbRadu
{
    public class TensorManualTests : MonoBehaviour
    {
        public Device TestDevice;
        public Vector2Int MatShape = new Vector2Int(64, 64);
        public int Runs = 100;

        private void Start()
        {
            // CPUvsGPU();
            TestDense();
            
        }

        void CPUvsGPU_Speed()
        {
            if(TestDevice == Device.CPU)
            {
                Tensor x = Tensor.Random01(10, 100);

                TimerX.Start();
                for (int i = 0; i < Runs; i++)
                {
                    Tensor.Expand(x, 1, 10);
                }
                TimerX.Stop();
            }
            else
            {
                TensorGPU x = TensorGPU.Random01(10, 100);

                TimerX.Start();
                for (int i = 0; i < Runs; i++)
                {
                    TensorGPU.Expand(x, 1, 10);
                }
                TimerX.Stop();
            }
        }

        void TestDense()
        {
            Dense dense = new Dense(MatShape.x, MatShape.y, device: TestDevice);
            
            Tensor input = Tensor.Random01(8, MatShape.x);
            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {            
                var pred = dense.Forward(input);
                var loss = dense.Backward(pred);
            }
            TimerX.Stop();
        }
        void TestConv2d()
        {
            Conv2D conv2d = new Conv2D((1, 28, 28), 5, 3);

            Tensor input = Tensor.Random01(1, 1, 28, 28);

            for (int i = 0; i < Runs; i++)
            {
                TimerX.Start();
                var pred = conv2d.Forward(input);
                var loss = conv2d.Backward(pred);
                print(input.Shape.ToCommaSeparatedString());
                print(pred.Shape.ToCommaSeparatedString());
                print(loss.Shape.ToCommaSeparatedString());
                TimerX.Stop();
            }

        }
        void TestRot180d()
        {
            Tensor kernels = Tensor.Random01(64, 1, 3, 3);
            int outChannels = kernels.Size(-4);
            int inChannels = kernels.Size(-3);
            int kernelSize = kernels.Size(-1);
            Tensor rot180dKernels = Tensor.Zeros(kernels.Shape);
            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int h = 0; h < kernelSize; h++)
                        {
                            for (int w = 0; w < kernelSize; w++)
                            {
                                rot180dKernels[oc, ic, kernelSize - h - 1, kernelSize - w - 1] = kernels[oc, ic, h, w];
                            }
                        }
                    }
                }
            }
            // for (int i = 0; i < Runs; i++)
            // {
            //     Parallel.For(0, outChannels, oc =>
            //     {
            //         for (int ic = 0; ic < inChannels; ic++)
            //         {
            //             for (int h = 0; h < kernelSize; h++)
            //             {
            //                 for (int w = 0; w < kernelSize; w++)
            //                 {
            //                     rot180dKernels[oc, ic, kernelSize - h - 1, kernelSize - w - 1] = kernels[oc, ic, h, w];
            //                 }
            //             }
            //         }
            //     });
            // }
            TimerX.Stop();
            
           
            print(kernels);
            print(rot180dKernels);
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
            if(TestDevice == Device.CPU)
            {
                var t1 = Tensor.Random01(MatShape.x, MatShape.y);
                var t2 = Tensor.Random01(MatShape.x, MatShape.y);

                TimerX.Start();
                for (int i = 0; i < Runs; i++)
                {
                    Tensor.MatMul(t1, t2);
                }
                TimerX.Stop();
            }
            else
            {
                var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
                var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);

                TimerX.Start();
                for (int i = 0; i < Runs; i++)
                {
                    TensorGPU.MatMul(t1, t2);
                }
                TimerX.Stop();
            }

        }
        void Benchmark_TensorGPUTime()
        {
            var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
            var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);
           
            
            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {
                TensorGPU.MatMul(t1, t2);
            }
            TimerX.Stop();

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

