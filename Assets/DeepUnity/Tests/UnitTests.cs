using UnityEngine;
using DeepUnity;
using Unity.VisualScripting;
using System.Collections.Generic;
namespace kbRadu
{
    public class UnitTests : MonoBehaviour
    {
        public Device TestDevice;
        public Vector2Int MatShape = new Vector2Int(64, 64);
        public int batchSize = 32;
        public int Runs = 100;
        public RNN rnn_network;
        public Sequential dnn_network;

        public Sprite sprite;
        private void Start()
        {
            // Conv2DBenchmark();
            // Conv2DTest();
            // Conv2DLearnTest();
            // CorrelationTest();

            // DenseTest();
            // foreach (var module in dnn_network.modules)
            // {
            //     if(module != null)
            //         print(module.GetType().Name);   
            // }
            // MNISTForwardBenchmark();
            // MaxPoolBenchmark();        
            //TestMaxPool();
        }

        void MNISTForwardBenchmark()
        {
            var network = new Sequential(
                new Conv2D((1, 28, 28), 5, 3),                    // outs (5, 26, 26)
                new ReLU(),
                new MaxPool2D(2),                               // outs (5, 13, 13)

                new Conv2D((5, 13, 13), 10, 3)    ,            // outs (10, 11, 11)
                 new ReLU(),
                new MaxPool2D(2),                               // outs (10, 5, 5)

                new Flatten(-3, -1),                            // outs (250)
                new Dense(250, 128, device: TestDevice),
                new Dropout(0.2f),
                new Dense(128, 10),
                new Softmax()
                );

            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {
                var output = network.Forward(Tensor.Random01(32, 1, 28, 28));
                network.Backward(output);
            }
            
            TimerX.Stop();
        }
        void MaxPoolBenchmark()
        {
            Tensor input = Tensor.Random01(batchSize, 1, MatShape.x, MatShape.y);

            MaxPool2D mp = new MaxPool2D(5);

            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {
                mp.Predict(input);
            }
            TimerX.Stop();
        }
        void TestMaxPool()
        {
            Tensor input = Tensor.Random01(2, 26, 26);
            print(input);
            MaxPool2D mp = new MaxPool2D(2);
            Tensor output = mp.Forward(input);
            print(output);
            mp.Backward(output);
            
        }

        void TestRNNCell()
        {
            var rnn = new RNNCell(10, 20);
            var input = Tensor.Split(Tensor.RandomNormal(6, 3, 10), 0, 1);
            var hx = Tensor.RandomNormal(3, 20);
            var output = new List<Tensor>();
            for (int i = 0; i < 6; i++)
            {
                hx = rnn.Forward(input[i].Squeeze(0), hx);
                output.Add(hx);
            }

            for (int i = 5; i >= 0; i--)
            {
                rnn.Backward(output[i]);
            }
            

            // print(output.ToLineSeparatedString());
        }
        void TestRNN()
        {
            rnn_network = new RNN(10, 20, 2);
            var input = Tensor.RandomNormal(6, 3, 10); // (L, B, H_in) 
            var h0 = Tensor.RandomNormal(2, 3, 20);  // (num_layers, B, H_out)
            var output = rnn_network.Forward(input, h0);
            rnn_network.Backward(output.Item1);

            rnn_network.Save("Some rnn");
            // print("output" + output.Item1);
            // print("h_n" + output.Item2);
        }

        void Conv2DLearnTest() 
        {
            Conv2D conv2d = new Conv2D((3, 28, 28), 1, 3, device: TestDevice);
            Tensor input = Tensor.RandomNormal(3, 28, 28);
            Tensor target = Tensor.RandomNormal(1, 26, 26);
            Optimizer optim = new Adam(new Learnable[] { conv2d }, lr:0.001f);

            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {
                var pred = conv2d.Forward(input);
                var loss = (pred - target) * (pred - target);
                var lossderiv = (pred - target) * 2;
                conv2d.Backward(lossderiv);
                optim.Step();
                print("Loss: " + Tensor.Mean(loss.Reshape(loss.Count()),0)[0]);
            }
            TimerX.Stop();
        }
        void CorrelationTest()
        {
            Tensor input = Tensor.Constant(new float[,] { { 1, 6, 2 }, { 5, 3, 1 }, { 7, 0, 4 } });
            Tensor kernel = Tensor.Constant(new float[,] {{ 1, 2 }, { -1, 0 } });
            print(input);
            print(kernel);
            print(Tensor.Correlate2D(input, kernel, CorrelationMode.Valid));
            print(Tensor.Convolve2D(input, kernel, CorrelationMode.Valid));
        }
        void Conv2DTest()
        {
            Conv2D conv = new Conv2D((1, 4, 4), 3, 2, device: Device.CPU);
            Tensor input = Tensor.Random01(3, 1, 4, 4);


            Tensor outputCPU = conv.Forward(input);
            Tensor lossCPU = conv.Backward(outputCPU);
            print(outputCPU);
            print(lossCPU);
            print("gammaGrad " + conv.gammaGrad);

            conv.gammaGrad = Tensor.Zeros(conv.gammaGrad.Shape);
            conv.betaGrad = Tensor.Zeros(conv.betaGrad.Shape);

            conv.device = Device.GPU;
            Tensor outputGPU = conv.Forward(input);
            Tensor lossGPU = conv.Backward(outputGPU);
            print(outputGPU);
            print(lossGPU);
            print("gammaGrad " + conv.gammaGrad);


        }
        void Conv2DBenchmark()
        {
            Conv2D c = new Conv2D((1, 28, 28), 64, 3, device: TestDevice);

            Tensor input = Tensor.Random01(32, 1, 28, 28);
            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {
                var outp = c.Forward(input);
                c.Backward(outp); 
            }
            TimerX.Stop();
        }


        void DenseTest()
        {
           
            Tensor input = Tensor.Fill(2, 32, 64);

            print("64x64 CPU");
            Dense dense = new Dense(64, 64, init: InitType.Debug, device: Device.CPU);
            var outp = dense.Forward(input);
            print(outp);
            var loss = dense.Backward(outp);
            print(loss);

            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("64x64 GPU");
            dense = new Dense(64, 64, init: InitType.Debug, device: Device.GPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("64x1 CPU");
            dense = new Dense(64, 1, init: InitType.Debug, device: Device.CPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("64x1 GPU");
            dense = new Dense(64, 1, init: InitType.Debug, device: Device.GPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad); ;





            input = Tensor.Fill(2, batchSize, 1);
            print("1x64 CPU");
            dense = new Dense(1, 64, init: InitType.Debug, device: Device.CPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("1x64 GPU");
            dense = new Dense(1, 64, init: InitType.Debug, device: Device.GPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("-------no batch---------------------");

            input = Tensor.Fill(2, 64);
            print("64x64 CPU");
            dense = new Dense(64, 64, init: InitType.Debug, device: Device.CPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("64x64 GPU");
            dense = new Dense(64, 64, init: InitType.Debug, device: Device.GPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);




            print("64x1 CPU");
            dense = new Dense(64, 1, init: InitType.Debug, device: Device.CPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("64x1 GPU");
            dense = new Dense(64, 1, init: InitType.Debug, device: Device.GPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);


            input = Tensor.Fill(2, 1);
            print("1x64 CPU");
            dense = new Dense(1, 64, init: InitType.Debug, device: Device.CPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);

            print("1x64 GPU");
            dense = new Dense(1, 64, init: InitType.Debug, device: Device.CPU);
            outp = dense.Forward(input);
            print("output" + outp);
            loss = dense.Backward(outp);
            print("back" + loss);
            print("GammaGrad: " + dense.gammaGrad);
            print("BetaGrad: " + dense.betaGrad);
        }
        void DenseVSDenseGPU_Benchmark()
        {
            Tensor input = Tensor.Fill(2, batchSize, MatShape.x);


            Dense dense = new Dense(MatShape.x, MatShape.y, init: InitType.Debug, device: TestDevice);
            TimerX.Start();
            for (int i = 0; i < Runs; i++)
            {
                var outp = dense.Forward(input);
                dense.Backward(outp);
            }
            TimerX.Stop();
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

