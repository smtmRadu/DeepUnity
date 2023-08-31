using UnityEngine;
using DeepUnity;
using System.Collections.Generic;
using UnityEngine.UI;
using System;

namespace kbRadu
{
    public class UnitTests : MonoBehaviour
    {
        public Device TestDevice;
        public Vector2Int MatShape = new Vector2Int(64, 64);
        public int batchSize = 32;
        public float lr = 0.001f;
        public int Runs = 100;
        public Optimizer optim;
        public RNN rnn_network;
        public Sequential net;
        public Image image;
        public Image image2;

        public PerformanceGraph graph = new PerformanceGraph();

        public int index = 0;
        public int whatIsIt = 0;
        List<(Tensor, Tensor)> train;

        /* private void Start()
         {
             Conv2DLearnTest();
         }

         Tensor input = Tensor.RandomNormal(64, 1, 28, 28);
         Tensor target = Tensor.RandomNormal(64, 5 * 26 * 26);
         void Conv2DLearnTest()
         {
             net = new Sequential(
                new Conv2D((1, 28, 28), out_channels: 5, kernel_size: 3, gamma_init: InitType.Random_Normal, beta_init: InitType.Random_Normal, device: TestDevice),
                new Flatten());

             optim = new Adam(net.Parameters(), lr: lr);
         }

         private void Update()
         {
             var pred = net.Forward(input);
             var loss = Loss.MSE(pred, target);
             net.Backward(loss.Derivative);
             // optim.ClipGradNorm(1f);
             optim.Step();
             print($"Epoch {Time.frameCount} | Loss: {loss.Item}");
             graph.Append(loss.Item);
         }*/


        void TestCrossEntropy()
        {
            var net = new Sequential(
                new Dense(10, 100),
                new Tanh(),
                new Dense(100, 5),
                new Softmax());
            var optim = new Adam(net.Parameters());

            Tensor input = Tensor.Random01(2048, 10);
            Tensor targets = Tensor.Zeros(2048, 5);
            for (int b = 0; b < 2048; b++)
            {
                Tensor mean = input.Mean(1);
                float val = mean[0];
                if (val < 0.4)
                    targets[b, 0] = 1;
                else if (val < 0.5)
                    targets[b, 1] = 1;
                else if(val < 0.6)
                    targets[b, 2] = 1;
                else if (val < 0.7)
                    targets[b, 3] = 1;
                else 
                    targets[b, 4] = 1;
            }

            var inputBatched = input.Split(0, 32);
            var targetBatches = targets.Split(0, 32);

            for (int i = 0; i < inputBatched.Length; i++)
            {
                var preds = net.Forward(inputBatched[i]);
                var loss = Loss.CrossEntropy(preds, targetBatches[i]);

                net.Backward(loss.Derivative);
                optim.Step();
                float acc = Metrics.Accuracy(preds, targetBatches[i]);
                graph.Append(acc);
                print(acc);   
            }
        }
        void RunAllModules()
        {
            var net = new Sequential(
                new Conv2D((1, 28, 28), 5, 3, device: TestDevice),
                new MaxPool2D(2),
                new Conv2D((5, 13, 13), 1, 3, device: TestDevice),
                new Flatten(),
                new ReLU(),
                new BatchNorm(121),
                new Dense(121, 100),
                new LayerNorm(),
                new Sigmoid(),
                new Dropout(),
                new Dense(100, 10),
                new Softmax()
                );

            var x = Tensor.Random01(2, 1, 28, 28);
            print(x);
            print(net.Forward(x));
            net.Backward(net.Forward(x));

            x = Tensor.Random01(32, 1, 28, 28);
            print(x);
            print(net.Forward(x));
            net.Backward(net.Forward(x));

            x = Tensor.Random01(1, 28, 28);
            print(x);
            print(net.Forward(x));
            net.Backward(net.Forward(x));
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

            ClockTimer.Start();
            for (int i = 0; i < Runs; i++)
            {
                var output = network.Forward(Tensor.Random01(32, 1, 28, 28));
                network.Backward(output);
            }
            
            ClockTimer.Stop();
        }
        void MaxPoolBenchmark()
        {
            Tensor input = Tensor.Random01(batchSize, 1, MatShape.x, MatShape.y);

            MaxPool2D mp = new MaxPool2D(5);

            ClockTimer.Start();
            for (int i = 0; i < Runs; i++)
            {
                mp.Predict(input);
            }
            ClockTimer.Stop();
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
            rnn_network = new RNN(10, 20, 2).CreateAsset("rnn");
            var input = Tensor.RandomNormal(6, 3, 10); // (L, B, H_in) 
            var h0 = Tensor.RandomNormal(2, 3, 20);  // (num_layers, B, H_out)
            var output = rnn_network.Forward(input, h0);
            rnn_network.Backward(output.Item1);
            // print("output" + output.Item1);
            // print("h_n" + output.Item2);
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
       


        void DenseTest()
        {
           
           

            print("64x64--------------------------------------------------------------------------------------------------------------");
            Dense dense = new Dense(64, 64, device: Device.CPU);
            Tensor input = Tensor.Random01(batchSize, 64);

            var out_cpu = dense.Forward(input);        
            var back_cpu = dense.Backward(out_cpu);    
            var gammaGrad_cpu = dense.gammaGrad;       
            var betaGrad_cpu = dense.betaGrad;         
              
            dense.device = Device.GPU;
            dense.gammaGrad = Tensor.Zeros(dense.gammaGrad.Shape);
            dense.betaGrad = Tensor.Zeros(dense.betaGrad.Shape);
            var out_GPU = dense.Forward(input);
            var back_GPU = dense.Backward(out_GPU);
            var gammaGrad_GPU = dense.gammaGrad;
            var betaGrad_GPU = dense. betaGrad;

            print("out: " + out_cpu.Equals(out_GPU));
            print("back " + back_cpu.Equals(back_GPU));
            print("gammaG " + gammaGrad_cpu.Equals(gammaGrad_GPU));
            print("betaG " + betaGrad_cpu.Equals(betaGrad_GPU));

            print(out_cpu);
            print(back_cpu);
            print(gammaGrad_cpu);
            print(betaGrad_cpu);

            print(out_GPU);
            print(back_GPU);
            print(gammaGrad_GPU);
            print(betaGrad_GPU);



            print("64x1--------------------------------------------------------------------------------------------------------------");
            dense = new Dense(64, 1, device: Device.CPU);
            input = Tensor.Random01(batchSize, 64);

            out_cpu = dense.Forward(input);
            back_cpu = dense.Backward(out_cpu);
            gammaGrad_cpu = dense.gammaGrad;
            betaGrad_cpu = dense.betaGrad;

            dense.device = Device.GPU; 
            dense.gammaGrad = Tensor.Zeros(dense.gammaGrad.Shape);
            dense.betaGrad = Tensor.Zeros(dense.betaGrad.Shape);

            out_GPU = dense.Forward(input);
            back_GPU = dense.Backward(out_GPU);
            gammaGrad_GPU = dense.gammaGrad;
            betaGrad_GPU = dense.betaGrad;

            print("out: " + out_cpu.Equals(out_GPU));
            print("back " + back_cpu.Equals(back_GPU));
            print("gammaG " + gammaGrad_cpu.Equals(gammaGrad_GPU));
            print("betaG " + betaGrad_cpu.Equals(betaGrad_GPU));

            print(out_cpu);
            print(back_cpu);
            print(gammaGrad_cpu);
            print(betaGrad_cpu);

            print(out_GPU);
            print(back_GPU);
            print(gammaGrad_GPU);
            print(betaGrad_GPU);






           

            print("1x64 --------------------------------------------------------------------------------------------------------------");
            dense = new Dense(1, 64, device: Device.CPU);
            input = Tensor.Random01(batchSize, 1);

            out_cpu = dense.Forward(input);
            back_cpu = dense.Backward(out_cpu);
            gammaGrad_cpu = dense.gammaGrad;
            betaGrad_cpu = dense.betaGrad;

            dense.device = Device.GPU;
            dense.gammaGrad = Tensor.Zeros(dense.gammaGrad.Shape);
            dense.betaGrad = Tensor.Zeros(dense.betaGrad.Shape);

            out_GPU = dense.Forward(input);
            back_GPU = dense.Backward(out_GPU);
            gammaGrad_GPU = dense.gammaGrad;
            betaGrad_GPU = dense.betaGrad;

            print("out: " + out_cpu.Equals(out_GPU));
            print("back " + back_cpu.Equals(back_GPU));
            print("gammaG " + gammaGrad_cpu.Equals(gammaGrad_GPU));
            print("betaG " + betaGrad_cpu.Equals(betaGrad_GPU));

            print(out_cpu);
            print(back_cpu);
            print(gammaGrad_cpu);
            print(betaGrad_cpu);

            print(out_GPU);
            print(back_GPU);
            print(gammaGrad_GPU);
            print(betaGrad_GPU);


            print("=============================================no batch===============================================================================");

           

            print("64x64--------------------------------------------------------------------------------------------------------------");
            dense = new Dense(64, 64, device: Device.CPU);
            input = Tensor.Random01(64);

            out_cpu = dense.Forward(input);
            back_cpu = dense.Backward(out_cpu);
            gammaGrad_cpu = dense.gammaGrad;
            betaGrad_cpu = dense.betaGrad;

            dense.device = Device.GPU;
            dense.gammaGrad = Tensor.Zeros(dense.gammaGrad.Shape);
            dense.betaGrad = Tensor.Zeros(dense.betaGrad.Shape);

            out_GPU = dense.Forward(input);
            back_GPU = dense.Backward(out_GPU);
            gammaGrad_GPU = dense.gammaGrad;
            betaGrad_GPU = dense.betaGrad;

            print("out: " + out_cpu.Equals(out_GPU));
            print("back " + back_cpu.Equals(back_GPU));
            print("gammaG " + gammaGrad_cpu.Equals(gammaGrad_GPU));
            print("betaG " + betaGrad_cpu.Equals(betaGrad_GPU));

            print(out_cpu);
            print(back_cpu);
            print(gammaGrad_cpu);
            print(betaGrad_cpu);

            print(out_GPU);
            print(back_GPU);
            print(gammaGrad_GPU);
            print(betaGrad_GPU);



            print("64x1--------------------------------------------------------------------------------------------------------------");
            dense = new Dense(64, 1, device: Device.CPU);
            input = Tensor.Random01(64);

            out_cpu = dense.Forward(input);
            back_cpu = dense.Backward(out_cpu);
            gammaGrad_cpu = dense.gammaGrad;
            betaGrad_cpu = dense.betaGrad;

            dense.device = Device.GPU;
            dense.gammaGrad = Tensor.Zeros(dense.gammaGrad.Shape);
            dense.betaGrad = Tensor.Zeros(dense.betaGrad.Shape);

            out_GPU = dense.Forward(input);
            back_GPU = dense.Backward(out_GPU);
            gammaGrad_GPU = dense.gammaGrad;
            betaGrad_GPU = dense.betaGrad;

            print("out: " + out_cpu.Equals(out_GPU));
            print("back " + back_cpu.Equals(back_GPU));
            print("gammaG " + gammaGrad_cpu.Equals(gammaGrad_GPU));
            print("betaG " + betaGrad_cpu.Equals(betaGrad_GPU));

            print(out_cpu);
            print(back_cpu);
            print(gammaGrad_cpu);
            print(betaGrad_cpu);

            print(out_GPU);
            print(back_GPU);
            print(gammaGrad_GPU);
            print(betaGrad_GPU);








            print("1x64 --------------------------------------------------------------------------------------------------------------");
            dense = new Dense(1, 64, device: Device.CPU);
            input = Tensor.Random01(1);


            out_cpu = dense.Forward(input);
            back_cpu = dense.Backward(out_cpu);
            gammaGrad_cpu = dense.gammaGrad;
            betaGrad_cpu = dense.betaGrad;

            dense.device = Device.GPU;
            dense.gammaGrad = Tensor.Zeros(dense.gammaGrad.Shape);
            dense.betaGrad = Tensor.Zeros(dense.betaGrad.Shape);

            out_GPU = dense.Forward(input);
            back_GPU = dense.Backward(out_GPU);
            gammaGrad_GPU = dense.gammaGrad;
            betaGrad_GPU = dense.betaGrad;

            print("out: " + out_cpu.Equals(out_GPU));
            print("back " + back_cpu.Equals(back_GPU));
            print("gammaG " + gammaGrad_cpu.Equals(gammaGrad_GPU));
            print("betaG " + betaGrad_cpu.Equals(betaGrad_GPU));

            print(out_cpu);
            print(back_cpu);
            print(gammaGrad_cpu);
            print(betaGrad_cpu);

            print(out_GPU);
            print(back_GPU);
            print(gammaGrad_GPU);
            print(betaGrad_GPU);
        }
        void DenseBenchmark()
        {
            Tensor input = Tensor.Fill(2, batchSize, MatShape.x);


            Dense dense = new Dense(MatShape.x, MatShape.y, device: TestDevice);
            ClockTimer.Start();
            for (int i = 0; i < Runs; i++)
            {
                var outp = dense.Forward(input);
                dense.Backward(outp);
            }
            ClockTimer.Stop();
        }


        void Conv2DTest()
        {
           
            print("Batch included");
            Tensor input = Tensor.Random01(1, 1, 28, 28);
            Conv2D conv2d = new Conv2D((1, 28, 28), 5, 3, device: Device.CPU);
            var output = conv2d.Forward(input);
            var back = conv2d.Backward(output);
            var gammaG = conv2d.gammaGrad;
            var betaG = conv2d.betaGrad;
            print(output);
            print(back);
            print(gammaG);
            print(betaG);

            conv2d.device = Device.GPU;
            conv2d.gammaGrad = Tensor.Zeros(conv2d.gammaGrad.Shape);
            conv2d.betaGrad = Tensor.Zeros(conv2d.betaGrad.Shape);
            output = conv2d.Forward(input);
            back = conv2d.Backward(output);
            gammaG = conv2d.gammaGrad;
            betaG = conv2d.betaGrad;
            print(output);
            print(back);
            print(gammaG);
            print(betaG);



            print("Batch not included");
            input = Tensor.Random01(1, 28, 28);
            conv2d = new Conv2D((1, 28, 28), 5, 3, device: Device.CPU);
            output = conv2d.Forward(input);
            back = conv2d.Backward(output);
            gammaG = conv2d.gammaGrad;
            betaG = conv2d.betaGrad;
            print(output);
            print(back);
            print(gammaG);
            print(betaG);

            conv2d.device = Device.GPU;
            conv2d.gammaGrad = Tensor.Zeros(conv2d.gammaGrad.Shape);
            conv2d.betaGrad = Tensor.Zeros(conv2d.betaGrad.Shape);
            output = conv2d.Forward(input);
            back = conv2d.Backward(output);
            gammaG = conv2d.gammaGrad;
            betaG = conv2d.betaGrad;
            print(output);
            print(back);
            print(gammaG);
            print(betaG);
        }
        void Conv2DBenchmark()
        {
            Conv2D c = new Conv2D((1, 28, 28), 64, 3, device: TestDevice);

            Tensor input = Tensor.Random01(batchSize, 1, 28, 28);
            ClockTimer.Start();
            for (int i = 0; i < Runs; i++)
            {
                var outp = c.Forward(input);
                c.Backward(outp);
            }
            ClockTimer.Stop();
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

                ClockTimer.Start();
                for (int i = 0; i < Runs; i++)
                {
                    Tensor.MatMul(t1, t2);
                }
                ClockTimer.Stop();
            }
            else
            {
                var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
                var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);

                ClockTimer.Start();
                for (int i = 0; i < Runs; i++)
                {
                    TensorGPU.MatMul(t1, t2);
                }
                ClockTimer.Stop();
            }

        }
        void Benchmark_TensorGPUTime()
        {
            var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
            var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);
           
            
            ClockTimer.Start();
            for (int i = 0; i < Runs; i++)
            {
                TensorGPU.MatMul(t1, t2);
            }
            ClockTimer.Stop();

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

