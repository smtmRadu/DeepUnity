using UnityEngine;
using DeepUnity;
using System.Collections.Generic;
using UnityEngine.UI;

namespace kbRadu
{
    public class UnitTests : MonoBehaviour
    {
        public Material material;
        public Device TestDevice;
        public PerformanceGraph graph = new PerformanceGraph();
        public Vector2Int MatShape = new Vector2Int(64, 64);
        public int batchSize = 32;
        public float lr = 0.001f;
        public int Runs = 100;
        public Optimizer optim;
        public RNN rnn_network;
        public NeuralNetwork net;
        public RawImage image;
        public RawImage image2;

      

        public int index = 0;
        public int whatIsIt = 0;

        public NeuralNetwork network;
        public AgentBehaviour beh;
        private void Start()
        {
            PReLU prelu = new PReLU(0.01f);
            print(prelu.ParametersCount());
            print(prelu.Parameters());
        }

        public void TestCPU()
        {
            Tensor tensor1 = Tensor.RandomNormal(MatShape.x, MatShape.y);

            Tensor tensor2 = Tensor.RandomNormal(MatShape.x, MatShape.y);


            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                Tensor.MatMul(tensor1, tensor2);
            }
            
            TimeKeeper.Stop();
        }

        public void TestGPU()
        {
            TensorGPU tensor1 = TensorGPU.RandomNormal(MatShape.x, MatShape.y);

            TensorGPU tensor2 = TensorGPU.RandomNormal(MatShape.x, MatShape.y);


            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                TensorGPU.MatMul(tensor1, tensor2);
            }

            TimeKeeper.Stop();
        }
        // private void Start()
        // {
        //     rnn_network = new RNN(10, 20, 2).CreateAsset("rnn");
        //     optim = new Adam(rnn_network.Parameters(), lr: 1e-4f);
        // }
        // 
        // Tensor input = Tensor.RandomNormal(6, 64, 10); // (L, B, H_in) 
        // Tensor h0 = Tensor.Zeros(2, 64, 20);  // (num_layers, B, H_out)
        // Tensor targ = Tensor.RandomNormal(6, 64, 20);
        // private void Update()
        // {
        //     
        //     optim.ZeroGrad();
        //     var output = rnn_network.Forward((input, h0));
        //     var loss = Loss.MSE(output.Item1, targ);
        //     rnn_network.Backward((loss.Derivative, null));
        //     optim.Step();
        // 
        //     graph.Append(loss.Item);
        // }
        void TestAttention()
        {
            Attention att = new Attention((28, 28), 6);

            Tensor input = Tensor.Random01(28, 28);

            print(input);

            print(att.Forward(input));
        }

        void TestCrossEntropy()
        {
            var net = new NeuralNetwork(
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
                var loss = Loss.CE(preds, targetBatches[i]);

                net.Backward(loss.Derivative);
                optim.Step();
                float acc = Metrics.Accuracy(preds, targetBatches[i]);
                graph.Append(acc);
                print(acc);   
            }
        }
        void RunAllModules()
        {
            var net = new NeuralNetwork(
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
            var network = new NeuralNetwork(
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

            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                var output = network.Forward(Tensor.Random01(32, 1, 28, 28));
                network.Backward(output);
            }
            
            TimeKeeper.Stop();
        }
        void MaxPoolBenchmark()
        {
            Tensor input = Tensor.Random01(batchSize, 1, MatShape.x, MatShape.y);

            MaxPool2D mp = new MaxPool2D(5);

            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                mp.Predict(input);
            }
            TimeKeeper.Stop();
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
            var rnn = new RecurrentDense(10, 20);
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



        void CorrelationTest()
        {
            Tensor input = Tensor.Constant(new float[,] { { 1, 6, 2 }, { 5, 3, 1 }, { 7, 0, 4 } });
            Tensor kernel = Tensor.Constant(new float[,] {{ 1, 2 }, { -1, 0 } });
            print(input);
            print(kernel);
            print(Tensor.Correlate2D(input, kernel, CorrelationMode.Valid));
            print(Tensor.Convolve2D(input, kernel, CorrelationMode.Valid));
        }
       



        void DenseBenchmark()
        {
            Tensor input = Tensor.Fill(2, batchSize, MatShape.x);


            Dense dense = new Dense(MatShape.x, MatShape.y, device: TestDevice);
            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                var outp = dense.Forward(input);
                dense.Backward(outp);
            }
            TimeKeeper.Stop();
        }

        void Conv2DBenchmark()
        {
            Conv2D c = new Conv2D((1, 28, 28), 64, 3, device: TestDevice);

            Tensor input = Tensor.Random01(batchSize, 1, 28, 28);
            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                var outp = c.Forward(input);
                c.Backward(outp);
            }
            TimeKeeper.Stop();
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

                TimeKeeper.Start();
                for (int i = 0; i < Runs; i++)
                {
                    Tensor.MatMul(t1, t2);
                }
                TimeKeeper.Stop();
            }
            else
            {
                var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
                var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);

                TimeKeeper.Start();
                for (int i = 0; i < Runs; i++)
                {
                    TensorGPU.MatMul(t1, t2);
                }
                TimeKeeper.Stop();
            }

        }
        void Benchmark_TensorGPUTime()
        {
            var t1 = TensorGPU.Random01(MatShape.x, MatShape.y);
            var t2 = TensorGPU.Random01(MatShape.x, MatShape.y);
           
            
            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                TensorGPU.MatMul(t1, t2);
            }
            TimeKeeper.Stop();

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

