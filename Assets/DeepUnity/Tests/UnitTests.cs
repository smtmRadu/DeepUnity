using UnityEngine;
using DeepUnity;
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
        public NeuralNetwork net;
        public RawImage image;
        public RawImage image2;

      

        public int index = 0;
        public int whatIsIt = 0;

        public NeuralNetwork network;
        public AgentBehaviour beh;
        private void Start()
        {
            Tensor stackObs = Tensor.RandomNormal(8);
            print(stackObs);
            print(stackObs.Reshape(2, 4));
            Tensor obs = Tensor.RandomNormal(6, 10);
            print(obs);
        }

        void TestRNN()
        {
            net = new NeuralNetwork(
               new RNNCell(10, 64, HiddenStates.ReturnAll),
               new RNNCell(64, 64, HiddenStates.ReturnAll),
               new RNNCell(64, 10, HiddenStates.ReturnLast));
            optim = new Adam(net.Parameters());
            Tensor input = Tensor.Random01(batchSize, 6, 10);
            Tensor target = Tensor.Ones(batchSize, 10);
            net.SetDevice(TestDevice);
            TimeKeeper.Start();
            for (int i = 0; i < Runs; i++)
            {
                var predict = net.Forward(input);
                Loss mse = Loss.MSE(predict, target);
                optim.ZeroGrad();
                net.Backward(mse.Gradient);
                optim.Step();
                graph.Append(mse.Item);
            }
            TimeKeeper.Stop();
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

    }
}

