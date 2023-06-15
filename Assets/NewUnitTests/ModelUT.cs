using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class ModelUT : MonoBehaviour
    {
        public NeuralNetwork net;
        public int hiddenSize = 64;
        public int samples = 1024;
        public int batch_size = 32;

        [Space]
        public float rotationSpeed = 0.4f;
        public float dataScale = 1f;

        private Tensor[] trainXbatches;
        private Tensor[] trainYbatches;

        private Tensor[] testXbatches;
        private Tensor[] testYbatches;

        private Vector3[] trainPoints = null;
        private Vector3[] testPoints = null;
        public int drawScale = 10;

        private int epoch = 0;
        private int i = 0;
        public void Start()
        {
            Settings.Device = Device.CPU;
            if (net == null)
            {
                net = new NeuralNetwork(
                 new Dense(2, hiddenSize),
                 new ReLU(),
                 new Dense(hiddenSize, hiddenSize),
                 new ReLU(),
                 new Dense(hiddenSize, 1)
                 );
                net.Compile(new SGD(), "somenet");
            }

            trainPoints = new Vector3[samples];
            testPoints = new Vector3[samples];

            // Prepare train batches
            Tensor x1 = Tensor.RandomNormal(samples, 1) * dataScale;
            Tensor x2 = Tensor.RandomNormal(samples, 1) * dataScale;
            Tensor y = Tensor.Sqrt(Tensor.Pow(x1, 2) + Tensor.Pow(x2, 2));

            trainXbatches = Tensor.Split(Tensor.Join(1, x1, x2), 0, batch_size);
            trainYbatches = Tensor.Split(y, 0, batch_size);

            // Prepare test batches
            x1 = Tensor.RandomNormal(samples, 1) * dataScale;
            x2 = Tensor.RandomNormal(samples, 1) * dataScale;
            y = Tensor.Sqrt(Tensor.Pow(x1, 2) + Tensor.Pow(x2, 2));

            testXbatches = Tensor.Split(Tensor.Join(1, x1, x2), 0, batch_size);
            testYbatches = Tensor.Split(y, 0, batch_size);
        }


        List<float> trainAcc = new List<float>();
        List<float> testAcc = new List<float>();

        public void Update()
        {
            if (i == samples / batch_size)
            {

                Debug.Log($"Epoch {++epoch} | Train Accuracy {trainAcc.Average() * 100f}% | Test Accuracy {testAcc.Average() * 100f}%");
                trainAcc.Clear();
                testAcc.Clear();
                i = 0;
                return;
            }
            var trainPrediction = net.Forward(trainXbatches[i]);
            var loss = Loss.MSE(trainPrediction, trainYbatches[i]);

            net.ZeroGrad();
            net.Backward(loss);
            net.Step();

            // Compute train accuracy
            float trainacc = Metrics.Accuracy(trainPrediction, trainYbatches[i]);
            trainAcc.Add(trainacc);

            // Compute test accuracy
            var testPrediction = net.Forward(testXbatches[i]);
            float testacc = Metrics.Accuracy(testPrediction, testYbatches[i]);
            testAcc.Add(testacc);



            for (int j = 0; j < batch_size; j++)
            {
                trainPoints[j + i * batch_size] = new Vector3(trainXbatches[i][j, 0], trainPrediction[j , 0], trainXbatches[i][j, 1]);
                testPoints[j + i * batch_size] = new Vector3(testXbatches[i][j, 0], testPrediction[j, 0], testXbatches[i][j, 1]);

            }
            i++;
        }

        public void LateUpdate()
        {
            transform.RotateAround(Vector3.zero, Vector3.up, rotationSpeed);
        }
        public void OnDrawGizmos()
        {

            if (trainPoints == null)
                return;
            try
            {
                Gizmos.color = Color.blue;
                for (int i = 0; i < samples; i++)
                {
                    Gizmos.DrawCube(trainPoints[i] * drawScale, Vector3.one);
                }

                Gizmos.color = Color.red;
                for (int i = 0; i < samples; i++)
                {

                    Gizmos.DrawSphere(testPoints[i] * drawScale, 1f);
                }
            }
            catch { }
        }
    }
}