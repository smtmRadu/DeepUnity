using DeepUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class ModelTest : MonoBehaviour
    {
        public Device device;
        public NeuralNetwork net;       
        public int hiddenSize = 64;
        public int samples = 1024;
        public int batch_size = 32;

        [Space]      
        public float rotationSpeed = 0.4f;
        public float dataScale = 1f;

        private NDArray[] trainXbatches;
        private NDArray[] trainYbatches;
               
        private NDArray[] testXbatches;
        private NDArray[] testYbatches;

        private Vector3[] trainPoints = null;
        private Vector3[] testPoints = null;
        public int drawScale = 10;

        private int epoch = 0;
        private int i = 0;
        public void Start()
        {
            if(net == null)
            {
                net = new NeuralNetwork(
                 new Dense(2, hiddenSize),
                 new ReLU(),
                 new Dense(hiddenSize, hiddenSize),
                 new ReLU(),
                 new Dense(hiddenSize, 1)
                 );
                net.Compile(new Adam(), "somenet");
            }

            trainPoints = new Vector3[samples];
            testPoints = new Vector3[samples];

            // Prepare train batches
            NDArray x1 = NDArray.RandomNormal(1, samples) * dataScale;
            NDArray x2 = NDArray.RandomNormal(1, samples) * dataScale;
            NDArray y = NDArray.Sqrt(NDArray.Exp(x1) + NDArray.Exp(x2));

            trainXbatches = NDArray.Split(NDArray.Join(0, x1, x2), 1, batch_size);
            trainYbatches = NDArray.Split(y, 1, batch_size);

            // Prepare test batches
            x1 = NDArray.RandomNormal(1, samples) * dataScale;
            x2 = NDArray.RandomNormal(1, samples) * dataScale;
            y = NDArray.Sqrt(NDArray.Exp(x1) + NDArray.Exp(x2));

            testXbatches = NDArray.Split(NDArray.Join(0, x1, x2), 1, batch_size);
            testYbatches = NDArray.Split(y, 1, batch_size);


        }


        List<float> trainAcc = new List<float>();
        List<float> testAcc = new List<float>();

        public void Update()
        {
            if(i == samples/batch_size)
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

           

            for(int j = 0; j < batch_size; j++)
            {
                trainPoints[j + i * batch_size] = new Vector3(trainXbatches[i][0, j], trainPrediction[0, j], trainXbatches[i][1, j]);
                testPoints[j + i * batch_size] = new Vector3(testXbatches[i][0, j], testPrediction[0, j], testXbatches[i][1, j]);
            
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

