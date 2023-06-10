using DeepUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class ModelTest : MonoBehaviour
    {

        public NeuralNetwork net;
        public Device device;
        public int hiddenSize = 64;
        public float rotationSpeed = 0.4f;

        [Space]
        public int samples = 100;
        public float dataSigma = 1f;

        private Tensor[] trainInputs;
        private Tensor[] trainLabels;

        private Tensor[] testInputs;
        private Tensor[] testLabels;

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
                 new Dense(2, hiddenSize, device: device),
                 new ReLU(),
                 new Dense(hiddenSize, hiddenSize, device: device),
                 new ReLU(),
                 new Dense(hiddenSize, 1, device: device)
                 );
                net.Compile(new Adam(), "somenet");
            }

            trainInputs = new Tensor[samples];
            trainLabels = new Tensor[samples];
            testInputs = new Tensor[samples];
            testLabels = new Tensor[samples];

            trainPoints = new Vector3[samples];
            testPoints = new Vector3[samples];

            for (int i = 0; i < samples; i++)
            {
                float x = Utils.Random.Gaussian(0, dataSigma);
                float y = Utils.Random.Gaussian(0, dataSigma);
                trainInputs[i] = Tensor.Constant(new float[] { x, y });
                trainLabels[i] = Tensor.Constant(MathF.Sqrt(MathF.Pow(1, 2) + MathF.Pow(x,2) + MathF.Pow(y, 2)));

                float xTest = Utils.Random.Gaussian(0, dataSigma);
                float yTest = Utils.Random.Gaussian(0, dataSigma);
                testInputs[i] = Tensor.Constant(new float[] { xTest, yTest });
                testLabels[i] = Tensor.Constant(MathF.Sqrt(MathF.Pow(1, 2) + MathF.Pow(xTest, 2) + MathF.Pow(yTest, 2)));
            }

        }


        List<float> trainAcc = new List<float>();
        List<float> testAcc = new List<float>();

        public void Update()
        {
            if(i == samples)
            {

                Debug.Log($"Epoch {++epoch} | Train Accuracy {trainAcc.Average() * 100f}% | Test Accuracy {testAcc.Average() * 100f}%");
                trainAcc.Clear();
                testAcc.Clear();
                i = 0;
                return;
            }

            var prediction = net.Forward(trainInputs[i]);
            var loss = Loss.MSE(prediction, trainLabels[i]);

            net.ZeroGrad();
            net.Backward(loss);
            net.Step();

            // Compute accuracy
            trainPoints[i] = new Vector3(trainInputs[i][0], prediction[0], trainInputs[i][1]); 
            
            float acc = Metrics.Accuracy(prediction, trainLabels[i]);
            trainAcc.Add(acc);


            var testPrediction = net.Forward(testInputs[i]);
            // Compute test accuracy
            testPoints[i] = new Vector3(testInputs[i][0], testPrediction[0], testInputs[i][1]);

            float testacc = Metrics.Accuracy(testPrediction, testLabels[i]);
            testAcc.Add(testacc);

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

