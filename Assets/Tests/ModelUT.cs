using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class ModelUT : MonoBehaviour
    {
        public Sequential net;
        public Optimizer optimizer;
        public StepLR scheduler;
        public int hiddenSize = 64;
        public int trainingSamples = 1024;
        public int batch_size = 32;
        public int validationSamples = 128;
        public int scheduler_step_size = 10;
        public float scheduler_gamma = 0.9f;

        [Space]
        public float rotationSpeed = 0.4f;
        public float dataScale = 1f;

        private Tensor[] trainXbatches;
        private Tensor[] trainYbatches;

        private Tensor validationInputs;
        private Tensor validationTargets;

        private Vector3[] trainPoints = null;
        private Vector3[] validationPoints = null;
        public int drawScale = 10;

        private int epoch = 0;
        private int i = 0;

        public void Start()
        {
            if (net == null)
            {
                net = new Sequential(
                 new Dense(2, hiddenSize),
                 //new BatchNorm(hiddenSize),
                 new TanH(),
                 //new Dropout(0.1f),
                 
                 new Dense(hiddenSize, hiddenSize),

                 //new LayerNorm(),
                 new ReLU(),
                
                 new Dense(hiddenSize, 1)
                 );
            }

            optimizer = new Adamax(net.Parameters());
            scheduler = new StepLR(optimizer, scheduler_step_size, scheduler_gamma);


            trainPoints = new Vector3[trainingSamples];
            validationPoints = new Vector3[validationSamples];

            // Prepare train batches
            Tensor x1 = Tensor.RandomNormal((0, 1), trainingSamples, 1) * dataScale;
            Tensor x2 = Tensor.RandomNormal((0, 1), trainingSamples, 1) * dataScale;
            Tensor y = Tensor.Sqrt(Tensor.Pow(x1, 2) + Tensor.Pow(x2, 2));

            trainXbatches = Tensor.Split(Tensor.Join(Dim.width, x1, x2), Dim.height, batch_size);
            trainYbatches = Tensor.Split(y, Dim.height, batch_size);

            // Prepare test batches
            x1 = Tensor.RandomNormal((0, 1), validationSamples, 1) * dataScale;
            x2 = Tensor.RandomNormal((0, 1), validationSamples, 1) * dataScale;
            y = Tensor.Sqrt(Tensor.Pow(x1, 2) + Tensor.Pow(x2, 2));

            validationInputs = Tensor.Join(Dim.width, x1, x2);
            validationTargets = y;
        }


        List<float> trainAcc = new List<float>();
        List<float> validationAcc = new List<float>();

        public void Update()
        {
            if (i == trainingSamples / batch_size)
            {

                Debug.Log($"Epoch {++epoch} | Train Accuracy {trainAcc.Average() * 100f}% | Validation Accuracy {validationAcc.Average() * 100f}% | LR {scheduler.CurrentLR}");
                trainAcc.Clear();
                validationAcc.Clear();
                scheduler.Step();
                if (epoch % 10 == 0)
                    net.Save("test");
                i = 0;
                return;
            }
            var trainPrediction = net.Forward(trainXbatches[i]);
            var loss = Loss.MSE(trainPrediction, trainYbatches[i]);

            optimizer.ZeroGrad();
            net.Backward(loss);
            optimizer.ClipGradNorm(0.5f);
            optimizer.Step();
            

            // Compute train accuracy
            float trainacc = Metrics.Accuracy(trainPrediction, trainYbatches[i]);
            trainAcc.Add(trainacc);

            // Compute test accuracy
            var testPrediction = net.Predict(validationInputs);
            float testacc = Metrics.Accuracy(testPrediction, validationTargets);
            for (int j = 0; j < validationInputs.Height; j++)
            {
                validationPoints[j] = new Vector3(validationInputs[j, 0], testPrediction[j, 0], validationInputs[j, 1]);
            }

            validationAcc.Add(testacc);



            for (int j = 0; j < batch_size; j++)
            {
                trainPoints[j + i * batch_size] = new Vector3(trainXbatches[i][j, 0], trainPrediction[j , 0], trainXbatches[i][j, 1]);               
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
                for (int i = 0; i < trainingSamples; i++)
                {
                    Gizmos.DrawCube(trainPoints[i] * drawScale, Vector3.one);
                }

                Gizmos.color = Color.red;
                for (int i = 0; i < trainingSamples; i++)
                {

                    Gizmos.DrawSphere(validationPoints[i] * drawScale, 1f);
                }
            }
            catch { }
        }
    }
}