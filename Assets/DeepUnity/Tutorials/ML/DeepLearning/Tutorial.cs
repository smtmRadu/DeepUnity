using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class Tutorial : MonoBehaviour
    {
        [Header("Learning z = x^2 + y^2.")]
        [SerializeField] private NeuralNetwork network;
        [SerializeField] private PerformanceGraph trainLossGraph = new PerformanceGraph();
        [SerializeField] private PerformanceGraph validLossGraph = new PerformanceGraph();

        private Optimizer optim;
        private LRScheduler scheduler;

        private Tensor train_inputs;
        private Tensor train_targets;
        private Tensor valid_inputs;
        private Tensor valid_targets;

        public void Start()
        {
            if (network == null)
            {
                network = new NeuralNetwork(
                    new Dense(2, 64),
                    new Tanh(),
                    new Dense(64, 64, device: Device.GPU),
                    new ReLU(),
                    new Dense(64, 1)).CreateAsset("TutorialModel");
            }
            optim = new Adam(network.GetLearnables(), 0.001f);
            scheduler = new LRScheduler(optim, 30, 0.1f);

            // Generate training dataset
            int data_size = 1024;
            Tensor x = Tensor.RandomNormal(data_size, 1);
            Tensor y = Tensor.RandomNormal(data_size, 1);
            train_inputs = Tensor.Cat(1, x, y);
            train_targets = x * x + y * y;

            // Generate validation set
            int valid_size = 64;
            x = Tensor.RandomNormal(valid_size, 1);
            y = Tensor.RandomNormal(valid_size, 1);
            valid_inputs = Tensor.Cat(1, x, y);
            valid_targets = x * x + y * y;
        }

        public void Update()
        {
            // Training. Split the dataset into batches of 32.
            float train_loss = 0f;
            Tensor[] input_batches = train_inputs.Split(0, 32);
            Tensor[] target_batches = train_targets.Split(0, 32);
            for (int i = 0; i < input_batches.Length; i++)
            {
                Tensor prediction = network.Forward(input_batches[i]);
                Loss loss = Loss.MSE(prediction, target_batches[i]);

                optim.ZeroGrad();
                network.Backward(loss.Derivative);
                optim.ClipGradNorm(0.5f);
                optim.Step();

                train_loss += loss.Item;
            }
            train_loss /= input_batches.Length;
            trainLossGraph.Append(train_loss);

            // Validation
            Tensor valid_prediction = network.Predict(valid_inputs);
            float valid_loss = Metrics.MeanSquaredError(valid_prediction, valid_targets);
            validLossGraph.Append(valid_loss);

            print($"Epoch: {Time.frameCount} - Train Loss: {train_loss} - Valid Loss: {valid_loss}");

            scheduler.Step();
            network.Save();
        }
    }
}