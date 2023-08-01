using UnityEngine;
using DeepUnity;

public class Tutorial : MonoBehaviour
{
    [Header("Learning z = x^2 + y^2.")]
    [SerializeField] private Sequential network;
    [SerializeField] private PerformanceGraph lossGraph = new PerformanceGraph();
    [SerializeField] private PerformanceGraph trainAccuracyGraph = new PerformanceGraph();
    [SerializeField] private PerformanceGraph validationAccuracyGraph = new PerformanceGraph();
   
    private Optimizer optim;
    private StepLR scheduler;

    private Tensor train_inputs;
    private Tensor train_targets;
    private Tensor valid_inputs;
    private Tensor valid_targets;

    public void Start()
    {
        if (network == null)
        {
            network = new Sequential(
                new Dense(2, 64),
                new Tanh(),
                new Dense(64, 64),
                new ReLU(),
                new Dense(64, 1));
        }
        optim = new Adam(network.Parameters);
        scheduler = new StepLR(optim, 100);

        // Generate training dataset
        int data_size = 1024;
        Tensor x = Tensor.RandomNormal(data_size, 1);
        Tensor y = Tensor.RandomNormal(data_size, 1);
        train_inputs = Tensor.Cat(1, x, y);
        train_targets = x.Zip(y, (a, b) => a * a + b * b);

        // Generate validation set
        int valid_size = 64;
        x = Tensor.RandomNormal(valid_size, 1);
        y = Tensor.RandomNormal(valid_size, 1);
        valid_inputs = Tensor.Cat(1, x, y);
        valid_targets = x.Zip(y, (a, b) => a * a + b * b);

    }

    public void Update()
    {
        // Split dataset into batches
        int batch_size = 32;
        Tensor[] input_batches = train_inputs.Split(0, batch_size);
        Tensor[] target_batches = train_targets.Split(0, batch_size);

        // Update the network for each batch
        for (int i = 0; i < input_batches.Length; i++)
        {
            Tensor prediction = network.Forward(input_batches[i]);
            Loss loss = DeepUnity.Loss.MSE(prediction, target_batches[i]);

            optim.ZeroGrad();
            network.Backward(loss);
            optim.ClipGradNorm(0.5f);
            optim.Step();

            lossGraph.Append(loss.Value);
            trainAccuracyGraph.Append(Metrics.Accuracy(prediction, target_batches[i]) * 100f);
        }
        validationAccuracyGraph.Append(Metrics.Accuracy(network.Predict(valid_inputs), valid_targets) * 100f);

        scheduler.Step();
        network.Save("Tutorial");       
    }
}