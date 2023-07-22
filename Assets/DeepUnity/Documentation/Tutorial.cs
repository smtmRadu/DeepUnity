using UnityEngine;
using DeepUnity;
using System.Collections.Generic;
using System.Linq;

public class Tutorial : MonoBehaviour
{
    [Header("Learning z = x^2 + y^2.")]
    [SerializeField]
    private Sequential network;
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

        // Learning z = x^2 + y^2 function.
        // Generate dataset
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
        List<float> epoch_train_accuracies = new List<float>();

        // Split dataset into batches
        int batch_size = 32;
        Tensor[] input_batches = Tensor.Split(train_inputs, 0, batch_size);
        Tensor[] target_batches = Tensor.Split(train_targets, 0, batch_size);

        // Update the network for each batch
        for (int i = 0; i < input_batches.Length; i++)
        {
            Tensor prediction = network.Forward(input_batches[i]);
            Tensor loss = Loss.MSEDerivative(prediction, target_batches[i]);

            optim.ZeroGrad();
            network.Backward(loss);
            optim.ClipGradNorm(0.5f);
            optim.Step();

            float train_acc = Metrics.Accuracy(prediction, target_batches[i]);
            epoch_train_accuracies.Add(train_acc);
        }

        scheduler.Step();
        network.Save("tutorial_model");

        float valid_acc = Metrics.Accuracy(network.Predict(valid_inputs), valid_targets);
        print($"[Epoch {Time.frameCount} | Train Accuracy: {epoch_train_accuracies.Average() * 100f}% | Validation Accuracy: {valid_acc * 100f}%]");
    }
}