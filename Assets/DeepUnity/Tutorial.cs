using UnityEngine;
using DeepUnity;
using System.Collections.Generic;
using System.Linq;

public class Tutorial : MonoBehaviour
{
    [SerializeField] 
    private Sequential network;
    private Optimizer optim;
    private StepLR scheduler;

    private Tensor train_inputs;
    private Tensor train_targets;

    private Tensor valid_inputs;
    private Tensor valid_targets;

    private List<float> train_accs = new List<float>();

    public void Start()
    {
        if(network == null)
        {
            network = new Sequential(
                new Dense(2, 64),
                new ReLU(),
                new Dense(64, 64),                
                new ReLU(),
                new Dense(64, 1));
        }

        optim = new Adam(network.Parameters());
        scheduler = new StepLR(optim, 100);

        // Generate dataset - learning x^2 + y^2 function.
        int data_size = 1024;
        Tensor x = Tensor.RandomNormal((0, 0.5f), data_size, 1);
        Tensor y = Tensor.RandomNormal((0, 0.5f), data_size, 1);
        train_inputs = Tensor.Join(Dim.width, x, y);
        train_targets = x.Zip(y, (x, y) => x * x + y * y);

        // Generate validation set
        int valid_size = 64;
        x = Tensor.RandomNormal((0, 0.5f), valid_size, 1);
        y = Tensor.RandomNormal((0, 0.5f), valid_size, 1);
        valid_inputs = Tensor.Join(Dim.width, x, y);
        valid_targets = x.Zip(y, (x, y) => x * x + y * y);

    }

    public void Update()
    {
        train_accs.Clear();

        // Split dataset into batches
        int batch_size = 32;
        Tensor[] input_batches = Tensor.Split(train_inputs, 0, batch_size);
        Tensor[] target_batches = Tensor.Split(train_targets, 0, batch_size);

        // Update the network for each batch
        for (int i = 0; i < input_batches.Length; i++)
        {
            Tensor prediction = network.Forward(input_batches[i]);
            Tensor loss = Loss.MSE(prediction, target_batches[i]);

            optim.ZeroGrad();
            network.Backward(loss);
            optim.ClipGradNorm(0.5f);
            optim.Step();
            
            float train_acc = Metrics.Accuracy(prediction, target_batches[i]);
            train_accs.Add(train_acc);       
        }

        scheduler.Step();
        network.Save("tutorial");

        float valid_acc = Metrics.Accuracy(network.Predict(valid_inputs), valid_targets);
        print($"Epoch {Time.frameCount} | Train Accuracy: {train_accs.Average() * 100f}% | Validation Accuracy: {valid_acc * 100f}%");
    }
}

