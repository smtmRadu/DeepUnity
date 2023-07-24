using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Tutorial2 : MonoBehaviour
{
    [Header("Learning z = x^2 + y^2. Visible on Gizmos.")]
    public Device device;
    public Sequential net;
    public bool save;
    public Optimizer optimizer;
    public StepLR scheduler;
    public int hiddenSize = 64;
    public int trainingSamples = 1024;
    public int batch_size = 32;
    public int validationSamples = 128;
    public int scheduler_step_size = 10;
    public float scheduler_gamma = 0.9f;

    [Space]
    public float rotationSpeed = 0.35f;
    public float dataScale = 0.15f;

    [Space]
    public int timerStopEpoch = 50;
    public int drawScale = 200;

    private Tensor[] trainXbatches;
    private Tensor[] trainYbatches;

    private Tensor validationInputs;
    private Tensor validationTargets;

    private Vector3[] trainPoints = null;
    private Vector3[] validationPoints = null;
    
    private int epoch = 0;
    private int i = 0;

    public void Start()
    {
        if (net == null)
        {
            net = new Sequential(
             new Dense(2, hiddenSize),
             new LayerNorm(hiddenSize),
             new ReLU(),
             new Dense(hiddenSize, hiddenSize, device: device),
             new ReLU(),
             new Dense(hiddenSize, hiddenSize, device: device),
             new ReLU(),
             new Dense(hiddenSize, 1)
             );; 
        }

        optimizer = new Adam(net.Parameters());
        scheduler = new StepLR(optimizer, scheduler_step_size, scheduler_gamma);


        trainPoints = new Vector3[trainingSamples];
        validationPoints = new Vector3[validationSamples];

        // Prepare train batches
        Tensor x1 = Tensor.RandomNormal(trainingSamples, 1) * dataScale;
        Tensor x2 = Tensor.RandomNormal(trainingSamples, 1) * dataScale;
        Tensor y = Tensor.Sqrt(Tensor.Pow(x1, 2) + Tensor.Pow(x2, 2));

        trainXbatches = Tensor.Split(Tensor.Cat(1, x1, x2), 0, batch_size);
        trainYbatches = Tensor.Split(y, 0, batch_size);

        // Prepare test batches
        x1 = Tensor.RandomNormal(validationSamples, 1) * dataScale;
        x2 = Tensor.RandomNormal(validationSamples, 1) * dataScale;
        y = Tensor.Sqrt(Tensor.Pow(x1, 2) + Tensor.Pow(x2, 2));

        validationInputs = Tensor.Cat(1, x1, x2);
        validationTargets = y;

        TimerX.Start();
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
            if (save && epoch % 10 == 0)
                 net.Save("test");
            i = 0;

            if (epoch == timerStopEpoch)
                TimerX.Stop();

            return;
        }
        var trainPrediction = net.Forward(trainXbatches[i]);
        var loss = Loss.MSEDerivative(trainPrediction, trainYbatches[i]);

        optimizer.ZeroGrad();
        net.Backward(loss);
        //optimizer.ClipGradNorm(0.5f);
        optimizer.Step();
        

        // Compute train accuracy
        float trainacc = Metrics.Accuracy(trainPrediction, trainYbatches[i]);
        trainAcc.Add(trainacc);

        // Compute test accuracy
        var testPrediction = net.Predict(validationInputs);
        float testacc = Metrics.Accuracy(testPrediction, validationTargets);
        for (int j = 0; j < validationInputs.Size(-2); j++)
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
