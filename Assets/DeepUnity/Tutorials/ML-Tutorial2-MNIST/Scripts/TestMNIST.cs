using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TestMNIST : MonoBehaviour
{
	[SerializeField] private Sequential network;
    
	List<(Tensor, Tensor)> test = new();
    List<float> accs = new List<float> ();

    public string completed = "0/0";
    public string accuracy = "0%";
    public PerformanceGraph graph = new PerformanceGraph ();
    int sample_index = 0;
    public void Start()
    {
        Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out _, out test, DatasetSettings.LoadTestOnly);
        Debug.Log("MNIST Dataset test loaded.");

        if(network == null)
        {
            Debug.Log("Please load a network to test.");
        }
        print($"Total test samples {test.Count}.");
        print("Network used: " + network.Summary());
        Utils.Shuffle(test);
    }

    public void Update()
    {
        if (sample_index == test.Count)
            return; // case test finished

        (Tensor, Tensor) sample = test[sample_index++];

        var input = sample.Item1;
        var label = sample.Item2;

        var output = network.Predict(input);
        float acc = Metrics.Accuracy(output, label);
        graph.Append(acc);
        accs.Add(acc);

        completed = $"{sample_index}/{test.Count}";
        accuracy = $"{(accs.Average() * 100)}%";


    }

}


