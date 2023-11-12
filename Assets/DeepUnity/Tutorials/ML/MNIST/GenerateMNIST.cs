using DeepUnity;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class GenerateMNIST : MonoBehaviour
{
	[SerializeField] NeuralNetwork generator;
	[SerializeField] Transform displaysHolder;

    private List<RawImage> displays = new List<RawImage>();
    private void Awake()
    {
        for (int i = 0; i < displaysHolder.childCount; i++)
        {
            displays.Add(displaysHolder.GetChild(i).GetComponent<RawImage>());
        }
        foreach (var item in displays)
        {
            item.texture = new Texture2D(28, 28);
        }
    }

    public void Update()
    {
        if (displays.Count == 0)
            return;

        var paramst = generator.Parameters();

        foreach (var item in paramst)
        {
            item.device = Device.CPU;
        }

        foreach (var dis in displays)
        {
            if (dis == null)
                continue;

            if (dis.enabled == false)
                continue;

            var sample = generator.Predict(GeneratorInput(1, 10)).Squeeze(0);
            Texture2D display = dis.texture as Texture2D;
            display.SetPixels(Utils.TensorToColorArray(sample));
            display.Apply();
        }

        foreach (var item in paramst)
        {
            item.device = Device.GPU;
        }
    }
    private Tensor GeneratorInput(int batch_size, int latent_dim)
    {
        return Tensor.RandomNormal(batch_size, latent_dim);
    }
}


