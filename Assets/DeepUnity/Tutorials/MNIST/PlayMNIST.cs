using DeepUnity;
using System.Text;
using TMPro.EditorUtilities;
using UnityEngine;
using UnityEngine.UI;

public class PlayMNIST : MonoBehaviour
{
   
    [SerializeField] public Image image;
    [SerializeField] public TMPro.TMP_Text digitPredictionText;
    [SerializeField] public TMPro.TMP_Text softmaxOutputText;


    [SerializeField] public Sequential network;
    [SerializeField] public int brushRadius = 1;
	[SerializeField] public float brushStrength = 0.1f;


    public void Start()
    {
        if(network == null)
        {
            Debug.Log("Please load a neural network to use.");
        }
    }
    public void Update()
    {
        Draw();
        Remove();
        Clear();
        Predict();
         
    }

    public void Predict()
    {
        if(!Input.GetMouseButtonUp(0) && !Input.GetKeyUp(KeyCode.R) && !Input.GetMouseButtonUp(1))
        {
            return;
        }
        


        Texture2D texture = image.sprite.texture;
        Tensor input = Tensor.Constant(texture, true);
        var prediction = network.Predict(input);

        // Display prediction
        float digit = prediction.ArgMax(-1)[0];
        digitPredictionText.text = digit.ToString() + ".";

        // display the confidence
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < 10; i++)
        {
            if (i == digit)
            {
                stringBuilder.AppendLine($"<color=green>{i} - {(prediction[i] * 100f).ToString("00.00")}%</color>");
            }
            else
                stringBuilder.AppendLine($"<color=red>{i} - {(prediction[i] * 100f).ToString("00.00")}%</color>");
        }
        softmaxOutputText.text = stringBuilder.ToString();

    }
    public void Draw()
    {
        if (!Input.GetMouseButton(0))
            return;

        Vector2 mousePosition = Input.mousePosition;

        if (mousePosition.x > Screen.width)
            return;

        int x = (int)(mousePosition.x / Screen.height * 28f);
        int y = (int)(mousePosition.y / Screen.height * 28f);

        for (int xb = x - brushRadius; xb < x + brushRadius; xb++)
        {
            for (int yb = y - brushRadius; yb < y + brushRadius; yb++)
            {
                float xDistance = x - xb;
                float yDistance = y - yb;
                float distanceFromCenter = xDistance * xDistance + yDistance * yDistance;
                if (distanceFromCenter < brushRadius * brushRadius)
                {
                    Color alreadyColor = image.sprite.texture.GetPixel(xb, yb);
                    image.sprite.texture.SetPixel(xb, yb, alreadyColor + brushStrength * Color.white * (1f / (distanceFromCenter + Utils.EPSILON)));
                }
            }
        }

        image.sprite.texture.Apply();
    }
    public void Remove()
    {
        if (!Input.GetMouseButton(1))
            return;

        Vector2 mousePosition = Input.mousePosition;

        if (mousePosition.x > Screen.width)
            return;

        int x = (int)(mousePosition.x / Screen.height * 28f);
        int y = (int)(mousePosition.y / Screen.height * 28f);

        for (int xb = x - brushRadius; xb < x + brushRadius; xb++)
        {
            for (int yb = y - brushRadius; yb < y + brushRadius; yb++)
            {
                float xDistance = x - xb;
                float yDistance = y - yb;
                float distanceFromCenter = xDistance * xDistance + yDistance * yDistance;
                if (distanceFromCenter < brushRadius * brushRadius)
                {
                    Color alreadyColor = image.sprite.texture.GetPixel(xb, yb);
                    image.sprite.texture.SetPixel(xb, yb, Color.black);
                }
            }
        }

        image.sprite.texture.Apply();
    }
    public void Clear()
    {
        if (!Input.GetKeyDown(KeyCode.R))
            return;

        var pixels = image.sprite.texture.GetPixels();
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = Color.black;
        }

        image.sprite.texture.SetPixels(pixels);
        image.sprite.texture.Apply();
    }
}

