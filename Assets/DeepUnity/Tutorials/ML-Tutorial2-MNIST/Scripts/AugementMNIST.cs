using DeepUnity;
using UnityEngine;
using System.IO;
using UnityEngine.UI;

namespace DeepUnityTutorials
{
    public class AugementMNIST : MonoBehaviour
    {
        public Image image;
        public Image image2;

        /// <summary>
        /// Good Values for training:
        /// noise_prob: 0.1
        /// noise_size: 0.25
        /// angle: -60, 60
        /// zoom: 0.70, 1.50
        /// offset: 5
        /// </summary>
        [Range(0f, 1f)] public float noise_prob;
        [Range(0f, 1f)] public float noise_size;
        [Range(-360f, 360f)] public float rot_angle;
        [Range(0.1f, 4f)] public float zoom_fac;
        [Range(-5f, 5f)] public float x_offset;
        [Range(-5f, 5f)] public float y_offset;


        private Texture2D originalTexture;
        public void Start()
        {
            originalTexture = LoadTexture("C:\\Users\\radup\\OneDrive\\Desktop\\TRAIN\\3\\7.png");

            image.sprite = Sprite.Create(originalTexture, new Rect(0, 0, originalTexture.width, originalTexture.height), new Vector2(0.5f, 0.5f));
            image2.sprite = Sprite.Create(originalTexture, new Rect(0, 0, originalTexture.width, originalTexture.height), new Vector2(0.5f, 0.5f));

            image.sprite.texture.filterMode = FilterMode.Point;
            image2.sprite.texture.filterMode = FilterMode.Point;
        }


        public void Update()
        {
            Tensor orgTex = Tensor.Constant(originalTexture.GetPixels(), (1, 28, 28));
            originalTexture.SetPixels(Utils.TensorToPixels(orgTex));

            Texture2D newTexture = Utils.ImageProcessing.Zoom(originalTexture, zoom_fac);
            newTexture = Utils.ImageProcessing.Rotate(newTexture, rot_angle);
            newTexture = Utils.ImageProcessing.Offset(newTexture, x_offset, y_offset);
            newTexture = Utils.ImageProcessing.Noise(newTexture, noise_prob, noise_size);

            // Create a new texture for image2.sprite
            Texture2D newImage2Texture = new Texture2D(newTexture.width, newTexture.height);
            newImage2Texture.filterMode = FilterMode.Point;
            newImage2Texture.SetPixels(newTexture.GetPixels());
            newImage2Texture.Apply();

            image2.sprite = Sprite.Create(newImage2Texture, new Rect(0, 0, newImage2Texture.width, newImage2Texture.height), new Vector2(0.5f, 0.5f));
        }

        private static Texture2D LoadTexture(string filePath)
        {
            Texture2D tex = null;
            byte[] fileData;

            if (File.Exists(filePath))
            {
                fileData = File.ReadAllBytes(filePath);
                tex = new Texture2D(28, 28);
                tex.LoadImage(fileData);
            }
            else
            {
                Debug.LogError($"File at path '{filePath}' not found");
            }
            return tex;
        }
    }

}

