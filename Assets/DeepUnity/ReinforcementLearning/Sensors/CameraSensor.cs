using UnityEngine;
using UnityEditor;
using System.IO;
using System.Linq;
using System.Collections;
using System.Text.RegularExpressions;
using Unity.VisualScripting;

namespace DeepUnity
{
    [AddComponentMenu("DeepUnity/CameraSensor")]
    public class CameraSensor : MonoBehaviour, ISensor
    {
        public int Width => width;
        public int Height => height;

        [SerializeField] private Camera cam;
        [SerializeField, Min(16)]private int width = 640;
        [SerializeField, Min(9)]private int height = 480;
        [SerializeField] private CaptureType type = CaptureType.RGB;
        [SerializeField] private CompressionType compression = CompressionType.PNG;

        [SerializeField, HideInInspector] public RenderTexture renderTexture; // remains active for the rest of the game
        private void Awake()
        {
            if (cam == null)
                Debug.Log("Please attach a camera to CamSensor");         
        }



        /// <summary>
        /// Returns the image pixels converted into float numbers.
        /// </summary>
        /// <returns>Returns a float[] with length = <b>3 * width * height</b>, or <b>1 * width * height</b> for Grayscale capture.</returns>
        public float[] GetObservationsVector()
        {
            Color[] pixels = GetObservationPixels();
            int channels = type == CaptureType.RGB ? 3 : 1;
            float[] vector = new float[pixels.Length * channels];
            int index = 0;
            foreach (var item in pixels)
            {
                if (type == CaptureType.RGB)
                    vector[index++] = item.grayscale;
                else
                {
                    vector[index++] = item.r;
                    vector[index++] = item.g;
                    vector[index++] = item.b;
                }
            }
            return vector;
        }
        /// <summary>
        /// Returns the grayscale image pixels converted into float numbers.
        /// </summary>
        /// <returns>Returns a float[] with length = <b>width * height</b>, as if the capture would have been Grayscale.</returns>
        public float[] GetCompressedObservationsVector()
        {
            CaptureType oldCaptureType = type;
            float[] vec = GetObservationsVector();
            type = oldCaptureType;
            return vec;
        }
        /// <summary>
        /// Returns the pixels of the camera rendered image. The pixels are not affected by Grayscale type.
        /// </summary>
        /// <returns></returns>
        public Color[] GetObservationPixels()
        {
            if (cam == null)
            {
                Debug.LogError("<color=red>CamSensor Cam not set to an instance of an object.</color>");
                return null;
            }
            if (renderTexture == null)
                renderTexture = new RenderTexture(width, height, 0);

            RenderTexture activeRT = RenderTexture.active; // this gets the previous rt, and while we render our part we assign it back
            RenderTexture.active = cam.targetTexture;
            Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
            image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
            image.Apply();
            RenderTexture.active = activeRT;

            Color[] pixels = image.GetPixels();
            Destroy(image);
            return pixels;
        }
       



        /// <summary>
        /// Do not use anywhere without Destroying the Texture2D afterwards!
        /// </summary>
        /// <returns></returns>
        private Texture2D GetObservationTexture()
        {
            if (cam == null)
            {
                Debug.LogError("<color=red>CamSensor Cam not set to an instance of an object.</color>");
                return null;
            }
            if (renderTexture == null)
                renderTexture = new RenderTexture(width, height, 0);


            cam.targetTexture = renderTexture;

            RenderTexture activeRT = RenderTexture.active;
            RenderTexture.active = cam.targetTexture;
            Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
            image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
            image.Apply();
            RenderTexture.active = activeRT;

            if (type == CaptureType.Grayscale)
                MakeTextureGray(image);

            return image;
        }
        private static void MakeTextureGray(Texture2D texture)
        {
            texture.SetPixels(texture.GetPixels().Select(x => new Color(x.grayscale, x.grayscale, x.grayscale, x.a)).ToArray());
        }

        public void SaveImage()
        {
            if (cam == null)
            {
                Debug.LogError("CameraSensor Cam object reference not set to an instance of an object. Please assign a Camera for the first field!");
                return;
            }

            if (!Directory.Exists("Assets/CamShots"))
                Directory.CreateDirectory("Assets/CamShots");

            string[] guids = AssetDatabase.FindAssets("t:Texture", new string[] { "Assets/CamShots" });

            if(guids.Length == 0)
                ExecuteShotImageSave("Assets/CamShots/Frame1.png");
            else
            {
                string lastPath = AssetDatabase.GUIDToAssetPath(guids.Last());

                // Extract the number from the last path using a regular expression
                Match match = Regex.Match(lastPath, @".*Frame(\d+)");

                string pathToSave = "Assets/CamShots/Frame1.png";
                if (match.Success)
                {
                    string numberString = match.Groups[1].Value;
                    int newNumber = int.Parse(numberString) + 1;
                    string newPath = lastPath.ToString().Replace(numberString, newNumber.ToString());
                    pathToSave = newPath;
                }

                ExecuteShotImageSave(pathToSave);
            }            

            AssetDatabase.Refresh();
        }  
        private void ExecuteShotImageSave(string atPath)
        {
            Texture2D image = GetObservationTexture();
            switch (compression)
            {
                case CompressionType.PNG:
                    File.WriteAllBytes(atPath, image.EncodeToPNG());
                    break;
                case CompressionType.JPG:
                    File.WriteAllBytes(atPath, image.EncodeToJPG());
                    break;
                case CompressionType.EXG:
                    File.WriteAllBytes(atPath, image.EncodeToEXR());
                    break;
                case CompressionType.TGA:
                    File.WriteAllBytes(atPath, image.EncodeToTGA());
                    break;
                default: throw new System.NotImplementedException("Unhandled compression type.");
            }
            DestroyImmediate(image);
        }
    }



    #region Editor
    [CustomEditor(typeof(CameraSensor)), CanEditMultipleObjects]
    class ScriptlessCameraSensor : Editor
    {
        public static string[] dontInclude = new string[] { "m_Script" };
        public override void OnInspectorGUI()
        {
            CameraSensor script = (CameraSensor)target;
            SerializedProperty cam = serializedObject.FindProperty("cam");

            // Display the rendered image
            // if (script.renderTexture != null)
            // {
            //     EditorGUILayout.Space();
            //     Rect previewRect = GUILayoutUtility.GetRect(100,100);
            //     EditorGUI.DrawPreviewTexture(previewRect, script.renderTexture);
            // 
            //     GUIStyle centeredStyle = new GUIStyle(GUI.skin.label);
            //     centeredStyle.alignment = TextAnchor.UpperCenter;
            //     centeredStyle.normal.textColor = Color.white;
            //     centeredStyle.fontSize = 15;
            //     centeredStyle.fontStyle = FontStyle.Bold;
            //     GUI.Label(previewRect, "Camera Preview", centeredStyle);
            // 
            // }


            

            if (cam.objectReferenceValue == null)
                EditorGUILayout.HelpBox("Camera not attached to Cam Sensor.", MessageType.Warning);


            DrawPropertiesExcluding(serializedObject, dontInclude);

            SerializedProperty type = serializedObject.FindProperty("type");
            SerializedProperty w = serializedObject.FindProperty("width");
            SerializedProperty h = serializedObject.FindProperty("height");

            int vecDim = type.enumValueIndex == (int)CaptureType.Grayscale ?
                            w.intValue * h.intValue :
                            3 * w.intValue * h.intValue;

            int compVecDim = w.intValue * h.intValue;


            if (cam.objectReferenceValue != null)
                EditorGUILayout.HelpBox($"Observations Vector contains {vecDim} float values. Compressed Observations Vector contains {compVecDim} flaot values.", MessageType.Info);
            else
                EditorGUILayout.HelpBox($"Cannot compute Observations Vector size until attaching a Camera.", MessageType.Info);

            serializedObject.ApplyModifiedProperties();

            EditorGUILayout.Separator();
            if(GUILayout.Button("Save image"))
            {
                script.SaveImage();
            }
        }
    }
    #endregion
}
