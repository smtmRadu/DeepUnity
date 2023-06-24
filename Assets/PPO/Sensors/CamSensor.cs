using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Linq;
using System.Collections;
using System.Text.RegularExpressions;
using Unity.VisualScripting;

namespace DeepUnity
{
    [AddComponentMenu("DeepUnity/Cam Sensor")]
    public class CamSensor : MonoBehaviour, ISensor
    {
        public Camera cam;
        [Min(16)]public int Width = 640;
        [Min(9)]public int Height = 480;
        public CaptureType type = CaptureType.RGB;


        public IEnumerable GetObservations()
        {
            return Capture().GetPixels().Cast<float>();
        }

        private Texture2D Capture()
        {
            if (cam == null)
            {
                Debug.LogError("<color=red>CamSensor Cam not set to an instance of an object.</color>");
                return null;
            }
            cam.targetTexture = new RenderTexture(Width, Height, 0);


            RenderTexture activeRT = RenderTexture.active;
            RenderTexture.active = cam.targetTexture;

            cam.Render();

            Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
            image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
            image.Apply();
            RenderTexture.active = activeRT;

            switch (type)
            {
                case CaptureType.RGB:
                    break;
                case CaptureType.Greyscale:
                    image.SetPixels(image.GetPixels().Select(x => new Color(x.grayscale, x.grayscale, x.grayscale, x.a)).ToArray());
                    break;
            }

            return image;
        }

        public void TakeShot()
        {
            if (cam == null)
            {
                Debug.LogError("<color=red>CameraSensor Cam object reference not set to an instance of an object.</color>");
                return;
            }
            if (cam.targetTexture == null)
                cam.targetTexture = new RenderTexture(Width, Height, 0);


            if (!Directory.Exists("Assets/CamShots"))
                Directory.CreateDirectory("Assets/CamShots");

            string[] guids = AssetDatabase.FindAssets("t:Texture", new string[] { "Assets/CamShots" });

            string lastPath = AssetDatabase.GUIDToAssetPath(guids.Last());

            Debug.Log(guids.ToLineSeparatedString());
            // Extract the number from the last path using a regular expression
            Match match = Regex.Match(lastPath, @".*Frame(\d+)");
            if (match.Success)
            {
                string numberString = match.Groups[1].Value;
                int newNumber = int.Parse(numberString) + 1;
                string newPath = lastPath.ToString().Replace(numberString, newNumber.ToString());

                Debug.Log(newNumber);
                Debug.Log(newPath);
                File.WriteAllBytes(newPath, Capture().EncodeToPNG());
                
            }
            else
            {
                File.WriteAllBytes("Assets/CamShots/Frame1.png", Capture().EncodeToPNG());
            }

            AssetDatabase.Refresh();
        }  
    }

    public enum CaptureType
    {
        RGB,
        Greyscale,
    }

    #region Editor
    [CustomEditor(typeof(CamSensor)), CanEditMultipleObjects]
    class ScriptlessCameraSensor : Editor
    { 
        public override void OnInspectorGUI()
        {
            List<string> dontInclude = new List<string>() { "m_Script" };
            CamSensor script = (CamSensor)target;
          
            DrawPropertiesExcluding(serializedObject, dontInclude.ToArray());
            serializedObject.ApplyModifiedProperties();

            EditorGUILayout.Separator();
            if(GUILayout.Button("Take a shot"))
            {
                script.TakeShot();
            }
        }
    }
    #endregion
}
