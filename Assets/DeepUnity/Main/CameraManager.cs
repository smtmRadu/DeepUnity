using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A simple way to change between the cameras using keycodes 1, 2, 3, 4...
    /// </summary>
    public class CameraManager : MonoBehaviour
    {
        [Header("Add all Cameras you want to control.\n" +
            "Use keys 1, 2, 3 etc. to select the camera at runtime.")]
        [SerializeField] private List<Camera> cameras = new List<Camera>();
        private int currentIndex = -1;

        private void Awake()
        {
            foreach (Camera camera in cameras)
            {
                camera.gameObject.SetActive(false);
            }

            for (int i = 0; i < cameras.Count; i++)
            {
                if (cameras[i].gameObject.activeSelf)
                {
                    currentIndex = i;
                    break;
                }
            }
            if (currentIndex == -1 && cameras.Count > 0)
            {
                currentIndex = 0;
                cameras[currentIndex].gameObject.SetActive(true);
            }
        }

        private void Update()
        {
            if (Input.anyKeyDown && cameras.Count > 0)
            {
                for (int i = 0; i < cameras.Count; i++)
                {
                    if (Input.GetKeyDown(KeyCode.Alpha1 + i))
                    {
                        SwitchCamera(i);
                        break;
                    }
                }
            }
        }

        private void SwitchCamera(int newIndex)
        {
            if (currentIndex == newIndex)
                return;

            cameras[currentIndex].gameObject.SetActive(false);
            cameras[newIndex].gameObject.SetActive(true);

            currentIndex = newIndex;
        }

        // Finds all cameras in the current scene and assignes them to the list
        public void AssignAllSceneCameras()
        {
            GameObject[] allGameObjects = UnityEngine.SceneManagement.SceneManager.GetActiveScene().GetRootGameObjects();

            cameras = cameras == null ? new() : cameras;
            
            cameras = GameObject.FindSceneObjectsOfType(typeof(Camera)).OfType<Camera>().ToList();
            
        }
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(CameraManager), true), CanEditMultipleObjects]
    sealed class CameraManagerEditor : Editor
    {
        string[] drawNow = new string[] { "m_Script" };
        public override void OnInspectorGUI()
        {

            var script = (CameraManager)target;
            if (GUILayout.Button("Find all scene cameras"))
                script.AssignAllSceneCameras();

            DrawPropertiesExcluding(serializedObject, drawNow);

            

            serializedObject.ApplyModifiedProperties();
            serializedObject.Update();
        }
    }
#endif

}
