using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A simple way to change between the cameras using keycodes 1, 2, 3, 4...
    /// </summary>
    public class CameraManager : MonoBehaviour
    {
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
            if (cameras.Count > 0)
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
    }

}
