using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{
    [SerializeField] CarScript carScript;
    [SerializeField] List<Camera> cameras;

    [SerializeField] float acceleration;
    [SerializeField] float steer;
    [SerializeField] bool manualBreak;


    private void Start()
    {
        if (cameras.Count > 0)
        {
            for (int i = 0; i < cameras.Count; i++)
            {
                cameras[i].gameObject.SetActive(i == 0);
            }
        }
    }
    private int currentCameraIndex = 0;
    private void Update()
    {
         acceleration = Input.GetAxis("Vertical");
         steer = Input.GetAxis("Horizontal");
        manualBreak = Input.GetKey(KeyCode.Space);
        carScript.Accelerate(acceleration);
        carScript.Steer(steer);
        carScript.Break(manualBreak);

        if (Input.GetKeyDown(KeyCode.Alpha1))
            SwitchCamera(0);
        
        else if(Input.GetKeyDown(KeyCode.Alpha2))
            SwitchCamera(1);

        else if(Input.GetKeyDown(KeyCode.Alpha3))
            SwitchCamera(2);


    }

    private void SwitchCamera(int newCameraIndex)
    {
        try
        {
            cameras[currentCameraIndex].gameObject.SetActive(false);
            cameras[newCameraIndex].gameObject.SetActive(true);
            currentCameraIndex = newCameraIndex;
        }
        catch { }
    }
}


