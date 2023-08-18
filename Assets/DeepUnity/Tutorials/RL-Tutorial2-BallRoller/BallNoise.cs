using DeepUnity;
using UnityEngine;

public class BallNoise : MonoBehaviour
{
	public float noise = 1f;
    Rigidbody rb;
    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }
    private void FixedUpdate()
    {
        float zDirNoise = Utils.Random.Range(-1f, 1f);
        float xDirNoise = Utils.Random.Range(-1f, 1f);
        rb.AddForce(new Vector3(xDirNoise, 0, zDirNoise) * noise);
    }
}


