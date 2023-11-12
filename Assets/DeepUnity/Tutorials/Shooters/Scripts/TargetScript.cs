using DeepUnity;
using DeepUnityTutorials;
using UnityEngine;

public class TargetScript : MonoBehaviour
{
    public ShooterScript gunner;

    private void Start()
    {
        Reposition();
    }
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Enemy"))
            gunner.AddReward(1);

       Reposition();
    }

    private void Reposition()
    {
        float random_x = Utils.Random.Range(-10, 10);
        float random_z = Utils.Random.Range(-10, 10);
        transform.position = new Vector3(random_x, transform.position.y, random_z);
        transform.rotation = Quaternion.LookRotation(gunner.transform.position) * Quaternion.Euler(90, 90, 0);
    }
}


