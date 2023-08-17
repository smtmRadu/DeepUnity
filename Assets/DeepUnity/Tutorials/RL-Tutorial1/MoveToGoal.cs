using UnityEngine;
using DeepUnity;

public class MoveToGoal : Agent
{
    [Header("Properties")]
    public float speed = 10f;
    public Transform target;

    public override void OnEpisodeBegin()
    {
        float xrand = Random.Range(-5f, 5f);
        float zrand = Random.Range(-5f, 5f);
        target.localPosition = new Vector3(xrand, 0, zrand);

        xrand = Random.Range(-5f, 5f);
        zrand = Random.Range(-5f, 5f);
        transform.localPosition = new Vector3(xrand, 0, zrand);
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(transform.localPosition.x / 5f);
        sensorBuffer.AddObservation(transform.localPosition.z / 5f);
        sensorBuffer.AddObservation(target.transform.localPosition.x / 5f);
        sensorBuffer.AddObservation(target.transform.localPosition.z / 5f);
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        float xmov = actionBuffer.ContinuousActions[0];
        float zmov = actionBuffer.ContinuousActions[1];

        transform.position += new Vector3(xmov, 0, zmov) * Time.fixedDeltaTime * speed;
    }
    public override void Heuristic(ActionBuffer actionBuffer)
    {
        float xmov = 0;
        float zmov = 0;

        if (Input.GetKey(KeyCode.W))
            zmov = 1;
        else if(Input.GetKey(KeyCode.S))
            zmov = -1;

        if (Input.GetKey(KeyCode.D))
            xmov = 1;
        else if (Input.GetKey(KeyCode.A))
            xmov = -1;

        actionBuffer.ContinuousActions[0] = xmov;
        actionBuffer.ContinuousActions[1] = zmov;
    }  
    private void OnCollisionEnter(Collision collision)
    {
        if(collision.collider.TryGetComponent<Goal>(out _))
        {
            SetReward(1f);
            EndEpisode();
        }    
        if(collision.collider.TryGetComponent<Wall>(out _))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }
}


