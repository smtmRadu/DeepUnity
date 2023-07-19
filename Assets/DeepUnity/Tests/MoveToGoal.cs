using UnityEngine;
using DeepUnity;

public class MoveToGoal : Agent
{
    [Header("Properties")]
    public float speed = 10f;
    public Transform target; // referenced manually

    public override void FixedUpdate()
    {
        base.FixedUpdate();
        AddReward(-0.001f);
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        // Space size 4
        sensorBuffer.AddObservation(transform.localPosition.x);
        sensorBuffer.AddObservation(transform.localPosition.z);
        sensorBuffer.AddObservation(target.transform.localPosition.x);
        sensorBuffer.AddObservation(target.transform.localPosition.z);
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        // Continuous actions 2 | Discrete actions 0
        float xmov = actionBuffer.ContinuousActions[0];
        float zmov = actionBuffer.ContinuousActions[1];

        transform.position += new Vector3(xmov, 0, zmov) * Time.fixedDeltaTime * speed;
    }
    public override void Heuristic(ActionBuffer actionBuffer)
    {
        // Control the agent manually
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
    
    public override void OnEpisodeBegin()
    {
        return;
        // Randomly position the target on each episode begin
        float xrand = Random.Range(-5, 5);
        float zrand = Random.Range(-5, 5);

        target.position = new Vector3(xrand, 0, zrand);
    }
    private void OnCollisionEnter(Collision collision)
    {
        if(collision.collider.CompareTag("Target"))
        {
            AddReward(1f);
            EndEpisode();
        }    
        else if(collision.collider.CompareTag("Wall"))
        {
            AddReward(-1f);
            EndEpisode();
        }
    }
}


