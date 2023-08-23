using UnityEngine;
using DeepUnity;

public class MoveToGoal : Agent
{
    [Header("Properties")]
    public float speed = 10f;
    public Transform target;
    public float norm_scale = 8f;
    public override void OnEpisodeBegin()
    {
        float xrand = Random.Range(-norm_scale, norm_scale);
        float zrand = Random.Range(-norm_scale, norm_scale);
        target.localPosition = new Vector3(xrand, 2.25f, zrand);
        
        xrand = Random.Range(-norm_scale, norm_scale);
        zrand = Random.Range(-norm_scale, norm_scale);
        transform.localPosition = new Vector3(xrand, 2.25f, zrand);
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(transform.localPosition.x / norm_scale);
        sensorBuffer.AddObservation(transform.localPosition.z / norm_scale);
        sensorBuffer.AddObservation(target.transform.localPosition.x / norm_scale);
        sensorBuffer.AddObservation(target.transform.localPosition.z / norm_scale);
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        float xmov = actionBuffer.ContinuousActions[0];
        float zmov = actionBuffer.ContinuousActions[1];

        transform.position += new Vector3(xmov, 0, zmov) * Time.fixedDeltaTime * speed;

        AddReward(-0.0025f);
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

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Goal>(out _))
        {
            SetReward(1f);
            EndEpisode();
        }
        if (other.TryGetComponent<Wall>(out _))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }
    private void OnTriggerStay(Collider other)
    {
        if (other.TryGetComponent<Goal>(out _))
        {
            SetReward(1f);
            EndEpisode();
        }
        if (other.TryGetComponent<Wall>(out _))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }
}


