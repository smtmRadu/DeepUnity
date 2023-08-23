using DeepUnity;
using UnityEngine;

public class PushBlock : Agent
{
    [Header("Properties")]
    public Transform box;
    public Transform arena;
    public float speed = 3000f;
    public float rotationSpeed = 100f;
    private Rigidbody rb;


    public override void Awake()
    {
        base.Awake();
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // // Replace box;
        // float randx = Utils.Random.Range(7f, 30f);
        // float randz = Utils.Random.Range(12f, 20f);
        // box.localPosition = new Vector3(randx, box.localPosition.y, randz);
        // 
        // // Reposition agent
        // randx = Utils.Random.Range(5, 32f);
        // randz = Utils.Random.Range(20f, 31f);
        // transform.localPosition = new Vector3(randx, transform.localPosition.y, randz);

    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        // + 60
        sensorBuffer.AddObservation(rb.velocity); // + 3
        
        // They are already received from the RaySensor
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        rb.AddForce(new Vector3(actionBuffer.ContinuousActions[0], 0, actionBuffer.ContinuousActions[1]) * speed);
        rb.AddTorque(new Vector3(0, actionBuffer.ContinuousActions[2], 0) * rotationSpeed, ForceMode.Impulse);

        AddReward(-0.0025f);
    }
    public override void Heuristic(ActionBuffer actionBuffer)
    {
        actionBuffer.ContinuousActions[0] = Input.GetAxis("Horizontal");
        actionBuffer.ContinuousActions[1] = Input.GetAxis("Vertical");

        if (Input.GetKey(KeyCode.Q))
            actionBuffer.ContinuousActions[2] = -1f;
        else if (Input.GetKey(KeyCode.E))
            actionBuffer.ContinuousActions[2] = 1f;
        else
            actionBuffer.ContinuousActions[2] = 0f;
    }

    private void OnTriggerEnter(Collider other)
    {
        if(other.CompareTag("Wall"))
        {
            AddReward(-1f);
            EndEpisode();
        }    
    }

    private void OnCollisionEnter(Collision collision)
    {
        if(collision.collider.CompareTag("Box"))
        {
            AddReward(+0.0025f);
        }
    }
}


