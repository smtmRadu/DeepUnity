using UnityEngine;
using DeepUnity.ReinforcementLearning;
using DeepUnity;

public class MoveToGoal : Agent
{
    [Button("SetDefaultHP")]
    [Header("5 discrete or 2 continuous actions")]
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
    public override void CollectObservations(StateVector sensorBuffer)
    {
        sensorBuffer.AddObservation(transform.localPosition.x / norm_scale);
        sensorBuffer.AddObservation(transform.localPosition.z / norm_scale);
        sensorBuffer.AddObservation(target.transform.localPosition.x / norm_scale);
        sensorBuffer.AddObservation(target.transform.localPosition.z / norm_scale);
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        if(model.IsUsingContinuousActions)
        {
            float xmov = actionBuffer.ContinuousActions[0];
            float zmov = actionBuffer.ContinuousActions[1];

            transform.position += new Vector3(xmov, 0, zmov) * Time.fixedDeltaTime * speed;
        }

        if(model.IsUsingDiscreteActions)
        {
            switch(actionBuffer.DiscreteAction)
            {
                case 0:
                    break;
                case 1:
                    transform.position += Vector3.forward * Time.fixedDeltaTime * speed;
                    break;
                case 2:
                    transform.position += -Vector3.right * Time.fixedDeltaTime * speed;
                    break;
                case 3:
                    transform.position += -Vector3.forward * Time.fixedDeltaTime * speed;
                    break;
                case 4:
                    transform.position += Vector3.right * Time.fixedDeltaTime * speed;
                    break;


            }
        }

        AddReward(-0.001f);
    }
    public override void Heuristic(ActionBuffer actionsOut)
    {
        if(model.IsUsingContinuousActions)
        {
            float xmov = 0;
            float zmov = 0;

            if (Input.GetKey(KeyCode.W))
                zmov = 1;
            else if (Input.GetKey(KeyCode.S))
                zmov = -1;

            if (Input.GetKey(KeyCode.D))
                xmov = 1;
            else if (Input.GetKey(KeyCode.A))
                xmov = -1;

            actionsOut.ContinuousActions[0] = xmov;
            actionsOut.ContinuousActions[1] = zmov;
        }
        
        if(model.IsUsingDiscreteActions)
        {

            actionsOut.DiscreteAction = 0;
            if (Input.GetKey(KeyCode.W))
                actionsOut.DiscreteAction = 1;
            else if(Input.GetKey(KeyCode.A))
                actionsOut.DiscreteAction = 2;
            else if (Input.GetKey(KeyCode.S))
                actionsOut.DiscreteAction = 3;
            else if (Input.GetKey(KeyCode.D))
                actionsOut.DiscreteAction = 4;
        }
    }  
    private void OnTriggerStay(Collider other) //possible to spawn right on the apple collider (we use stay)
    {
        if (other.CompareTag("Goal"))
        {
            SetReward(1f);
            EndEpisode();
        }
        if (other.CompareTag("Wall"))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }

    public void SetDefaultHP()
    {
        model.config.actorLearningRate = 1e-3f;
        model.config.criticLearningRate = 1e-3f;
        model.config.batchSize = 128;
        model.config.bufferSize = 2048;
        model.standardDeviationValue = 2;
        model.config.timescale = 50;

        print("Config changed for Balance ball");
    }
}


