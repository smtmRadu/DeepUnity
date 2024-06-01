using UnityEngine;
using DeepUnity.ReinforcementLearning;

namespace DeepUnity.Tutorials
{
    public class BalanceBall : Agent
    {
        [Button("SetDefaultHP")]
        [SerializeField] Rigidbody ball;
        [SerializeField] const float rotationSpeed = 1f;

        public override void CollectObservations(StateVector sensorBuffer)
        {
            // 10 float observations
            sensorBuffer.AddObservation(transform.rotation); // 4
            sensorBuffer.AddObservation(ball.velocity); // 3
            sensorBuffer.AddObservation(ball.gameObject.transform.position - transform.position); // 3
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // 2 continuous actions
            float xRot = actionBuffer.ContinuousActions[0];
            float zRot = actionBuffer.ContinuousActions[1];

            transform.Rotate(new Vector3(1, 0, 0), xRot * rotationSpeed);
            transform.Rotate(new Vector3(0, 0, 1), zRot * rotationSpeed);

            SetReward(0.025f);
            if (ball.gameObject.transform.position.y < transform.position.y)
                EndEpisode();
        }

        public override void Heuristic(ActionBuffer actionBuffer)
        {
            float xRot = 0f;
            float zRot = 0f;

            if (Input.GetKey(KeyCode.D))
                xRot = 1f;
            else if (Input.GetKey(KeyCode.A))
                xRot = -1f;

            if (Input.GetKey(KeyCode.W))
                zRot = 1f;
            else if (Input.GetKey(KeyCode.S))
                zRot = -1f;

            actionBuffer.ContinuousActions[0] = xRot;
            actionBuffer.ContinuousActions[1] = zRot;
        }



        // This exist because balance ball is the best env for testing out. (in 1 minute it must get around 273 mean steps)
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



}
