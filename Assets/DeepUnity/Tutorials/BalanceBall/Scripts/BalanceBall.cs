using UnityEngine;
using DeepUnity.ReinforcementLearning;

namespace DeepUnity.Tutorials
{
    public class BalanceBall : Agent
    {
        [Button("SetPPOHP")]
        // [Button("SetDefaultHP")]
        [SerializeField] Rigidbody ball;

        [Button("SetSACHP")]
        [SerializeField] float rotationSpeed = 1f;

        public override void CollectObservations(StateVector sensorBuffer)
        {
            // 10 float observations
            sensorBuffer.AddObservation(transform.rotation); // (-1 to 1 values)
            sensorBuffer.AddObservation(ball.velocity);
            sensorBuffer.AddObservation(ball.gameObject.transform.position - transform.position);
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
        public void SetPPOHP()
        {
            Utils.Random.Seed = 0;
            model.config.trainer = TrainerType.PPO;
            model.config.actorLearningRate = 1e-3f;
            model.config.criticLearningRate = 1e-3f;
            model.config.batchSize = 128;
            model.config.bufferSize = 2048;
            model.standardDeviationValue = 2;
            model.config.timescale = 20;

            print("Config changed for Balance ball");
        }

        public void SetSACDeprHP()
        {
            SetSACHP();
            model.config.trainer = TrainerType.SACDepr; // same verified setup, deprecated CPU trainer (~10x slower)
            print("Config changed for Balance ball (SAC CPU, deprecated)");
        }

        // Verified-convergent SAC setup (2026-06-10, see ReinforcementLearning/FIXES.md):
        // takeoff at ~30k decisions, episodes then reach the 10k maxStep cap.
        public void SetSACHP()
        {
            Utils.Random.Seed = 0;

            model.config.trainer = TrainerType.SAC;
            model.config.actorLearningRate = 1e-3f;
            model.config.criticLearningRate = 1e-3f;
            model.config.gamma = 0.99f;
            model.config.LRSchedule = false;
            model.config.replayBufferSize = 1_000_000;
            model.config.minibatchSize = 64;
            model.config.updateInterval = 1;      // UTD ~1 (the old default of 50 undertrains 50x)
            model.config.updateAfter = 1024;
            model.config.updatesNum = 1;
            model.config.alpha = 0.005f;          // scaled to the 0.025/step reward
            model.config.tau = 0.005f;
            model.config.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            model.config.timescale = 20;

            model.standardDeviationScale = 1.5f;
            model.normalize = false;
            model.trainingDevice = Device.GPU;

            var requester = GetComponent<DecisionRequester>();
            if (requester != null)
            {
                requester.decisionPeriod = 5;     // action repeat: decisive for dQ/da signal
                requester.takeActionsBetweenDecisions = true;
                requester.maxStep = 10_000;
            }

#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(model.config);
            UnityEditor.EditorUtility.SetDirty(model);
            UnityEditor.EditorUtility.SetDirty(this);
            if (requester != null) UnityEditor.EditorUtility.SetDirty(requester);
#endif
            print("Config changed for Balance ball (SAC): updateInterval=1, alpha=0.005, decisionPeriod=5");
        }
    }



}
