using DeepUnity;
using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    /// <summary>
    /// A tiny multi-step 1D control task used to validate off-policy bootstrapping.
    /// The agent observes its scalar position x in [-1, 1], applies a scalar action a in [-1, 1],
    /// and the state evolves as x <- clamp(x + 0.1 * a, -1, 1).
    /// Reward is highest near zero and episodes terminate on success or after a short horizon.
    /// </summary>
    public sealed class OneDimReachAgent : Agent
    {
        [SerializeField] private float spawnRange = 1f;
        [SerializeField] private float stepScale = 0.1f;
        [SerializeField] private float successThreshold = 0.05f;

        private float position;

        public override void OnEpisodeBegin()
        {
            do
            {
                position = Utils.Random.Range(-spawnRange, spawnRange);
            } while (Mathf.Abs(position) < successThreshold * 2f);
        }

        public override void CollectObservations(StateVector sensorBuffer)
        {
            sensorBuffer.AddObservation(position);
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            float action = Mathf.Clamp(actionBuffer.ContinuousActions[0], -1f, 1f);
            position = Mathf.Clamp(position + stepScale * action, -1f, 1f);

            float distance = Mathf.Abs(position);
            float reward = 1f - distance;

            if (distance <= successThreshold)
            {
                reward += 0.5f;
                SetReward(reward);
                EndEpisode();
                return;
            }

            SetReward(reward);
        }

        public override void Heuristic(ActionBuffer actionOut)
        {
            actionOut.ContinuousActions[0] = Mathf.Clamp(-position / Mathf.Max(stepScale, 1e-6f), -1f, 1f);
        }
    }
}
