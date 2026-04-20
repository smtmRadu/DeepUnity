using DeepUnity;
using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    /// <summary>
    /// A 2D one-step continuous bandit. The optimal policy is to output the observed target directly.
    /// This is intentionally tiny so the real DeepUnity SAC stack can be validated quickly.
    /// </summary>
    public sealed class QuadraticBanditAgent : Agent
    {
        [SerializeField] private float targetRange = 1f;

        private Vector2 target;

        public override void OnEpisodeBegin()
        {
            target = new Vector2(
                Utils.Random.Range(-targetRange, targetRange),
                Utils.Random.Range(-targetRange, targetRange));
        }

        public override void CollectObservations(StateVector sensorBuffer)
        {
            sensorBuffer.AddObservation(target.x);
            sensorBuffer.AddObservation(target.y);
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            float dx = actionBuffer.ContinuousActions[0] - target.x;
            float dy = actionBuffer.ContinuousActions[1] - target.y;

            float squaredError = dx * dx + dy * dy;
            float reward = 1f - 0.5f * squaredError;

            SetReward(reward);
            EndEpisode();
        }

        public override void Heuristic(ActionBuffer actionOut)
        {
            actionOut.ContinuousActions[0] = target.x;
            actionOut.ContinuousActions[1] = target.y;
        }
    }
}
