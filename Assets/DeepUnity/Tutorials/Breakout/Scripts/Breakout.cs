using DeepUnity;
using UnityEngine;
using DeepUnity.Sensors;
using DeepUnity.ReinforcementLearning;

namespace DeepUnityTutorials
{
    public class Breakout : Agent
    {
        CameraSensor view;
        public GameObject blocksHolder;
        public Rigidbody ballrb;
        public float ballSpeed = 1.0f;
        public float platformSpeed = 1.0f;
        public int score = 0;

        public override void Awake()
        {
            base.Awake();
            view = GetComponent<CameraSensor>();
        }
        public override void OnEpisodeBegin()
        {
            score = 0;
            for (int i = 0; i < blocksHolder.transform.childCount; i++)
            {
                var blk = blocksHolder.transform.GetChild(i);
                blk.gameObject.SetActive(true);
            }
            Vector2 randDir = new Vector3(Utils.Random.Value, Utils.Random.Value).normalized;

            float fac = 1;
            if (Utils.Random.Bernoulli())
                fac = -1;
            randDir = new Vector2(randDir.x * fac, -randDir.y);
            ballrb.AddForce(randDir * ballSpeed);
        }
        public override void CollectObservations(out Tensor state)
        {
            var pixels = view.GetPixels();
            state = Tensor.Constant(pixels, (1,32, 40));

        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // Assuming platformSpeed is a positive value
            float translation = actionBuffer.ContinuousActions[0] * platformSpeed;

            // Clamp the x-axis translation within the range of -14 to 14
            float clampedTranslation = Mathf.Clamp(transform.position.x + translation, -14f, 14f);

            // Apply the clamped translation to the object's position
            transform.position = new Vector3(clampedTranslation, transform.position.y, transform.position.z);

            if (score == blocksHolder.transform.childCount)
            {
                AddReward(+1);
                EndEpisode();
            }
        }
    }



}
