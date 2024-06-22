using UnityEngine;
using DeepUnity.ReinforcementLearning;
using DeepUnity.Sensors;

namespace DeepUnity.Tutorials
{
    public class BalanceBallVision : Agent
    {
        [Header("This is the balance ball environment but trained with CNN for visual top-down observation.")]
        [Button("SetDefaultHP")]
        [SerializeField] Rigidbody ball;
        [SerializeField] const float rotationSpeed = 1f;
        private CameraSensor cam;

        public override void Awake()
        {
            base.Awake();
            cam = GetComponent<CameraSensor>();
        }
        public override void CollectObservations(out Tensor stateTensor)
        {
            var pixels = cam.GetPixels();
            Tensor image = Tensor.Constant(pixels, (3, 16, 16));
            stateTensor = image;
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
