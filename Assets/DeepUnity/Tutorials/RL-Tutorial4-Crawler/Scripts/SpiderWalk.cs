using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class SpiderWalk : Agent
    {
        [SerializeField] Transform target;
        [SerializeField] Transform directionArrow;

        [SerializeField] JointScript shin1;
        [SerializeField] JointScript shin2;
        [SerializeField] JointScript shin3;
        [SerializeField] JointScript shin4;

        [SerializeField] JointScript thigh1;
        [SerializeField] JointScript thigh2;
        [SerializeField] JointScript thigh3;
        [SerializeField] JointScript thigh4;

        [SerializeField] float speed = 5f;


        private Rigidbody selfRB;

        public override void Awake()
        {
            base.Awake();
            selfRB = GetComponent<Rigidbody>();
        }
        public override void OnEpisodeBegin()
        {
            float random_angle = Utils.Random.Range(0f, 360f);
            const float distance = 15f;

            float random_rad = Mathf.Rad2Deg * random_angle;
            float x = distance * Mathf.Cos(random_rad);
            float z = distance * Mathf.Sin(random_rad);

            target.localPosition = new Vector3(x, target.localPosition.y, z);
            // target.localPosition = new Vector3(Random.Range(-35f, 35f), target.position.y, Random.Range(-35f, 35f));
        }
        public override void CollectObservations(StateBuffer sensorBuffer)
        {
            // RaySensor
            // +1

            // + 10
            sensorBuffer.AddObservation(transform.rotation.normalized);
            sensorBuffer.AddObservation(selfRB.velocity.normalized);
            sensorBuffer.AddObservation(selfRB.angularVelocity.normalized);

            // + 40
            sensorBuffer.AddObservation(thigh1.transform.rotation.normalized);
            sensorBuffer.AddObservation(thigh1.rb.velocity.normalized);
            sensorBuffer.AddObservation(thigh1.rb.angularVelocity.normalized);
            sensorBuffer.AddObservation(thigh2.transform.rotation.normalized);
            sensorBuffer.AddObservation(thigh2.rb.velocity.normalized);
            sensorBuffer.AddObservation(thigh2.rb.angularVelocity.normalized);
            sensorBuffer.AddObservation(thigh3.transform.rotation.normalized);
            sensorBuffer.AddObservation(thigh3.rb.velocity.normalized);
            sensorBuffer.AddObservation(thigh3.rb.angularVelocity.normalized);
            sensorBuffer.AddObservation(thigh4.transform.rotation.normalized);
            sensorBuffer.AddObservation(thigh4.rb.velocity.normalized);
            sensorBuffer.AddObservation(thigh4.rb.angularVelocity.normalized);

            // + 40
            sensorBuffer.AddObservation(shin1.transform.rotation.normalized);
            sensorBuffer.AddObservation(shin1.rb.velocity.normalized);
            sensorBuffer.AddObservation(shin1.rb.angularVelocity.normalized);
            sensorBuffer.AddObservation(shin2.transform.rotation.normalized);
            sensorBuffer.AddObservation(shin2.rb.velocity.normalized);
            sensorBuffer.AddObservation(shin2.rb.angularVelocity.normalized);
            sensorBuffer.AddObservation(shin3.transform.rotation.normalized);
            sensorBuffer.AddObservation(shin3.rb.velocity.normalized);
            sensorBuffer.AddObservation(shin3.rb.angularVelocity.normalized);
            sensorBuffer.AddObservation(shin4.transform.rotation.normalized);
            sensorBuffer.AddObservation(shin4.rb.velocity.normalized);
            sensorBuffer.AddObservation(shin4.rb.angularVelocity.normalized);

            // + 3
            sensorBuffer.AddObservation((target.position - transform.position).normalized);

            // Total 94 
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            thigh1.SetAngularVelocity(actionBuffer.ContinuousActions[0] * speed, actionBuffer.ContinuousActions[1] * speed, 0);
            thigh2.SetAngularVelocity(actionBuffer.ContinuousActions[2] * speed, actionBuffer.ContinuousActions[3] * speed, 0);
            thigh3. SetAngularVelocity(actionBuffer.ContinuousActions[4] * speed, actionBuffer.ContinuousActions[5] * speed, 0);
            thigh4.SetAngularVelocity(actionBuffer.ContinuousActions[6] * speed, actionBuffer.ContinuousActions[7] * speed, 0);

            shin1.SetAngularVelocity(actionBuffer.ContinuousActions[8] * speed, 0, 0);
            shin2.SetAngularVelocity(actionBuffer.ContinuousActions[9] * speed, 0, 0);
            shin3.SetAngularVelocity(actionBuffer.ContinuousActions[10] * speed, 0, 0);
            shin4.SetAngularVelocity(actionBuffer.ContinuousActions[11] * speed, 0, 0);

            // AddReward(+0.0005f);
            AddReward(transform.localPosition.y / 1000f);
            AddReward(Mathf.Clamp(1f / Vector3.Distance(transform.position, target.position), 0, 1f) / 500f);

            // Point the arrow towards the target
            directionArrow.rotation = Quaternion.LookRotation(target.position - transform.position) * Quaternion.Euler(0, 90f, 0);
        }

        public override void Heuristic(ActionBuffer actionBuffer)
        {
            float hor = Input.GetAxis("Horizontal");
            float vert = Input.GetAxis("Vertical");

            for (int i = 0; i < 8; i += 2)
            {
                actionBuffer.ContinuousActions[i] = vert;
            }
            for (int i = 8; i < 12; i++)
            {
                actionBuffer.ContinuousActions[i] = hor;
            }

        }

        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.CompareTag("Floor"))
            {
                AddReward(-1f);
                EndEpisode();
            }
          
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal"))
            {
                AddReward(1f);
                EndEpisode();
            }
        }

        private void OnTriggerStay(Collider other)
        {
            if (other.CompareTag("Goal"))
            {
                AddReward(1f);
                EndEpisode();
            }
        }
    }


}


