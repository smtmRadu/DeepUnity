using UnityEngine;
using DeepUnity;
using System.Collections.Generic;

namespace DeepUnityTutorials
{
    public class Dodger : Agent
    {
        public GameObject ballPrefab;
        public float speed = 2.5f;
        public float rotationSpeed = 2.5f;
        public float ballSpawnOnSteps = 50f;
        public float ballForce = 4f;
        public float ballDistanceSpawn = 3f;
        public float ballHeightSpawn = 2f;
        public int maxBalls = 10;
        private Rigidbody rb;
        private List<GameObject> balls = new List<GameObject>();

        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();
            foreach (var item in balls)
            {
                Destroy(item.gameObject);
            }
            balls.Clear();
        }
        public override void CollectObservations(StateVector stateVector)
        {
            stateVector.AddObservation(transform.position.normalized);
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            Vector3 movement;
            switch (actionBuffer.DiscreteAction)
            {
                case 0:
                    break; //No Action
                case 1:
                    movement = transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 2:
                    movement = -transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 3:
                    movement = transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 4:
                    movement = -transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 5:
                    transform.Rotate(0, rotationSpeed, 0); break;
                case 6:
                    transform.Rotate(0, -rotationSpeed, 0); break;

            }

            if (EpisodeStepCount % ballSpawnOnSteps == 0)
            {
                float random_angle = Utils.Random.Range(0f, 360f);

                float random_rad = Mathf.Rad2Deg * random_angle;
                float x = ballDistanceSpawn * Mathf.Cos(random_rad);
                float z = ballDistanceSpawn * Mathf.Sin(random_rad);

                GameObject g = Instantiate(ballPrefab, transform.position + new Vector3(x, transform.position.y + ballHeightSpawn, z), Quaternion.identity);
                g.transform.localScale *= 0.15f;
                balls.Add(g);
                if (balls.Count > maxBalls)
                { 
                    Destroy(balls[0]);
                    balls.RemoveAt(0);
                }
                g.GetComponent<Rigidbody>().AddForce((transform.position - g.transform.position).normalized * ballForce);
            }
           
        }


        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.CompareTag("Ball"))
            {
                EndEpisode();
                AddReward(-1);
            }
        }
        private void OnTriggerStay(Collider other)
        {
            AddReward(+0.0025f);
        }
    }
}



