using UnityEngine;
using DeepUnity;
using System.Linq.Expressions;

namespace DeepUnityTutorials
{
    public class SurvivorScript : Agent
    {
        public GunScript gun;
        public float health = 1f;
        public float speed = 2.5f;
        public float rotationSpeed = 2.5f;

        private Rigidbody rb;


        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();

        }

        public override void CollectObservations(StateVector stateVector)
        {
            // gun = 3
            // view = 150
            // here = 4
            stateVector.AddObservation(gun.currentAmmo / (float)gun.CAPACITY);
            stateVector.AddOneHotObservation((int)gun.state, 3);
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
                case 7:
                    gun.Fire();
                    AddReward(-0.0025f);
                    break;
                case 8:
                    gun.Reload(); break;

            }

            AddReward(0.0025f); // Surviving reward
        }

        public override void Heuristic(ActionBuffer actionOut)
        {
            if (Input.GetKey(KeyCode.W))
                actionOut.DiscreteAction = 4;
            else if (Input.GetKey(KeyCode.S))
                actionOut.DiscreteAction = 3;
            else if (Input.GetKey(KeyCode.D))
                actionOut.DiscreteAction = 2;
            else if (Input.GetKey(KeyCode.A))
                actionOut.DiscreteAction = 1;
            else if (Input.GetKey(KeyCode.E))
                actionOut.DiscreteAction = 5;
            else if (Input.GetKey(KeyCode.Q))
                actionOut.DiscreteAction = 6;
            else if (Input.GetMouseButton(0))
                actionOut.DiscreteAction = 7;
            else if(Input.GetKey(KeyCode.R))
                actionOut.DiscreteAction = 8;
            else
                actionOut.DiscreteAction = 0;

        }
        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.CompareTag("Enemy"))
            {
                ZombieEscapeManager.NewEpisode();
            }
        }
    }

}


