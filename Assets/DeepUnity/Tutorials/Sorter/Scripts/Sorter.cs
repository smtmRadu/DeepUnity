using UnityEngine;
using DeepUnity;
using System.Collections.Generic;
using UnityEngine.UI;
using System.Text;
using DeepUnity.ReinforcementLearning;
namespace DeepUnityTutorials
{
    public class Sorter : Agent
    {
        public List<GameObject> tiles = new List<GameObject>();
        public Text text;
        public float speed = 3f;
        public float rotationSpeed = 5f;
        private const float tileDistance = 3.3f;
        private const float tileHeight = 0.4f;
        int to_find = 0;
        Rigidbody rb;

        public override void OnEpisodeBegin()
        {
            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < tiles.Count; i++)
            {
                stringBuilder.Append($"[{i}]");
            }
            text.text = stringBuilder.ToString();

            to_find = 0;

            Utils.Shuffle(tiles);

            for (int i = 0; i < tiles.Count; i++)
            {
                float angle = i * 2 * Mathf.PI / tiles.Count; // Convert to radians
                float x = Mathf.Cos(angle);
                float z = Mathf.Sin(angle);

                tiles[i].SetActive(true);
                tiles[i].transform.position = new Vector3(x * tileDistance, tileHeight, z * tileDistance);
                tiles[i].transform.rotation = Quaternion.Euler(0, -angle * Mathf.Rad2Deg - 90f, 0); // Convert angle to degrees
            }
        }

        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
        }
        public override void CollectObservations(StateVector stateVector)
        {
            // + 6
            stateVector.AddOneHotObservation(to_find, 10);
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

            AddReward(-0.0025f);
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Ground"))
                return;

            if (collision.collider.CompareTag(to_find.ToString()))
            {
                to_find++;

                StringBuilder stringBuilder = new StringBuilder();
                stringBuilder.Append("<color=green>");
                for (int i = 0; i < to_find; i++)
                {
                    stringBuilder.Append($"[{i}]");
                }
                stringBuilder.Append("</color><color=white>");
                for (int i = to_find; i < tiles.Count; i++)
                {
                    stringBuilder.Append($"[{i}]");
                }
                stringBuilder.Append("</color>");
                text.text = stringBuilder.ToString();


                
                AddReward(1f);
                collision.collider.gameObject.SetActive(false);
                

                if(to_find == tiles.Count) // If it's done
                {
                    EndEpisode();
                    to_find = 0;
                }
            }
            else // Wall or wrong tile
            {
                AddReward(-1f);
                EndEpisode();
            }
           
        }

    }


}


