using UnityEngine;
using DeepUnity;
using DeepUnity.ReinforcementLearning;

namespace DeepUnityTutorials
{
    // Note: because i forgot to send input if the dragon is alive or not, the guy that get's the key basically knows it, and he can go for the door and then the episode ends.
    // This might be resolved actually if i send to all the agents that they have the key.... we ll see...
    public class KnightScript : Agent
    {
        [SerializeField] public float initialHealth = 100f;
        [ViewOnly] public float health = 100f;
        [SerializeField] private float speed = 10f;
        [SerializeField] private float jump_speed = 10f;
        [SerializeField] private float rot_speed = 5f;
        [SerializeField] private float gravity_strength = 1000f;

        [SerializeField] private DungeonManager dungeonManager;
        [SerializeField] private Transform weaponHand;
        [SerializeField] private Transform keyHand;
        [SerializeField] private GameObject healthBar;

        public bool IHaveKey { get => keyHand.transform.Find("Key") != null; }
        private Rigidbody rb;
        public bool grounded { get; private set; } = true;
        public bool interact { get; private set; } = false;

        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
            health = initialHealth;
        }


        public override void FixedUpdate()
        {
            base.FixedUpdate();
            rb.AddForce(new Vector3(0f, -1f, 0f) * gravity_strength);
            healthBar.transform.localScale = new Vector3(health / 10f, healthBar.transform.localScale.y, healthBar.transform.localScale.z);

            if (health <= 0f)
                this.transform.gameObject.SetActive(false);
        }

        public override void OnEpisodeBegin()
        {
            health = initialHealth;
            interact = false;
            grounded = true;
        }
        public override void CollectObservations(StateVector stateBuffer)
        {
            //240 from sensors

            // 10 new inputs
            Vector3 dirToKey = dungeonManager.key.transform.position - transform.position;
            Vector3 dirToDoor = dungeonManager.door1.transform.position - transform.position;

            stateBuffer.AddObservation(grounded);
            stateBuffer.AddObservation(interact);
            stateBuffer.AddObservation(IHaveKey);

            stateBuffer.AddObservation(transform.rotation.y % 360 / 360f);
            stateBuffer.AddObservation(dirToKey.normalized);
            stateBuffer.AddObservation(dirToDoor.normalized);

        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            Vector3 movement;
            switch (actionBuffer.DiscreteAction)
            {
                case 0:
                    movement = transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 1:
                    movement = transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 2:
                    movement = -transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 3:
                    movement = -transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 4:
                    if (grounded)
                    {
                        movement = transform.up * jump_speed;
                        rb.velocity = new Vector3(rb.velocity.x, movement.y, rb.velocity.z);
                        grounded = false;
                    }
                    break;
                case 5:
                    transform.Rotate(0, -rot_speed, 0);
                    break;
                case 6:
                    transform.Rotate(0, rot_speed, 0);
                    break;
                case 7: //Interact
                    interact = !interact;
                    if (interact)
                    {
                        weaponHand.Rotate(-60f, 0, 0);
                    }
                    else
                    {
                        weaponHand.Rotate(60, 0, 0);
                    }
                    break;

            }
        }

        public override void Heuristic(ActionBuffer actionOut)
        {
            int action;
            
            if (Input.GetKey(KeyCode.A))
                action = 0;
            else if (Input.GetKey(KeyCode.S))
                action = 1;
            else if (Input.GetKey(KeyCode.D))
                action = 2;
            else if (Input.GetKey(KeyCode.W))
                action = 3;
            else if (Input.GetKey(KeyCode.Space))
                action = 4;
            else if (Input.GetKey(KeyCode.Q))
                action = 5;
            else if (Input.GetKey(KeyCode.E))
                action = 6;
            else if (Input.GetMouseButtonDown(0))
                action = 7;
            else
                action = 8;

            actionOut.DiscreteAction = action;
        }
        private void OnCollisionEnter(Collision collision)
        {
            if(collision.gameObject.name == "Key")
            {
                collision.gameObject.transform.position = keyHand.position;
                collision.gameObject.transform.rotation = transform.rotation * Quaternion.Euler(0, -90, 0);
                collision.gameObject.transform.parent = keyHand.transform;
                collision.gameObject.GetComponent<BoxCollider>().enabled = false;
                collision.gameObject.GetComponent<Rigidbody>().isKinematic = true;
            }
            if(collision.gameObject.CompareTag("Box") && IHaveKey) // When hitting the door with the key they open the doors
            {
                dungeonManager.UnlockTheDoors();
            }

            if (collision.collider.CompareTag("Target"))
                dungeonManager.EndDungeonEpisode(true);

            if (collision.collider.CompareTag("Ground"))
                grounded = true;
        }
        private void OnCollisionStay(Collision collision)
        {
            if (collision.collider.CompareTag("Ground"))
                grounded = true;
        }
        private void OnParticleCollision(GameObject particles)
        {
            health -= 1f;
        }
    }
}



