using DeepUnity;
using DeepUnity.ReinforcementLearning;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class MarioScript : Agent
    {
        [Header("Mario adds 6 more normalized observations")]
        public GameObject coinsParent;
        public Transform Pauline;
        public AudioClip jump_sound;
        public AudioClip get_coin;
        public float speed = 10f;
        public float jumpPower = 8f;
        public float climbSpeed = 1f;
        public float gameGravity = -9.81f;
      
        [ViewOnly] public MarioState state;
        [ViewOnly] public float stepReward;       
        private Rigidbody2D rb;
        private Animator animator;
        private Collider2D coll;
        private AudioSource audioSource;
        private List<Transform> coins;

        public override void Awake()
        {
            Physics2D.gravity = new Vector2(0, gameGravity);
            base.Awake();
            rb = GetComponent<Rigidbody2D>();
            animator = GetComponent<Animator>();
            coll = GetComponent<Collider2D>();
            audioSource = GetComponent<AudioSource>();

            coins = new();
            for (int i = 0; i < coinsParent.transform.childCount; i++)
            {
                coins.Add(coinsParent.transform.GetChild(i));
            }
        }


        public override void OnEpisodeBegin()
        {
            foreach (var item in MonkeyScript.barrels)
            {
                Destroy(item);
            }
            MonkeyScript.barrels.Clear();
            foreach (var item in coins)
            {
                item.gameObject.SetActive(true);
            }

        }
        public override void CollectObservations(StateVector stateVector)
        {
            stateVector.AddOneHotObservation((int)state, 4);
            float normalized_x_pos = transform.position.x / 30f;
            float normalized_y_pos = transform.position.y / 30f;
            stateVector.AddObservation(normalized_x_pos);
            stateVector.AddObservation(normalized_y_pos);
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            switch(actionBuffer.DiscreteAction)
            {
                case 0: // Up
                    if (state == MarioState.Grounded) // JUMP
                    {
                        state = MarioState.Jumping;
                        rb.velocity = new Vector2(rb.velocity.x, jumpPower);
                        animator.SetBool("isJumping", true);
                        audioSource.clip = jump_sound;
                        audioSource.Play();
                    }
                    else if(state == MarioState.Climbing) // CLIMP UP
                    {
                        rb.velocity = Vector2.up * climbSpeed;
                        animator.SetFloat("climbSpeed", 1);
                    }                  
                    break;



                case 1: // Left
                    rb.velocity = new Vector2(-speed, rb.velocity.y);
                    break;

                case 2: // Down
                    if(state == MarioState.Climbing)
                    {
                        rb.velocity = -Vector2.up * climbSpeed;   
                        animator.SetFloat("climbSpeed", 1);
                    }
                    break;



                case 3: // Right
                    rb.velocity = new Vector2(speed, rb.velocity.y);
                    break;

                default:
                    if (state == MarioState.Idle)
                        rb.velocity = new Vector2(0, rb.velocity.y);

                    else if (state == MarioState.Grounded)
                        rb.velocity = new Vector2(0, rb.velocity.y);

                    else if(state == MarioState.Jumping)
                        rb.velocity = new Vector2(0, rb.velocity.y);

                    else if (state == MarioState.Climbing)
                        rb.velocity = Vector2.zero;

                    animator.SetFloat("climbSpeed", 0);
                    break;
            }

            animator.SetBool("isClimbing", state == MarioState.Climbing);
            animator.SetBool("isJumping", state == MarioState.Jumping);
            animator.SetFloat("speed", rb.velocity.x);             
        }
        public override void Heuristic(ActionBuffer actionOut)
        {
            if (Input.GetKey(KeyCode.W))
                actionOut.DiscreteAction = 0;
            else if (Input.GetKey(KeyCode.A))
                actionOut.DiscreteAction = 1;
            else if(Input.GetKey(KeyCode.S))
                actionOut.DiscreteAction = 2;
            else if (Input.GetKey(KeyCode.D))
                actionOut.DiscreteAction = 3;
            else
                actionOut.DiscreteAction = 4;
        }

        private void OnCollisionEnter2D(Collision2D collision)
        {
            if (collision.collider.CompareTag("Box"))
            {
                EndEpisode();
            }
        }
        private void OnCollisionStay2D(Collision2D collision)
        {
            if (collision.collider.CompareTag("Ground"))
            {
                if(state != MarioState.Climbing)
                    state = MarioState.Grounded;
            }
        }
        private void OnTriggerEnter2D(Collider2D collision)
        {
            if (collision.name == "Pauline")
            {
                AddReward(3f);
                EndEpisode();
                audioSource.clip = get_coin;
                audioSource.Play();
            }

            if (collision.name == "OutOfBounds")
            {
                AddReward(-1f);
                EndEpisode();
            }

            if(collision.CompareTag("Box"))
            {
                EndEpisode();
            }

            if(collision.CompareTag("Coin"))
            {
                AddReward(+0.5f);
                collision.gameObject.SetActive(false);
            }
        }
        private void OnTriggerStay2D(Collider2D collision)
        {
            if(collision.CompareTag("Ladder"))
            {
                state = MarioState.Climbing;
                coll.isTrigger = true;
                rb.velocity = Vector2.zero;
            }

        }
        private void OnTriggerExit2D(Collider2D collision)
        {
            if (collision.CompareTag("Ladder"))
            {
                state = MarioState.Idle;
                coll.isTrigger = false;
            }

        }



        public enum MarioState
        {
            Idle,
            Grounded,
            Jumping,
            Climbing
        }
    }

}


