using UnityEngine;
using DeepUnity;

namespace DeepUnity.Tutorials
{


    public class BarrelScript : MonoBehaviour
    {
        [Header("Best chance value 0.3 - 0.4.")]
        public float getLadderChance = 0.3f;
        public float speed = 5f;
        public float verticalSpeed;
        private Rigidbody2D rb;
        private Animator animator;
        private Collider2D col;
        [SerializeField] private BarrelState state = BarrelState.Horizontal;

        [ViewOnly] public bool isOnLadder;
        private void Awake()
        {
         
            rb = GetComponent<Rigidbody2D>();
            animator = GetComponent<Animator>();
            col = GetComponent<CircleCollider2D>();
        }
        private void Start()
        {
            rb.AddForce(Vector2.right);
        }
        private void FixedUpdate()
        {
            if (state == BarrelState.Horizontal)
                RollHorizontal();
            else
                RollVertical();
        }
        private void OnDrawGizmos()
        {
            Gizmos.DrawRay(transform.position, Vector2.down * 1.5f);
            Gizmos.DrawRay(transform.position - new Vector3(0f, 1.3f, 0), (Vector2.down + Vector2.left ) * 0.5f);
            Gizmos.DrawRay(transform.position - new Vector3(0f, 1.3f, 0), (Vector2.down + Vector2.right) * 0.5f);
        }

        private void RollVertical()
        {
            state = BarrelState.Vertical;
            animator.SetBool("isRollingVertical", true);
            col.isTrigger = true;
            rb.rotation = 0f;
            rb.angularVelocity = 0f;

            rb.velocity = new Vector2(0, -verticalSpeed);
        }
        private void RollHorizontal()
        {
            state = BarrelState.Horizontal;
            animator.SetBool("isRollingVertical", false);
            col.isTrigger = false;

            if(!isOnLadder)
            {
                int mask = 1 << LayerMask.NameToLayer("Default");
                RaycastHit2D hit = Physics2D.Raycast(transform.position, Vector2.down, 1.5f, ~0 & (~mask));
                
                if (hit.collider == null)
                {
                    rb.velocity = new Vector2(rb.velocity.x * 0.9f, rb.velocity.y); 
                }
                 
                else
                { 
                    RaycastHit2D hitLeft =  Physics2D.Raycast(transform.position - new Vector3(0f, 1.3f, 0), Vector2.down + Vector2.left, 0.5f, ~0 & (~mask));
                    RaycastHit2D hitRight = Physics2D.Raycast(transform.position - new Vector3(0f, 1.3f, 0), Vector2.down + Vector2.right, 0.5f, ~0 & (~mask));

                    if (hitLeft.collider == null && hitRight.collider == null)
                    {
                         /// nothing
                    }
                    else if(hitLeft.collider == null)
                    {
                        rb.velocity = new Vector2(-speed, rb.velocity.y);
                    }
                    else if(hitRight.collider == null)
                    {
                        rb.velocity = new Vector2(+speed, rb.velocity.y);
                    }
                    else if(hitRight.distance > hitLeft.distance)
                    {
                        rb.velocity = new Vector2(speed, rb.velocity.y);
                    }
                    else if (hitRight.distance < hitLeft.distance)
                    {
                        rb.velocity = new Vector2(-speed, rb.velocity.y);
                    }
                }
                    
            }
        }
        private void OnTriggerStay2D(Collider2D collision) // This method is usually called 8-9 times per ladder interaction.. so lets say 0.1 for max chance
        {
          
            if (collision.name == "Ladder" && Random.value < getLadderChance * 0.125f)
            {
                isOnLadder = true;
                transform.position = new Vector2(collision.transform.position.x, transform.position.y); //align barrel with the stairs
                state = BarrelState.Vertical;
            }

            if(collision.name == "OutOfBounds")
            {
                Destroy(this.gameObject);
                MonkeyScript.barrels.Remove(this.gameObject);
            }
        }
        private void OnTriggerExit2D(Collider2D collision)
        {
            if (collision.name == "Ladder")
            {
                isOnLadder = false;
                state = BarrelState.Horizontal;
            }
                
        }


        public enum BarrelState
        {
            Horizontal,
            Vertical
        }
    }
}

