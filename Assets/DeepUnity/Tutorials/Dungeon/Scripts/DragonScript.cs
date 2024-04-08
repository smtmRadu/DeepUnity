using UnityEngine;
using DeepUnity;

namespace DeepUnity.Tutorials
{
    public class DragonScript : MonoBehaviour
    {
        [SerializeField] private DungeonManager dungeonManager;
        [SerializeField] private GameObject healthBar;
        [SerializeField] private ParticleSystem fire; 
        [SerializeField] private float speed = 1;
        [SerializeField] public int initialHealth;
        [SerializeField] public Vector2 fireRateRange = new Vector2(6f, 8f);
        [ViewOnly] public int health = 10;
        private Rigidbody rb;
        private float timeSinceLastFireBurst;

        private void Awake()
        {
            timeSinceLastFireBurst = Utils.Random.Range(fireRateRange.x, fireRateRange.y);
            health = initialHealth;
            rb = GetComponent<Rigidbody>();
        }

        private void FixedUpdate()
        {
            rb.velocity = new Vector3(0, 0, speed);
            healthBar.transform.localScale = new Vector3(health / 5, healthBar.transform.localScale.y, healthBar.transform.localScale.z);

            if (health <= 0)
                dungeonManager.DragonIsDead();

            timeSinceLastFireBurst -= Time.fixedDeltaTime;
            if(timeSinceLastFireBurst <= 0)
            {
                fire.Play();
                timeSinceLastFireBurst = Utils.Random.Range(fireRateRange.x, fireRateRange.y);
            }
        }


        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.name == "Portal")
            {
                dungeonManager.EndDungeonEpisode(false);
            }
            if(collision.collider.name == "Weapon")
            {
                GameObject weapon = collision.collider.gameObject;
                Transform hand = weapon.transform.parent;
                KnightScript agent = hand.transform.parent.GetComponent<KnightScript>();
                if(agent.interact)
                {
                    agent.AddReward(0.01f);
                    health -= 1;
                }             
            }
        }
    }


}

