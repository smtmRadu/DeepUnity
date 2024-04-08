using UnityEngine;
using DeepUnity;
using System.Collections.Generic;

namespace DeepUnity.Tutorials
{
    public class DungeonManager : MonoBehaviour
    {
        public float maxSecondsPerEpsiode = 60f;
        [ViewOnly] public float timeLeftFromEpisode;
        public GameObject key;
        public Transform keySpawnPoint;
        public DragonScript dragon;
        public Transform dragonSpawnPoint;
        public List<KnightScript> agents;
        public GameObject door1;
        public GameObject door2;

        private void Awake()
        {
            timeLeftFromEpisode = maxSecondsPerEpsiode;
        }
        /// Dungeon Events 

        public void FixedUpdate()
        {
            bool are_all_dead = true;
            foreach (var item in agents)
            {
                if(item.gameObject.activeSelf)
                {
                    are_all_dead = false;
                    break;
                }    
            }
            if (are_all_dead)
                EndDungeonEpisode(false);

            timeLeftFromEpisode -= Time.deltaTime;
            if(timeLeftFromEpisode <= 0)
                EndDungeonEpisode(false);
                
        }

        public void UnlockTheDoors()
        {
            key.SetActive(false);

            // Add 1 reward for the guy that unlocks the door
            foreach (var item in agents)
            {
                if(item.IHaveKey)
                {
                    item.AddReward(1);
                    break;
                }
            }

            door1.GetComponent<Rigidbody>().isKinematic = false;
            door2.GetComponent<Rigidbody>().isKinematic = false;
        }
        public void EndDungeonEpisode(bool agents_won)
        {
            // Reset timer
            timeLeftFromEpisode = maxSecondsPerEpsiode;

            // Reset key
            key.SetActive(true);
            key.transform.position = keySpawnPoint.position;    
            key.transform.parent = this.transform;
            key.GetComponent<Rigidbody>().isKinematic = false;
            key.GetComponent<BoxCollider>().enabled = true;

            // Reset dragon
            dragon.health = dragon.initialHealth;
            dragon.gameObject.SetActive(true);
            dragon.transform.position = dragonSpawnPoint.position;

            // Reset agents
            foreach (var item in agents)
            {
                item.gameObject.SetActive(true);
                item.AddReward(agents_won ? 1 : -1);
                item.EndEpisode();
            }

            // Reset doors
            var rb_door1 = door1.GetComponent<Rigidbody>();
            var rb_door2 = door2.GetComponent<Rigidbody>();
            door1.transform.position = Vector3.zero;
            door2.transform.position = Vector3.zero;
            door1.transform.rotation = Quaternion.identity;
            door2.transform.rotation = Quaternion.identity;
            rb_door1.rotation = Quaternion.identity;
            rb_door2.rotation = Quaternion.identity;
            
            // rb_door1.angularVelocity = Vector3.zero;
            // rb_door2.angularVelocity = Vector3.zero;    
            // rb_door1.velocity = Vector3.zero;
            // rb_door2.velocity = Vector3.zero;

            rb_door1.isKinematic = true;
            rb_door2.isKinematic = true;

        }
        public void DragonIsDead()
        {
            dragon.gameObject.SetActive(false);
        }
    }

}


