using System.Collections.Generic;
using UnityEngine;
using DeepUnity;
using System.Linq;

namespace DeepUnityTutorials
{
    public class ZombieSurvivalManager : MonoBehaviour
    {
        public static ZombieSurvivalManager Instance;
        public SurvivorScript survivor;
        public GameObject zombiePrefab;
        public GameObject zombieSpawnersParent;
        public GameObject survivorSpawnersParent;

        public float zombieSpawnOnSeconds = 5f;
        private float nextSpawnTimeRemained = 0f;

        private LinkedList<GameObject> zombies = new LinkedList<GameObject>();
        private List<Transform> zombieSpawners = new();
        private List<Transform> survivorSpawners = new();   
        private void Awake()
        {
            Instance = this;
            for (int i = 0; i < zombieSpawnersParent.transform.childCount; i++)
            {
                zombieSpawners.Add(zombieSpawnersParent.transform.GetChild(i));
            }
            for (int i = 0; i < survivorSpawnersParent.transform.childCount; i++)
            {
                survivorSpawners.Add(survivorSpawnersParent.transform.GetChild(i));
            }
        }
        public static void NewEpisode()
        {
            Instance.nextSpawnTimeRemained = 0f;
            foreach (GameObject zombie in Instance.zombies)
            {
                Destroy(zombie.gameObject);
            }
            Instance.zombies.Clear();
            Instance.RespawnSurvivor();
        }

        public void Update()
        {
            nextSpawnTimeRemained -= Time.deltaTime;
            if(nextSpawnTimeRemained <= 0f)
            {
                SpawnZombie();
                nextSpawnTimeRemained = zombieSpawnOnSeconds;
            }
        }


        private void SpawnZombie()
        {
            Transform closestePoint = zombieSpawners.OrderBy(x => Vector3.Distance(x.position, survivor.transform.position)).FirstOrDefault();
            List<Transform> toChooseFrom = zombieSpawners.Where(x => x != closestePoint).ToList();

            GameObject zombie = Instantiate(zombiePrefab, Utils.Random.Sample(toChooseFrom).position, Quaternion.identity);
            zombies.AddLast(zombie);
            zombie.GetComponent<ZombieScript>().survivor = survivor;
        }

        private void RespawnSurvivor()
        {
            Transform farthestPoint = Utils.Random.Sample(survivorSpawners);
            survivor.transform.position = farthestPoint.position;
            survivor.gun.currentAmmo = survivor.gun.CAPACITY;
            survivor.gun.state = GunScript.WeaponState.WithAmmo;
            survivor.EndEpisode();
            
        }

    }


}


