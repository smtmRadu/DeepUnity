using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class Autoshooter : MonoBehaviour
    {
        [Header("This script is used to continuosly shoot the agent with some projectiles from upper hemisphere during training.")]
        [SerializeField] private GameObject projectilePrefab;
        [Min(0.001f), SerializeField] private float shootInterval = 1.5f;
        [Min(0.001f), SerializeField] private float distance = 5f;
        [Min(0.001f), SerializeField] private float power = 5f;

        GameObject projectile;
        float lastShoot;

        private void Start()
        {
            lastShoot = Time.time;
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, distance + 0.1f);
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(transform.position, distance);         
            Gizmos.DrawRay(transform.position, Vector3.up * distance);
            Gizmos.DrawRay(transform.position, Vector3.right * distance);
            Gizmos.DrawRay(transform.position, -Vector3.right * distance);
            Gizmos.DrawRay(transform.position, Vector3.forward * distance);
            Gizmos.DrawRay(transform.position, -Vector3.forward * distance);
        }

        private void Update()
        {
            if (Time.time - shootInterval > lastShoot)
                Shoot();
        }

        void Shoot()
        {
            if(projectile != null)
                Destroy(projectile);


           

            Vector3 start = Utils.Random.OnUnitSphere * distance;
            var correctedY = Mathf.Clamp(Mathf.Abs(start.y), 0.1f, 1f);
            start = transform.position + new Vector3(start.x, correctedY, start.z);
            
            var direction = transform.position - start;

            projectile = Instantiate(projectilePrefab, start, Quaternion.identity * Quaternion.Euler(Random.value * 360f, Random.value * 360f, Random.value * 360f));
            projectile.GetComponent<Rigidbody>().AddForce(direction * power, ForceMode.Impulse);

            lastShoot = Time.time;
        }
    }
}



