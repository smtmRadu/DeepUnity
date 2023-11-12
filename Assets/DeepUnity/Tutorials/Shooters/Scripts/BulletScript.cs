using UnityEngine;

namespace DeepUnityTutorials
{
    public class BulletScript : MonoBehaviour
    {
        public float lifetime = 2f;
        public bool dieOnCollision = true;

        private void FixedUpdate()
        {
            lifetime -= Time.fixedDeltaTime;
            if(lifetime <= 0)
                Destroy(gameObject);
        }
        private void OnCollisionEnter(Collision collision)
        {
            if (dieOnCollision)
                Destroy(gameObject);
        }
    }
}



