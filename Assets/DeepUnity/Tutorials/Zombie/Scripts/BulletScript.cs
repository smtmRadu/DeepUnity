using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class BulletScript : MonoBehaviour
    {
        public float lifetime = 1f;

        private void Update()
        {
            lifetime -= Time.deltaTime;

            if(lifetime <= 0f )
                Destroy( this.gameObject );
        }
    }


}

