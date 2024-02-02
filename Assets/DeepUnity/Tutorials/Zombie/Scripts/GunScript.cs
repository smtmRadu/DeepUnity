using DeepUnity;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;



namespace DeepUnityTutorials
{
    public class GunScript : MonoBehaviour
    {
        [Button("Fire")]
        public float FIRE_POWER = 1000f;
        public float FIRE_RATE = 0.15f;
        public float RELOAD_TIME = 4f;
        public int CAPACITY = 12;
        private float nextTimeTillFire = 0f;
       
        private Transform muzzle;
        public Text text;
        public GameObject bulletPrefab;
        public AudioClip pistolShot;
        public AudioClip pistolReload;
        public AudioClip pistolCock;

        private ParticleSystem particles;
        private AudioSource audioSource;
        public int currentAmmo { get; set; }
        public WeaponState state { get; set; } = WeaponState.WithAmmo;

        private void Awake()
        {
            currentAmmo = CAPACITY;
            muzzle = transform.GetChild(0);
            audioSource = gameObject.GetComponent<AudioSource>();
            particles = gameObject.GetComponent<ParticleSystem>();
        }

        private void Update()
        {
            nextTimeTillFire -= Time.deltaTime;
           
        }
        private void FixedUpdate()
        {
            if (state == WeaponState.Reloading)
                text.text = $"Reloading";
            else
                text.text = $"{currentAmmo}/{CAPACITY}";
        }
        public void Fire()
        {
            if (nextTimeTillFire >= 0f)
                return;

            if (state != WeaponState.WithAmmo)
                return;

            audioSource.clip = pistolShot;
            audioSource.Play();
            nextTimeTillFire = FIRE_RATE;

            GameObject bullet = Instantiate(bulletPrefab, muzzle.position, Quaternion.identity);
            bullet.GetComponent<Rigidbody>().AddForce(-transform.forward * FIRE_POWER);
            particles.Play();
            currentAmmo--;

            if (currentAmmo == 0)
                state = WeaponState.WithoutAmmo;
        }
        public void Reload()
        {
            if (currentAmmo == CAPACITY)
                return;

            if (state == WeaponState.Reloading)
                return;
            
            StartCoroutine("ReloadProcess");           
        }

        private IEnumerator ReloadProcess()
        {
            audioSource.clip = pistolReload;
            audioSource.Play();
            state = WeaponState.Reloading;
            yield return new WaitForSeconds(RELOAD_TIME - pistolCock.length);
            audioSource.clip = pistolCock;
            audioSource.Play();
            yield return new WaitForSeconds(pistolCock.length);
            currentAmmo = CAPACITY;
            state = WeaponState.WithAmmo;
        }

        public enum WeaponState
        {
            WithAmmo,
            WithoutAmmo,
            Reloading
        }
    }

}


