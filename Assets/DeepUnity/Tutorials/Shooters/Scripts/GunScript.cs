using UnityEngine;
using DeepUnity;
using System.Collections;
using UnityEngine.UI;
using System;

namespace DeepUnityTutorials
{
    public class GunScript : MonoBehaviour
    {
        [Button("Fire")]
        [SerializeField] private Transform muzzle;
        [Button("Reload")]
        [SerializeField] private GameObject bulletPrefab;
        [SerializeField] private AudioSource shootingAudio;
        [SerializeField] private AudioSource reloadAudio;

        [SerializeField] private Text ammo_text;
        [SerializeField] private float POWER = 1000f;
        [SerializeField] private int MAG_CAPACITY = 12;
        [SerializeField] private float FIRE_RATE = 0.1f;
        [SerializeField] private float RELOAD_TIME = 3f;
        [ReadOnly] public int currentMagAmmo;

        private ParticleSystem particleSystem;


        public WeaponState weaponState { get; private set; } = WeaponState.IdleCharged;

        event Action OnStateChanged;
        private void Awake()
        {
            particleSystem = GetComponent<ParticleSystem>();

            currentMagAmmo = MAG_CAPACITY;
            OnStateChanged += UpdateText;
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.red;
            Gizmos.DrawRay(muzzle.position, -transform.forward * 25);
        }
        public void Fire()
        {
            if (weaponState == WeaponState.IdleCharged)
                StartCoroutine(FireProcess());
        }
        public void Reload()
        {
            if ((weaponState == WeaponState.IdleCharged
                || weaponState == WeaponState.EmptyMagazine)
                && currentMagAmmo != MAG_CAPACITY)
                StartCoroutine(ReloadProcess());
        }
        private void UpdateText()
        {
            ammo_text.text = $"{currentMagAmmo}/{MAG_CAPACITY}";
        }

        private IEnumerator FireProcess()
        {
            currentMagAmmo--;
            weaponState = WeaponState.Shooting;
            OnStateChanged.Invoke();
            shootingAudio?.Play();
            particleSystem?.Play();
            var bullet = Instantiate(bulletPrefab, muzzle.position, transform.rotation);
            bullet.GetComponent<Rigidbody>().AddForce(-bullet.transform.forward * POWER, ForceMode.Impulse);
            yield return new WaitForSeconds(1f / FIRE_RATE);
            weaponState = WeaponState.IdleCharged;

            if (currentMagAmmo == 0)
            {
                weaponState = WeaponState.EmptyMagazine;
                OnStateChanged.Invoke();
            }

        }
        private IEnumerator ReloadProcess()
        {
            weaponState = WeaponState.Reloading;
            OnStateChanged.Invoke();
            reloadAudio.pitch *= 0.5f;
            reloadAudio?.Play();
            yield return new WaitForSeconds(RELOAD_TIME);
            currentMagAmmo = MAG_CAPACITY;
            weaponState = WeaponState.IdleCharged;
            OnStateChanged.Invoke();
            reloadAudio.pitch *= 2f;
            reloadAudio?.Play();

        }
        public enum WeaponState
        {
            IdleCharged,
            EmptyMagazine,
            Reloading,
            Shooting
        }

    }


}

