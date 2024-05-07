using System.Collections;
using UnityEngine;


namespace DeepUnity.Tutorials
{
    public class GlockScript : MonoBehaviour
    {
        const float FIRE_RATE = 0.15f;
        const int MAG_CAPACITY = 20;
        const float POWER = 8000f;
        const float RECOIL_STRENGTH = 800;

        public int[] StateOneHot { get
            {
                int[] stateX = new int[3];
                stateX[(int)state] = 1;
                return stateX;
            }
        }

        public int Ammo => mag_ammo;


        [Button("Fire")]
        [SerializeField] private Transform muzzle;
        [Button("Reload")]
        [SerializeField] private AudioClip shootSFX;
        [SerializeField] private AudioClip emptyFireSFX;
        [SerializeField] private AudioClip reloadSFX;
        [SerializeField] private AudioClip cockSFX;

        [SerializeField, ViewOnly] private int mag_ammo;

        [SerializeField, ViewOnly] private WeaponState state;
        private Animator animator;
        private AudioSource audioSource;
        private ParticleSystem partSystem;
        private FixedJoint hand1;

        
        private void Awake()
        {
            mag_ammo = MAG_CAPACITY;
            state = WeaponState.IDLE;
            animator = GetComponent<Animator>();
            audioSource = GetComponent<AudioSource>();
            hand1 = GetComponent<FixedJoint>();
            partSystem = GetComponent<ParticleSystem>();
        }
        public void Fire(out Collider hit)
        {
            hit = null;
            if (state == WeaponState.IDLE)
            {
                if(mag_ammo > 0)
                {
                    state = WeaponState.FIRE;
                    mag_ammo--;

                    Ray ray = new Ray(muzzle.position, muzzle.right);
                    RaycastHit hitRC;
                    if (Physics.Raycast(ray, out hitRC, 10000))
                    {
                        hit = hitRC.collider;
                        Rigidbody rb;
                        if (hit.TryGetComponent(out rb))
                            rb.AddForce(muzzle.right * POWER);
                    }
                    hand1.connectedBody.AddForce(-muzzle.right * RECOIL_STRENGTH);
                    
                    audioSource.clip = shootSFX;
                    audioSource.Play();
                    animator.Play("Shoot");
                    partSystem.Play();
                    StartCoroutine(FireRateWait());
                }
                   
                else
                {
                    audioSource.clip = emptyFireSFX;
                    audioSource.Play();
                }

            }
               
        }
        public void Reload()
        {
            if (state == WeaponState.IDLE && mag_ammo < MAG_CAPACITY)
            {
                state = WeaponState.RELOAD;
                StartCoroutine(ReloadProcess());
            }
                

        }
        public void OnDrawGizmos()
        {
            Gizmos.color = Color.black;
            Gizmos.DrawRay(muzzle.position, muzzle.right * 10);
        }
        IEnumerator FireRateWait()
        {
            yield return new WaitForSeconds(FIRE_RATE);
            state = WeaponState.IDLE;
        }
       
        IEnumerator ReloadProcess()
        {
            animator.Play("Reload");
            audioSource.clip = reloadSFX;
            audioSource.Play();
            yield return new WaitForSeconds(reloadSFX.length);
            audioSource.clip = cockSFX;
            audioSource.Play();
            yield return new WaitForSeconds(cockSFX.length);
            mag_ammo = MAG_CAPACITY;
            state = WeaponState.IDLE;

        }
        private enum WeaponState
        {
            IDLE,
            FIRE,
            RELOAD
        }
    }


}

