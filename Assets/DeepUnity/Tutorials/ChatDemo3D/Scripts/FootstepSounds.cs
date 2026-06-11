using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Plays footstep clips timed to the CharacterController's ground speed —
    /// faster cadence while sprinting, silent while airborne, rolling or attacking.
    /// </summary>
    [RequireComponent(typeof(CharacterController), typeof(AudioSource))]
    public class FootstepSounds : MonoBehaviour
    {
        [SerializeField] private AudioClip[] clips;
        [SerializeField] private float runInterval = 0.38f;
        [SerializeField] private float sprintInterval = 0.29f;
        [SerializeField] private float volume = 0.55f;

        private CharacterController cc;
        private SoulsPlayerController player;
        private AudioSource source;
        private float timer;
        private int lastClip = -1;

        private void Start()
        {
            cc = GetComponent<CharacterController>();
            player = GetComponent<SoulsPlayerController>();
            source = GetComponent<AudioSource>();
        }

        private void Update()
        {
            Vector3 v = cc.velocity;
            v.y = 0f;
            float speed = v.magnitude;

            // cc.isGrounded only reflects the controller's LAST Move() call — a purely
            // horizontal move over flat ground reports false, which silenced all steps on
            // open terrain. Ray-check the ground instead and only treat real falls as airborne.
            bool grounded = cc.isGrounded
                         || Physics.Raycast(transform.position + Vector3.up * 0.3f, Vector3.down, 0.75f);

            if (!grounded || speed < 0.5f || (player != null && player.IsBusy))
            {
                timer = 0.1f;   // first step lands quickly when movement resumes
                return;
            }

            timer -= Time.deltaTime;
            if (timer > 0f || clips == null || clips.Length == 0)
                return;

            timer = speed > 5.5f ? sprintInterval : runInterval;
            int i;
            do { i = Random.Range(0, clips.Length); } while (clips.Length > 1 && i == lastClip);
            lastClip = i;
            // pitched low — an armored knight on stone, not sneakers on pavement
            source.pitch = Random.Range(0.78f, 0.92f);
            source.PlayOneShot(clips[i], volume);
        }
    }
}
