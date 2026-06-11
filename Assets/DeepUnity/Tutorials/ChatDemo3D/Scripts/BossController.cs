using System.Collections;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// The Sentinel of the Mist — a towering hollow knight with a halberd guarding the boss
    /// chamber. Dormant before his statue until the player traverses the fog wall (or hits him),
    /// then closes in and swings telegraphed halberd combos. Rolling through swings i-frames
    /// them; blocking halves the damage. Resets to dormant full health when the player dies or
    /// flees the arena; stays felled once killed.
    /// </summary>
    [RequireComponent(typeof(CharacterController))]
    public class BossController : MonoBehaviour
    {
        public static BossController Instance { get; private set; }

        private enum BossState { Dormant, Intro, Combat, Dead }

        [SerializeField, ViewOnly] private BossState state = BossState.Dormant;
        [SerializeField] private string bossName = "Sentinel of the Mist";
        [SerializeField] private float maxHealth = 320f;
        [SerializeField] private float moveSpeed = 2.6f;
        [SerializeField] private float attackRange = 3.0f;
        [SerializeField] private float attackDamage = 30f;
        [SerializeField] private float leashDistance = 18f;
        [Tooltip("Seconds per halberd swing; the builder fills these from the clip lengths / playback speed.")]
        [SerializeField] private float[] attackDurations = { 1.2f, 1.2f, 1.6f };

        [Header("Lunge (gap closer)")]
        [SerializeField] private float lungeRange = 7.5f;     // triggers when the player kites in this band
        [SerializeField] private float lungeSpeed = 8.5f;
        [SerializeField] private float lungeDamage = 36f;
        [SerializeField] private float lungeDuration = 1.1f;  // builder fills from the dash clip length

        [SerializeField] private SoulsPlayerController player;
        [SerializeField] private RectTransform healthFill;
        [SerializeField] private CanvasGroup barGroup;
        [SerializeField] private DeathScreen deathScreen;

        [Header("Music")]
        [SerializeField] private AudioSource musicSource;      // boss theme, looping, starts silent
        [SerializeField] private AudioSource ambienceSource;   // overworld music, ducked during the fight
        [SerializeField] private float musicVolume = 0.45f;
        [SerializeField] private float ambienceVolume = 0.3f;

        private CharacterController cc;
        private Animator animator;
        private float health;
        private float verticalVelocity;
        private float nextAttackAt;
        private float nextLungeAt;
        private bool busy;
        private string locomotionState = "";
        private Vector3 homePos;
        private Quaternion homeRot;

        public string BossName => bossName;

        private void Awake()
        {
            Instance = this;
            cc = GetComponent<CharacterController>();
            animator = GetComponentInChildren<Animator>();
            health = maxHealth;
            homePos = transform.position;
            homeRot = transform.rotation;
        }

        /// <summary>Wakes the Sentinel — called by the mist door when the player steps through.</summary>
        public void Activate()
        {
            if (state != BossState.Dormant || player == null) return;
            state = BossState.Intro;
            StartCoroutine(Intro());
        }

        private IEnumerator Intro()
        {
            FacePlayer(instant: true);
            CrossFade("Idle", 0.3f);
            yield return new WaitForSeconds(1.1f);   // a beat of stillness while the bar slides in
            if (state == BossState.Intro) state = BossState.Combat;
        }

        private void Update()
        {
            // gravity (always, so he settles onto the floor even while dormant)
            if (cc.enabled)
            {
                if (cc.isGrounded && verticalVelocity < 0f) verticalVelocity = -2f;
                verticalVelocity -= 25f * Time.deltaTime;
                cc.Move(Vector3.up * verticalVelocity * Time.deltaTime);
            }

            // boss bar: visible while the fight is on
            bool fightOn = state == BossState.Intro || state == BossState.Combat;
            if (healthFill != null)
                healthFill.anchorMax = new Vector2(Mathf.Clamp01(health / maxHealth), 1f);
            if (barGroup != null)
                barGroup.alpha = Mathf.MoveTowards(barGroup.alpha, fightOn ? 1f : 0f, Time.deltaTime * 2.5f);

            // music: boss theme swells in as the fight starts, fades back to the ambience after
            if (musicSource != null)
            {
                musicSource.volume = Mathf.MoveTowards(musicSource.volume, fightOn ? musicVolume : 0f,
                                                       Time.deltaTime * (fightOn ? 0.35f : 0.15f));
                if (fightOn && !musicSource.isPlaying) musicSource.Play();
                if (!fightOn && musicSource.isPlaying && musicSource.volume <= 0.001f) musicSource.Stop();
            }
            if (ambienceSource != null)
                ambienceSource.volume = Mathf.MoveTowards(ambienceSource.volume, fightOn ? 0.04f : ambienceVolume,
                                                          Time.deltaTime * 0.12f);

            if (state != BossState.Combat || busy || player == null) return;

            Vector3 to = player.transform.position - transform.position;
            to.y = 0f;
            float dist = to.magnitude;

            // the player fled the arena — return to the statue and sleep again
            if (dist > leashDistance)
            {
                ResetFight();
                return;
            }

            FacePlayer();

            if (dist > attackRange)
            {
                // kiting in the mid band gets punished with a dashing halberd lunge
                if (dist < lungeRange && Time.time >= nextLungeAt)
                {
                    StartCoroutine(Lunge());
                    return;
                }
                cc.Move(to.normalized * moveSpeed * Time.deltaTime);
                PlayLocomotion("Run");
            }
            else
            {
                PlayLocomotion("Idle");
                if (Time.time >= nextAttackAt)
                    StartCoroutine(Attack());
            }
        }

        // dash attack: covers ground fast during the first half, blade live mid-dash — rolling
        // sideways evades it, backpedaling does not
        private IEnumerator Lunge()
        {
            busy = true;
            CrossFade("Lunge", 0.12f);
            float t = 0f;
            bool dealt = false;
            while (t < lungeDuration)
            {
                if (state == BossState.Dead) yield break;
                t += Time.deltaTime;

                Vector3 to = player.transform.position - transform.position;
                to.y = 0f;
                if (t < lungeDuration * 0.2f) FacePlayer();   // aims only at the start — commit, then fly
                if (t < lungeDuration * 0.55f)
                    cc.Move(transform.forward * lungeSpeed * Time.deltaTime);

                if (!dealt && t >= lungeDuration * 0.35f && t <= lungeDuration * 0.75f)
                {
                    float surfaceGap = to.magnitude - 0.6f - 0.35f;
                    if (surfaceGap < 2.0f && Vector3.Angle(transform.forward, to) < 55f)
                    {
                        dealt = true;
                        player.TakeDamage(lungeDamage);
                    }
                }
                yield return null;
            }
            busy = false;
            locomotionState = "";
            nextLungeAt = Time.time + Random.Range(4f, 7f);
            nextAttackAt = Time.time + 0.4f;   // can follow up quickly if he lands next to you
        }

        private IEnumerator Attack()
        {
            busy = true;
            int swing = Random.Range(0, attackDurations.Length);
            CrossFade("Attack" + (swing + 1), 0.15f);
            float duration = attackDurations[Mathf.Min(swing, attackDurations.Length - 1)];
            float t = 0f;
            bool dealt = false;
            while (t < duration)
            {
                if (state == BossState.Dead) yield break;
                t += Time.deltaTime;
                if (t < duration * 0.3f) FacePlayer();   // tracks early, commits late — roll the swing
                // halberd contact window: timed to where the swing visually crosses the front
                // (~50-72%), checked per-frame so rolling through it actually evades; distance is
                // surface-to-surface and the arc is tight enough that side-stepping works
                if (!dealt && t >= duration * 0.50f && t <= duration * 0.72f)
                {
                    Vector3 to = player.transform.position - transform.position;
                    to.y = 0f;
                    float surfaceGap = to.magnitude - 0.6f - 0.35f;   // boss radius + player radius
                    if (surfaceGap < 2.3f && Vector3.Angle(transform.forward, to) < 60f)
                    {
                        dealt = true;
                        player.TakeDamage(attackDamage);
                    }
                }
                yield return null;
            }
            busy = false;
            locomotionState = "";
            nextAttackAt = Time.time + Random.Range(0.5f, 1.5f);
        }

        public void TakeDamage(float damage)
        {
            if (state == BossState.Dead) return;
            if (state == BossState.Dormant) { Activate(); }   // striking the dormant Sentinel wakes him

            health = Mathf.Max(0f, health - damage);
            if (health <= 0f)
            {
                StopAllCoroutines();
                state = BossState.Dead;
                StartCoroutine(Die());
                return;
            }
            if (!busy && state == BossState.Combat && Random.value < 0.35f)
                StartCoroutine(Stagger());
        }

        private IEnumerator Stagger()
        {
            busy = true;
            CrossFade("Hit", 0.08f);
            yield return new WaitForSeconds(0.55f);
            busy = false;
            locomotionState = "";
        }

        private IEnumerator Die()
        {
            busy = true;
            CrossFade("Death", 0.15f);
            cc.enabled = false;   // the corpse is scenery now
            if (deathScreen != null) deathScreen.ShowVictory();
            yield return null;
        }

        /// <summary>Back to dormant at full health (player died or fled). A felled boss stays felled.</summary>
        public void ResetFight()
        {
            if (state == BossState.Dead) return;
            StopAllCoroutines();
            busy = false;
            health = maxHealth;
            state = BossState.Dormant;
            nextAttackAt = 0f;
            cc.enabled = false;
            transform.SetPositionAndRotation(homePos, homeRot);
            cc.enabled = true;
            locomotionState = "";
            CrossFade("Idle", 0.25f);
        }

        private void FacePlayer(bool instant = false)
        {
            Vector3 to = player.transform.position - transform.position;
            to.y = 0f;
            if (to.sqrMagnitude < 1e-4f) return;
            Quaternion look = Quaternion.LookRotation(to.normalized);
            transform.rotation = instant
                ? look
                : Quaternion.Slerp(transform.rotation, look, 1f - Mathf.Exp(-5f * Time.deltaTime));
        }

        private void PlayLocomotion(string s)
        {
            if (locomotionState == s) return;
            locomotionState = s;
            CrossFade(s, 0.2f);
        }

        private void CrossFade(string s, float fade)
        {
            if (animator != null)
                animator.CrossFadeInFixedTime(s, fade, 0);
        }
    }
}
