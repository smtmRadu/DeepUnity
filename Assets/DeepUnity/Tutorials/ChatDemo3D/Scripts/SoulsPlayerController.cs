using System.Collections;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Souls-like third person character. WASD moves relative to the camera, Shift sprints,
    /// Space dodge-rolls, left click swings the sword (3 hit combo), holding right click blocks
    /// with the shield. The Animator is driven from code via CrossFade onto plain states
    /// (one per clip, no transition graph): Idle, Walk, Run, Sprint, Roll,
    /// Attack1..3, BlockIdle, Hit. The scene builder wires the controller and clip durations.
    /// </summary>
    [RequireComponent(typeof(CharacterController))]
    public class SoulsPlayerController : MonoBehaviour
    {
        public enum PlayerMode { Exploring, Interaction }

        [SerializeField, ViewOnly] private PlayerMode mode = PlayerMode.Exploring;
        [SerializeField] public SoulsCameraRig cam;

        [Header("Locomotion")]
        [SerializeField] private float runSpeed = 4.3f;
        [SerializeField] private float sprintSpeed = 7.0f;
        [SerializeField] private float blockMoveSpeed = 1.7f;
        [SerializeField] private float turnSharpness = 12f;
        [SerializeField] private float gravity = 25f;

        [Header("Dodge roll")]
        [SerializeField] private float rollDistance = 5.0f;
        [SerializeField] private float rollDuration = 0.7f;   // builder overwrites with the real clip length

        [Header("Attack")]
        [SerializeField] private float attackStepDistance = 0.55f;
        [Tooltip("Seconds per combo swing; the scene builder fills these from the actual clip lengths.")]
        [SerializeField] private float[] attackDurations = { 0.8f, 0.8f, 1.0f };

        [Header("Stamina")]
        [SerializeField] private float maxStamina = 100f;
        [SerializeField] private float sprintDrainPerSec = 16f;
        [SerializeField] private float regenPerSec = 26f;
        [SerializeField] private float rollCost = 20f;
        [SerializeField] private float attackCost = 17f;
        [Tooltip("After running the bar dry, sprinting stays locked until stamina recovers to this value.")]
        [SerializeField] private float sprintRecoverStamina = 30f;

        [Header("Mist traversal")]
        [SerializeField] private float mistWalkSpeed = 1.25f;
        [SerializeField] private float interactDuration = 1.0f;   // builder overwrites with the Interact clip length

        [Header("Audio")]
        [SerializeField] private AudioClip rollClip;

        [Header("Combat")]
        [SerializeField] private float maxHealth = 100f;
        [SerializeField] private float swordDamage = 34f;
        [SerializeField] private DeathScreen deathScreen;

        [Header("Heal flask")]
        [SerializeField] private int maxFlaskCharges = 5;
        [SerializeField] private float flaskHealAmount = 45f;
        [SerializeField] private float drinkDuration = 1.7f;
        [SerializeField] private GameObject flaskObject;   // glowing bottle in the left hand, hidden unless drinking
        [SerializeField] private AudioClip drinkClip;      // glass clink as the flask comes up
        [SerializeField] private AudioClip healClip;       // soft chime when the heal lands

        private CharacterController controller;
        private Animator animator;
        private AudioSource audioSource;
        private float verticalVelocity;
        private float stamina;
        private float health;
        private bool exhausted;     // sprint lockout until stamina recovers (prevents drain/regen flicker)
        private bool busy;          // mid-roll or mid-swing
        private bool rolling;       // dodge i-frames
        private bool dead;
        private bool blocking;
        private string locomotionState = "";
        private float armRaise, armRaiseTarget;   // reach-into-the-mist overlay weight
        private float drinkWeight, drinkWeightTarget;   // flask-to-mouth overlay weight
        private int flaskCharges;
        private bool drinking;
        private Coroutine drinkCo;
        private Vector3 spawnPos;
        private Quaternion spawnRot;

        public bool IsBusy => busy;
        public bool IsExploring => mode == PlayerMode.Exploring && !busy;
        public Animator Animator => animator;
        public float Stamina01 => stamina / maxStamina;
        public float Health01 => health / maxHealth;
        public int FlaskCharges => flaskCharges;

        private void Start()
        {
            controller = GetComponent<CharacterController>();
            animator = GetComponentInChildren<Animator>();
            audioSource = GetComponent<AudioSource>();
            stamina = maxStamina;
            health = maxHealth;
            flaskCharges = maxFlaskCharges;
            spawnPos = transform.position;
            spawnRot = transform.rotation;
            if (flaskObject != null) flaskObject.SetActive(false);
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        /// <summary>Freezes gameplay input and frees the mouse so the player can type in the chat.</summary>
        public void EnterInteractiveMode()
        {
            mode = PlayerMode.Interaction;
            blocking = false;
            PlayLocomotion("Idle");
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        public void ExitInteractiveMode()
        {
            mode = PlayerMode.Exploring;
            locomotionState = "";   // force a locomotion refresh in case a dialogue pose is playing
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        /// <summary>Typing gesture during dialogue — mirrors the NPC's talking animation.
        /// Eases in quickly, eases back out slowly so it never snaps.</summary>
        public void PlayDialoguePose(bool talking)
        {
            locomotionState = "";
            CrossFade(talking ? "Talking" : "Idle", talking ? 0.25f : 0.55f);
        }

        private void Update()
        {
            ApplyGravity();

            if (mode != PlayerMode.Exploring || busy)
                return;

            Vector3 wishDir = WishDirection();
            blocking = Input.GetMouseButton(1);

            bool sprinting = Input.GetKey(KeyCode.LeftShift) && !exhausted && wishDir != Vector3.zero && !blocking;
            if (sprinting)
            {
                stamina -= sprintDrainPerSec * Time.deltaTime;
                if (stamina <= 0f)
                {
                    // lock sprint out until the bar recovers — without this the drain/regen
                    // pair flips the Sprint/Run state every frame and the animation freezes
                    stamina = 0f;
                    exhausted = true;
                    sprinting = false;
                }
            }
            else
            {
                stamina = Mathf.Min(maxStamina, stamina + regenPerSec * (blocking ? 0.4f : 1f) * Time.deltaTime);
                if (exhausted && stamina >= sprintRecoverStamina)
                    exhausted = false;
            }

            // sprinting turns the character into the movement direction (Elden Ring style);
            // otherwise the camera-locked yaw keeps the back to the camera so the mouse steers
            if (cam != null)
            {
                Quaternion targetRot = sprinting
                    ? Quaternion.LookRotation(wishDir)
                    : Quaternion.Euler(0f, cam.Yaw, 0f);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, 1f - Mathf.Exp(-turnSharpness * Time.deltaTime));
            }

            if (Input.GetKeyDown(KeyCode.Space) && stamina >= rollCost * 0.5f)
            {
                stamina = Mathf.Max(0f, stamina - rollCost);
                StartCoroutine(Roll(wishDir));
                return;
            }
            if (Input.GetMouseButtonDown(0) && stamina >= attackCost * 0.5f)
            {
                StartCoroutine(AttackCombo());
                return;
            }
            if (Input.GetKeyDown(KeyCode.R) && flaskCharges > 0 && health < maxHealth)
            {
                drinkCo = StartCoroutine(DrinkFlask());
                return;
            }

            float speed = wishDir == Vector3.zero ? 0f
                        : blocking ? blockMoveSpeed
                        : sprinting ? sprintSpeed
                        : runSpeed;

            if (wishDir != Vector3.zero)
                controller.Move(wishDir * speed * Time.deltaTime);

            if (blocking) PlayLocomotion(speed > 0f ? "Walk" : "BlockIdle");
            else if (speed == 0f) PlayLocomotion("Idle");
            else if (sprinting) PlayLocomotion("Sprint");
            // mostly-backward movement plays the jog reversed (RunB) instead of moonwalking
            else PlayLocomotion(Vector3.Dot(transform.forward, wishDir) < -0.5f ? "RunB" : "Run");
        }

        /// <summary>
        /// Scripted mist-door entry: jog to the threshold, reach a hand into the fog
        /// (Interact clip), then push slowly through to the far side. Input is locked
        /// for the duration; MistDoor drives the white flash and the blocking collider.
        /// </summary>
        public IEnumerator TraverseMist(Vector3 doorPoint, Vector3 exitPoint)
        {
            busy = true;
            blocking = false;

            yield return MoveTo(doorPoint, runSpeed * 0.85f, "Run");

            FaceTowards(exitPoint);
            CrossFade("Interact", 0.2f);
            yield return new WaitForSeconds(interactDuration * 0.9f);

            CrossFade("MistWalk", 0.35f);
            armRaiseTarget = 0.6f;   // extra forward reach layered on the push pose
            yield return MoveTo(exitPoint, mistWalkSpeed, null);
            armRaiseTarget = 0f;

            CrossFade("Idle", 0.45f);
            busy = false;
            locomotionState = "";
        }

        private IEnumerator MoveTo(Vector3 point, float speed, string state)
        {
            if (state != null) CrossFade(state, 0.2f);
            float timeout = 6f;
            while (timeout > 0f)
            {
                timeout -= Time.deltaTime;
                Vector3 to = point - transform.position;
                to.y = 0f;
                if (to.magnitude < 0.18f) break;
                transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(to.normalized),
                                                      1f - Mathf.Exp(-10f * Time.deltaTime));
                controller.Move(to.normalized * speed * Time.deltaTime);
                yield return null;
            }
        }

        // world-axis bone overlays so they work regardless of the rig's bone axes;
        // LateUpdate runs after the Animator writes the pose, so the offsets survive
        private void LateUpdate()
        {
            if (animator == null) return;

            // reach-into-the-mist: right arm forward
            armRaise = Mathf.MoveTowards(armRaise, armRaiseTarget, Time.deltaTime * 2f);
            if (armRaise > 0.001f)
            {
                var upperArm = animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
                if (upperArm != null)
                    upperArm.rotation = Quaternion.AngleAxis(-55f * armRaise, transform.right) * upperArm.rotation;
                var lowerArm = animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
                if (lowerArm != null)
                    lowerArm.rotation = Quaternion.AngleAxis(-20f * armRaise, transform.right) * lowerArm.rotation;
            }

            // drinking: left arm lifts the flask to the mouth, head tips back
            drinkWeight = Mathf.MoveTowards(drinkWeight, drinkWeightTarget, Time.deltaTime * 3f);
            if (drinkWeight > 0.001f)
            {
                var upperArm = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
                if (upperArm != null)
                    upperArm.rotation = Quaternion.AngleAxis(-72f * drinkWeight, transform.right) * upperArm.rotation;
                var lowerArm = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
                if (lowerArm != null)
                    lowerArm.rotation = Quaternion.AngleAxis(-58f * drinkWeight, transform.right) * lowerArm.rotation;
                var head = animator.GetBoneTransform(HumanBodyBones.Head);
                if (head != null)
                    head.rotation = Quaternion.AngleAxis(-16f * drinkWeight, transform.right) * head.rotation;
            }
        }

        /// <summary>Instantly turns the character (used by the NPC to face him during dialogue).</summary>
        public void FaceTowards(Vector3 worldPoint)
        {
            Vector3 dir = worldPoint - transform.position;
            dir.y = 0f;
            if (dir.sqrMagnitude > 1e-4f)
                transform.rotation = Quaternion.LookRotation(dir.normalized);
        }

        private void ApplyGravity()
        {
            if (controller.isGrounded && verticalVelocity < 0f)
                verticalVelocity = -2f;
            verticalVelocity -= gravity * Time.deltaTime;
            controller.Move(Vector3.up * verticalVelocity * Time.deltaTime);
        }

        private Vector3 WishDirection()
        {
            float h = Input.GetAxisRaw("Horizontal");
            float v = Input.GetAxisRaw("Vertical");
            Vector3 fwd = cam != null ? cam.PlanarForward : Vector3.forward;
            Vector3 right = Vector3.Cross(Vector3.up, fwd);
            Vector3 dir = fwd * v + right * h;
            return dir.sqrMagnitude > 1e-4f ? dir.normalized : Vector3.zero;
        }

        private IEnumerator Roll(Vector3 dir)
        {
            busy = true;
            rolling = true;   // i-frames: the boss can't hurt a rolling player
            if (dir == Vector3.zero)
                dir = transform.forward;
            transform.rotation = Quaternion.LookRotation(dir);
            CrossFade("Roll", 0.06f);
            if (rollClip != null && audioSource != null)
            {
                audioSource.pitch = Random.Range(0.82f, 0.92f);   // cloth-and-plate tumble
                audioSource.PlayOneShot(rollClip, 0.75f);
            }

            float t = 0f;
            while (t < rollDuration)
            {
                t += Time.deltaTime;
                // souls-style i-frames: invulnerable through the tumble, vulnerable in recovery —
                // rolling INTO an attack with good timing evades it, panic-rolling late doesn't
                if (rolling && t >= rollDuration * 0.70f)
                    rolling = false;
                // front-loaded speed curve so the dodge bites at the start like a souls roll
                float k = 2f * (1f - Mathf.Clamp01(t / rollDuration));
                controller.Move(dir * (rollDistance / rollDuration) * k * Time.deltaTime);
                ApplyGravity();
                yield return null;
            }
            rolling = false;
            EndAction();
        }

        private IEnumerator AttackCombo()
        {
            busy = true;
            int swing = 0;
            while (true)
            {
                stamina = Mathf.Max(0f, stamina - attackCost);
                CrossFade("Attack" + (swing + 1), 0.06f);
                float duration = attackDurations[Mathf.Min(swing, attackDurations.Length - 1)];
                float t = 0f;
                bool queuedNext = false, dealtHit = false;
                while (t < duration)
                {
                    t += Time.deltaTime;
                    // small forward push during the first part of the swing
                    if (t < duration * 0.4f)
                        controller.Move(transform.forward * (attackStepDistance / (duration * 0.4f)) * Time.deltaTime);
                    // the blade is live for a window mid-swing (not one instant — a single-frame
                    // check whiffs when the boss steps into range a moment later); distance is
                    // measured to the boss's body surface, not his root, so big targets feel fair
                    if (!dealtHit && t > duration * 0.32f && t < duration * 0.62f)
                    {
                        var boss = BossController.Instance;
                        if (boss != null)
                        {
                            Vector3 to = boss.transform.position - transform.position;
                            to.y = 0f;
                            float surfaceGap = to.magnitude - 0.6f;   // boss controller radius
                            if (surfaceGap < 1.9f && Vector3.Angle(transform.forward, to) < 70f)
                            {
                                dealtHit = true;
                                boss.TakeDamage(swordDamage);
                            }
                        }
                    }
                    if (t > duration * 0.35f && Input.GetMouseButtonDown(0))
                        queuedNext = true;
                    ApplyGravity();
                    yield return null;
                }
                if (!queuedNext || swing >= attackDurations.Length - 1 || stamina < attackCost * 0.5f)
                    break;
                swing++;
            }
            EndAction();
        }

        /// <summary>Estus, basically: locks input, raises the flask to the helm (overlay in
        /// LateUpdate), heals mid-drink. Getting hit interrupts the drink and wastes the charge.</summary>
        private IEnumerator DrinkFlask()
        {
            busy = true;
            drinking = true;
            blocking = false;
            flaskCharges--;
            if (flaskObject != null) flaskObject.SetActive(true);
            CrossFade("Idle", 0.2f);
            drinkWeightTarget = 1f;
            if (drinkClip != null && audioSource != null)
            {
                audioSource.pitch = Random.Range(0.92f, 1.0f);
                audioSource.PlayOneShot(drinkClip, 0.6f);
            }

            float t = 0f;
            bool healed = false;
            while (t < drinkDuration)
            {
                t += Time.deltaTime;
                if (!healed && t >= drinkDuration * 0.55f)
                {
                    healed = true;
                    health = Mathf.Min(maxHealth, health + flaskHealAmount);
                    if (healClip != null && audioSource != null)
                    {
                        audioSource.pitch = 1f;
                        audioSource.PlayOneShot(healClip, 0.45f);
                    }
                }
                yield return null;
            }

            drinkWeightTarget = 0f;
            if (flaskObject != null) flaskObject.SetActive(false);
            drinking = false;
            EndAction();
        }

        private void InterruptDrink()
        {
            if (drinkCo != null) StopCoroutine(drinkCo);
            drinking = false;
            drinkWeightTarget = 0f;
            if (flaskObject != null) flaskObject.SetActive(false);
            busy = false;
            locomotionState = "";
        }

        /// <summary>Boss (or any hazard) damage entry point. Rolling grants i-frames; blocking
        /// halves the hit; death plays the Death clip, shows YOU DIED and respawns at the gate.</summary>
        public void TakeDamage(float damage)
        {
            if (dead || rolling || mode != PlayerMode.Exploring) return;

            health = Mathf.Max(0f, health - (blocking ? damage * 0.45f : damage));
            if (health <= 0f)
            {
                StopAllCoroutines();   // cut any roll/attack/traverse mid-flight before dying
                StartCoroutine(Die());
                return;
            }
            if (drinking)
            {
                InterruptDrink();   // greedy heals get punished — charge is spent, no heal
                StartCoroutine(Flinch());
            }
            else if (!busy) StartCoroutine(Flinch());
        }

        private IEnumerator Flinch()
        {
            busy = true;
            CrossFade("Hit", 0.08f);
            yield return new WaitForSeconds(0.35f);
            EndAction();
        }

        private IEnumerator Die()
        {
            dead = true;
            busy = true;
            rolling = false;
            blocking = false;
            armRaiseTarget = 0f;
            drinking = false;
            drinkWeightTarget = 0f;
            if (flaskObject != null) flaskObject.SetActive(false);
            CrossFade("Death", 0.1f);
            if (deathScreen != null) deathScreen.ShowDeath();
            yield return new WaitForSeconds(4.5f);

            // respawn at the gate, souls-style: full bars, boss resets to his arena
            controller.enabled = false;
            transform.SetPositionAndRotation(spawnPos, spawnRot);
            controller.enabled = true;
            health = maxHealth;
            stamina = maxStamina;
            exhausted = false;
            flaskCharges = maxFlaskCharges;   // flasks refill on respawn, bonfire-style
            dead = false;
            CrossFade("Idle", 0.3f);
            EndAction();
            BossController.Instance?.ResetFight();
        }

        private void EndAction()
        {
            busy = false;
            locomotionState = "";   // force the next PlayLocomotion to crossfade
        }

        private void PlayLocomotion(string state)
        {
            if (locomotionState == state) return;
            locomotionState = state;
            CrossFade(state, 0.15f);
        }

        private void CrossFade(string state, float fade)
        {
            if (animator != null)
                animator.CrossFadeInFixedTime(state, fade, 0);
        }
    }
}
