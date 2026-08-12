using System.Collections;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Corvus' ash staff in the PLAYER's hands — the demo's second purchasable, and the ranged
    /// answer to Velmire's sword: a cast lobs a glintstone-blue spark bolt from the staff's
    /// crystal (Elden Ring pebble energy — user 2026-08-12). LEFT CLICK casts while the staff is
    /// the only weapon owned; Q casts always, so buying Velmire's sword later does not brick the
    /// sorcery. The staff shares the rig's single Weapon.R mount with the sword, so the moment
    /// the sword is owned the staff stows out of sight (the cast then leaves from the free
    /// shoulder) — one mount, no clipping, nothing to hand-tune. Bolt, trail, glow and burst are
    /// all built in code: no prefabs, no textures, same as the rest of the demo.
    /// </summary>
    public class GlintstoneStaff : MonoBehaviour
    {
        [SerializeField, Tooltip("The held staff on the weapon mount — bolts leave from its crystal (local +0.72 on the prop); hidden while the sword owns the mount.")]
        private Transform staff;
        [SerializeField, Tooltip("Ownership — casting only works once PlayerGear.GrantStaff() ran.")]
        private PlayerGear gear;
        [SerializeField, Tooltip("Damage per bolt against the Sentinel (a sword swing lands ~34).")]
        private float magicDamage = 26f;
        [SerializeField] private float boltSpeed = 16f;
        [SerializeField] private float castCooldown = 0.9f;

        Animator anim;
        float nextCast;

        void Awake()
        {
            anim = GetComponentInChildren<Animator>();
            if (gear == null) gear = GetComponent<PlayerGear>();
            if (gear != null) gear.GearChanged += Reseat;
        }

        void Start() => Reseat();
        void OnDestroy() { if (gear != null) gear.GearChanged -= Reseat; }

        // One weapon mount on the chibi rig: the sword evicts the staff visual. Runs after every
        // ownership change (PlayerGear activates the object on grant; this immediately applies
        // the mount rule on top).
        void Reseat()
        {
            if (staff == null || gear == null) return;
            staff.gameObject.SetActive(gear.HasStaff && !gear.HasSword);
        }

        void Update()
        {
            if (gear == null || !gear.HasStaff || Time.time < nextCast) return;
            // gameplay only — an unlocked cursor means a dialogue/menu is open, and a 'q' typed
            // into the chat input must not fire sorcery at the NPC
            if (Cursor.lockState != CursorLockMode.Locked) return;
            bool primary = Input.GetMouseButtonDown(0) && !gear.HasSword;   // staff-only loadout: left click casts
            bool always = Input.GetKeyDown(KeyCode.Q);                      // works alongside the sword
            if (!primary && !always) return;
            nextCast = Time.time + castCooldown;
            if (anim != null) anim.CrossFadeInFixedTime("Attack1", 0.06f);  // the swing doubles as the cast gesture
            StartCoroutine(Cast());
        }

        IEnumerator Cast()
        {
            yield return new WaitForSeconds(0.16f);   // release on the gesture's forward beat
            Vector3 dir = (transform.forward + Vector3.up * 0.02f).normalized;
            Vector3 from = staff != null && staff.gameObject.activeInHierarchy
                ? staff.TransformPoint(new Vector3(0f, 0.72f, 0f))          // the crystal (prop is built along +Y)
                : transform.position + Vector3.up * 1.35f - transform.right * 0.15f + transform.forward * 0.3f;
            SpawnBolt(from, dir);
        }

        void SpawnBolt(Vector3 from, Vector3 dir)
        {
            var bolt = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            bolt.name = "GlintBolt";
            Destroy(bolt.GetComponent<Collider>());   // flight is raycast-based; a collider would hit the caster
            bolt.transform.position = from;
            bolt.transform.localScale = Vector3.one * 0.16f;
            bolt.GetComponent<MeshRenderer>().sharedMaterial = GlintMat();
            var glow = new GameObject("Glow").AddComponent<Light>();
            glow.transform.SetParent(bolt.transform, false);
            glow.type = LightType.Point;
            glow.color = new Color(0.55f, 0.75f, 1f);
            glow.intensity = 2.6f;
            glow.range = 6f;
            glow.shadows = LightShadows.None;
            AddSparkles(bolt, trail: true);
            bolt.AddComponent<GlintBolt>().Init(dir * boltSpeed, magicDamage);
        }

        static Material glintMat;
        static Material GlintMat()
        {
            if (glintMat != null) return glintMat;
            glintMat = new Material(Shader.Find("Standard"));
            glintMat.color = new Color(0.70f, 0.86f, 1f);
            glintMat.EnableKeyword("_EMISSION");
            glintMat.SetColor("_EmissionColor", new Color(0.45f, 0.80f, 1.70f));
            return glintMat;
        }

        /// <summary>Cyan sparkle particles, code-built: a per-distance wake on flying bolts, a
        /// one-shot burst on impacts. Shared by the bolt and the burst so they read as one magic.</summary>
        internal static void AddSparkles(GameObject on, bool trail)
        {
            var ps = on.AddComponent<ParticleSystem>();
            var main = ps.main;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.35f, 0.6f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(0.05f, trail ? 0.4f : 2.2f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.03f, 0.09f);
            main.startColor = new ParticleSystem.MinMaxGradient(new Color(0.65f, 0.85f, 1f), Color.white);
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            var emission = ps.emission;
            emission.rateOverTime = 0f;
            if (trail) emission.rateOverDistance = 45f;
            else emission.SetBursts(new[] { new ParticleSystem.Burst(0f, 60) });
            var shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Sphere;
            shape.radius = 0.06f;
            var psr = on.GetComponent<ParticleSystemRenderer>();
            psr.sharedMaterial = SparkleMat();
        }

        static Material sparkleMat;
        static Material SparkleMat()
        {
            if (sparkleMat != null) return sparkleMat;
            var sh = Shader.Find("Legacy Shaders/Particles/Additive");
            if (sh == null) sh = Shader.Find("Particles/Standard Unlit");
            sparkleMat = new Material(sh);
            return sparkleMat;
        }
    }

    /// <summary>One flying glintstone bolt: straight line, sparkle wake, burst on whatever it
    /// meets. The Sentinel takes the damage; walls and floor just take the light show. Player and
    /// NPC models live on layer 2 (Ignore Raycast), so the caster never blocks his own bolt.</summary>
    public class GlintBolt : MonoBehaviour
    {
        Vector3 velocity;
        float damage;
        float dieAt;

        public void Init(Vector3 v, float dmg) { velocity = v; damage = dmg; dieAt = Time.time + 4f; }

        void Update()
        {
            float step = velocity.magnitude * Time.deltaTime;
            if (Physics.Raycast(transform.position, velocity.normalized, out RaycastHit hit, step + 0.15f))
            { Burst(hit.point); return; }
            transform.position += velocity * Time.deltaTime;

            var boss = BossController.Instance;
            if (boss != null)
            {
                // same fairness rule as the sword: measured to his body, not his root
                Vector3 center = boss.transform.position + Vector3.up * 1.2f;
                if (Vector3.Distance(transform.position, center) < 1.35f)
                { boss.TakeDamage(damage); Burst(transform.position); return; }
            }
            if (Time.time >= dieAt) Burst(transform.position);
        }

        void Burst(Vector3 at)
        {
            var b = new GameObject("GlintBurst");
            b.transform.position = at;
            GlintstoneStaff.AddSparkles(b, trail: false);
            var l = b.AddComponent<Light>();
            l.type = LightType.Point;
            l.color = new Color(0.55f, 0.75f, 1f);
            l.intensity = 4f;
            l.range = 7f;
            l.shadows = LightShadows.None;
            b.AddComponent<GlintBurstFade>();
            Destroy(gameObject);
        }
    }

    /// <summary>Burst afterglow: the light fades over ~0.5 s, the object dies at 1.2 s — after
    /// the last spark has burned out.</summary>
    public class GlintBurstFade : MonoBehaviour
    {
        float t;
        void Update()
        {
            t += Time.deltaTime;
            var l = GetComponent<Light>();
            if (l != null) l.intensity = Mathf.Max(0f, 4f * (1f - t / 0.5f));
            if (t > 1.2f) Destroy(gameObject);
        }
    }
}
