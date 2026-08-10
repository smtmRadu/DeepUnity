using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Procedural walk cycle for the primitive-built farm animals (there are no animal rigs in
    /// the CC0 art set, so the cow and pig are cubes — this is their whole skeleton). Diagonal
    /// leg pairs swing in opposite phase around their hip pivots, the head bobs with the steps,
    /// the tail sways; at rest the head periodically drops into a graze. Whoever moves the body
    /// (stroll group, pen wanderer) writes <see cref="Speed"/>; everything else is local.
    /// </summary>
    public class QuadrupedGait : MonoBehaviour
    {
        [Header("Pivots (wired by the scene builder)")]
        [SerializeField] private Transform legFL;
        [SerializeField] private Transform legFR;
        [SerializeField] private Transform legRL;
        [SerializeField] private Transform legRR;
        [SerializeField] private Transform head;
        [SerializeField] private Transform tail;

        [SerializeField] private float strideLength = 0.55f;   // meters per full cycle — sets cadence
        [SerializeField] private float swingDegrees = 24f;
        [SerializeField] private float grazeDegrees = 38f;

        /// <summary>Current ground speed in m/s. 0 = standing (graze cycle takes over).</summary>
        public float Speed { get; set; }

        float phase;          // walk cycle phase, radians
        float swing;          // smoothed swing amplitude so stops settle instead of freezing mid-air
        float graze;          // 0 = head up, 1 = head down in the grass
        float grazeFlipAt;
        bool grazing;

        void Update()
        {
            float dt = Time.deltaTime;
            phase += (Speed / Mathf.Max(0.1f, strideLength)) * 2f * Mathf.PI * dt;
            swing = Mathf.MoveTowards(swing, Speed > 0.05f ? 1f : 0f, dt * 3f);

            float s = Mathf.Sin(phase) * swingDegrees * swing;
            if (legFL != null) legFL.localRotation = Quaternion.Euler(+s, 0f, 0f);
            if (legRR != null) legRR.localRotation = Quaternion.Euler(+s, 0f, 0f);
            if (legFR != null) legFR.localRotation = Quaternion.Euler(-s, 0f, 0f);
            if (legRL != null) legRL.localRotation = Quaternion.Euler(-s, 0f, 0f);

            if (tail != null)
                tail.localRotation = Quaternion.Euler(0f, Mathf.Sin(Time.time * 1.7f) * 14f, 0f);

            if (head != null)
            {
                if (swing < 0.05f)
                {
                    // standing: ease between upright idling and long grazes
                    if (Time.time > grazeFlipAt)
                    {
                        grazing = !grazing;
                        grazeFlipAt = Time.time + (grazing ? Random.Range(2.5f, 5f) : Random.Range(2f, 6f));
                    }
                    graze = Mathf.MoveTowards(graze, grazing ? 1f : 0f, dt * 1.2f);
                }
                else
                {
                    graze = Mathf.MoveTowards(graze, 0f, dt * 2f);
                    grazing = false;
                    grazeFlipAt = Time.time + Random.Range(2f, 5f);
                }
                float bob = Mathf.Sin(phase * 2f) * 3.5f * swing;
                head.localRotation = Quaternion.Euler(bob + graze * grazeDegrees, 0f, 0f);
            }
        }
    }
}
