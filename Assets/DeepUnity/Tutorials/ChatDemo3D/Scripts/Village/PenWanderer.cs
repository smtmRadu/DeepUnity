using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Aimless shuffling inside a fenced rectangle — the pig's whole life. Picks a point in the
    /// pen, ambles there, stands around (the gait's graze cycle plays), repeats. The 3D cousin
    /// of the 2D farm demo's CritterWander2D.
    /// </summary>
    public class PenWanderer : MonoBehaviour
    {
        [SerializeField] private Vector3 penCenter;
        [SerializeField] private Vector2 penHalfExtents = new Vector2(1.6f, 1.6f);
        [SerializeField] private float speed = 0.35f;
        [SerializeField] private float turnDegPerSec = 160f;
        [SerializeField] private QuadrupedGait gait;

        Vector3 target;
        float restUntil;

        /// <summary>Live ground speed, for animation drivers that aren't a QuadrupedGait.</summary>
        public float CurrentSpeed { get; private set; }

        void Start() => PickTarget();

        void Update()
        {
            if (Time.time < restUntil)
            {
                CurrentSpeed = 0f;
                if (gait != null) gait.Speed = 0f;
                return;
            }

            Vector3 to = target - transform.position;
            to.y = 0f;
            if (to.magnitude < 0.15f)
            {
                restUntil = Time.time + Random.Range(2f, 7f);
                PickTarget();
                return;
            }

            transform.rotation = Quaternion.RotateTowards(
                transform.rotation, Quaternion.LookRotation(to.normalized), turnDegPerSec * Time.deltaTime);
            // only advance roughly along the facing so the turn reads as a waddle, not a slide
            float facing = Mathf.Max(0f, Vector3.Dot(transform.forward, to.normalized));
            transform.position += transform.forward * (speed * facing * Time.deltaTime);
            CurrentSpeed = speed * facing;
            if (gait != null) gait.Speed = CurrentSpeed;
        }

        void PickTarget()
        {
            target = penCenter + new Vector3(
                Random.Range(-penHalfExtents.x, penHalfExtents.x), 0f,
                Random.Range(-penHalfExtents.y, penHalfExtents.y));
            target.y = transform.position.y;
        }
    }
}
