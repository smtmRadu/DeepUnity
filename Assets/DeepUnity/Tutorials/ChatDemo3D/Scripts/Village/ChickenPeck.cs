using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Chicken body language on top of a PenWanderer: quick double-pecks at the ground while
    /// standing, a waddle roll while moving. The whole bird is three boxes — this script is
    /// its entire skeleton, like QuadrupedGait is for the cow and pig.
    /// </summary>
    public class ChickenPeck : MonoBehaviour
    {
        [SerializeField] private Transform head;      // pivot at the neck base
        [SerializeField] private Transform body;      // pivot for the waddle roll
        [SerializeField] private PenWanderer wanderer;

        float peckAt;
        int pecksLeft;
        float peckPhase;

        void Start() => peckAt = Time.time + Random.Range(1f, 4f);

        void Update()
        {
            float speed = wanderer != null ? wanderer.CurrentSpeed : 0f;

            if (body != null)
            {
                float roll = speed > 0.02f ? Mathf.Sin(Time.time * 9f) * 5f : 0f;
                body.localRotation = Quaternion.Euler(0f, 0f, roll);
            }

            if (head == null) return;
            if (speed > 0.02f)
            {
                // head bobs forward-back with the steps, no pecking on the move
                head.localRotation = Quaternion.Euler(Mathf.Sin(Time.time * 9f) * 9f, 0f, 0f);
                pecksLeft = 0;
                peckAt = Time.time + Random.Range(1.5f, 4f);
                return;
            }

            if (pecksLeft == 0 && Time.time > peckAt)
            {
                pecksLeft = Random.Range(2, 4);
                peckPhase = 0f;
            }
            if (pecksLeft > 0)
            {
                peckPhase += Time.deltaTime * 7f;          // ~0.45 s per peck
                float t = Mathf.PingPong(peckPhase, 1f);
                head.localRotation = Quaternion.Euler(t * 55f, 0f, 0f);
                if (peckPhase >= 2f)
                {
                    peckPhase = 0f;
                    if (--pecksLeft == 0) peckAt = Time.time + Random.Range(2f, 6f);
                }
            }
            else
            {
                head.localRotation = Quaternion.Slerp(head.localRotation, Quaternion.identity, Time.deltaTime * 6f);
            }
        }
    }
}
