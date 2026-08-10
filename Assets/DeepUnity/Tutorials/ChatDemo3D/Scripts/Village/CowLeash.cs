using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// The rope from the peddler's hand to the cow trailing behind him: a LineRenderer strung
    /// between the two anchors every frame with a simple gravity sag, so it stays attached
    /// through the walk cycle, the stop-and-turn on E, and the dialogue itself.
    /// </summary>
    [ExecuteAlways]   // the rope reads in edit mode too (scene view, builder screenshots)
    public class CowLeash : MonoBehaviour
    {
        [SerializeField] private Transform holder;      // hand bone (or a shoulder-height socket)
        [SerializeField] private Transform cowAnchor;   // the cow's neck
        [SerializeField] private LineRenderer line;
        [SerializeField] private float sag = 0.22f;

        const int POINTS = 9;

        void LateUpdate()
        {
            if (holder == null || cowAnchor == null || line == null) return;
            Vector3 a = holder.position, b = cowAnchor.position;
            // shorter rope hangs less; a taut one (pair pulling ahead) flattens out
            float slack = Mathf.Clamp01(2.6f - Vector3.Distance(a, b) * 0.55f) * sag;
            for (int i = 0; i < POINTS; i++)
            {
                float t = i / (POINTS - 1f);
                Vector3 p = Vector3.Lerp(a, b, t);
                p.y -= 4f * slack * t * (1f - t);   // parabolic dip, zero at both ends
                line.SetPosition(i, p);
            }
        }
    }
}
