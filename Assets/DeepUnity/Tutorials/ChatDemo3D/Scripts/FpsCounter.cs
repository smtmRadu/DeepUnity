using TMPro;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// FPS readout, top-right. The frame time is smoothed (EMA) so the number reads stable,
    /// and the label refreshes a few times per second — a per-frame TMP re-layout is itself
    /// a small cost the counter shouldn't add.
    /// </summary>
    public class FpsCounter : MonoBehaviour
    {
        [SerializeField] private TMP_Text label;

        private float emaDt = -1f;
        private float nextRefresh;

        private void Update()
        {
            float dt = Time.unscaledDeltaTime;
            emaDt = emaDt < 0f ? dt : Mathf.Lerp(emaDt, dt, 0.08f);

            if (label == null || Time.unscaledTime < nextRefresh) return;
            nextRefresh = Time.unscaledTime + 0.25f;
            label.text = Mathf.RoundToInt(1f / Mathf.Max(emaDt, 1e-5f)) + " FPS";
        }
    }
}
