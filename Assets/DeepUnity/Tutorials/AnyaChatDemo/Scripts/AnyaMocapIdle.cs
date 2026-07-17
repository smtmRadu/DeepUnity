using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Play-mode idle driven by REAL captured human facial motion (see <see cref="AnyaMocapTrack"/>).
    /// Assign the .bytes track extracted from video; she replays that person's actual blinks, gaze,
    /// micro-expressions and head motion, looped. No LLM, no TTS, no procedural synthesis.
    /// </summary>
    public class AnyaMocapIdle : MonoBehaviour
    {
        [SerializeField] TextAsset track;
        [SerializeField, Range(0f, 1.5f)] float weightScale = 1f;
        [SerializeField, Range(0f, 1.5f)] float headScale = 0.6f;
        [SerializeField, Range(0f, 0.95f)] float smooth = 0.35f;

        readonly AnyaMocapTrack mocap = new AnyaMocapTrack();
        float t0;

        void Start()
        {
            var smr = GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr == null || track == null) { enabled = false; return; }
            mocap.Init(smr, track.bytes);
            if (!mocap.Ready) { enabled = false; return; }
            t0 = Time.time;
        }

        void LateUpdate()
        {
            mocap.WeightScale = weightScale;
            mocap.HeadScale = headScale;
            mocap.Smooth = smooth;
            mocap.Evaluate(Time.time - t0);
        }
    }
}
