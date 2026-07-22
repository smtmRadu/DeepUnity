using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// The BASE face-sync component for talking NPCs — one general, multi-configurable script that
    /// owns how a character's face follows its voice, and (as the system grows) the umbrella the
    /// other animation layers coordinate through: audio-driven mouth today; full-face ARKit-52
    /// curves once the NVIDIA Audio2Face-3D ports land (regression v2.3 first, diffusion v3.0
    /// later); idle/expression/body layers plugging in around it.
    ///
    /// Backends (swappable in the inspector):
    ///  - AmplitudeSpectral — self-contained, model-free driver: reads the LIVE samples of the
    ///    AudioSource the TTS streams through on this GameObject (any engine — no phoneme timing,
    ///    no packages), derives a loudness envelope + a coarse 2-band vowel shape, and drives the
    ///    ARKit mouth/jaw blendshapes. Always available.
    ///  - A2F_Regression_v2 / A2F_Diffusion_v3 — the planned DeepUnity compute-shader ports of
    ///    Audio2Face-3D (fed from the TTS ring's UNPLAYED samples so inference latency hides
    ///    behind playback). Until a port is resident, selecting one warns once and falls back to
    ///    AmplitudeSpectral, so the character always talks.
    ///
    /// Runs in LateUpdate at a LATE execution order so it layers ON TOP of the idle systems
    /// (e.g. AnyaBehaviourIdle at order 10), blending the mouth channels the idle just wrote
    /// toward the spoken viseme by the current loudness envelope: silence leaves the idle mouth
    /// untouched, speech takes over. Blendshapes are resolved by ARKit name token, so any
    /// ARKit-52-style rig works.
    /// </summary>
    [DefaultExecutionOrder(100)]   // after the idle layers (behaviour idles run at order <= 10)
    public class FaceSync : MonoBehaviour
    {
        /// <summary>Which engine drives the mouth (see class doc).</summary>
        public enum Backend { AmplitudeSpectral, A2F_Regression_v2, A2F_Diffusion_v3 }

        [Tooltip("Mouth driver. AmplitudeSpectral = built-in loudness/spectral driver (no model). A2F_* = NVIDIA Audio2Face-3D DeepUnity ports (regression v2.3 first, diffusion v3.0 later) — fall back to AmplitudeSpectral with a one-time warning while unported.")]
        [SerializeField] Backend backend = Backend.AmplitudeSpectral;
        bool warnedBackendFallback;

        [Header("Viseme extents (ARKit weight 0..100 at full loudness)")]
        [SerializeField, Range(0f, 100f)] float jawMax = 40f;
        [SerializeField, Range(0f, 100f)] float funnelMax = 32f;
        [SerializeField, Range(0f, 100f)] float stretchMax = 30f;

        [Header("Loudness mapping")]
        [Tooltip("RMS at or below this reads as silence (mouth stays idle).")]
        [SerializeField] float silence = 0.004f;
        [Tooltip("RMS that maps to a fully open mouth.")]
        [SerializeField] float loud = 0.11f;
        [Tooltip("Per-frame smoothing toward a LOUDER target (fast = crisp syllable onsets; lower = softer, less machine-gun jaw).")]
        [SerializeField, Range(0.05f, 0.95f)] float attack = 0.22f;
        [Tooltip("Per-frame smoothing toward a QUIETER target (slow = lips don't snap shut mid-word).")]
        [SerializeField, Range(0.05f, 0.95f)] float release = 0.10f;

        SkinnedMeshRenderer smr;
        AudioSource src;
        Mesh mesh;
        int jawOpen, funnel, pucker, stretchL, stretchR;
        float env;                    // smoothed loudness, 0..1
        float smRound, smSpread;      // smoothed vowel shape — raw per-frame spectra flicker, which
                                      // made the mouth shape jitter robotically between frames

        const int NSAMP = 1024;
        const int NSPEC = 512;
        readonly float[] samp = new float[NSAMP];
        readonly float[] spec = new float[NSPEC];

        void Start()
        {
            smr = GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr == null || smr.sharedMesh == null) { enabled = false; return; }
            mesh = smr.sharedMesh;
            jawOpen = Idx("JawOpen");
            funnel = Idx("MouthFunnel");
            pucker = Idx("MouthPucker");
            stretchL = Idx("MouthStretchLeft");
            stretchR = Idx("MouthStretchRight");
            if (jawOpen < 0) enabled = false;   // no mouth to drive on this rig
        }

        void LateUpdate()
        {
            // the AudioSource is auto-added at runtime by the TTS voice component (RequireComponent) —
            // grab it lazily once it appears
            if (src == null) { src = GetComponent<AudioSource>(); if (src == null) return; }

            // A2F seam: once the Audio2Face-3D port lands, its per-frame full-face curves take over
            // here. Until the engine is resident, fall back to the amplitude driver below.
            if (backend != Backend.AmplitudeSpectral && !warnedBackendFallback)
            {
                warnedBackendFallback = true;
                Debug.LogWarning($"[FaceSync] {backend} selected but the Audio2Face-3D port isn't available yet — falling back to AmplitudeSpectral.");
            }

            // ---- loudness envelope from the live output ------------------------------------------
            float rms = 0f;
            bool playing = src.isPlaying;
            if (playing)
            {
                src.GetOutputData(samp, 0);
                for (int i = 0; i < NSAMP; i++) rms += samp[i] * samp[i];
                rms = Mathf.Sqrt(rms / NSAMP);
            }
            float target = Mathf.Clamp01(Mathf.InverseLerp(silence, loud, rms));
            env += (target - env) * (target > env ? attack : release);
            if (env <= 0.001f) return;   // fully silent: leave the idle mouth exactly as it was set

            // ---- vowel shape from two spectral bands of the playing audio ------------------------
            float low = 0f, high = 0f;
            if (playing)
            {
                src.GetSpectrumData(spec, 0, FFTWindow.BlackmanHarris);
                float binHz = (AudioSettings.outputSampleRate * 0.5f) / NSPEC;
                for (int i = 1; i < NSPEC; i++)
                {
                    float hz = i * binHz;
                    if (hz < 1200f) low += spec[i];
                    else if (hz < 5000f) high += spec[i];
                }
            }
            float tot = low + high + 1e-6f;
            float rounded = low / tot;    // 0..1 -> oo / oh
            float spread = high / tot;    // 0..1 -> ee / ss
            // smooth the vowel shape so the mouth doesn't jitter between spectral frames
            smRound += (rounded - smRound) * 0.18f;
            smSpread += (spread - smSpread) * 0.18f;

            // ---- viseme targets, blended over the idle mouth by env ------------------------------
            // quiet speech barely moves the jaw (env^1.4): full opening is reserved for loud peaks
            float openness = Mathf.Pow(Mathf.Clamp01(env), 1.4f);
            float jawT = jawMax * Mathf.Lerp(0.55f, 1f, smRound) * Mathf.Lerp(0.65f, 1f, openness);
            float funT = funnelMax * smRound;
            float strT = stretchMax * smSpread;

            Blend(jawOpen, jawT);
            Blend(funnel, funT);
            Blend(pucker, funT * 0.6f);
            Blend(stretchL, strT);
            Blend(stretchR, strT);
        }

        // set = lerp(idle-weight-this-frame, viseme target, env) so silence keeps the idle mouth and
        // full loudness reaches the viseme
        void Blend(int idx, float target)
        {
            if (idx < 0) return;
            float idle = smr.GetBlendShapeWeight(idx);
            smr.SetBlendShapeWeight(idx, Mathf.Clamp(Mathf.Lerp(idle, target, Mathf.Clamp01(env)), 0f, 100f));
        }

        int Idx(string token)
        {
            for (int i = 0; i < mesh.blendShapeCount; i++)
                if (mesh.GetBlendShapeName(i).Contains(token)) return i;
            return -1;
        }
    }
}
