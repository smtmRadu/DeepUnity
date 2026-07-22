using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    // ============================== TEXT-CONDITIONED SPEECH EXPRESSIONS ==========================
    // While Anya talks, her face reacts to WHAT she is saying. The pipeline:
    //
    //   PocketTTSVoice.OnClauseSpoken(text, duration)      (audio for that text just started)
    //        -> AnyaBehaviourIdle.OnClauseSpoken            (the manager's handler)
    //        -> AnyaSpeechCueDetector.Detect(...)           (tiny lexical scan, no ML)
    //        -> List<AnyaSpeechCue> {kind, atTime, strength} appended to
    //        -> AnyaTalkingExpressionsBehaviour             (a normal AnyaBehaviour unit)
    //           which renders each cue as a timed envelope (ramp ~0.25 s, hold, decay ~0.6 s),
    //           scaled by the speak blend and the inspector's talkingExpressiveness.
    //
    // A trigger word's moment is approximated as (charOffset / totalChars) * duration from the
    // clause start — crude but reads right at sentence granularity. Max 2 cues render at once.
    //
    // ---- HOW TO ADD A NEW CUE KIND (drop-in) ----------------------------------------------------
    //  1. Add a value to AnyaCueKind (e.g. Blush).
    //  2. DETECTOR RULE: in AnyaSpeechCueDetector, add a word set (or punctuation rule) and emit
    //     `new AnyaSpeechCue { kind = AnyaCueKind.Blush, atTime = ..., strength = ... }` in
    //     ScanSentence — atTime = startT + (charOffset / (float)total) * dur.
    //  3. ENVELOPE RENDERER: in AnyaTalkingExpressionsBehaviour add a case to Render() (blendshape
    //     weights via rig.Add, head degrees via rig.Pitch/Yaw/Roll — head motion is automatically
    //     rounded by the head spring) and a hold time in HoldFor().
    //  Cues from other sources (not TTS text) can also be queued directly:
    //     idle.TalkingExpressions.Cues.Add(new AnyaSpeechCue { ... });
    // =============================================================================================

    public enum AnyaCueKind
    {
        Smile,      // positive/warm word -> shallow Duchenne bloom
        Question,   // '?' / interrogative -> brow raise + tiny curious tilt
        Emphasis,   // '!' / intensity word -> eye widen + brow flash + small nod
        Sympathy,   // negative/soft-sad word -> subtle frown + inner-brow raise
        Engaged     // no cue found -> light "engaged speaking" baseline (never dead-faced)
    }

    public struct AnyaSpeechCue
    {
        public AnyaCueKind kind;
        public float atTime;     // idle-clock seconds when the envelope starts ramping
        public float strength;   // 0..1 per-cue intensity
    }

    /// <summary>
    /// Tiny deterministic lexical cue detector. Sentence-splits the spoken chunk (a chunk may hold
    /// up to ~3 sentences), scans words against small cue lexicons, uses the end punctuation of
    /// each sentence, and guarantees an Engaged baseline cue for sentences with no hits.
    /// </summary>
    public static class AnyaSpeechCueDetector
    {
        static readonly string[] Positive = {
            "love", "loved", "great", "wonderful", "happy", "happiness", "glad", "fun", "funny",
            "haha", "hehe", "thanks", "thank", "awesome", "nice", "beautiful", "cool", "enjoy",
            "enjoyed", "yay", "sweet", "lovely", "laugh", "laughed", "smile", "excited",
            "exciting", "favorite", "favourite", "perfect", "delighted", "adorable", "cute"
        };
        static readonly string[] Interrog = { "what", "why", "how", "when", "where", "who", "which" };
        static readonly string[] Emph = {
            "wow", "amazing", "incredible", "absolutely", "totally", "really", "never", "huge",
            "best", "so", "definitely", "unbelievable"
        };
        static readonly string[] Negative = {
            "sorry", "sad", "afraid", "bad", "miss", "missed", "unfortunately", "hard", "tough",
            "hurt", "lonely", "worry", "worried", "scared", "cry", "terrible", "awful", "alone",
            "difficult", "wish"
        };

        const float MinSpacing = 0.7f;   // s between scheduled cues (any kind)
        const int MaxPerClause = 4;      // queue cap per spoken chunk

        /// <summary>Scan <paramref name="text"/> (audio started at idle-time <paramref name="startT"/>,
        /// lasts <paramref name="duration"/> s) and append timed cues to <paramref name="into"/>.</summary>
        public static void Detect(string text, float startT, float duration, List<AnyaSpeechCue> into)
        {
            if (string.IsNullOrEmpty(text) || duration <= 0.05f || into == null) return;
            int n = text.Length;
            float lastAt = float.NegativeInfinity;
            int added = 0;
            int sentStart = 0;
            for (int i = 0; i <= n; i++)
            {
                bool end = i == n || text[i] == '.' || text[i] == '!' || text[i] == '?';
                if (!end) continue;
                if (i > sentStart)
                {
                    char punct = i < n ? text[i] : '.';
                    ScanSentence(text, sentStart, i, punct, startT, duration, n, into, ref lastAt, ref added);
                }
                while (i < n - 1 && (text[i + 1] == '.' || text[i + 1] == '!' || text[i + 1] == '?')) i++;   // "?!", "..."
                sentStart = i + 1;
            }
        }

        static void ScanSentence(string text, int s, int e, char punct, float startT, float dur,
                                 int total, List<AnyaSpeechCue> into, ref float lastAt, ref int added)
        {
            bool any = false, hasQ = false, hasEmph = false;

            int i = s;
            while (i < e && added < MaxPerClause)
            {
                if (!char.IsLetter(text[i])) { i++; continue; }
                int w0 = i;
                while (i < e && char.IsLetter(text[i])) i++;
                string word = text.Substring(w0, i - w0).ToLowerInvariant();

                AnyaCueKind kind; float str;
                if (Has(Positive, word)) { kind = AnyaCueKind.Smile; str = 0.75f + 0.25f * H(text, w0); }
                else if (Has(Negative, word)) { kind = AnyaCueKind.Sympathy; str = 0.7f + 0.3f * H(text, w0); }
                else if (punct == '?' && Has(Interrog, word)) { kind = AnyaCueKind.Question; str = 0.9f; hasQ = true; }
                else if (Has(Emph, word))
                {   // "so" is mostly a conjunction — keep it barely-there
                    kind = AnyaCueKind.Emphasis; str = word.Length <= 2 ? 0.4f : 0.7f + 0.3f * H(text, w0);
                    hasEmph = true;
                }
                else continue;

                float at = startT + (w0 / (float)total) * dur;
                any = true;                              // sentence is not cue-less, even if we skip below
                if (at - lastAt < MinSpacing) continue;  // too close to the previous cue
                into.Add(new AnyaSpeechCue { kind = kind, atTime = at, strength = str });
                lastAt = at; added++;
            }

            // sentence-final punctuation, when the words alone didn't already say it
            if (added < MaxPerClause && punct == '?' && !hasQ)
                any |= TryAdd(into, AnyaCueKind.Question, At(s, e, 0.35f, startT, dur, total), 0.85f, ref lastAt, ref added);
            if (added < MaxPerClause && punct == '!' && !hasEmph)
                any |= TryAdd(into, AnyaCueKind.Emphasis, At(s, e, 0.8f, startT, dur, total), 0.8f, ref lastAt, ref added);

            // no cue at all -> light engaged baseline so she is never dead-faced mid-sentence
            if (!any && added < MaxPerClause)
                TryAdd(into, AnyaCueKind.Engaged, At(s, e, 0.35f, startT, dur, total),
                       0.5f + 0.5f * H(text, s), ref lastAt, ref added);
        }

        static float At(int s, int e, float frac, float startT, float dur, int total)
            => startT + (Mathf.Lerp(s, e, frac) / total) * dur;

        static bool TryAdd(List<AnyaSpeechCue> into, AnyaCueKind kind, float at, float str,
                           ref float lastAt, ref int added)
        {
            if (at - lastAt < MinSpacing) return false;
            into.Add(new AnyaSpeechCue { kind = kind, atTime = at, strength = str });
            lastAt = at; added++;
            return true;
        }

        static bool Has(string[] set, string word) => System.Array.IndexOf(set, word) >= 0;

        // deterministic per-text-position variation
        static float H(string text, int idx)
            => AnyaFaceRig.Hash01(text.Length * 131 + idx * 17 + text[Mathf.Clamp(idx, 0, text.Length - 1)]);
    }

    /// <summary>
    /// Renders the queued <see cref="AnyaSpeechCue"/>s as timed facial envelopes while she talks.
    /// A regular <see cref="AnyaBehaviour"/> unit — registered before the idle smile so it can
    /// suppress it (via <see cref="AnyaFaceRig.TalkSmile"/>) and avoid double-smile stacking.
    /// Everything scales with the speak blend (silence -> nothing) and the global
    /// <see cref="Expressiveness"/> dial. Head contributions go through the head spring, so cue
    /// nods/tilts are automatically smooth. Only touches channels lip-sync does NOT own
    /// (smile/cheek/squint/brow/wide/frown + head) — the mouth stays AnyaLipSync's.
    /// </summary>
    public class AnyaTalkingExpressionsBehaviour : AnyaBehaviour
    {
        /// <summary>Pending + active cues (appended by the manager's OnClauseSpoken handler).</summary>
        public readonly List<AnyaSpeechCue> Cues = new List<AnyaSpeechCue>();
        /// <summary>Global 0..1 multiplier (inspector: talkingExpressiveness).</summary>
        public float Expressiveness = 0.8f;

        const int MaxActive = 2;            // cap simultaneous cues so combinations stay natural
        const float Ramp = 0.25f, Decay = 0.6f;

        int smileL, smileR, cheekL, cheekR, squintL, squintR;
        int browInner, browOutL, browOutR, wideL, wideR, frownL, frownR;

        public override void Init(AnyaFaceRig rig)
        {
            smileL = rig.Shape("MouthSmileLeft"); smileR = rig.Shape("MouthSmileRight");
            cheekL = rig.Shape("CheekSquintLeft"); cheekR = rig.Shape("CheekSquintRight");
            squintL = rig.Shape("EyeSquintLeft"); squintR = rig.Shape("EyeSquintRight");
            browInner = rig.Shape("BrowInnerUp");
            browOutL = rig.Shape("BrowOuterUpLeft"); browOutR = rig.Shape("BrowOuterUpRight");
            wideL = rig.Shape("EyeWideLeft"); wideR = rig.Shape("EyeWideRight");
            frownL = rig.Shape("MouthFrownLeft"); frownR = rig.Shape("MouthFrownRight");
        }

        static float HoldFor(AnyaCueKind k)
        {
            switch (k)
            {
                case AnyaCueKind.Smile: return 1.3f;
                case AnyaCueKind.Question: return 0.9f;
                case AnyaCueKind.Emphasis: return 0.35f;
                case AnyaCueKind.Sympathy: return 1.2f;
                default: return 0.8f;   // Engaged
            }
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            // prune finished cues (keeps the queue bounded even if speech is interrupted)
            for (int i = Cues.Count - 1; i >= 0; i--)
                if (f.t > Cues[i].atTime + Ramp + HoldFor(Cues[i].kind) + Decay)
                    Cues.RemoveAt(i);

            float gate = f.speak * Mathf.Clamp01(Expressiveness);
            if (Cues.Count == 0 || gate <= 0.001f) return;

            int active = 0;
            for (int i = 0; i < Cues.Count && active < MaxActive; i++)
            {
                var c = Cues[i];
                float ts = f.t - c.atTime;
                if (ts < 0f) continue;   // scheduled later in the clause
                float hold = HoldFor(c.kind);
                float e;
                if (ts < Ramp) e = AnyaFaceRig.Smooth01(ts / Ramp);
                else if (ts < Ramp + hold) e = 1f;
                else e = 1f - AnyaFaceRig.Smooth01((ts - Ramp - hold) / Decay);
                if (e <= 0f) continue;
                active++;
                Render(rig, c.kind, e * c.strength * gate, (int)(c.atTime * 97.13f));
            }
        }

        // a = final 0..1 amplitude for this cue this frame; amplitudes deliberately SHALLOW
        void Render(AnyaFaceRig rig, AnyaCueKind kind, float a, int seed)
        {
            switch (kind)
            {
                case AnyaCueKind.Smile:
                    float asym = 0.9f + 0.1f * AnyaFaceRig.Hash01(seed);
                    rig.Add(smileL, a * 36f); rig.Add(smileR, a * 36f * asym);
                    rig.Add(cheekL, a * 20f); rig.Add(cheekR, a * 20f * asym);
                    rig.Add(squintL, a * 9f); rig.Add(squintR, a * 9f);
                    rig.TalkSmile = Mathf.Max(rig.TalkSmile, Mathf.Clamp01(a));   // suppress idle Duchenne
                    break;
                case AnyaCueKind.Question:
                    rig.Add(browInner, a * 24f);
                    rig.Add(browOutL, a * 19f); rig.Add(browOutR, a * 19f);
                    rig.Add(wideL, a * 6f); rig.Add(wideR, a * 6f);
                    rig.Roll += (AnyaFaceRig.Hash01(seed + 1) < 0.5f ? 1f : -1f) * a * 1.3f;   // tiny curious tilt
                    break;
                case AnyaCueKind.Emphasis:
                    rig.Add(wideL, a * 15f); rig.Add(wideR, a * 15f);
                    rig.Add(browInner, a * 15f);
                    rig.Add(browOutL, a * 12f); rig.Add(browOutR, a * 12f);
                    rig.Pitch += a * 2.0f;   // small emphasis nod (chin dip), spring-smoothed
                    break;
                case AnyaCueKind.Sympathy:
                    rig.Add(frownL, a * 13f); rig.Add(frownR, a * 13f);
                    rig.Add(browInner, a * 22f);
                    rig.Add(squintL, a * 6f); rig.Add(squintR, a * 6f);
                    rig.Pitch += a * 0.8f;   // barely-there head drop
                    break;
                case AnyaCueKind.Engaged:
                    rig.Add(browInner, a * 7f);
                    rig.Add(smileL, a * 8f); rig.Add(smileR, a * 7f);
                    rig.Add(cheekL, a * 4f); rig.Add(cheekR, a * 4f);
                    break;
            }
        }
    }
}
