using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Two-track fps diagnostic for the demo scene. Opt-in: does nothing unless `record` is ticked.
    ///
    /// Track 1 — ProbeLogs/frame_spikes.csv: every frame longer than spikeMs, tagged with what the
    /// LLM was doing (LLM.CurrentPhase), whether a GC collection landed on it, the TTS pipeline
    /// stage that ran THAT frame (PocketTTS.LastHeavyTick, taken only when its frame stamp says it
    /// is fresh — the 2026-08-02 hunt chased a "flush_push storm" that was mostly a stale tag), and
    /// the voice pump's own account of the frame (state / heavy ticks issued vs allowed / ring
    /// seconds, from PocketTTSVoice.Pump*).
    ///
    /// Track 2 — ProbeLogs/fps_timeline.csv: one row per second for the WHOLE session, spikes or
    /// not: fps, mean/max frame ms, spike + GC counts, dominant pump state and ring level. This is
    /// the "what was the game doing at minute 2:41" view the spike list can't give.
    /// </summary>
    public class FrameSpikeProbe : MonoBehaviour
    {
        [Tooltip("Diagnostic only — flip on when hunting fps dips. Off = no console reports, no CSV, near-zero overhead.")]
        public bool record = false;   // public so the scene builder can arm it for a hunt
        [SerializeField] float spikeMs = 18f;

        readonly List<string> rows = new List<string>(1024);
        int lastGcCount;
        int spikeCount;
        float worstMs;
        string worstPhase = "";
        float nextReportAt;

        // ---- per-second timeline accumulators ----
        readonly List<string> timeline = new List<string>(1024);
        float bucketStart = -1f;
        int bFrames, bSpikes, bGc;
        float bSumMs, bMaxMs;
        string bPump = "";          // last pump state seen in the bucket ("" = pump never ran)
        float bRing;
        string bLlm = "";

        private void Update()
        {
            if (!record) return;
            float ms = Time.unscaledDeltaTime * 1000f;
            int gc = System.GC.CollectionCount(0);
            bool gcThisFrame = gc != lastGcCount;
            lastGcCount = gc;

            string phase = LLM.CurrentPhase;

            // freshness: a tag/pump snapshot is evidence about this frame only if it was written
            // this frame or the previous one (Update-order skew) — anything older is history.
            int now = Time.frameCount;
            int tickAge = now - PocketTTSModeling.PocketTTS.LastHeavyTickFrame;
            string ttsTick = (tickAge >= 0 && tickAge <= 1)
                ? (PocketTTSModeling.PocketTTS.LastHeavyTick ?? "") : "";
            bool pumpFresh = now - PocketTTSModeling.PocketTTSVoice.PumpFrame <= 1;
            string pump = pumpFresh ? PocketTTSModeling.PocketTTSVoice.PumpState : "";
            int pumpTicks = pumpFresh ? PocketTTSModeling.PocketTTSVoice.PumpTicks : 0;
            int pumpCap = pumpFresh ? PocketTTSModeling.PocketTTSVoice.PumpTickCap : 0;
            float ringS = pumpFresh ? PocketTTSModeling.PocketTTSVoice.PumpRingSeconds : 0f;

            // ---- timeline bucket ----
            float t = Time.unscaledTime;
            if (bucketStart < 0f) bucketStart = Mathf.Floor(t);
            if (t - bucketStart >= 1f)
            {
                CloseBucket();
                bucketStart = Mathf.Floor(t);
            }
            bFrames++;
            bSumMs += ms;
            if (ms > bMaxMs) bMaxMs = ms;
            if (gcThisFrame) bGc++;
            if (pump != "") { bPump = pump; bRing = ringS; }
            bLlm = phase;

            if (ms < spikeMs) return;

            string phaseGc = phase + (gcThisFrame ? "+GC" : "");
            spikeCount++;
            bSpikes++;
            rows.Add($"{t:0.00},{System.DateTime.Now:HH:mm:ss.ff},{ms:0.0},{phaseGc},{ttsTick}," +
                     $"{pump},{pumpTicks},{pumpCap},{ringS:0.00}");
            if (ms > worstMs) { worstMs = ms; worstPhase = $"{phaseGc}/{(ttsTick != "" ? ttsTick : pump)}"; }

            if (Time.unscaledTime >= nextReportAt)
            {
                nextReportAt = Time.unscaledTime + 10f;
                Debug.Log($"[FrameSpikeProbe] {spikeCount} frames over {spikeMs} ms — worst {worstMs:0.0} ms during '{worstPhase}'");
            }
        }

        void CloseBucket()
        {
            if (bFrames == 0) return;
            timeline.Add($"{bucketStart:0},{System.DateTime.Now:HH:mm:ss},{bFrames}," +
                         $"{bSumMs / bFrames:0.0},{bMaxMs:0.0},{bSpikes},{bGc},{bLlm},{bPump},{bRing:0.00}");
            bFrames = bSpikes = bGc = 0;
            bSumMs = bMaxMs = 0f;
            bPump = ""; bRing = 0f;
        }

        private void OnDestroy()
        {
            CloseBucket();
            if (rows.Count == 0 && timeline.Count == 0) return;
            try
            {
                Directory.CreateDirectory("ProbeLogs");
                var all = new List<string>(rows.Count + 1)
                    { "time_s,clock,frame_ms,llm_phase,tts_tick,pump,ticks,cap,ring_s" };
                all.AddRange(rows);
                File.WriteAllLines(Path.Combine("ProbeLogs", "frame_spikes.csv"), all);
                var tl = new List<string>(timeline.Count + 1)
                    { "time_s,clock,fps,avg_ms,max_ms,spikes,gc,llm_phase,pump,ring_s" };
                tl.AddRange(timeline);
                File.WriteAllLines(Path.Combine("ProbeLogs", "fps_timeline.csv"), tl);
                Debug.Log($"[FrameSpikeProbe] wrote {rows.Count} spikes + {timeline.Count}s timeline to ProbeLogs/");
            }
            catch { /* probe must never break shutdown */ }
        }
    }
}
