using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Records every frame longer than spikeMs together with what the LLM machinery was doing at
    /// that moment (LLM.CurrentPhase) and whether a GC collection landed on the frame — so fps
    /// dips get attributed to a phase ("boot", "kv-restore", "decode", or plain "idle" gameplay)
    /// instead of guessed at. Logs a compact summary to the console every 10 s while spikes keep
    /// happening, and dumps the full list to ProbeLogs/frame_spikes.csv when the scene ends.
    /// Opt-in: does nothing unless `record` is ticked in the inspector.
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

        private void Update()
        {
            if (!record) return;
            float ms = Time.unscaledDeltaTime * 1000f;
            int gc = System.GC.CollectionCount(0);
            bool gcThisFrame = gc != lastGcCount;
            lastGcCount = gc;

            if (ms < spikeMs) return;

            // llm_phase says what the LLM was doing; tts_tick is the LAST heavy TTS slice issued
            // (prefill_text / ar_frame / mimi_decode / flush_push...) — together with the wall
            // clock they line the spike up against the [GPU]/[PocketTTSVoice] console lines, which
            // is how a "freeze during load-up / warmup" gets attributed instead of guessed at.
            string phase = LLM.CurrentPhase + (gcThisFrame ? "+GC" : "");
            string ttsTick = PocketTTSModeling.PocketTTS.LastHeavyTick ?? "";
            spikeCount++;
            rows.Add($"{Time.unscaledTime:0.00},{System.DateTime.Now:HH:mm:ss.ff},{ms:0.0},{phase},{ttsTick}");
            if (ms > worstMs) { worstMs = ms; worstPhase = $"{phase}/{ttsTick}"; }

            if (Time.unscaledTime >= nextReportAt)
            {
                nextReportAt = Time.unscaledTime + 10f;
                Debug.Log($"[FrameSpikeProbe] {spikeCount} frames over {spikeMs} ms — worst {worstMs:0.0} ms during '{worstPhase}'");
            }
        }

        private void OnDestroy()
        {
            if (rows.Count == 0) return;
            try
            {
                Directory.CreateDirectory("ProbeLogs");
                var all = new List<string>(rows.Count + 1) { "time_s,clock,frame_ms,llm_phase,tts_tick" };
                all.AddRange(rows);
                File.WriteAllLines(Path.Combine("ProbeLogs", "frame_spikes.csv"), all);
                Debug.Log($"[FrameSpikeProbe] wrote {rows.Count} spikes to ProbeLogs/frame_spikes.csv");
            }
            catch { /* probe must never break shutdown */ }
        }
    }
}
