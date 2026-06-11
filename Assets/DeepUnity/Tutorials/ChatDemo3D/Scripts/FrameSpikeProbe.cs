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
    /// </summary>
    public class FrameSpikeProbe : MonoBehaviour
    {
        [SerializeField] float spikeMs = 18f;

        readonly List<string> rows = new List<string>(1024);
        int lastGcCount;
        int spikeCount;
        float worstMs;
        string worstPhase = "";
        float nextReportAt;

        private void Update()
        {
            float ms = Time.unscaledDeltaTime * 1000f;
            int gc = System.GC.CollectionCount(0);
            bool gcThisFrame = gc != lastGcCount;
            lastGcCount = gc;

            if (ms < spikeMs) return;

            string phase = LLM.CurrentPhase + (gcThisFrame ? "+GC" : "");
            spikeCount++;
            rows.Add($"{Time.unscaledTime:0.00},{ms:0.0},{phase}");
            if (ms > worstMs) { worstMs = ms; worstPhase = phase; }

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
                var all = new List<string>(rows.Count + 1) { "time_s,frame_ms,llm_phase" };
                all.AddRange(rows);
                File.WriteAllLines(Path.Combine("ProbeLogs", "frame_spikes.csv"), all);
                Debug.Log($"[FrameSpikeProbe] wrote {rows.Count} spikes to ProbeLogs/frame_spikes.csv");
            }
            catch { /* probe must never break shutdown */ }
        }
    }
}
