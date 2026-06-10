using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Opt-in smoothness diagnostic — NOT active unless you add it to a scene (or call
    /// <see cref="Ensure"/> once). Logs every frame longer than 20 ms with the frame index and
    /// whether a GC collection ran on that frame; cross-reference the entries with your own phase
    /// logs (or Editor.log) to pinpoint exactly what a freeze is. `GC: YES` means an allocation
    /// problem; a spike with no GC around model inference means a GPU-bound frame.
    /// Used to find every freeze documented in LLMs/OPTIMIZATIONS.md.
    /// </summary>
    public class FrameSpikeLogger : MonoBehaviour
    {
        const double SPIKE_MS = 20.0;

        public static void Ensure()
        {
            if (FindObjectOfType<FrameSpikeLogger>() == null)
                new GameObject("[FrameSpikeLogger]").AddComponent<FrameSpikeLogger>();
        }

        readonly System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        int lastGc0, lastGc1, lastGc2;
        bool first = true;

        void Update()
        {
            double ms = sw.Elapsed.TotalMilliseconds;
            sw.Restart();
            int gc0 = GC.CollectionCount(0);
            int gc1 = GC.CollectionCount(1);
            int gc2 = GC.CollectionCount(2);

            if (!first && ms > SPIKE_MS)
                Debug.Log($"[Spike] frame {Time.frameCount}: {ms:0.0} ms | GC this frame: " +
                          $"gen0 {(gc0 != lastGc0 ? "YES" : "no")}, gen1 {(gc1 != lastGc1 ? "YES" : "no")}, gen2 {(gc2 != lastGc2 ? "YES" : "no")}");

            lastGc0 = gc0; lastGc1 = gc1; lastGc2 = gc2;
            first = false;
        }
    }
}
