#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        using Cfg = PocketTTSConfig;

        // #StreamArBatch gate — K-blocked streaming AR must be BIT-EXACT with the per-frame path.
        //
        // What changed (2026-07-30): SynthesizeStreaming issues StreamArBatchFrames chained
        // GPU-resident frames per combined [eos|latent] readback instead of one, to amortize the
        // readback latency that capped production at ~1 latent/Unity-frame on the GTX 1650 (the
        // "blips"). The construction argument (frame f's latent never depends on later frames;
        // slot index only selects buffer offsets; overshoot rows never attended) says the samples
        // cannot change — this probe MEASURES that instead of trusting it: same text + same
        // injected noise at K=1 (the old per-frame schedule, ramp disabled) vs K=3 (ragged blocks)
        // vs K=4 ramped (the shipping config) must give maxAbs EXACTLY 0 and identical frame
        // counts. Two clauses: A ends within the first steady block (EOS-inside-block edge),
        // B spans many blocks (ragged tail + flush interleave).
        //
        //   menu:  DeepUnity/PocketTTS/#StreamArBatch K-Block Parity
        //   batch: Unity.exe -batchmode -projectPath <repo> ^
        //            -logFile ProbeLogs/pockettts_stream_batch.log ^
        //            -executeMethod DeepUnity.PocketTTSModeling.PocketTTSStreamBatchProbe.Run
        // No -quit (the method exits itself: 0 on PASS, 1 on FAIL) and NO -nographics: this probe
        // runs real compute shaders.
        public static class PocketTTSStreamBatchProbe
        {
            const string WEIGHTS_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string WEIGHTS_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
            const string REPORT = "ProbeLogs/pockettts_stream_batch.md";
            const string DONE = "ProbeLogs/pockettts_stream_batch.done";

            const string CLAUSE_A = "Fifty coppers, and not a bell more.";
            const string CLAUSE_B = "You will find the gate barred at dusk, and the watchman is not a patient man tonight.";
            const int NOISE_FRAMES = 160;   // cap; EOS ends both clauses well before this

            static string WEIGHTS = WEIGHTS_FP16;
            static readonly StringBuilder report = new StringBuilder();
            static int failures;

            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketStreamBatch] " + s); }
            static void Fail(string s) { failures++; report.AppendLine(s); Debug.LogError("[PocketStreamBatch] " + s); }

            [MenuItem("DeepUnity/PocketTTS/#StreamArBatch K-Block Parity")]
            public static void RunInteractive() { WEIGHTS = WEIGHTS_FP16; Execute(exitWhenDone: false); }

            public static void Run() { WEIGHTS = WEIGHTS_FP16; Execute(exitWhenDone: true); }

            [MenuItem("DeepUnity/PocketTTS/#StreamArBatch K-Block Parity (int8)")]
            public static void RunInteractiveInt8() { WEIGHTS = WEIGHTS_INT8; Execute(exitWhenDone: false); }

            public static void RunInt8() { WEIGHTS = WEIGHTS_INT8; Execute(exitWhenDone: true); }

            static void Execute(bool exitWhenDone)
            {
                report.Clear();
                failures = 0;
                Directory.CreateDirectory("ProbeLogs");
                PocketTTS tts = null;
                int savedK = PocketTTS.StreamArBatchFrames;
                int[] savedRamp = PocketTTS.StreamArBatchRamp;
                try
                {
                    Log($"# pocket-tts #StreamArBatch — K-block streaming parity — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {WEIGHTS}");
                    Log("");
                    tts = new PocketTTS(WEIGHTS);
                    tts.AsyncReadback = false;      // probes drain MoveNext tightly: sync readback, deterministic
                    tts.StreamChunkFrames = 8;      // pin the flush cadence (it slices the wav, so pin it)
                    tts.LoadBlocking();
                    Log($"weights resident, {tts.WeightBytes / (1024f * 1024f):F1} MB");

                    int[] idsA = tts.Tokenize(CLAUSE_A);
                    int[] idsB = tts.Tokenize(CLAUSE_B);
                    var noise = Noise(NOISE_FRAMES, seed: 20260730);

                    // reference: K=1, ramp OFF — every frame issues and reads back alone, which is
                    // the exact pre-2026-07-30 per-frame schedule.
                    PocketTTS.StreamArBatchFrames = 1; PocketTTS.StreamArBatchRamp = null;
                    float[] a1 = Synth(tts, idsA, noise, out int fA1);
                    float[] b1 = Synth(tts, idsB, noise, out int fB1);
                    Alive("K=1 (per-frame reference) clause A", a1, fA1);
                    Alive("K=1 (per-frame reference) clause B", b1, fB1);

                    // K=3 flat: ragged blocks, EOS mid-block, block-count coprime with chunk 8.
                    PocketTTS.StreamArBatchFrames = 3; PocketTTS.StreamArBatchRamp = null;
                    float[] a3 = Synth(tts, idsA, noise, out int fA3);
                    float[] b3 = Synth(tts, idsB, noise, out int fB3);
                    Exact($"K=3 flat vs K=1, clause A ({fA1}/{fA3} frames)", a3, a1);
                    Exact($"K=3 flat vs K=1, clause B ({fB1}/{fB3} frames)", b3, b1);

                    // K=4 with the shipping ramp {1,2}: the exact in-game configuration.
                    PocketTTS.StreamArBatchFrames = 4; PocketTTS.StreamArBatchRamp = new[] { 1, 2 };
                    float[] a4 = Synth(tts, idsA, noise, out int fA4);
                    float[] b4 = Synth(tts, idsB, noise, out int fB4);
                    Exact($"K=4 ramped vs K=1, clause A ({fA1}/{fA4} frames)", a4, a1);
                    Exact($"K=4 ramped vs K=1, clause B ({fB1}/{fB4} frames)", b4, b1);
                }
                catch (Exception e)
                {
                    Fail($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    PocketTTS.StreamArBatchFrames = savedK;
                    PocketTTS.StreamArBatchRamp = savedRamp;
                    tts?.Dispose();
                    Log("");
                    Log(failures == 0 ? "## RESULT: PASS" : $"## RESULT: FAIL ({failures} failure(s))");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failures == 0 ? "PASS" : "FAIL");
                }
                if (exitWhenDone) EditorApplication.Exit(failures == 0 ? 0 : 1);
            }

            // ------------------------------------------------ async-readback path (in-game path)
            //
            // The 2026-07-30 in-game crash (ArgumentException in ReadbackYielding, voice dead for
            // the whole session) went through a path NO probe above exercises: AsyncReadback=true.
            // Probes drain MoveNext tightly inside one editor update, where AsyncGPUReadback
            // requests never complete — so they all pin AsyncReadback=false, and the sync GetData
            // fallback happened to tolerate what the async NativeArray.CopyTo did not (a dst
            // larger than the ramped block's count). This entry runs the SAME synthesis on
            // EditorApplication.update so the async requests actually complete, and gates the
            // samples against the sync path (same dispatches — data must be identical).

            static PocketTTS aTts;
            static IEnumerator aJob;
            static List<float> aAcc;
            static float[] aRef;
            static double aT0;

            [MenuItem("DeepUnity/PocketTTS/#StreamArBatch Async-Path Parity")]
            public static void RunInteractiveAsync()
            {
                WEIGHTS = WEIGHTS_FP16;
                report.Clear();
                failures = 0;
                Directory.CreateDirectory("ProbeLogs");
                try
                {
                    Log($"# pocket-tts #StreamArBatch — ASYNC readback parity — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {WEIGHTS}");
                    aTts = new PocketTTS(WEIGHTS);
                    aTts.AsyncReadback = false;
                    aTts.StreamChunkFrames = 8;
                    aTts.LoadBlocking();
                    int[] ids = aTts.Tokenize(CLAUSE_B);
                    var noise = Noise(NOISE_FRAMES, seed: 20260730);
                    aRef = Synth(aTts, ids, noise, out int fr);   // sync reference, shipping K/ramp
                    Alive("sync reference (shipping K/ramp)", aRef, fr);

                    aTts.AsyncReadback = true;                    // the in-game configuration
                    aAcc = new List<float>(1 << 16);
                    aJob = aTts.SynthesizeStreaming(ids, w => { if (w != null) aAcc.AddRange(w); }, injectNoise: noise);
                    aT0 = EditorApplication.timeSinceStartup;
                    EditorApplication.update += AsyncTick;
                    Log("async synthesis pumping on EditorApplication.update — result lands in " +
                        "ProbeLogs/pockettts_stream_batch_async.done");
                }
                catch (Exception e)
                {
                    Fail($"**EXCEPTION (setup)**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                    FinishAsync(ran: false);
                }
            }

            static void AsyncTick()
            {
                try
                {
                    // Force every pending readback to completion FIRST. An idle, unfocused editor
                    // stops pumping AsyncGPUReadback entirely (no repaints -> fences never signal
                    // -> both earlier versions of this pump froze at exactly 19200 samples), and
                    // ReadbackYielding's hardwait guard counts Time.frameCount, which does not
                    // advance in edit mode. Play mode pumps requests every frame, so this is a
                    // HARNESS problem, not a product one — WaitAllRequests only closes the fence;
                    // the tested path (request -> done -> explicit-length copy) is unchanged.
                    UnityEngine.Rendering.AsyncGPUReadback.WaitAllRequests();
                    double budgetEnd = EditorApplication.timeSinceStartup + 0.05;
                    while (EditorApplication.timeSinceStartup < budgetEnd)
                    {
                        if (!aJob.MoveNext()) { FinishAsync(ran: true); return; }
                        // a fresh request was just issued: complete it now rather than spin on it
                        if (ReferenceEquals(aJob.Current, PocketTTS.GpuWait))
                            UnityEngine.Rendering.AsyncGPUReadback.WaitAllRequests();
                    }
                    EditorApplication.QueuePlayerLoopUpdate();   // keep updates coming while unfocused
                    if (EditorApplication.timeSinceStartup - aT0 > 60)
                    {
                        Fail($"**TIMEOUT**: async synthesis incomplete after 60 s ({aAcc.Count} samples so far).");
                        FinishAsync(ran: false);
                    }
                }
                catch (Exception e)
                {
                    Fail($"**EXCEPTION on the async path**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                    FinishAsync(ran: false);
                }
            }

            static void FinishAsync(bool ran)
            {
                EditorApplication.update -= AsyncTick;
                if (ran)
                    Exact($"async vs sync readback, clause B ({aAcc.Count} samples)", aAcc.ToArray(), aRef);
                aTts?.Dispose(); aTts = null; aJob = null; aAcc = null; aRef = null;
                Log("");
                Log(failures == 0 ? "## RESULT: PASS" : $"## RESULT: FAIL ({failures} failure(s))");
                File.WriteAllText("ProbeLogs/pockettts_stream_batch_async.md", report.ToString());
                File.WriteAllText("ProbeLogs/pockettts_stream_batch_async.done", failures == 0 ? "PASS" : "FAIL");
            }

            // ---------------------------------------------------------------- helpers

            static float[] Synth(PocketTTS tts, int[] ids, float[][] noise, out int frames)
            {
                var acc = new List<float>(1 << 16);
                var e = tts.SynthesizeStreaming(ids, w => { if (w != null) acc.AddRange(w); }, injectNoise: noise);
                while (e.MoveNext()) { }
                frames = tts.StreamLastTokenCount;
                return acc.ToArray();
            }

            static float[][] Noise(int frames, int seed)
            {
                var rng = new System.Random(seed);
                float sigma = Mathf.Sqrt(Cfg.TEMPERATURE);
                var rows = new float[frames][];
                for (int t = 0; t < frames; t++)
                {
                    rows[t] = new float[Cfg.LDIM];
                    for (int i = 0; i < Cfg.LDIM; i++)
                    {
                        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();   // u1 != 0 for Log
                        rows[t][i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)) * sigma;
                    }
                }
                return rows;
            }

            /// <summary>Guard against a "parity" that only holds because both runs produced nothing.</summary>
            static void Alive(string name, float[] w, int frames)
            {
                double rms = 0;
                bool finite = true;
                foreach (float s in w) { finite &= !float.IsNaN(s) && !float.IsInfinity(s); rms += (double)s * s; }
                rms = Math.Sqrt(rms / Math.Max(w.Length, 1));
                if (frames > 0 && w.Length > 0 && finite && rms > 0.001)
                    Log($"##   {name}: {frames} frames, {w.Length} samples, rms {rms:F4}  PASS (audible)");
                else
                    Fail($"##   {name}: {frames} frames, {w.Length} samples, rms {rms:F4}, finite {finite}" +
                         "  <-- FAIL (nothing was synthesised — parity below would be vacuous)");
            }

            static void Exact(string name, float[] a, float[] b)
            {
                if (a.Length != b.Length)
                {
                    Fail($"## {name}: LENGTH DIFFERS {a.Length} vs {b.Length} samples  <-- FAIL " +
                         "(the two schedules did not even generate the same frame count)");
                    return;
                }
                double mx = 0;
                for (int i = 0; i < a.Length; i++)
                {
                    double d = Math.Abs(a[i] - b[i]);
                    if (d > mx) mx = d;
                }
                if (mx == 0) Log($"## {name}: {a.Length} samples, maxAbs 0 — BIT-IDENTICAL  PASS");
                else Fail($"## {name}: maxAbs {mx:E3}  <-- FAIL (K-blocking changed the samples)");
            }
        }
    }
}
#endif
