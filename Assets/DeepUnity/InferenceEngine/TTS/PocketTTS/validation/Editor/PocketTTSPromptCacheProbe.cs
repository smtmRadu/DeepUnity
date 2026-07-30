#if UNITY_EDITOR
using System;
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

        // #32 gate — the RETAINED voice-prompt KV path must be BIT-EXACT with the full prefill.
        //
        // What #32 does: rows [bbv | voicePrompt] of the flow-LM KV cache are identical in content
        // AND in absolute position on every clause, so they are kept across clauses and only the
        // ~25 text rows are prefilled — through the per-row decode path (DecodeStepKVIssue), not the
        // block prefill. That swap of kernels is the whole risk, so it is not argued here, it is
        // MEASURED: same text + same injected noise on both paths must give the same samples with
        // maxAbs EXACTLY 0. injectNoise makes SynthesizeStreaming deterministic (deterministic =
        // injectNoise != null), which is what makes a sample-exact comparison meaningful at all.
        //
        // Six runs, one engine, one noise block (order matters — it is what warms and grows things):
        //   1  voice2, clause A  -> FULL prefill on a COLD cache        = the voice2 reference
        //   2  voice1, clause A  -> FULL (voice swap: key mismatch)     = the clause-A reference
        //   3  voice1, clause B  -> FULL (longer clause grows kvCap)      warms the retained prompt
        //   4  voice1, clause A  -> RETAINED                            == run 2, sample-exact
        //   5  voice1, clause B  -> RETAINED                            == run 3, sample-exact
        //   6  voice2, clause A  -> FULL (voice swap fires the fallback)== run 1, sample-exact
        // Run 6 is the dangerous case: voice2 has the SAME row count as voice1, so only the identity
        // of the voicePrompt array separates them, and rows of voice1's prompt are still physically
        // in the cache while run 6 prefills over them.
        // LastPrefillRows is asserted on every run, so a silent reuse (or a silent fallback that made
        // the parity trivial) fails instead of passing quietly.
        //
        //   menu:  DeepUnity/PocketTTS/#32 Retained Voice-Prompt KV Parity
        //   batch: Unity.exe -batchmode -projectPath <repo> ^
        //            -logFile ProbeLogs/pockettts_prompt_cache.log ^
        //            -executeMethod DeepUnity.PocketTTSModeling.PocketTTSPromptCacheProbe.Run
        // No -quit (the method exits itself: 0 on PASS, 1 on FAIL) and NO -nographics: this probe
        // runs real compute shaders.
        public static class PocketTTSPromptCacheProbe
        {
            const string WEIGHTS_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string WEIGHTS_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
            const string REPORT = "ProbeLogs/pockettts_prompt_cache.md";
            const string DONE = "ProbeLogs/pockettts_prompt_cache.done";

            // ~10 and ~20 tokens: clause B MUST be the longer one, so its Lp + maxFrames is what
            // sizes kvCap — otherwise run 4 would legitimately fall back on a cache too small for it
            // and the headline parity check would never exercise the retained path.
            const string CLAUSE_A = "Fifty coppers, and not a bell more.";
            const string CLAUSE_B = "You will find the gate barred at dusk, and the watchman is not a patient man tonight.";
            const int NOISE_FRAMES = 160;   // cap; EOS ends both clauses well before this

            static string WEIGHTS = WEIGHTS_FP16;
            static readonly StringBuilder report = new StringBuilder();
            static int failures;

            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketPromptKV] " + s); }
            static void Fail(string s) { failures++; report.AppendLine(s); Debug.LogError("[PocketPromptKV] " + s); }

            [MenuItem("DeepUnity/PocketTTS/#32 Retained Voice-Prompt KV Parity")]
            public static void RunInteractive() { WEIGHTS = WEIGHTS_FP16; Execute(exitWhenDone: false); }

            /// <summary>Batch entry (-executeMethod). Exits 0 on PASS, 1 on FAIL.</summary>
            public static void Run() { WEIGHTS = WEIGHTS_FP16; Execute(exitWhenDone: true); }

            // int8 quantizes the weights, not the kernel ROUTING (CoalEligible keys on in_dim only),
            // so the bit-exactness claim has to hold on the q8 GEMV/GEMM twins as well.
            [MenuItem("DeepUnity/PocketTTS/#32 Retained Voice-Prompt KV Parity (int8)")]
            public static void RunInteractiveInt8() { WEIGHTS = WEIGHTS_INT8; Execute(exitWhenDone: false); }

            public static void RunInt8() { WEIGHTS = WEIGHTS_INT8; Execute(exitWhenDone: true); }

            static void Execute(bool exitWhenDone)
            {
                report.Clear();
                failures = 0;
                Directory.CreateDirectory("ProbeLogs");
                PocketTTS tts = null;
                try
                {
                    Log($"# pocket-tts #32 — retained voice-prompt KV parity — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {WEIGHTS}");
                    Log("");
                    tts = new PocketTTS(WEIGHTS);
                    tts.AsyncReadback = false;      // probes drain MoveNext tightly: sync readback, deterministic
                    tts.StreamChunkFrames = 8;      // pin the flush cadence (it slices the wav, so pin it)
                    tts.LoadBlocking();
                    Log($"weights resident, {tts.WeightBytes / (1024f * 1024f):F1} MB");

                    int[] idsA = tts.Tokenize(CLAUSE_A);
                    int[] idsB = tts.Tokenize(CLAUSE_B);
                    if (idsB.Length <= idsA.Length)
                        Fail($"**SETUP**: clause B ({idsB.Length} tokens) must be LONGER than clause A " +
                             $"({idsA.Length}) or the retained path is never reached.");

                    float[] voice1 = tts.CurrentVoicePrompt;                  // baked 'jean'
                    int voiceFrames = voice1.Length / Cfg.DIM;
                    float[] voice2 = SecondVoicePrompt(voice1);               // same rows, different speaker
                    int Lv = 1 + voiceFrames;
                    int LpA = Lv + idsA.Length, LpB = Lv + idsB.Length;
                    Log($"prompt {voiceFrames} frames -> Lv {Lv} retained rows; " +
                        $"clause A {idsA.Length} tokens (Lp {LpA}), clause B {idsB.Length} tokens (Lp {LpB})");
                    Log($"PREFILL ROWS  before(full) A {LpA} / B {LpB}   after(retained) A {idsA.Length} / B {idsB.Length}");
                    Log("");

                    var noise = Noise(NOISE_FRAMES, seed: 20260728);

                    // ---- 1) voice2 on a cold cache: the reference for the fallback in run 6 ----
                    if (!tts.BindRawVoicePrompt(voice2, "probe-voice2"))
                    { Fail("**SETUP**: BindRawVoicePrompt rejected the synthetic second prompt."); goto done; }
                    if (tts.RetainedPromptRows != 0)
                        Fail($"**COLD**: a fresh engine already claims {tts.RetainedPromptRows} retained rows.");
                    float[] wV2Cold = Synth(tts, idsA, noise, out int rows1, out int frames1, out int tk1, out double pm1);
                    Expect("run 1  voice2/A  cold", rows1, LpA, tk1, pm1);
                    Alive("run 1  voice2/A  cold", wV2Cold, frames1);

                    // ---- 2) voice1, clause A: full prefill (the voice swap must not reuse voice2) ----
                    if (!tts.BindRawVoicePrompt(voice1, "probe-voice1"))
                    { Fail("**SETUP**: BindRawVoicePrompt rejected the baked prompt."); goto done; }
                    float[] wA_full = Synth(tts, idsA, noise, out int rows2, out int frames2, out int tk2, out double pm2);
                    Expect("run 2  voice1/A  full", rows2, LpA, tk2, pm2);
                    Alive("run 2  voice1/A  full", wA_full, frames2);
                    if (tts.RetainedPromptRows != Lv)
                        Fail($"**RETAIN**: after a completed full prefill the engine retains " +
                             $"{tts.RetainedPromptRows} rows, expected {Lv}.");

                    // ---- 3) voice1, clause B: full (longer clause -> EnsureKV grows -> re-retained) ----
                    float[] wB_full = Synth(tts, idsB, noise, out int rows3, out int frames3, out int tk3, out double pm3);
                    Expect("run 3  voice1/B  full", rows3, LpB, tk3, pm3);
                    Alive("run 3  voice1/B  full", wB_full, frames3);

                    // ---- 4) HEADLINE: clause A again, retained prompt, same noise -> same samples ----
                    float[] wA_ret = Synth(tts, idsA, noise, out int rows4, out int frames4, out int tk4, out double pm4);
                    Expect("run 4  voice1/A  RETAINED", rows4, idsA.Length, tk4, pm4);
                    Alive("run 4  voice1/A  RETAINED", wA_ret, frames4);
                    Exact("clause A: retained vs full prefill", wA_ret, wA_full);
                    Log($"##   clause A: {tk2} prefill ticks / {pm2:F0} ms whole synth (FULL) -> " +
                        $"{tk4} ticks / {pm4:F0} ms (RETAINED), saved {pm2 - pm4:F0} ms");

                    // ---- 5) same for the longer clause (more text rows through the per-row path) ----
                    float[] wB_ret = Synth(tts, idsB, noise, out int rows5, out int frames5, out int tk5, out double pm5);
                    Expect("run 5  voice1/B  RETAINED", rows5, idsB.Length, tk5, pm5);
                    Exact("clause B: retained vs full prefill", wB_ret, wB_full);
                    Log($"##   clause B: {tk3} prefill ticks / {pm3:F0} ms whole synth (FULL) -> " +
                        $"{tk5} ticks / {pm5:F0} ms (RETAINED), saved {pm3 - pm5:F0} ms");

                    // ---- 6) voice swap: the fallback must FIRE and still be exact ----
                    if (!tts.BindRawVoicePrompt(voice2, "probe-voice2")) { Fail("**SETUP**: rebind voice2 failed."); goto done; }
                    float[] wV2_fb = Synth(tts, idsA, noise, out int rows6, out int frames6, out int tk6, out double pm6);
                    Expect("run 6  voice2/A  FALLBACK", rows6, LpA, tk6, pm6);
                    Exact("voice swap: fallback vs cold full prefill", wV2_fb, wV2Cold);
                    // and the two voices must actually differ, or run 6 proves nothing
                    if (SameSamples(wV2Cold, wA_full))
                        Fail("**SETUP**: voice1 and voice2 produce identical audio — the swap test is vacuous.");
                    else
                        Log("## voice1 vs voice2 audio differs: PASS (the swap test is meaningful)");

                    // ---- invalidation: a weight defetch must drop the retained rows ----
                    tts.Defetch(slow: false);
                    if (tts.RetainedPromptRows != 0)
                        Fail($"**INVALIDATE**: Defetch left {tts.RetainedPromptRows} retained rows.");
                    else
                        Log("## Defetch drops the retained prompt: PASS");

                done: ;
                }
                catch (Exception e)
                {
                    Fail($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    tts?.Dispose();
                    Log("");
                    Log(failures == 0 ? "## RESULT: PASS" : $"## RESULT: FAIL ({failures} failure(s))");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failures == 0 ? "PASS" : "FAIL");
                }
                if (exitWhenDone) EditorApplication.Exit(failures == 0 ? 0 : 1);
            }

            // ---------------------------------------------------------------- helpers

            /// <summary>Drain SynthesizeStreaming synchronously, concatenating the pushed blocks. The
            /// prefill cost comes from the engine's own LastPrefill* counters, NOT from watching
            /// LastHeavyTick: that string is only written at yield sites that actually execute, and
            /// with AsyncReadback off the AR readback enumerators complete WITHOUT yielding, so the
            /// last prefill label bleeds across AR ticks and a probe-side tick count over-reports
            /// (it read 25 "prefill" ticks for a retained clause that issued 2).
            /// TICKS is the number that matters in the game, but the pump does NOT end the frame on each
            /// FrameBreak (corrected 2026-07-28) — it breaks at maxHeavyTicks, 6 on Smooth down to 2 on
            /// Very Fast, so 24 ticks is 4-12 frames rather than 24.</summary>
            static float[] Synth(PocketTTS tts, int[] ids, float[][] noise, out int prefillRows,
                                 out int frames, out int prefillTicks, out double prefillMs)
            {
                var acc = new List<float>(1 << 16);
                var e = tts.SynthesizeStreaming(ids, w => { if (w != null) acc.AddRange(w); }, injectNoise: noise);
                while (e.MoveNext()) { }
                prefillRows = tts.LastPrefillRows;
                prefillTicks = tts.LastPrefillTicks;
                // GenMs (whole synth, GPU included) — not LastPrefillMs, which is CPU ISSUE time only:
                // the prefill dispatches are queued, not awaited, so the GPU cost of the rows shows up
                // later at the first readback. Two runs of the SAME clause differ only in the prefill,
                // and the outputs are bit-identical, so the GenMs delta IS the prefill saving.
                prefillMs = tts.GenMs;
                frames = tts.StreamLastTokenCount;
                return acc.ToArray();
            }

            /// <summary>A second speaker with the SAME frame count as the baked one: per-frame reversal
            /// plus a channel-dependent tilt. Deterministic (no RNG), well away from the baked prompt,
            /// and still an [T,1024] buffer with a healthy RMS so PromptIsValid accepts it. It only has
            /// to be a DIFFERENT conditioning — it does not have to sound like anybody.</summary>
            static float[] SecondVoicePrompt(float[] src)
            {
                int dim = Cfg.DIM, T = src.Length / dim;
                var dst = new float[src.Length];
                for (int t = 0; t < T; t++)
                {
                    int s = (T - 1 - t) * dim, d = t * dim;
                    for (int c = 0; c < dim; c++)
                        dst[d + c] = src[s + c] * (c % 2 == 0 ? 0.85f : -1.15f);
                }
                return dst;
            }

            /// <summary>Fixed-seed Gaussian noise rows, scaled like the runtime sampler
            /// (Gauss(LDIM, sqrt(TEMPERATURE))) so the injected utterances stay realistic.</summary>
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

            /// <summary>Which prefill path actually ran. Asserted on EVERY run: without it a bug that
            /// always fell back (or always reused) would make the parity checks meaningless.</summary>
            static void Expect(string name, int rows, int expected, int ticks, double ms)
            {
                string cost = $"({ticks} prefill ticks, whole synth {ms:F0} ms)";
                if (rows == expected) Log($"## {name}: prefilled {rows} rows {cost}  PASS");
                else Fail($"## {name}: prefilled {rows} rows {cost}, expected {expected} rows" +
                          "  <-- FAIL (wrong path taken)");
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

            /// <summary>Sample-exact or FAIL. maxAbs must be EXACTLY 0 — see the header for why that
            /// bar is reachable at all.</summary>
            static void Exact(string name, float[] a, float[] b)
            {
                if (a.Length != b.Length)
                {
                    Fail($"## {name}: LENGTH DIFFERS {a.Length} vs {b.Length} samples  <-- FAIL " +
                         "(the two paths did not even generate the same frame count)");
                    return;
                }
                double mx = 0;
                int bad = 0, first = -1;
                for (int i = 0; i < a.Length; i++)
                {
                    double d = Math.Abs(a[i] - b[i]);
                    if (d != 0) { bad++; if (first < 0) first = i; }
                    if (d > mx) mx = d;
                }
                if (bad == 0) Log($"## {name}: {a.Length} samples, maxAbs 0 — BIT-IDENTICAL  PASS");
                else Fail($"## {name}: {bad}/{a.Length} samples differ, maxAbs {mx:E6}, " +
                          $"first at {first} ({a[first]:E9} vs {b[first]:E9})  <-- FAIL");
            }

            static bool SameSamples(float[] a, float[] b)
            {
                if (a.Length != b.Length) return false;
                for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
                return true;
            }
        }
    }
}
#endif
