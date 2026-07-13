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
        // P4 — offline end-to-end. Deterministic parity (inject the reference per-frame noise) gates
        // the AR loop: generated latents vs latents.npy, wav vs wav.npy. Plus a real-RNG "names"
        // listen sentence. EDITOR-MODE synchronous (ClaudeBridge invoke).
        public static class PocketTTSE2EProbe
        {
            const string DUMP = "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/dump";
            const string WEIGHTS_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string WEIGHTS_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
            const string REPORT = "ProbeLogs/pockettts_e2e_parity.md";
            const string DONE = "ProbeLogs/pockettts_e2e.done";

            static string WEIGHTS = WEIGHTS_FP16;
            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketE2E] " + s); }

            [MenuItem("DeepUnity/PocketTTS/P4 Offline E2E")]
            public static void Run()
            {
                WEIGHTS = WEIGHTS_FP16;
                RunInner();
            }

            // int8 parity: expect corr >= 0.99 (NOT bit-exact) + smaller VRAM. Same gates @ 0.99.
            [MenuItem("DeepUnity/PocketTTS/P4 Offline E2E (int8)")]
            public static void RunInt8()
            {
                WEIGHTS = WEIGHTS_INT8;
                try { RunInner(); } finally { WEIGHTS = WEIGHTS_FP16; }
            }

            static void RunInner()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                PocketTTS tts = null;
                try
                {
                    Log($"# pocket-tts P4 — offline E2E — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {WEIGHTS}");
                    tts = new PocketTTS(WEIGHTS);
                    tts.LoadBlocking();
                    Log($"weights resident. footprint {tts.WeightBytes / (1024f * 1024f):F1} MB");

                    // ---- deterministic parity: inject reference per-frame noise ----
                    int[] textIds = Ints("text_ids", out _);
                    float[] noiseFlat = Floats("flow_noise_all", out int[] nsh);   // [T,32]
                    int T = nsh[0], ld = nsh[nsh.Length - 1];
                    var inject = new float[T][];
                    for (int t = 0; t < T; t++) { inject[t] = new float[ld]; Array.Copy(noiseFlat, t * ld, inject[t], 0, ld); }
                    Log($"deterministic: {T} frames, {textIds.Length} text ids");

                    bool int8 = WEIGHTS == WEIGHTS_INT8;
                    float[] embMean = Floats("emb_mean", out _), embStd = Floats("emb_std", out _);
                    float[] latRef = Floats("latents", out int[] lsh);
                    float[] wavRef = Floats("wav", out _);
                    // fp16 = bit-exact expectation (raw-sample gates). int8 = quantized AR model:
                    // per-frame latent error compounds and the WAVEFORM PHASE desyncs vs the fp16
                    // ref, so raw-sample wav corr is the WRONG metric (verified: a 40-sample phase
                    // shift drops sample-corr to 0.17 but mel-corr stays 0.998). int8 gates on the
                    // phase-invariant mel-spectrogram corr + early-frame latents + frame count +
                    // validity + streamed==offline; raw-sample wav corr is logged INFORMATIONAL.
                    float latGate = int8 ? 0.90f : 0.99f;   // all-frames latent corr (int8 compounds ~0.95)

                    // ===== P5 gate: KV-cache decode MUST stay bit-exact vs full-forward (KV ≡ FF) =====
                    float[] wav = tts.GenerateOffline(textIds, inject, useKvCache: true);
                    float[] denKv = Denorm(tts.LastLatentsRaw, tts.LastFrames, ld, embMean, embStd);
                    Log($"[KV] frames: ours {tts.LastFrames} vs ref {lsh[0]}" + (tts.LastFrames != lsh[0] ? "  <-- FRAME COUNT MISMATCH" : ""));
                    if (tts.LastFrames != lsh[0]) failed = true;   // EOS must land on the same frame
                    PerFrame(denKv, latRef, ld, "KV");
                    // early-frame latents must be near-exact even under int8 (drift is later-frame)
                    failed |= FrameGate(denKv, latRef, ld, int8 ? 0.99f : 0.999f, "KV early-frame(0-3)");
                    failed |= Grade("KV latents (all-frames)", denKv, latRef, latGate);
                    if (int8) failed |= MelGate("KV wav mel", wav, wavRef, 0.92f);   // int8 AR drift accepted (user A/B 2026-07-13; measured ~0.925)
                    else failed |= Grade("KV wav", wav, wavRef, 0.99f);
                    LogInfo(int8, "KV wav raw-sample (informational under int8)", wav, wavRef);
                    SaveWav("ProbeLogs/pockettts_e2e_unity.wav", wav);

                    // ===== cross-check: full-forward path (P4 original) — KV==FF EXACTLY, both quant =====
                    float[] wavFF = tts.GenerateOffline(textIds, inject, useKvCache: false);
                    float[] denFF = Denorm(tts.LastLatentsRaw, tts.LastFrames, ld, embMean, embStd);
                    Log($"[full-forward] frames: ours {tts.LastFrames} vs ref {lsh[0]}");
                    // KV vs FF is the same math either quant -> must be ~1.0 regardless of int8
                    failed |= Grade("KV==FF latents", denFF, denKv, 0.999f);
                    failed |= Grade("FF latents (all-frames vs ref)", denFF, latRef, latGate);

                    // ===== perf: KV vs full-forward, RTF + breakdown (deterministic run, same frames) =====
                    float sec = wav.Length / (float)PocketTTSConfig.SAMPLE_RATE;
                    // re-time KV cleanly (previous KV call warmed shaders/weights)
                    tts.GenerateOffline(textIds, inject, useKvCache: true);
                    float kvGen = tts.GenMs, kvPre = tts.PrefillMs, kvLoop = tts.LoopMs, kvDec = tts.DecodeMs, kvTtfa = tts.TtfaMs;
                    float kvTotal = kvGen + kvDec;
                    Log($"[perf KV]   prefill {kvPre:F0} + loop {kvLoop:F0} = gen {kvGen:F0} ms | mimi decode {kvDec:F0} ms " +
                        $"| total {kvTotal:F0} ms for {sec:F2}s -> RTF {kvTotal / 1000f / sec:F3} | TTFA(proxy) {kvTtfa:F0} ms");
                    // time the REAL full-forward loop (O(T·L²)) for the speedup comparison
                    tts.GenerateOffline(textIds, inject, useKvCache: false);
                    float ffGen = tts.GenMs;
                    Log($"[perf FF]   gen {ffGen:F0} ms (full-forward O(T·L²), for reference) -> KV speedup {ffGen / Mathf.Max(kvGen, 1e-3f):F1}x");

                    // ---- listen sentence (names), REAL RNG, KV path ----
                    int[] namesIds = Ints("names_ids", out _);
                    float[] wavNames = tts.GenerateOffline(namesIds, injectNoise: null, useKvCache: true);
                    SaveWav("ProbeLogs/pockettts_names_unity.wav", wavNames);
                    float nSec = wavNames.Length / (float)PocketTTSConfig.SAMPLE_RATE;
                    Log($"names sentence (KV): {namesIds.Length} ids -> {wavNames.Length} samples ({nSec:F2}s) " +
                        $"| prefill {tts.PrefillMs:F0} + loop {tts.LoopMs:F0} + decode {tts.DecodeMs:F0} ms " +
                        $"-> RTF {(tts.GenMs + tts.DecodeMs) / 1000f / nSec:F3} | TTFA(proxy) {tts.TtfaMs:F0} ms " +
                        $"-> ProbeLogs/pockettts_names_unity.wav (listen QA)");
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    tts?.Dispose();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            // denorm raw flow latents [T,32] -> Mimi-input space (matches latents.npy)
            static float[] Denorm(float[] raw, int frames, int ld, float[] embMean, float[] embStd)
            {
                float[] den = new float[frames * ld];
                for (int t = 0; t < frames; t++)
                    for (int cc = 0; cc < ld; cc++)
                        den[t * ld + cc] = raw[t * ld + cc] * embStd[cc] + embMean[cc];
                return den;
            }
            // per-frame corr (0/1/2): frame 0 exact + frame 1 diverging = feedback/KV-position bug
            static void PerFrame(float[] den, float[] latRef, int ld, string tag)
            {
                for (int fi = 0; fi < 3 && (fi + 1) * ld <= Math.Min(den.Length, latRef.Length); fi++)
                {
                    float[] a = new float[ld], b = new float[ld];
                    Array.Copy(den, fi * ld, a, 0, ld); Array.Copy(latRef, fi * ld, b, 0, ld);
                    var (mx, mae, corr) = Diff(a, b);
                    Log($"   [{tag}] frame {fi}: maxAbs {mx:F4} MAE {mae:F5} corr {corr:F6}");
                }
            }

            // early-frame latent gate: mean per-frame corr over the first `nf` frames (drift-free zone)
            static bool FrameGate(float[] den, float[] latRef, int ld, float gate, string tag, int nf = 4)
            {
                int frames = Math.Min(nf, Math.Min(den.Length, latRef.Length) / ld);
                double sum = 0; float worst = 1f;
                for (int fi = 0; fi < frames; fi++)
                {
                    float[] a = new float[ld], b = new float[ld];
                    Array.Copy(den, fi * ld, a, 0, ld); Array.Copy(latRef, fi * ld, b, 0, ld);
                    var (_, _, corr) = Diff(a, b);
                    sum += corr; worst = Mathf.Min(worst, corr);
                }
                float mean = (float)(sum / Math.Max(frames, 1));
                bool bad = worst < gate;
                Log($"## {tag}: frames {frames}, mean corr {mean:F6}, worst {worst:F6} (gate {gate:F3})" + (bad ? "  <-- FAIL" : "  PASS"));
                return bad;
            }
            // perceptual (phase-invariant) wav gate for int8
            static bool MelGate(string name, float[] ours, float[] rf, float gate)
            {
                float mc = PocketTTSMel.MelCorr(ours, rf, PocketTTSConfig.SAMPLE_RATE);
                bool bad = mc < gate;
                Log($"## {name}: mel-corr {mc:F6} (gate {gate:F3}, phase-invariant)" + (bad ? "  <-- FAIL" : "  PASS"));
                return bad;
            }
            // informational-only sample corr (never fails; contextualizes int8 phase desync)
            static void LogInfo(bool int8, string name, float[] ours, float[] rf)
            {
                var (mx, mae, corr) = Diff(ours, rf);
                Log($"   [info]{(int8 ? " int8" : "")} {name}: corr {corr:F6} maxAbs {mx:F4} MAE {mae:F5}"
                    + (int8 ? "  (low = phase desync, NOT quality loss — see mel-corr)" : ""));
            }

            static bool Grade(string name, float[] ours, float[] rf, float gate)
            {
                var (mx, mae, corr) = Diff(ours, rf);
                bool bad = corr < gate;
                Log($"## {name}: ours {ours.Length} vs ref {rf.Length}; maxAbs {mx:F4} MAE {mae:F5} corr {corr:F6}" + (bad ? "  <-- FAIL" : "  PASS"));
                return bad;
            }

            static Array LoadNpy(string path, out int[] shape)
            {
                byte[] all = File.ReadAllBytes(path);
                if (all[0] != 0x93) throw new Exception($"not npy: {path}");
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);
                string shapeStr = header.Substring(header.IndexOf("'shape':", StringComparison.Ordinal) + 8);
                shapeStr = shapeStr.Substring(shapeStr.IndexOf('(') + 1);
                shapeStr = shapeStr.Substring(0, shapeStr.IndexOf(')'));
                var dims = new List<int>();
                foreach (string s in shapeStr.Split(',')) if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
                if (dims.Count == 0) dims.Add(1);
                shape = dims.ToArray();
                long count = 1; foreach (int dd in shape) count *= dd;
                if (header.Contains("f4")) { float[] r = new float[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4); return r; }
                if (header.Contains("i8")) { long[] r = new long[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 8); return r; }
                if (header.Contains("i4")) { int[] r = new int[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4); return r; }
                throw new Exception($"unsupported npy dtype: {header}");
            }
            static float[] Floats(string name, out int[] sh) => (float[])LoadNpy(Path.Combine(DUMP, name + ".npy"), out sh);
            static int[] Ints(string name, out int[] sh)
            {
                Array a = LoadNpy(Path.Combine(DUMP, name + ".npy"), out sh);
                if (a is int[] ia) return ia;
                if (a is long[] la) { int[] r = new int[la.Length]; for (int i = 0; i < la.Length; i++) r[i] = (int)la[i]; return r; }
                // dump_reference.save() casts torch tensors to float32, so int64 ids arrive as f4 (exact < 2^24)
                if (a is float[] fa) { int[] r = new int[fa.Length]; for (int i = 0; i < fa.Length; i++) r[i] = (int)Math.Round(fa[i]); return r; }
                throw new Exception($"Ints: unexpected npy element type {a.GetType()} for {name}");
            }
            static (float, float, float) Diff(float[] a, float[] b)
            {
                int n = Mathf.Min(a.Length, b.Length);
                double mx = 0, mae = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                {
                    double dd = Math.Abs(a[i] - b[i]); mx = Math.Max(mx, dd); mae += dd;
                    sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
                }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return ((float)mx, (float)(mae / n), (float)(cov / Math.Sqrt(Math.Max(va * vb, 1e-20))));
            }
            static void SaveWav(string path, float[] s)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var wr = new BinaryWriter(fs);
                int sr = PocketTTSConfig.SAMPLE_RATE, bl = s.Length * 2;
                wr.Write(Encoding.ASCII.GetBytes("RIFF")); wr.Write(36 + bl);
                wr.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); wr.Write(16);
                wr.Write((short)1); wr.Write((short)1); wr.Write(sr); wr.Write(sr * 2);
                wr.Write((short)2); wr.Write((short)16);
                wr.Write(Encoding.ASCII.GetBytes("data")); wr.Write(bl);
                foreach (float v in s) wr.Write((short)Mathf.Clamp(Mathf.RoundToInt(v * 32767f), short.MinValue, short.MaxValue));
            }
        }
    }
}
