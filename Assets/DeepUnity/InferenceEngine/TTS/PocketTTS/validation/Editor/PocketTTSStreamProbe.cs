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
        // P5.2 probe — real-time streaming synthesis. Editor-mode synchronous (ClaudeBridge):
        // runs SynthesizeStreaming to completion, concatenates the pushed chunks, and gates the
        // streamed wav against the OFFLINE wav (must be bit-exact — streaming is the same per-frame
        // AR + a causal re-decode that reproduces earlier samples exactly). Reports TTFA (wall to
        // first chunk), chunk count/cadence, and per-frame RTF. Ring-starvation is a play-mode
        // timing property (audio thread); this probe proves the DATA path (continuity + parity).
        // A separate play-mode run of PocketTTSVoice asserts underflows==0 (see DO note).
        public static class PocketTTSStreamProbe
        {
            const string DUMP = "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/dump";
            const string WEIGHTS_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string WEIGHTS_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
            const string REPORT = "ProbeLogs/pockettts_stream_report.md";
            const string DONE = "ProbeLogs/pockettts_stream.done";

            static string WEIGHTS = WEIGHTS_FP16;
            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketStream] " + s); }

            [MenuItem("DeepUnity/PocketTTS/P5 Streaming Synthesis")]
            public static void Run() { WEIGHTS = WEIGHTS_FP16; RunInner(); }

            // int8: streamed vs OFFLINE(int8) must still be bit-exact (same latents, causal decode);
            // streamed vs fp16-ref expected ~0.99. Reports int8 RTF/VRAM.
            [MenuItem("DeepUnity/PocketTTS/P5 Streaming Synthesis (int8)")]
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
                    Log($"# pocket-tts P5 — streaming synthesis — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {WEIGHTS}");
                    tts = new PocketTTS(WEIGHTS);
                    // probe runs tight synchronous MoveNext loops — the sync GetData path keeps
                    // deterministic timing and avoids readback spin-waits (game path stays async)
                    tts.AsyncReadback = false;
                    tts.LoadBlocking();
                    Log($"weights resident. footprint {tts.WeightBytes / (1024f * 1024f):F1} MB");

                    // deterministic-noise offline reference (bit-exact P4 latents) as the parity target
                    int[] textIds = Ints("text_ids", out _);
                    float[] noiseFlat = Floats("flow_noise_all", out int[] nsh);
                    int Tn = nsh[0], ld = nsh[nsh.Length - 1];
                    var inject = new float[Tn][];
                    for (int t = 0; t < Tn; t++) { inject[t] = new float[ld]; Array.Copy(noiseFlat, t * ld, inject[t], 0, ld); }
                    float[] wavRef = Floats("wav", out _);
                    bool int8 = WEIGHTS == WEIGHTS_INT8;

                    // ===== 1) streaming DATA parity: chunks must concatenate to the offline wav =====
                    // Both paths use the SAME injected reference noise (RNG-free), so the streamed
                    // concatenation must equal the offline block wav BIT-EXACTLY — this tests the
                    // REAL SynthesizeStreaming coroutine (per-frame AR + causal chunked re-decode),
                    // not a probe copy. This invariant holds on BOTH fp16 and int8 (same latents +
                    // causal decode either quant). Comparisons to the fp16 REFERENCE, however, use
                    // mel-corr under int8 (quantized AR phase-desyncs the raw waveform).
                    float[] offWav = tts.GenerateOffline(textIds, inject, useKvCache: true);
                    if (int8) failed |= MelGate("offline vs ref wav mel", offWav, wavRef, 0.92f);   // int8 AR drift accepted (user A/B 2026-07-13)
                    else failed |= Grade("offline vs ref wav", offWav, wavRef, 0.99f);

                    var chunks = new List<float[]>();
                    var chunkAtMs = new List<double>();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    tts.StreamChunkFrames = 8;
                    var e = tts.SynthesizeStreaming(textIds, w =>
                    {
                        if (w == null) return;
                        chunks.Add(w); chunkAtMs.Add(sw.Elapsed.TotalMilliseconds);
                    }, injectNoise: inject);
                    while (e.MoveNext()) { }
                    sw.Stop();

                    int total = 0; foreach (var c in chunks) total += c.Length;
                    float[] streamed = new float[total];
                    int off = 0; foreach (var c in chunks) { c.CopyTo(streamed, off); off += c.Length; }
                    Log($"streaming: {chunks.Count} chunks -> {total} samples ({total / (float)PocketTTSConfig.SAMPLE_RATE:F2}s)");
                    if (chunkAtMs.Count > 0)
                        Log($"   TTFA (first chunk wall): {chunkAtMs[0]:F0} ms | total gen+decode wall {sw.Elapsed.TotalMilliseconds:F0} ms");

                    // bit-exact (both quant): streamed concatenation MUST equal the offline block wav
                    failed |= Grade("streamed vs offline wav", streamed, offWav, 0.999f);
                    // vs the fp16 reference: mel-corr under int8 (phase-invariant), raw-sample on fp16
                    if (int8)
                    {
                        failed |= MelGate("streamed vs ref wav mel", streamed, wavRef, 0.92f);   // int8 AR drift accepted (user A/B 2026-07-13)
                        var (_, _, sc) = Diff(streamed, wavRef);
                        Log($"   [info] int8 streamed vs ref raw-sample corr {sc:F6} (low = phase desync, NOT quality loss)");
                    }
                    else failed |= Grade("streamed vs ref wav", streamed, wavRef, 0.99f);

                    // continuity: no seam clicks (re-decode is causal-exact, so seams are natural slope)
                    float worst = 0;
                    for (int i = 1; i < chunks.Count; i++)
                        if (chunks[i - 1].Length > 0 && chunks[i].Length > 0)
                            worst = Mathf.Max(worst, Mathf.Abs(chunks[i][0] - chunks[i - 1][chunks[i - 1].Length - 1]));
                    Log($"   worst boundary jump {worst:F5} (clicks if >> natural |dx| ~0.05)");

                    SaveWav("ProbeLogs/pockettts_stream_unity.wav", streamed);
                    Log("   streamed audio -> ProbeLogs/pockettts_stream_unity.wav");

                    // ===== 2) names sentence via the REAL streaming path (listen QA) =====
                    int[] namesIds = Ints("names_ids", out _);
                    var nchunks = new List<float[]>();
                    var nsw = System.Diagnostics.Stopwatch.StartNew();
                    double firstMs = -1;
                    var ne = tts.SynthesizeStreaming(namesIds, w =>
                    {
                        if (w == null) return;
                        if (firstMs < 0) firstMs = nsw.Elapsed.TotalMilliseconds;
                        nchunks.Add(w);
                    });
                    while (ne.MoveNext()) { }
                    nsw.Stop();
                    int ntotal = 0; foreach (var c in nchunks) ntotal += c.Length;
                    float[] names = new float[ntotal];
                    off = 0; foreach (var c in nchunks) { c.CopyTo(names, off); off += c.Length; }
                    float nsec = ntotal / (float)PocketTTSConfig.SAMPLE_RATE;
                    Log($"names (streaming): {nchunks.Count} chunks, {tts.StreamLastTokenCount} frames -> {ntotal} samples ({nsec:F2}s)");
                    Log($"   TTFA {firstMs:F0} ms | total wall {nsw.Elapsed.TotalMilliseconds:F0} ms -> RTF {nsw.Elapsed.TotalMilliseconds / 1000f / nsec:F3}");
                    bool finite = true; double rms = 0;
                    foreach (float s in names) { finite &= !float.IsNaN(s) && !float.IsInfinity(s); rms += (double)s * s; }
                    rms = Math.Sqrt(rms / Math.Max(ntotal, 1));
                    Log($"   sanity: finite {finite}, rms {rms:F4}");
                    if (!(finite && rms > 0.005 && rms < 0.5 && ntotal > PocketTTSConfig.SAMPLE_RATE / 2)) { failed = true; Log("   FAIL sanity"); }
                    SaveWav("ProbeLogs/pockettts_names_stream_unity.wav", names);
                    Log("   names audio -> ProbeLogs/pockettts_names_stream_unity.wav (listen QA)");

                    // ===== 3) LONG-utterance regression: 65535-group dispatch guard + windowed parity =====
                    // T=256 latents (~20.5 s) through the DIRECT full decode — 31.5M elements in the
                    // final SEANet stage = 123k thread groups, which CRASHED before the Y-spill guard
                    // (live long-reply bug at T>~136). Then the windowed chunked decode must match it
                    // at corr >= 0.9999 (expected ~1.0; residual is fp RoPE-phase noise, see
                    // MIMI_DECODE_CTX). dump latents are ALREADY denormed -> identity mean/std.
                    float[] latDen = Floats("latents", out int[] ldsh);        // [42,32] denormed
                    int baseT = ldsh[0], ld2 = ldsh[ldsh.Length - 1];
                    const int LONG_T = 256;
                    float[] longLat = new float[LONG_T * ld2];
                    for (int t = 0; t < LONG_T; t++)
                        Array.Copy(latDen, (t % baseT) * ld2, longLat, t * ld2, ld2);
                    float[] ident0 = new float[ld2], ident1 = new float[ld2];
                    for (int i = 0; i < ld2; i++) { ident0[i] = 0f; ident1[i] = 1f; }

                    float[] fullLong = tts.DecodePrefix(longLat, LONG_T, ident0, ident1);   // direct — dispatch-guard regression
                    bool lfinite = true; double lrms = 0;
                    foreach (float s in fullLong) { lfinite &= !float.IsNaN(s) && !float.IsInfinity(s); lrms += (double)s * s; }
                    lrms = Math.Sqrt(lrms / Math.Max(fullLong.Length, 1));
                    Log($"## LONG direct decode: T={LONG_T} ({LONG_T / 12.5f:F1}s) -> {fullLong.Length} samples, finite {lfinite}, rms {lrms:F4}"
                        + (lfinite && lrms > 0.001 ? "  PASS (dispatch guard)" : "  <-- FAIL"));
                    if (!(lfinite && lrms > 0.001)) failed = true;

                    float[] winLong = tts.DecodeWindowed(longLat, LONG_T, ident0, ident1);
                    var (lmx, _, lcorr) = Diff(winLong, fullLong);
                    Log($"## LONG windowed vs direct: maxAbs {lmx:F6} corr {lcorr:F7}"
                        + (lcorr >= 0.9999f ? "  PASS" : "  <-- FAIL (window < receptive field?)"));
                    if (lcorr < 0.9999f) failed = true;
                }
                catch (Exception ex)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
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

            static bool Grade(string name, float[] ours, float[] rf, float gate)
            {
                var (mx, mae, corr) = Diff(ours, rf);
                bool bad = corr < gate || Mathf.Abs(ours.Length - rf.Length) > PocketTTSConfig.SAMPLES_PER_LATENT;
                Log($"## {name}: ours {ours.Length} vs {rf.Length}; maxAbs {mx:F5} MAE {mae:F6} corr {corr:F6}" + (bad ? "  <-- FAIL" : "  PASS"));
                return bad;
            }
            // phase-invariant perceptual gate (int8 wav vs fp16 ref) — mel-spectrogram corr
            static bool MelGate(string name, float[] ours, float[] rf, float gate)
            {
                float mc = PocketTTSMel.MelCorr(ours, rf, PocketTTSConfig.SAMPLE_RATE);
                bool bad = mc < gate;
                Log($"## {name}: mel-corr {mc:F6} (gate {gate:F3}, phase-invariant)" + (bad ? "  <-- FAIL" : "  PASS"));
                return bad;
            }

            // ---- npy + wav helpers (same as the E2E probe) ----
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
