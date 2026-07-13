using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // A5 probe — token-level streaming synthesis. Editor-mode synchronous (ClaudeBridge).
        // Measures TTFA (wall time to the first emitted samples), per-chunk timing, and audio
        // sanity (finite, continuous at chunk boundaries, plausible length vs token count).
        public static class CosyVoiceStreamProbe
        {
            const string TTS_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。";
            const string REPORT = "ProbeLogs/cosyvoice_stream_report.md";
            const string DONE = "ProbeLogs/cosyvoice_stream.done";

            static readonly StringBuilder report = new StringBuilder();

            static void Log(string line)
            {
                report.AppendLine(line);
                Debug.Log("[CosyVoiceStream] " + line);
            }

            static string paramsPath = "Assets/Resources/Weights/weights_cosyvoice3_fp16";
            static bool legacyPath = false;
            static bool gpuRas = false;

            // A6 — the same streaming run on int8 weights: TTFA/RTF benchmark
            [MenuItem("DeepUnity/CosyVoice/A6 Streaming INT8")]
            public static void RunInt8()
            {
                paramsPath = "Assets/Resources/Weights/weights_cosyvoice3_int8";
                try { Run(); } finally { paramsPath = "Assets/Resources/Weights/weights_cosyvoice3_fp16"; }
            }

            // A6-max A/B — the pre-A6 baseline (full re-solve + full re-vocode per chunk).
            // Same seed/tokens as the default run; compare RTF/TTFA and boundary jump.
            [MenuItem("DeepUnity/CosyVoice/A6-MAX Streaming LEGACY baseline")]
            public static void RunLegacy()
            {
                legacyPath = true;
                try { Run(); } finally { legacyPath = false; }
            }

            // A6-max stretch — GPU RAS sampler on int8 weights: the RTF < 1.0 configuration.
            // Different RNG stream than the CPU sampler -> different token stream; SANITY GATES
            // ONLY (finite/RMS/length + listen), plus the RTF/TTFA/breakdown numbers.
            [MenuItem("DeepUnity/CosyVoice/A6-MAX Streaming GPU-RAS INT8 (stretch)")]
            public static void RunGpuRas()
            {
                paramsPath = "Assets/Resources/Weights/weights_cosyvoice3_int8";
                gpuRas = true;
                try { Run(); }
                finally { paramsPath = "Assets/Resources/Weights/weights_cosyvoice3_fp16"; gpuRas = false; }
            }

            [MenuItem("DeepUnity/CosyVoice/A5 Streaming Synthesis")]
            public static void Run()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                CosyVoiceTTS tts = null;
                try
                {
                    Log($"# CosyVoice3 A5 — token-level streaming — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {paramsPath}");
                    tts = new CosyVoiceTTS(paramsPath, beginLoad: false) { Seed = 0 };
                    tts.FastStreaming = !legacyPath;
                    tts.LowLatencyFirstChunk = !legacyPath;   // legacy = the reference 25->50->100 schedule
                    tts.UseGpuSampler = gpuRas;
                    Log($"streaming path: {(legacyPath ? "LEGACY full re-solve/re-vocode, reference schedule" : "A6-max single-pass flow + windowed vocoder + low-latency first chunk")}"
                        + (gpuRas ? " + GPU RAS sampler (different token stream — sanity gates only)" : ""));
                    tts.LoadBlocking();
                    Log("weights resident.");

                    var chunks = new List<float[]>();
                    var chunkAt = new List<double>();
                    var swAll = System.Diagnostics.Stopwatch.StartNew();
                    IEnumerator e = tts.SynthesizeStreaming(TTS_TEXT, w =>
                    {
                        chunks.Add(w);
                        chunkAt.Add(swAll.Elapsed.TotalSeconds);
                    });
                    while (e.MoveNext()) { }
                    swAll.Stop();

                    if (chunks.Count == 0) { failed = true; Log("## FAIL: no audio chunks emitted"); }
                    else
                    {
                        int total = 0; foreach (var c in chunks) total += c.Length;
                        float[] wav = new float[total];
                        int off = 0;
                        foreach (var c in chunks) { c.CopyTo(wav, off); off += c.Length; }
                        float sec = total / (float)CosyVoiceConfig.SAMPLE_RATE;

                        bool finite = true; double rms = 0;
                        foreach (float s in wav) { finite &= !float.IsNaN(s) && !float.IsInfinity(s); rms += (double)s * s; }
                        rms = Math.Sqrt(rms / Math.Max(total, 1));

                        // boundary continuity: |last(prev) - first(next)| should be waveform-smooth.
                        // Per seam we also log the local natural slope (max |dx| over the 100
                        // samples on each side): with the cross-fade the seam pair is two
                        // CONSECUTIVE samples of the previous chunk's own vocode, so jump <=
                        // local max |dx| PROVES the seam is natural waveform slope, not a click.
                        float worstJump = 0;
                        var jumpStr = new StringBuilder();
                        for (int i = 1; i < chunks.Count; i++)
                            if (chunks[i - 1].Length > 0 && chunks[i].Length > 0)
                            {
                                float[] a = chunks[i - 1], b = chunks[i];
                                float j = Mathf.Abs(b[0] - a[a.Length - 1]);
                                float ctx = 0;
                                for (int k = Math.Max(0, a.Length - 100); k + 1 < a.Length; k++)
                                    ctx = Mathf.Max(ctx, Mathf.Abs(a[k + 1] - a[k]));
                                for (int k = 0; k + 1 < Math.Min(100, b.Length); k++)
                                    ctx = Mathf.Max(ctx, Mathf.Abs(b[k + 1] - b[k]));
                                worstJump = Mathf.Max(worstJump, j);
                                jumpStr.Append(jumpStr.Length > 0 ? ", " : "")
                                       .Append(j.ToString("F4")).Append(" (nat ").Append(ctx.ToString("F4")).Append(")");
                            }

                        Log($"## Streaming: {chunks.Count} chunks, {tts.LastTokenCount} tokens -> {total} samples ({sec:F2}s), RMS {rms:F4}");
                        Log($"   TTFA (first samples): {chunkAt[0]:F2}s | total wall {swAll.Elapsed.TotalSeconds:F1}s -> RTF {swAll.Elapsed.TotalSeconds / sec:F2}");
                        float ovPct = tts.StreamChunkMs > 0 ? 100f * tts.StreamChunkDuringMs / tts.StreamChunkMs : 100f;
                        Log($"   [breakdown] prefill {tts.StreamPrefillMs:F0} ms | LM decode+sample-wait {tts.StreamLmMs:F0} ms | " +
                            $"chunk synth {tts.StreamChunkMs:F0} ms ({tts.StreamChunkDuringMs:F0} during decode / {tts.StreamChunkAfterMs:F0} after " +
                            $"= {ovPct:F0}% overlapped) | finalize {tts.StreamFinalizeMs:F0} ms | " +
                            $"other {swAll.Elapsed.TotalMilliseconds - tts.StreamPrefillMs - tts.StreamLmMs - tts.StreamChunkMs - tts.StreamFinalizeMs:F0} ms");
                        Log("   [note] D3D11 is single-queue: LM and flow GPU work execute serially regardless of CPU scheduling. " +
                            "With 'after' ~0 the CPU-side overlap is exhausted and the wall floor = TOTAL GPU time + unhidden CPU, not max(LM, flow).");
                        for (int i = 0; i < chunks.Count; i++)
                            Log($"   chunk {i}: +{chunks[i].Length / (float)CosyVoiceConfig.SAMPLE_RATE:F2}s audio @ wall {chunkAt[i]:F2}s");
                        Log($"   boundary jumps per seam: [{jumpStr}]  worst {worstJump:F4} (clicks if >> typical |dx| ~0.05)");
                        Log($"   seams cross-faded: {tts.SeamsBlended} of {chunks.Count - 1} " +
                            (tts.SeamsBlended == chunks.Count - 1 ? "(fade engaged on every boundary incl. first+finalize)" : "<-- MISMATCH, investigate"));

                        bool sane = finite && rms > 0.005 && rms < 0.5 && total > CosyVoiceConfig.SAMPLE_RATE;
                        if (!sane) { failed = true; Log($"   FAIL sanity (finite {finite}, rms {rms:F4})"); }
                        SaveWav("ProbeLogs/cosyvoice_stream_unity.wav", wav, CosyVoiceConfig.SAMPLE_RATE);
                        Log("   audio -> ProbeLogs/cosyvoice_stream_unity.wav");
                    }
                }
                catch (Exception ex)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
                }
                finally
                {
                    tts?.Release();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            static void SaveWav(string path, float[] samples, int sr)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var w = new BinaryWriter(fs);
                int byteLen = samples.Length * 2;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + byteLen);
                w.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16);
                w.Write((short)1); w.Write((short)1); w.Write(sr); w.Write(sr * 2);
                w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(byteLen);
                foreach (float s in samples)
                    w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
            }
        }
    }
}
