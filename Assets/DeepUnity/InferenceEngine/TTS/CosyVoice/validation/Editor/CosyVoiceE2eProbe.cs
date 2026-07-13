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
        // A4 probe — offline end-to-end CosyVoiceTTS. Editor-mode synchronous (ClaudeBridge).
        //
        // Gates:
        //   tokenizer   EncodeIds(reference ZH text) == text_tokens dump (EXACT)
        //   e2e         Synthesize(seed 0) produces sane audio: length within the LM min/max
        //               token budget, finite samples, plausible RMS; wav saved for listening
        //   perf        LM tok/s + flow/vocoder ms + total RTF report
        public static class CosyVoiceE2eProbe
        {
            // TTS_TEXT of validation/dump_reference.py (official ZH example — voice is ZH-baked)
            const string TTS_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。";
            const string DUMP_DIR = "Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/validation/dump";
            const string REPORT = "ProbeLogs/cosyvoice_e2e_report.md";
            const string DONE = "ProbeLogs/cosyvoice_e2e.done";

            static readonly StringBuilder report = new StringBuilder();

            static void Log(string line)
            {
                report.AppendLine(line);
                Debug.Log("[CosyVoiceE2E] " + line);
            }

            [MenuItem("DeepUnity/CosyVoice/A4 E2E Synthesis")]
            public static void Run()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                CosyVoiceTTS tts = null;
                try
                {
                    Log($"# CosyVoice3 A4 — offline end-to-end — {DateTime.Now:yyyy-MM-dd HH:mm}");

                    // ---- tokenizer exact-match gate
                    var tok = new CosyVoiceTokenizer();
                    int[] ours = tok.EncodeIds(TTS_TEXT);
                    int[] rf = Ints("text_tokens");
                    bool exact = ours.Length == rf.Length;
                    if (exact) for (int i = 0; i < ours.Length; i++) exact &= ours[i] == rf[i];
                    Log($"## Tokenizer: {(exact ? "EXACT MATCH" : "MISMATCH")} (ours {ours.Length} vs ref {rf.Length} ids)");
                    if (!exact)
                    {
                        failed = true;
                        Log($"   ours: [{string.Join(",", ours)}]");
                        Log($"   ref:  [{string.Join(",", rf)}]");
                    }

                    // ---- e2e synthesis (seeded)
                    var swLoad = System.Diagnostics.Stopwatch.StartNew();
                    tts = new CosyVoiceTTS(beginLoad: false) { Seed = 0 };
                    tts.LoadBlocking();
                    swLoad.Stop();
                    Log($"weights resident (blocking, {swLoad.Elapsed.TotalSeconds:F1}s; {tts.TotalWeightBytes / 1e6:F0} MB).");

                    float[] wav = null;
                    var swAll = System.Diagnostics.Stopwatch.StartNew();
                    IEnumerator e = tts.Synthesize(TTS_TEXT, w => wav = w);
                    while (e.MoveNext()) { }
                    swAll.Stop();

                    if (wav == null) { failed = true; Log("## E2E: SYNTHESIS FAILED (null wav)"); }
                    else
                    {
                        float sec = wav.Length / (float)CosyVoiceConfig.SAMPLE_RATE;
                        double rms = 0; bool finite = true;
                        foreach (float s in wav) { rms += (double)s * s; finite &= !float.IsNaN(s) && !float.IsInfinity(s); }
                        rms = Math.Sqrt(rms / wav.Length);
                        int minTok = (int)(rf.Length * CosyVoiceConfig.MIN_TOKEN_TEXT_RATIO);
                        Log($"## E2E: {tts.LastTokenCount} speech tokens -> {wav.Length} samples ({sec:F2}s), RMS {rms:F4}");
                        bool sane = finite && tts.LastTokenCount >= minTok && rms > 0.005 && rms < 0.5;
                        if (!sane)
                        {
                            failed = true;
                            Log($"   FAIL sanity (finite {finite}, tokens >= {minTok}: {tts.LastTokenCount >= minTok}, rms in (0.005,0.5): {rms:F4})");
                        }
                        SaveWav("ProbeLogs/cosyvoice_e2e_unity.wav", wav, CosyVoiceConfig.SAMPLE_RATE);
                        float total = (float)swAll.Elapsed.TotalSeconds;
                        Log($"   [perf] LM {tts.LmMs:F0} ms ({tts.TokensPerSecond:F1} tok/s) | flow {tts.FlowMs:F0} ms | " +
                            $"vocoder {tts.VocoderMs:F0} ms | TOTAL {total:F1}s for {sec:F2}s audio -> RTF {total / sec:F2}");
                        Log("   audio -> ProbeLogs/cosyvoice_e2e_unity.wav (listen QA)");
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

            // A7 — the baked deep-EN "velmire" voice (make_voice.py, Kokoro am_onyx prompt).
            [MenuItem("DeepUnity/CosyVoice/A7 Velmire Voice Test")]
            public static void RunVelmire()
            {
                Directory.CreateDirectory("ProbeLogs");
                CosyVoiceTTS tts = null;
                try
                {
                    tts = new CosyVoiceTTS(voice: "velmire", beginLoad: false) { Seed = 1 };
                    tts.LoadBlocking();
                    float[] wav = null;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    IEnumerator e = tts.Synthesize(
                        "Ah... another lambkin wanders to my gate. Do come closer, little one. The dark beyond these walls has been so very patient with you.",
                        w => wav = w);
                    while (e.MoveNext()) { }
                    sw.Stop();
                    if (wav == null) { Debug.LogError("[Velmire] synthesis FAILED"); return; }
                    float sec = wav.Length / (float)CosyVoiceConfig.SAMPLE_RATE;
                    SaveWav("ProbeLogs/velmire_test.wav", wav, CosyVoiceConfig.SAMPLE_RATE);
                    Debug.Log($"[Velmire] {tts.LastTokenCount} tokens -> {sec:F2}s audio in {sw.Elapsed.TotalSeconds:F1}s " +
                              $"(LM {tts.TokensPerSecond:F1} tok/s) -> ProbeLogs/velmire_test.wav");
                }
                finally { tts?.Release(); }
            }

            // ---------------- helpers ----------------
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
                foreach (string s in shapeStr.Split(','))
                    if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
                if (dims.Count == 0) dims.Add(1);
                shape = dims.ToArray();
                long count = 1; foreach (int d in shape) count *= d;
                if (header.Contains("f4")) { float[] r = new float[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4); return r; }
                if (header.Contains("i8")) { long[] r = new long[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 8); return r; }
                if (header.Contains("i4")) { int[] r = new int[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4); return r; }
                throw new Exception($"unsupported npy dtype in {path}");
            }

            static int[] Ints(string name)
            {
                Array a = LoadNpy(Path.Combine(DUMP_DIR, name + ".npy"), out _);
                if (a is int[] i) return i;
                if (a is long[] l) { int[] r = new int[l.Length]; for (int j = 0; j < l.Length; j++) r[j] = (int)l[j]; return r; }
                float[] f = (float[])a;
                int[] r2 = new int[f.Length];
                for (int j = 0; j < f.Length; j++) r2[j] = (int)Math.Round(f[j]);
                return r2;
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
