using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    namespace STTValidation
    {
        // End-to-end GPU validation for the STT ports (QwenASR 0.6b/1.7b + Parakeet TDT v2/v3).
        // Runs the REAL GPU pipeline (weights streamed to VRAM, shaders dispatched) on the D0
        // reference clips and compares the produced transcript to reference_dumps/.../transcript.txt.
        // The CPU twins are already parity-green vs the python dumps (validation/harness); this
        // probe proves the GPU dispatch path itself (shader compile + kernel bindings — the class
        // of bug the CosyVoice port taught us fails SILENTLY) reproduces those transcripts.
        //
        // Play-mode probe (real coroutines + AsyncGPUReadback + background Tasks). Driven by
        // SttGpuProbeRunner; writes ProbeLogs/stt_gpu_report.md + .done (PASS/FAIL).
        public class SttGpuProbe : MonoBehaviour
        {
            const string QROOT = "Assets/DeepUnity/InferenceEngine/STT/QwenASR";
            const string PROOT = "Assets/DeepUnity/InferenceEngine/STT/Parakeet";
            const string REPORT = "ProbeLogs/stt_gpu_report.md";
            const string DONE = "ProbeLogs/stt_gpu.done";
            static readonly string[] CLIPS = { "clip1_hello", "clip2_numbers", "clip3_game" };

            readonly StringBuilder report = new StringBuilder();
            bool failed;

            void Log(string s) { report.AppendLine(s); Debug.Log("[SttGpu] " + s); }

            void Start() => StartCoroutine(Run());

            IEnumerator Run()
            {
                Directory.CreateDirectory("ProbeLogs");
                Log($"# STT GPU end-to-end — {DateTime.Now:yyyy-MM-dd HH:mm}");
                Log($"device: {SystemInfo.graphicsDeviceName}");
                Log("");

                // ---------------- Qwen3-ASR (0.6b + 1.7b) ----------------
                foreach (QwenASRSize size in new[] { QwenASRSize.B0_6, QwenASRSize.B1_7 })
                {
                    string label = QwenASRModeling.QwenASRConfig.SizeLabel(size);
                    Log($"## Qwen3-ASR {label}");
                    QwenASRSTT stt = null;
                    Exception ctorEx = null;
                    try { stt = new QwenASRSTT(size); } catch (Exception e) { ctorEx = e; }
                    if (ctorEx != null) { failed = true; Log($"- **CTOR FAILED**: {ctorEx.Message}"); Log(""); continue; }

                    yield return LoadAndTranscribe(
                        stt, $"{QROOT}/validation/reference_dumps/{label}",
                        (samples, cb) => stt.Transcribe(samples, cb));
                    stt.Release();
                    Log("");
                }

                // ---------------- Parakeet-TDT (v2 + v3) ----------------
                foreach (ParakeetModeling.ParakeetVariant v in
                         new[] { ParakeetModeling.ParakeetVariant.V2, ParakeetModeling.ParakeetVariant.V3 })
                {
                    string label = v == ParakeetModeling.ParakeetVariant.V3 ? "v3" : "v2";
                    Log($"## Parakeet-TDT 0.6b {label}");
                    ParakeetSTT stt = null;
                    Exception ctorEx = null;
                    try { stt = new ParakeetSTT(v); } catch (Exception e) { ctorEx = e; }
                    if (ctorEx != null) { failed = true; Log($"- **CTOR FAILED**: {ctorEx.Message}"); Log(""); continue; }

                    yield return LoadAndTranscribe(
                        stt, $"{PROOT}/validation/reference_dumps/{label}",
                        (samples, cb) => stt.Transcribe(samples, cb));
                    stt.Release();
                    Log("");
                }

                Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                File.WriteAllText(REPORT, report.ToString());
                File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
            }

            IEnumerator LoadAndTranscribe(STT stt, string dumpRoot,
                                          Func<float[], Action<string>, IEnumerator> transcribe)
            {
                // stream weights to VRAM as fast as the budget allows
                stt.Prefetch();
                stt.LoadBudgetBytesPerFrame = 256L * 1024 * 1024;
                var loadSw = System.Diagnostics.Stopwatch.StartNew();
                float timeout = 60f, t0 = Time.realtimeSinceStartup;
                while (!stt.IsReady)
                {
                    if (Time.realtimeSinceStartup - t0 > timeout)
                    { failed = true; Log("- **LOAD TIMEOUT** (weights never became ready)"); yield break; }
                    yield return null;
                }
                loadSw.Stop();
                Log($"- weights resident in {loadSw.ElapsedMilliseconds} ms " +
                    $"({stt.TotalWeightBytes / (1024 * 1024)} MB)");

                foreach (string clip in CLIPS)
                {
                    string wavPath = $"{Path.GetDirectoryName(Path.GetDirectoryName(dumpRoot))}";
                    // clips live at <module>/validation/clips/<clip>.wav
                    string clipsDir = Directory.GetParent(dumpRoot).Parent.FullName;
                    string wav = Path.Combine(clipsDir, "clips", clip + ".wav");
                    string refPath = Path.Combine(dumpRoot, clip, "transcript.txt");
                    if (!File.Exists(wav) || !File.Exists(refPath))
                    { failed = true; Log($"- {clip}: **MISSING** clip or reference ({wav})"); continue; }

                    float[] samples = LoadWav16kMono(wav);
                    // reference transcript.txt may carry the sentence on repeated lines (a NeMo
                    // reference-dump quirk on the synthetic clips); the harness compares line[0]
                    // only — match that convention.
                    string[] refLines = File.ReadAllLines(refPath);
                    string expected = (refLines.Length > 0 ? refLines[0] : "").Trim();

                    string got = null; bool done = false; Exception ex = null;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    IEnumerator it = null;
                    try { it = transcribe(samples, t => { got = t; done = true; }); }
                    catch (Exception e) { ex = e; }
                    if (ex == null)
                    {
                        while (true)
                        {
                            bool move;
                            try { move = it.MoveNext(); }
                            catch (Exception e) { ex = e; break; }
                            if (!move) break;
                            yield return it.Current;
                        }
                    }
                    sw.Stop();

                    if (ex != null) { failed = true; Log($"- {clip}: **EXCEPTION** {ex.Message}"); continue; }
                    if (!done || got == null) { failed = true; Log($"- {clip}: **NO OUTPUT**"); continue; }

                    string a = Normalize(got), b = Normalize(expected);
                    bool exact = got.Trim() == expected;
                    bool normMatch = a == b;
                    float audioSec = samples.Length / 16000f;
                    float rtf = sw.ElapsedMilliseconds / 1000f / Mathf.Max(0.001f, audioSec);
                    if (!exact && !normMatch) failed = true;
                    string verdict = exact ? "EXACT MATCH" : normMatch ? "match (punct/case only)" : "**MISMATCH**";
                    Log($"- {clip}: {verdict} | {sw.ElapsedMilliseconds} ms, RTF {rtf:F2}");
                    Log($"    got: \"{got.Trim()}\"");
                    if (!exact) Log($"    exp: \"{expected}\"");
                }
            }

            static string Normalize(string s)
            {
                var sb = new StringBuilder();
                foreach (char c in s.ToLowerInvariant())
                    if (char.IsLetterOrDigit(c) || c == ' ') sb.Append(c);
                return string.Join(" ", sb.ToString().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));
            }

            static float[] LoadWav16kMono(string path)
            {
                byte[] b = File.ReadAllBytes(path);
                int pos = 12;
                while (pos < b.Length - 8)
                {
                    string id = Encoding.ASCII.GetString(b, pos, 4);
                    int size = BitConverter.ToInt32(b, pos + 4);
                    if (id == "data")
                    {
                        int n = size / 2;
                        float[] r = new float[n];
                        for (int i = 0; i < n; i++)
                            r[i] = BitConverter.ToInt16(b, pos + 8 + 2 * i) / 32768f;
                        return r;
                    }
                    pos += 8 + size + (size & 1);
                }
                throw new IOException($"no data chunk in {path}");
            }
        }
    }
}
