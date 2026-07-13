using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Frame-pacing probe for the ChatDemo3D voice stack — measures per-frame ms through:
        //   A idle baseline (rotating-cube render load)
        //   B Kokoro weight prefetch (budgeted stream)
        //   C Qwen3.5-0.8B int8 load + system-prompt prefill
        //   C2 first Kokoro synthesis (includes one-time kernel/shader warmup)
        //   D Qwen generating WHILE KokoroVoice speaks the reply (the combined game path)
        // Report: ProbeLogs/qwen_kokoro_perf.md (+ .done marker). Driven by QwenKokoroPerfRunner.
        public class QwenKokoroPerfProbe : MonoBehaviour
        {
            const string REPORT = "ProbeLogs/qwen_kokoro_perf.md";
            const string DONE = "ProbeLogs/qwen_kokoro_perf.done";

            readonly List<float> cur = new List<float>();
            readonly List<string> rows = new List<string>();
            bool collecting;
            string phase = "";

            void Update()
            {
                if (collecting) cur.Add(Time.unscaledDeltaTime * 1000f);
            }

            void Begin(string name) { phase = name; cur.Clear(); collecting = true; }

            void End()
            {
                collecting = false;
                if (cur.Count > 0) cur.RemoveAt(0);   // first delta spans the pre-phase frame
                if (cur.Count == 0) { rows.Add($"| {phase} | 0 | - | - | - | - | - |"); return; }
                var a = new List<float>(cur);
                a.Sort();
                float avg = 0f; foreach (float v in a) avg += v; avg /= a.Count;
                float p95 = a[Mathf.Min(a.Count - 1, Mathf.FloorToInt(a.Count * 0.95f))];
                float max = a[a.Count - 1];
                int over16 = 0, over33 = 0;
                foreach (float v in a) { if (v > 16.7f) over16++; if (v > 33.4f) over33++; }
                rows.Add($"| {phase} | {a.Count} | {avg:F1} | {p95:F1} | {max:F1} | {over16} | {over33} |");
                Debug.Log($"[QwenKokoroPerf] {phase}: {a.Count} frames, avg {avg:F1} ms, p95 {p95:F1}, max {max:F1}, >16.7ms {over16}, >33ms {over33}");
            }

            IEnumerator Start()
            {
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = -1;
                Directory.CreateDirectory("ProbeLogs");

                // some real render load so frame pacing is meaningful
                var root = new GameObject("Cubes");
                for (int i = 0; i < 300; i++)
                {
                    var c = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    c.transform.SetParent(root.transform, false);
                    c.transform.localPosition = new Vector3((i % 20 - 10) * 1.2f, (i / 20 % 5) * 1.2f, 8f + i / 100 * 1.5f);
                    c.AddComponent<PerfSpin>();
                }
                yield return null;

                Begin("A idle baseline");
                yield return new WaitForSecondsRealtime(3f);
                End();

                // the probe owns the engine (int8 weights, the scene's velmire_elder voice) and
                // shares it with the KokoroVoice component — same path the demo takes
                sharedRef = new KokoroTTS("Assets/Resources/Weights/weights_kokoro_int8",
                                          voice: "velmire_elder", prefetch: false);
                KokoroVoice.SetSharedTTS(sharedRef);
                var kv = gameObject.AddComponent<KokoroVoice>();
                kv.streaming = true;
                kv.loadOnStart = false;
                kv.PrefetchNow();          // binds the shared engine to the component
                Begin("B Kokoro prefetch (int8)");
                sharedRef.Prefetch();
                float t0 = Time.realtimeSinceStartup;
                while (!kv.IsReady && Time.realtimeSinceStartup - t0 < 120f) yield return null;
                End();
                if (!kv.IsReady) { Finish(true, "Kokoro never became ready"); yield break; }

                Begin("C Qwen int8 load + prefill");
                var llm = new Qwen3_5ForCausalLM(quantization: LLMQuant.INT8, kv_quant: KVQuant.INT8);
                yield return llm.InitializeChat(system_prompt:
                    "You are Velmire, the Pale Herald, a soft-spoken emissary by a ruined gate. " +
                    "Keep your replies to two or three short sentences.");
                End();

                Begin("C2 kernel prewarm (one-time compiles)");
                kv.PrewarmKernels();
                yield return null;                       // let the prewarm coroutine start
                t0 = Time.realtimeSinceStartup;
                // the prewarm runs a tiny discarded synthesis through every kernel path
                kv.FeedText("The gate remembers every traveller, little lambkin. ");
                kv.FlushText();
                while (Time.realtimeSinceStartup - t0 < 60f)
                {
                    yield return null;
                    if (!kv.IsSpeaking && Time.realtimeSinceStartup - t0 > 1.5f) break;
                }
                End();

                // D split three ways to attribute the spikes: LLM alone, TTS alone, stacked
                Begin("D1 Qwen generate (silent)");
                yield return llm.Chat("Tell me of the sentinel beyond the mist.", max_new_tokens: 72,
                                      temperature: 0.8f, onTokenGenerated: _ => { });
                End();

                Begin("D2 Kokoro speak alone");
                kv.FeedText("The sentinel does not sleep, little lambkin. It waits in the gold haze, " +
                            "counting the bones of those who knocked before you. Knock softly. ");
                kv.FlushText();
                t0 = Time.realtimeSinceStartup;
                while (Time.realtimeSinceStartup - t0 < 90f)
                {
                    yield return null;
                    if (!kv.IsSpeaking && Time.realtimeSinceStartup - t0 > 1.5f) break;
                }
                End();

                Begin("D3 Qwen generate + Kokoro speak (combined)");
                bool chatDone = false;
                StartCoroutine(Chat(llm, kv, () => chatDone = true));
                t0 = Time.realtimeSinceStartup;
                while ((!chatDone || kv.IsSpeaking) && Time.realtimeSinceStartup - t0 < 180f) yield return null;
                End();

                llm.Release();

                // E: offline int8 listen artifact (same engine) — saved for human QA
                float[] wavOut = null;
                yield return sharedRef.Synthesize(
                    "The mist keeps what it takes, little lambkin. Walk softly, and it may only keep your name.",
                    w => wavOut = w);
                if (wavOut != null) SaveWav("ProbeLogs/kokoro_int8_velmire_elder.wav", wavOut);

                Finish(false, null);
            }

            KokoroTTS sharedRef;

            static void SaveWav(string path, float[] samples)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var w = new BinaryWriter(fs);
                int byteLen = samples.Length * 2;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + byteLen);
                w.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16);
                w.Write((short)1); w.Write((short)1); w.Write(KokoroTTS.SAMPLE_RATE); w.Write(KokoroTTS.SAMPLE_RATE * 2);
                w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(byteLen);
                foreach (float s in samples)
                    w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
            }

            IEnumerator Chat(LLM llm, KokoroVoice kv, Action done)
            {
                yield return llm.Chat("Greet me, then tell me what waits beyond the wall of golden mist.",
                    max_new_tokens: 96, temperature: 0.8f,
                    onTokenGenerated: t => kv.FeedText(t));
                kv.FlushText();
                done();
            }

            void Finish(bool failed, string why)
            {
                var sb = new StringBuilder();
                sb.AppendLine($"# Qwen3.5-0.8B int8 + Kokoro bm_george — frame pacing — {DateTime.Now:yyyy-MM-dd HH:mm}");
                sb.AppendLine($"vsync off, uncapped; 300 spinning cubes; {SystemInfo.graphicsDeviceName}");
                sb.AppendLine();
                sb.AppendLine("| phase | frames | avg ms | p95 ms | max ms | >16.7ms | >33ms |");
                sb.AppendLine("|---|---|---|---|---|---|---|");
                foreach (string r in rows) sb.AppendLine(r);
                if (failed) sb.AppendLine($"\n**FAIL**: {why}");
                File.WriteAllText(REPORT, sb.ToString());
                File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                Debug.Log("[QwenKokoroPerf] report -> " + REPORT);
            }
        }

        public class PerfSpin : MonoBehaviour
        {
            void Update() => transform.Rotate(31f * Time.deltaTime, 47f * Time.deltaTime, 0f);
        }
    }
}
