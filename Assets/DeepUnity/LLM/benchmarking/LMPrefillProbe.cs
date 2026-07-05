using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Prefill-speed probe. Boots ONE model (kind + quant set on the component), warms up kernels,
    // then times processing a fixed PREFILL of `prefillTokens` (2048 by default) into the KV cache
    // — the real prefill path (useCache + lastPosOnly), forced to GPU completion by reading a single
    // last-position logit (NOT the full all-positions readback Predict does, which would dwarf the
    // compute). Reports prefill tok/s = tokens / wall-seconds, plus the machine it ran on.
    //
    // Prefill is compute-bound (one big matmul-heavy forward over the whole prompt), unlike decode
    // which is memory-bandwidth-bound — so this is the natural companion axis to the decode tok/s
    // the QuantProbe already measures. Drive via LMPrefillProbeRunner. One model kind per editor run.
    public class LMPrefillProbe : MonoBehaviour
    {
        [Header("What to boot (one model per editor run)")]
        public ProbeModelKind model = ProbeModelKind.Qwen3_5_0_8B;
        public LLMQuant quant = LLMQuant.FP16;
        public KVQuant kvQuant = KVQuant.FP16;

        [Header("Prefill")]
        public int prefillTokens = 2048;
        public int prefillChunk = 32;  // tokens per Forward (production InitializeChat chunks the prompt too)
        public int warmupRuns = 2;     // untimed prefills (stabilize lazy/first-dispatch costs)
        public int timedRuns = 5;      // timed prefills; report best + median

        [Header("Output / safety")]
        public string reportDirectory;
        public float timeoutSeconds = 600f;

        readonly System.Diagnostics.Stopwatch lifeSw = System.Diagnostics.Stopwatch.StartNew();
        bool finished;

        void Start()
        {
            Application.runInBackground = true;
            StartCoroutine(Guarded());
        }

        void Update()
        {
            if (finished || !Application.isBatchMode) return;
            if (lifeSw.Elapsed.TotalSeconds > timeoutSeconds)
            {
                Status("ERROR: timeout");
                WriteReport(false, null, "(timed out)");
                Exit(2);
            }
        }

        IEnumerator Guarded()
        {
            var e = Run();
            while (true)
            {
                object cur;
                try
                {
                    if (!e.MoveNext()) break;
                    cur = e.Current;
                }
                catch (Exception ex)
                {
                    Status("ERROR: " + ex.Message + "\n" + ex.StackTrace);
                    WriteReport(false, null, ex.Message);
                    Exit(1);
                    yield break;
                }
                yield return cur;
            }
        }

        // ids well inside any model's vocab; prefill speed is token-id independent, only count matters.
        float[] SyntheticIds(int n)
        {
            var ids = new float[n];
            for (int i = 0; i < n; i++) ids[i] = 100 + (i % 20000);
            return ids;
        }

        IEnumerator Run()
        {
            Status($"constructing {LMProbeCommon.ModelLabel(model)} {quant}");
            LLM lm = LMProbeCommon.Build(model, quant, kvQuant);
            while (!lm.IsReady) yield return null;

            Status("warmup (kernel compile)");
            yield return lm.Warmup();

            // Bind the real prefill path on the concrete inner model: reset cache, forward the whole
            // prompt (lastPosOnly so only the final logits are produced), then read 1 logit row to
            // force the GPU to finish. Both inner models expose the same method shape.
            Action<float[]> prefill;
            switch (model)
            {
                case ProbeModelKind.Qwen3_5_0_8B:
                {
                    var m = ((Qwen3_5ForCausalLM)lm).model;
                    // Chunked prefill: feed the prompt in prefillChunk-token Forwards, each followed
                    // by a blocking SampleGreedy() readback (argmaxBuf.GetData). A single 2048-token
                    // Forward measured unreliably (repeated runs collapsed to ~1 ms — the giant scan
                    // dispatch wasn't being forced to complete), whereas small Forward+SampleGreedy
                    // steps time correctly (proven by the 4096-step decode-decay probe). This is also
                    // how production InitializeChat actually prefills, so it's the representative path.
                    prefill = ids =>
                    {
                        m.ResetCache();
                        for (int s = 0; s < ids.Length; s += prefillChunk)
                        {
                            int len = System.Math.Min(prefillChunk, ids.Length - s);
                            var part = new float[len];
                            System.Array.Copy(ids, s, part, 0, len);
                            m.Forward(Tensor.Constant(part), useCache: true, lastPosOnly: true);
                            m.SampleGreedy();
                        }
                    };
                    break;
                }
                case ProbeModelKind.Gemma3_270M:
                {
                    var m = ((Gemma3ForCausalLM)lm).model;
                    // Chunked prefill: feed the prompt in prefillChunk-token Forwards, each followed
                    // by a blocking SampleGreedy() readback (argmaxBuf.GetData). A single 2048-token
                    // Forward measured unreliably (repeated runs collapsed to ~1 ms — the giant scan
                    // dispatch wasn't being forced to complete), whereas small Forward+SampleGreedy
                    // steps time correctly (proven by the 4096-step decode-decay probe). This is also
                    // how production InitializeChat actually prefills, so it's the representative path.
                    prefill = ids =>
                    {
                        m.ResetCache();
                        for (int s = 0; s < ids.Length; s += prefillChunk)
                        {
                            int len = System.Math.Min(prefillChunk, ids.Length - s);
                            var part = new float[len];
                            System.Array.Copy(ids, s, part, 0, len);
                            m.Forward(Tensor.Constant(part), useCache: true, lastPosOnly: true);
                            m.SampleGreedy();
                        }
                    };
                    break;
                }
                default: throw new ArgumentOutOfRangeException();
            }

            float[] promptIds = SyntheticIds(prefillTokens);

            Status($"prefill warmup x{warmupRuns}");
            for (int i = 0; i < warmupRuns; i++) { prefill(promptIds); yield return null; }

            Status($"prefill timed x{timedRuns} ({prefillTokens} tokens)");
            var msRuns = new double[timedRuns];
            var sw = new System.Diagnostics.Stopwatch();
            for (int i = 0; i < timedRuns; i++)
            {
                sw.Restart();
                prefill(promptIds);
                msRuns[i] = sw.Elapsed.TotalMilliseconds;
                yield return null;
            }

            lm.Release();
            WriteReport(true, msRuns, null);
            Status($"DONE — report at {reportDirectory}");
            Exit(0);
        }

        void WriteReport(bool success, double[] msRuns, string note)
        {
            finished = true;
            // Force invariant culture so decimals are '.' (a comma-decimal locale would break JSON).
            var prevCulture = System.Threading.Thread.CurrentThread.CurrentCulture;
            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            try
            {
                Directory.CreateDirectory(reportDirectory);
                var md = new StringBuilder();
                md.AppendLine($"# Prefill speed — {LMProbeCommon.ModelLabel(model)} {quant}");
                md.AppendLine();
                md.AppendLine($"- success: {success}");
                md.AppendLine($"- prefill tokens: {prefillTokens}");
                if (note != null) md.AppendLine($"- note: {note}");
                md.AppendLine();
                md.AppendLine(LMProbeCommon.SystemInfoBlock());

                if (msRuns != null && msRuns.Length > 0)
                {
                    var sorted = msRuns.OrderBy(x => x).ToArray();
                    double best = sorted[0];
                    double median = sorted[sorted.Length / 2];
                    double mean = sorted.Average();
                    double Tps(double ms) => ms > 0 ? prefillTokens / (ms / 1000.0) : 0;

                    md.AppendLine("## Result");
                    md.AppendLine();
                    md.AppendLine("| metric | ms | tok/s |");
                    md.AppendLine("|---|---|---|");
                    md.AppendLine($"| best   | {best:0.0} | {Tps(best):0} |");
                    md.AppendLine($"| median | {median:0.0} | {Tps(median):0} |");
                    md.AppendLine($"| mean   | {mean:0.0} | {Tps(mean):0} |");
                    md.AppendLine();
                    md.AppendLine("### Per-run (ms)");
                    md.AppendLine("```");
                    md.AppendLine(string.Join(", ", msRuns.Select(x => x.ToString("0.0"))));
                    md.AppendLine("```");
                }

                File.WriteAllText(Path.Combine(reportDirectory, "report.md"), md.ToString());

                // summary.json — machine + headline metrics, for aggregating across runs.
                var js = new StringBuilder();
                js.Append("{\n");
                js.Append("  \"probe\": \"prefill_speed\",\n");
                js.Append("  \"model\": ").Append(LMProbeCommon.JsonStr(LMProbeCommon.ModelLabel(model))).Append(",\n");
                js.Append("  \"quant\": ").Append(LMProbeCommon.JsonStr(quant.ToString())).Append(",\n");
                js.Append("  \"kv\": ").Append(LMProbeCommon.JsonStr(kvQuant.ToString())).Append(",\n");
                js.Append("  \"success\": ").Append(success ? "true" : "false").Append(",\n");
                if (note != null) js.Append("  \"note\": ").Append(LMProbeCommon.JsonStr(note)).Append(",\n");
                js.Append("  \"prefill_tokens\": ").Append(prefillTokens).Append(",\n");
                js.Append("  \"machine\": ").Append(LMProbeCommon.MachineJson()).Append(",\n");
                if (msRuns != null && msRuns.Length > 0)
                {
                    var sorted = msRuns.OrderBy(x => x).ToArray();
                    double best = sorted[0], median = sorted[sorted.Length / 2], mean = sorted.Average();
                    double Tps(double ms) => ms > 0 ? prefillTokens / (ms / 1000.0) : 0;
                    js.Append($"  \"best_ms\": {best:0.0}, \"best_tok_s\": {Tps(best):0.0},\n");
                    js.Append($"  \"median_ms\": {median:0.0}, \"median_tok_s\": {Tps(median):0.0},\n");
                    js.Append($"  \"mean_ms\": {mean:0.0}, \"mean_tok_s\": {Tps(mean):0.0},\n");
                    js.Append("  \"runs_ms\": [").Append(string.Join(", ", msRuns.Select(x => x.ToString("0.0")))).Append("]\n");
                }
                else
                {
                    js.Append("  \"runs_ms\": []\n");
                }
                js.Append("}\n");
                File.WriteAllText(Path.Combine(reportDirectory, "summary.json"), js.ToString());

                Debug.Log($"[LMPrefillProbe] report written to {reportDirectory}");
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
            finally
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = prevCulture;
            }
        }

        static string StatusPath => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "prefill_probe_status.txt");
        void Status(string s)
        {
            try { Directory.CreateDirectory(Path.GetDirectoryName(StatusPath)); File.WriteAllText(StatusPath, $"[{DateTime.Now:HH:mm:ss}] {s}"); }
            catch { }
            Debug.Log("[LMPrefillProbe] " + s);
        }

        static void Exit(int code)
        {
#if UNITY_EDITOR
            // Batch: quit the process so the run returns. Interactive (menu): just stop play mode
            // and leave the editor open so the report can be inspected.
            if (Application.isBatchMode) UnityEditor.EditorApplication.Exit(code);
            else UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit(code);
#endif
        }
    }
}
