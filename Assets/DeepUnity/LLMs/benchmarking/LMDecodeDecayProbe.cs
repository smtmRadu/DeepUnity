using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Decode-speed decay probe. Boots ONE model (kind + quant), warms up, then decodes continuously
    // from an empty KV cache up to `maxTokens`, timing every single token. As the cache fills, each
    // step attends over more keys/values (all FP32 — KV is not quantized; see LLMQuant docs), so
    // tok/s drifts down with context length. The probe records the per-token curve and buckets it so
    // you can see the decay: tok/s at position ~0 vs near the limit, and the % drop.
    //
    // Decoding is greedy and the generated text is irrelevant — only the timing matters, so EOS is
    // ignored and the loop keeps feeding tokens until maxTokens. Yields every `yieldEvery` steps so
    // the editor stays responsive; the per-token stopwatch wraps only Forward+Sample, so the yields
    // don't pollute the timing. Drive via LMBenchmarkProbeRunner. One model kind per editor run.
    //
    // maxTokens is capped by the model's KV capacity (maxModelLength): Qwen3.5 = 8192, Gemma3 = 2048.
    // Keep maxTokens safely under that (the runner sets a lower default for Gemma).
    public class LMDecodeDecayProbe : MonoBehaviour
    {
        [Header("What to boot (one model per editor run)")]
        public ProbeModelKind model = ProbeModelKind.Qwen3_5_0_8B;
        public LLMQuant quant = LLMQuant.FP16;

        [Header("Decode")]
        public int maxTokens = 4096;   // total tokens to decode; keep under model KV capacity
        public int bucketSize = 128;   // tokens per reported bucket (decay granularity)
        public int warmupTokens = 32;  // untimed steps before the timed run (stabilize dispatch)
        public int yieldEvery = 16;    // hand a frame back every N decoded tokens

        [Header("Output / safety")]
        public string reportDirectory;
        public float timeoutSeconds = 900f;

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

        IEnumerator Run()
        {
            Status($"constructing {LMProbeCommon.ModelLabel(model)} {quant}");
            LLM lm = LMProbeCommon.Build(model, quant);
            while (!lm.IsReady) yield return null;

            Status("warmup (kernel compile)");
            yield return lm.Warmup();

            // Bind one decode step on the concrete inner model: forward the current token (append to
            // cache), sample greedily, return the next token. ResetCache clears the KV between runs.
            Action reset;
            Func<int, int> step;
            switch (model)
            {
                case ProbeModelKind.Qwen3_5_0_8B:
                {
                    var m = ((Qwen3_5ForCausalLM)lm).model;
                    reset = () => m.ResetCache();
                    step = t => { m.Forward(Tensor.Constant((float)t), useCache: true, lastPosOnly: true); return m.SampleGreedy(); };
                    break;
                }
                case ProbeModelKind.Gemma3_270M:
                {
                    var m = ((Gemma3ForCausalLM)lm).model;
                    reset = () => m.ResetCache();
                    step = t => { m.Forward(Tensor.Constant((float)t), useCache: true, lastPosOnly: true); return m.SampleGreedy(); };
                    break;
                }
                default: throw new ArgumentOutOfRangeException();
            }

            int seed = 100;

            // Untimed warmup decode (kernels already compiled by Warmup; this stabilizes first-touch).
            Status($"warmup decode x{warmupTokens}");
            reset();
            int tok = seed;
            for (int i = 0; i < warmupTokens; i++) { tok = step(tok); if (i % yieldEvery == 0) yield return null; }

            // Timed run from a clean cache so positions count from 0.
            Status($"decode x{maxTokens} (timed)");
            reset();
            tok = seed;
            var perToken = new double[maxTokens];
            var sw = new System.Diagnostics.Stopwatch();
            for (int i = 0; i < maxTokens; i++)
            {
                sw.Restart();
                tok = step(tok);
                perToken[i] = sw.Elapsed.TotalMilliseconds;
                if (i % yieldEvery == 0) yield return null;
            }

            lm.Release();
            WriteReport(true, perToken, null);
            Status($"DONE — report at {reportDirectory}");
            Exit(0);
        }

        struct Bucket { public int start, end; public double meanMs, tokS; }

        List<Bucket> Bucketize(double[] perToken)
        {
            var buckets = new List<Bucket>();
            for (int s = 0; s < perToken.Length; s += bucketSize)
            {
                int e = Math.Min(s + bucketSize, perToken.Length);
                double mean = 0;
                for (int i = s; i < e; i++) mean += perToken[i];
                mean /= (e - s);
                buckets.Add(new Bucket { start = s, end = e, meanMs = mean, tokS = mean > 0 ? 1000.0 / mean : 0 });
            }
            return buckets;
        }

        void WriteReport(bool success, double[] perToken, string note)
        {
            finished = true;
            var prevCulture = System.Threading.Thread.CurrentThread.CurrentCulture;
            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            try
            {
                Directory.CreateDirectory(reportDirectory);
                var buckets = perToken != null && perToken.Length > 0 ? Bucketize(perToken) : new List<Bucket>();

                // per-token CSV
                if (perToken != null)
                {
                    var csv = new StringBuilder("token_index,ms,tok_s\n");
                    for (int i = 0; i < perToken.Length; i++)
                        csv.Append(i).Append(',').Append(perToken[i].ToString("0.000")).Append(',')
                           .Append((perToken[i] > 0 ? 1000.0 / perToken[i] : 0).ToString("0.0")).Append('\n');
                    File.WriteAllText(Path.Combine(reportDirectory, "per_token.csv"), csv.ToString());
                }

                double firstTokS = buckets.Count > 0 ? buckets[0].tokS : 0;
                double lastTokS = buckets.Count > 0 ? buckets[buckets.Count - 1].tokS : 0;
                double decayPct = firstTokS > 0 ? (1.0 - lastTokS / firstTokS) * 100.0 : 0;

                var md = new StringBuilder();
                md.AppendLine($"# Decode-speed decay — {LMProbeCommon.ModelLabel(model)} {quant}");
                md.AppendLine();
                md.AppendLine($"- success: {success}");
                md.AppendLine($"- decoded tokens: {(perToken?.Length ?? 0)} | bucket: {bucketSize}");
                if (note != null) md.AppendLine($"- note: {note}");
                md.AppendLine($"- KV cache: FP32 (not quantized) — decay is attention-over-context growth");
                if (buckets.Count > 0)
                    md.AppendLine($"- tok/s: {firstTokS:0.0} (start) → {lastTokS:0.0} (end), **{decayPct:0.0}% drop**");
                md.AppendLine();
                md.AppendLine(LMProbeCommon.SystemInfoBlock());
                md.AppendLine("## Decay by context window");
                md.AppendLine();
                md.AppendLine("| token range | mean ms | tok/s |");
                md.AppendLine("|---|---|---|");
                foreach (var b in buckets)
                    md.AppendLine($"| {b.start}–{b.end} | {b.meanMs:0.00} | {b.tokS:0.0} |");
                File.WriteAllText(Path.Combine(reportDirectory, "report.md"), md.ToString());

                // summary.json
                var js = new StringBuilder();
                js.Append("{\n");
                js.Append("  \"probe\": \"decode_decay\",\n");
                js.Append("  \"model\": ").Append(LMProbeCommon.JsonStr(LMProbeCommon.ModelLabel(model))).Append(",\n");
                js.Append("  \"quant\": ").Append(LMProbeCommon.JsonStr(quant.ToString())).Append(",\n");
                js.Append("  \"success\": ").Append(success ? "true" : "false").Append(",\n");
                if (note != null) js.Append("  \"note\": ").Append(LMProbeCommon.JsonStr(note)).Append(",\n");
                js.Append("  \"decoded_tokens\": ").Append(perToken?.Length ?? 0).Append(",\n");
                js.Append("  \"bucket_size\": ").Append(bucketSize).Append(",\n");
                js.Append("  \"kv_cache_dtype\": \"fp32\",\n");
                js.Append($"  \"start_tok_s\": {firstTokS:0.0}, \"end_tok_s\": {lastTokS:0.0}, \"decay_pct\": {decayPct:0.0},\n");
                js.Append("  \"machine\": ").Append(LMProbeCommon.MachineJson()).Append(",\n");
                js.Append("  \"buckets\": [\n");
                for (int i = 0; i < buckets.Count; i++)
                {
                    var b = buckets[i];
                    js.Append($"    {{\"start\": {b.start}, \"end\": {b.end}, \"mean_ms\": {b.meanMs:0.000}, \"tok_s\": {b.tokS:0.0}}}")
                      .Append(i < buckets.Count - 1 ? ",\n" : "\n");
                }
                js.Append("  ]\n}\n");
                File.WriteAllText(Path.Combine(reportDirectory, "summary.json"), js.ToString());

                Debug.Log($"[LMDecodeDecayProbe] report written to {reportDirectory}");
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

        static string StatusPath => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "decode_decay_probe_status.txt");
        void Status(string s)
        {
            try { Directory.CreateDirectory(Path.GetDirectoryName(StatusPath)); File.WriteAllText(StatusPath, $"[{DateTime.Now:HH:mm:ss}] {s}"); }
            catch { }
            Debug.Log("[LMDecodeDecayProbe] " + s);
        }

        static void Exit(int code)
        {
#if UNITY_EDITOR
            if (Application.isBatchMode) UnityEditor.EditorApplication.Exit(code);
            else UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit(code);
#endif
        }
    }
}
