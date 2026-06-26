using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // A/B probe for weight-only INT8 vs FP16 on Qwen3.5-0.8B. Two sequential model instances
    // (FP16 released before INT8 boots): identical synthetic prefill + the FP16 path's greedy
    // tokens fed to both, full-vocab logit diffs per step, sync-decode tok/s, and a greedy
    // real-text Generate() sample from each so the decoding difference is visible as text.
    // Driven via FlashAttnProbeRunner.RunQwenInt8; status -> ClaudeBridge/flash_probe_status.txt.
    //
    // VISUALIZING THE AGGREGATE (paper): across the 6 configs (2 models x {FP16, INT8, INT4})
    // the axes are throughput (this probe), precision drift = KL/logit-diff from FP16 (this
    // probe; FP16 is the 0 reference), and boot time in seconds (LMBootProbe). An OVERLAID RADAR
    // works well here: min-max normalize each axis to [0,1] "goodness" (invert KL and boot so
    // outward = better), one polygon per config, color = model family, FP16/INT8/INT4 differ by
    // linestyle (dotted/dashed/solid) so it survives grayscale. Caption the normalization and
    // ship a companion table of raw numbers (radar = gestalt, table = exact values).
    //   Likely 4th+ axis: split throughput into PREFILL tok/s and DECODE tok/s as separate axes
    //   (different bottlenecks: prefill is compute-bound, decode is memory-bandwidth-bound, and
    //   quantization helps them unequally). Radar absorbs extra axes for free (it just becomes a
    //   quad/pentagon) — this is where it beats a 2D bubble plot, which caps at ~3 encodings.
    //   NOTE: this probe currently only TIMES decode (the sync-decode loop); Prefill() runs but
    //   isn't measured as tok/s — add a timed prefill pass before the prefill axis goes in.
    public class QuantProbe : MonoBehaviour
    {
        public string reportDirectory;
        public LLMQuant quant = LLMQuant.INT8;   // the format A/B'd against FP16

        const int PREFILL_TOKENS = 500;
        const int COMPARE_STEPS = 8;
        const int BENCH_TOKENS = 64;
        const int SAMPLE_TOKENS = 48;
        const int CHUNK = 8;

        readonly StringBuilder report = new StringBuilder();
        bool pass = true;

        static string StatusPath => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "flash_probe_status.txt");

        void Status(string s)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(StatusPath));
                File.WriteAllText(StatusPath, $"[{DateTime.Now:HH:mm:ss}] {s}");
            }
            catch { }
            Debug.Log("[QuantProbe] " + s);
        }

        void Start()
        {
            Application.runInBackground = true;
            StartCoroutine(Guarded());
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
                    WriteReport(false);
                    yield break;
                }
                yield return cur;
            }
        }

        static float[] SyntheticIds(int n)
        {
            var ids = new float[n];
            uint h = 2166136261u;
            for (int i = 0; i < n; i++)
            {
                h = (h ^ (uint)i) * 16777619u;
                ids[i] = 1000 + (h % 200000);
            }
            return ids;
        }

        static void Prefill(Qwen3_5ForCausalLM lm, float[] ids)
        {
            lm.model.ResetCache();
            for (int s = 0; s < ids.Length; s += CHUNK)
            {
                int len = Math.Min(CHUNK, ids.Length - s);
                float[] part = new float[len];
                Array.Copy(ids, s, part, 0, len);
                lm.model.Forward(Tensor.Constant(part), useCache: true, lastPosOnly: true);
            }
        }

        // One full pass over a model: logits at COMPARE_STEPS positions (feeding `feedTokens` if
        // given, else its own greedy choices, recorded into ownTokens), sync-decode bench, and a
        // greedy story continuation.
        IEnumerator Exercise(Qwen3_5ForCausalLM lm, float[] prompt, int[] feedTokens,
                             float[][] logitsOut, int[] ownTokens, double[] medianMsOut, string[] textOut, string label)
        {
            int V = Qwen3_5Modeling.Qwen3_5Config.VOCAB_SIZE;
            var w = lm.Warmup();
            while (w.MoveNext()) yield return w.Current;
            while (!lm.IsReady) yield return null;

            Status($"{label}: correctness prefill+decode");
            Prefill(lm, prompt);
            for (int k = 0; k < COMPARE_STEPS; k++)
            {
                Tensor lg = lm.model.ReadLogits(1);
                var arr = new float[V];
                int best = 0; float bv = float.NegativeInfinity;
                for (int i = 0; i < V; i++)
                {
                    float v = lg[i];
                    arr[i] = v;
                    if (v > bv) { bv = v; best = i; }
                }
                logitsOut[k] = arr;
                ownTokens[k] = best;
                int feed = feedTokens != null ? feedTokens[k] : best;
                if (k < COMPARE_STEPS - 1)
                    lm.model.Forward(Tensor.Constant((float)feed), useCache: true, lastPosOnly: true);
                yield return null;
            }

            Status($"{label}: sync decode bench");
            Prefill(lm, SyntheticIds(120));
            var ms = new double[BENCH_TOKENS];
            int tok = 2000;
            var sw = new System.Diagnostics.Stopwatch();
            for (int t = 0; t < BENCH_TOKENS; t++)
            {
                sw.Restart();
                lm.model.Forward(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
                tok = lm.model.SampleGreedy();
                ms[t] = sw.Elapsed.TotalMilliseconds;
            }
            medianMsOut[0] = ms.OrderBy(x => x).ToArray()[BENCH_TOKENS / 2];

            Status($"{label}: greedy text sample");
            string story = "Once upon a time in a small village by the sea, there lived an old fisherman who";
            (Tensor ids, Tensor _) = lm.tokenizer.Encode(story, add_special_tokens: false, truncation: true, max_length: 64);
            int n = ids.Size(-1);
            var idArr = new float[n];
            for (int i = 0; i < n; i++) idArr[i] = ids[i];
            var sb = new StringBuilder();
            var gen = lm.Generate(Tensor.Constant(idArr), tk => sb.Append(tk), max_new_tokens: SAMPLE_TOKENS, temperature: 0f);
            while (gen.MoveNext()) yield return gen.Current;
            textOut[0] = sb.ToString();
        }

        IEnumerator Run()
        {
            int V = Qwen3_5Modeling.Qwen3_5Config.VOCAB_SIZE;
            float[] prompt = SyntheticIds(PREFILL_TOKENS);

            report.AppendLine($"# {quant} vs FP16 probe (Qwen3.5-0.8B)");
            report.AppendLine();
            report.AppendLine($"- prefill {PREFILL_TOKENS} synthetic tokens, identical greedy token feed (FP16's choices)");
            report.AppendLine($"- GPU: {SystemInfo.graphicsDeviceName} | {SystemInfo.graphicsDeviceType}");
            report.AppendLine();

            // ---------------- FP16 reference ----------------
            Status("constructing FP16 Qwen3.5");
            var fp16 = new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, LLMQuant.FP16);
            var refLogits = new float[COMPARE_STEPS][];
            var refTok = new int[COMPARE_STEPS];
            var fp16Ms = new double[1];
            var fp16Text = new string[1];
            var ex = Exercise(fp16, prompt, null, refLogits, refTok, fp16Ms, fp16Text, "fp16");
            while (ex.MoveNext()) yield return ex.Current;
            fp16.Release();
            yield return null;

            // ---------------- quantized ----------------
            Status($"constructing {quant} Qwen3.5");
            var int8 = new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, quant);
            var qLogits = new float[COMPARE_STEPS][];
            var qTok = new int[COMPARE_STEPS];
            var int8Ms = new double[1];
            var int8Text = new string[1];
            ex = Exercise(int8, prompt, refTok, qLogits, qTok, int8Ms, int8Text, quant.ToString());
            while (ex.MoveNext()) yield return ex.Current;

            // ---------------- compare ----------------
            report.AppendLine($"## Logits ({quant} vs fp16, identical token feed)");
            report.AppendLine();
            report.AppendLine("| step | kv len | max abs diff | mean abs diff | argmax match |");
            report.AppendLine("|---|---|---|---|---|");
            int matches = 0;
            float worst = 0f;
            for (int k = 0; k < COMPARE_STEPS; k++)
            {
                float maxd = 0f; double sumd = 0;
                for (int i = 0; i < V; i++)
                {
                    float d = Math.Abs(qLogits[k][i] - refLogits[k][i]);
                    if (d > maxd) maxd = d;
                    sumd += d;
                }
                bool match = qTok[k] == refTok[k];
                if (match) matches++;
                if (maxd > worst) worst = maxd;
                report.AppendLine($"| {k} | {PREFILL_TOKENS + k} | {maxd:0.0000} | {sumd / V:0.000000} | {(match ? "yes" : $"NO ({qTok[k]} vs {refTok[k]})")} |");
            }
            // int8 is REAL quantization error, not fp reordering — argmax agreement is the gate.
            if (matches < COMPARE_STEPS - 1) pass = false;

            report.AppendLine();
            report.AppendLine("## Sync decode (greedy, cache 120, median ms/token)");
            report.AppendLine($"- fp16: {fp16Ms[0]:0.00} ms/tok ({1000 / fp16Ms[0]:0.0} tok/s)");
            report.AppendLine($"- {quant}: {int8Ms[0]:0.00} ms/tok ({1000 / int8Ms[0]:0.0} tok/s) — **{fp16Ms[0] / int8Ms[0]:0.00}x**");
            report.AppendLine();
            report.AppendLine("## Greedy story continuation (48 tokens)");
            report.AppendLine($"- fp16: `{fp16Text[0]}`");
            report.AppendLine($"- {quant}: `{int8Text[0]}`");
            int div = -1;
            int minLen = Math.Min(fp16Text[0].Length, int8Text[0].Length);
            for (int i = 0; i < minLen; i++) if (fp16Text[0][i] != int8Text[0][i]) { div = i; break; }
            if (div < 0 && fp16Text[0].Length != int8Text[0].Length) div = minLen;
            report.AppendLine($"- first divergence: {(div < 0 ? "none (identical)" : $"char {div}")}");
            report.AppendLine();
            report.AppendLine("## Summary");
            report.AppendLine($"- worst logit diff {worst:0.0000}, argmax match {matches}/{COMPARE_STEPS} -> {(pass ? "PASS" : "FAIL")}");

            WriteReport(pass);
            Status($"DONE {(pass ? "PASS" : "FAIL")} — report at {reportDirectory}");
        }

        void WriteReport(bool success)
        {
            try
            {
                Directory.CreateDirectory(reportDirectory);
                report.Insert(0, $"<!-- success: {success} -->\n");
                File.WriteAllText(Path.Combine(reportDirectory, "report.md"), report.ToString());
                Debug.Log($"[QuantProbe] report written to {reportDirectory}");
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }
    }
}
