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
        public Qwen3_5Size size = Qwen3_5Size.B0_8; // which Qwen3.5 export is probed

        // Aggregator-facing model tag, matching LMProbeCommon.ModelLabel.
        string ModelTag => size == Qwen3_5Size.B2 ? "qwen3.5-2B" : "qwen3.5-0.8B";

        // Standard benchmark pairing (see BENCHMARK.md): quantized weights ship with INT8 KV, the
        // fp16 reference keeps fp16 KV. So this A/B reports the drift of the FULL shipped config
        // (int8/int4 weights + int8 KV) vs the fp16 reference config — not weight-only. INT8 KV is
        // near-lossless, so the weight-quant story is unchanged; this just keeps Table 3 consistent
        // with the kv=int8 speed/boot rows and reflects what actually runs.
        readonly KVQuant kvForQuant = KVQuant.INT8;

        const int PREFILL_TOKENS = 500;
        // 8 → 96 (2026-08-04): the probe grew an Unsloth-style accuracy section — mean KL
        // divergence of the quantized distribution against FP16 over teacher-forced positions —
        // and 8 positions is anecdote, not a statistic. 96 keeps the run in editor-friendly
        // minutes (one full-vocab readback + one forward per position).
        const int COMPARE_STEPS = 96;

        // Real English text for the compared positions (2026-08-04). Unsloth's own methodology
        // note is the reason this is not SyntheticIds: perplexity/KLD measured on data resembling
        // the calibration set overfits — and measured on NOISE it means nothing at all. We don't
        // calibrate, but the distribution gap between quant and FP16 is only meaningful where the
        // model is actually in-distribution. Fixed constant → reproducible across runs/quants.
        const string KLD_TEXT =
            "The harbour town woke slowly in the grey light before dawn. Fishermen checked their " +
            "nets along the quay while the baker's chimney sent up its first thread of smoke, and " +
            "the narrow streets still held the cold of the night between their stone walls. By the " +
            "time the sun cleared the headland, carts were already rolling toward the market square, " +
            "loaded with crates of silver fish, bundles of herbs, and rounds of pale cheese wrapped " +
            "in cloth. An old clockmaker opened his shutters and set each of his instruments to the " +
            "chime of the church bell, as he had done every morning for forty years. Children ran " +
            "errands between the stalls, trading gossip for sweets, and the innkeeper swept yesterday's " +
            "sawdust into the gutter while arguing amiably with a carpenter about the price of oak. " +
            "Out beyond the breakwater, the last of the night boats turned for home, their lanterns " +
            "pale against the brightening water. The town had seen storms, sieges, and two fires that " +
            "each took a third of its roofs, yet the shape of its mornings had barely changed in a " +
            "century: bread, fish, bells, and the tide. A young cartographer who had arrived that " +
            "spring kept a journal of such details, convinced that the true map of a place was not " +
            "its coastline but the order in which it woke. She noted the baker before the fishmonger, " +
            "the bell before the gulls, the schoolmaster always last, hurrying with ink-stained " +
            "fingers past the fountain. On market days the square filled until the stalls touched, " +
            "and sailors from three kingdoms haggled in a patchwork of languages that everyone half " +
            "understood. When the fog rolled in, the lighthouse keeper wound the great lamp and the " +
            "town listened for the horn of the mail packet, which carried letters, newspapers, and " +
            "once a year the almanac that told the farmers when to plant. It was, the cartographer " +
            "wrote, a town that measured itself in small repetitions, and was the richer for it.";
        const int BENCH_TOKENS = 64;
        const int SAMPLE_TOKENS = 48;
        const int CHUNK = 8;

        readonly StringBuilder report = new StringBuilder();
        bool pass = true;
        bool finished;

        // Headless safety: in `-batchmode` nothing stops the play loop, so the probe quits the
        // process itself (Exit at the end of Run / on error / on timeout). Interactive bridge runs
        // (Unity open) leave play mode under the user's control. Mirrors GemmaQuantProbe.
        const float BATCH_TIMEOUT_SECONDS = 600f;
        readonly System.Diagnostics.Stopwatch lifeSw = System.Diagnostics.Stopwatch.StartNew();

        static void Exit(int code)
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.Exit(code);
#else
            Application.Quit(code);
#endif
        }

        void BatchExit(int code)
        {
            finished = true;
            if (Application.isBatchMode) Exit(code);
        }

        void Update()
        {
            if (finished || !Application.isBatchMode) return;
            if (lifeSw.Elapsed.TotalSeconds > BATCH_TIMEOUT_SECONDS)
            {
                Status("ERROR: batch timeout");
                WriteReport(false);
                BatchExit(2);
            }
        }

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
                    BatchExit(1);
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

            report.AppendLine($"# {quant} vs FP16 probe ({ModelTag})");
            report.AppendLine();
            report.AppendLine($"- real-text prefill (KLD_TEXT, cap {PREFILL_TOKENS} tok), identical greedy token feed (FP16's choices), {COMPARE_STEPS} compared positions");
            report.AppendLine($"- GPU: {SystemInfo.graphicsDeviceName} | {SystemInfo.graphicsDeviceType}");
            report.AppendLine();

            // ---------------- FP16 reference ----------------
            Status("constructing FP16 Qwen3.5");
            var fp16 = new Qwen3_5ForCausalLM(size, LLMQuant.FP16, kv_quant: KVQuant.FP16);
            // Tokenize the real-text prompt with the reference model's tokenizer (identical for
            // every quant of a size — the export never touches the tokenizer). WAIT for readiness
            // first: the tokenizer streams in with the weights, and encoding straight off the
            // constructor lost the race on the 2026-08-04 qwen2b_int4 run (three sibling runs won
            // it by luck — cached files load faster on repeat).
            while (!fp16.IsReady) yield return null;
            (Tensor pTok, Tensor _) = fp16.tokenizer.Encode(KLD_TEXT, add_special_tokens: false,
                                                            truncation: true, max_length: PREFILL_TOKENS);
            int pn = pTok.Size(-1);
            float[] prompt = new float[pn];
            for (int i = 0; i < pn; i++) prompt[i] = pTok[i];
            report.AppendLine($"- prompt tokens actually used: {pn}");
            report.AppendLine();
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
            var int8 = new Qwen3_5ForCausalLM(size, quant, kv_quant: kvForQuant);
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
            double meanDiffSum = 0;
            // #KLD (2026-08-04, Unsloth's metric): KL(P_fp16 || P_quant) per compared position,
            // full vocab, numerically via log-softmax with max-subtraction. Lower = the quant's
            // distribution mirrors FP16 more closely; Unsloth argues (and we agree) this beats
            // perplexity for quant quality because it also penalizes confident DISAGREEMENT.
            var kld = new double[COMPARE_STEPS];
            for (int k = 0; k < COMPARE_STEPS; k++)
            {
                float maxd = 0f; double sumd = 0;
                float mr = float.NegativeInfinity, mq = float.NegativeInfinity;
                for (int i = 0; i < V; i++)
                {
                    float d = Math.Abs(qLogits[k][i] - refLogits[k][i]);
                    if (d > maxd) maxd = d;
                    sumd += d;
                    if (refLogits[k][i] > mr) mr = refLogits[k][i];
                    if (qLogits[k][i] > mq) mq = qLogits[k][i];
                }
                double zr = 0, zq = 0;
                for (int i = 0; i < V; i++)
                {
                    zr += Math.Exp(refLogits[k][i] - mr);
                    zq += Math.Exp(qLogits[k][i] - mq);
                }
                double lnZr = Math.Log(zr), lnZq = Math.Log(zq), acc = 0;
                for (int i = 0; i < V; i++)
                {
                    double lpr = refLogits[k][i] - mr - lnZr;
                    double lpq = qLogits[k][i] - mq - lnZq;
                    acc += Math.Exp(lpr) * (lpr - lpq);
                }
                kld[k] = acc;
                meanDiffSum += sumd / V;
                bool match = qTok[k] == refTok[k];
                if (match) matches++;
                if (maxd > worst) worst = maxd;
                if (k < 8 || !match)   // the table stays readable at 96 steps: first 8 + every flip
                    report.AppendLine($"| {k} | {PREFILL_TOKENS + k} | {maxd:0.0000} | {sumd / V:0.000000} | {(match ? "yes" : $"NO ({qTok[k]} vs {refTok[k]})")} |");
            }
            var kSorted = kld.OrderBy(x => x).ToArray();
            double kMean = kld.Average(), kMedian = kSorted[COMPARE_STEPS / 2], kMax = kSorted[COMPARE_STEPS - 1];
            report.AppendLine();
            report.AppendLine("## KL divergence vs FP16 (Unsloth-style, teacher-forced real text)");
            report.AppendLine($"- mean KLD: **{kMean:0.000000}** nats | median: {kMedian:0.000000} | max: {kMax:0.000000}");
            report.AppendLine($"- top-1 agreement: {matches}/{COMPARE_STEPS} ({100.0 * matches / COMPARE_STEPS:0.0}%)");
            // int8 is REAL quantization error, not fp reordering — argmax agreement is the gate.
            // Percentage gates since the step count grew to 96 (2026-08-04): all-but-one was a
            // gate for 8 steps; at 96, int8 should hold ≥90% and int4's flips are the finding
            // itself, so its bar is lower and mostly informational.
            double agreeGate = quant == LLMQuant.INT4 ? 0.70 : 0.90;
            if (matches < COMPARE_STEPS * agreeGate) pass = false;

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

            WriteSummary(worst, meanDiffSum / COMPARE_STEPS, matches, div, fp16Ms[0], int8Ms[0]);
            WriteReport(pass);
            Status($"DONE {(pass ? "PASS" : "FAIL")} — report at {reportDirectory}");
            BatchExit(pass ? 0 : 1);
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

        // Machine-readable headline metrics for BENCHMARK.md aggregation. Invariant culture so a
        // comma-decimal locale (e.g. RO on the Pavilion box) can't emit invalid JSON.
        void WriteSummary(float maxDiff, double meanDiff, int matches, int divChar, double fp16Ms, double quantMs)
        {
            var prev = System.Threading.Thread.CurrentThread.CurrentCulture;
            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            try
            {
                double fpTokS = fp16Ms > 0 ? 1000.0 / fp16Ms : 0;
                double qTokS = quantMs > 0 ? 1000.0 / quantMs : 0;
                var js = new StringBuilder();
                js.Append("{\n");
                js.Append("  \"probe\": \"quant_quality\",\n");
                js.Append("  \"model\": ").Append(LMProbeCommon.JsonStr(ModelTag)).Append(",\n");
                js.Append("  \"quant\": ").Append(LMProbeCommon.JsonStr(quant.ToString())).Append(",\n");
                js.Append("  \"kv\": ").Append(LMProbeCommon.JsonStr(kvForQuant.ToString())).Append(",\n");
                js.Append("  \"success\": ").Append(pass ? "true" : "false").Append(",\n");
                js.Append("  \"compare_steps\": ").Append(COMPARE_STEPS).Append(",\n");
                js.Append($"  \"max_logit_diff\": {maxDiff:0.0000},\n");
                js.Append($"  \"mean_logit_diff\": {meanDiff:0.000000},\n");
                js.Append($"  \"argmax_match\": {matches},\n");
                js.Append($"  \"argmax_match_pct\": {100.0 * matches / COMPARE_STEPS:0.0},\n");
                js.Append($"  \"divergence_char\": {divChar},\n");
                js.Append($"  \"fp16_decode_ms\": {fp16Ms:0.00}, \"fp16_decode_tok_s\": {fpTokS:0.0},\n");
                js.Append($"  \"quant_decode_ms\": {quantMs:0.00}, \"quant_decode_tok_s\": {qTokS:0.0},\n");
                js.Append($"  \"decode_speedup\": {(quantMs > 0 ? fp16Ms / quantMs : 0):0.00},\n");
                js.Append("  \"machine\": ").Append(LMProbeCommon.MachineJson()).Append("\n");
                js.Append("}\n");
                Directory.CreateDirectory(reportDirectory);
                File.WriteAllText(Path.Combine(reportDirectory, "summary.json"), js.ToString());
            }
            catch (Exception e) { Debug.LogException(e); }
            finally { System.Threading.Thread.CurrentThread.CurrentCulture = prev; }
        }
    }
}
