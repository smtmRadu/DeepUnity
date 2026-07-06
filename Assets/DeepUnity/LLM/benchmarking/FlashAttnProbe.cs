using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // A/B probe for the FlashAttention path (fused `FlashAttention` kernel vs the legacy
    // scores/mask/softmax/attend chain) — runs against Gemma3-270m or Qwen3.5-0.8B.
    //
    //   1. Correctness — same synthetic prompt (700 tokens, prefilled in production-sized chunks
    //      of 8) + greedy decode steps under BOTH paths, feeding the legacy path's tokens to both
    //      so the contexts stay identical; compares full-vocab logits per step (max/mean abs
    //      diff) and argmax agreement.
    //   2. Benchmark — per-token sync decode wall time (Forward + greedy Sample readback) at a
    //      short and a long cache depth, paths interleaved L/F/L/F; re-prefilled per run so every
    //      run sees the same context (also resets Qwen's DeltaNet state).
    //   3. End-to-end Generate() — the production frame-paced path on a real tokenized prompt.
    //
    // Driven from the ClaudeBridge via FlashAttnProbeRunner.RunGemma / RunQwen; progress is
    // mirrored to ClaudeBridge/flash_probe_status.txt, the report lands in <reportDirectory>.
    public class FlashAttnProbe : MonoBehaviour
    {
        public enum LMKind { Gemma3, Qwen3_5 }

        public LMKind lmKind = LMKind.Gemma3;
        public string reportDirectory;

        const int PREFILL_TOKENS = 700;
        const int COMPARE_STEPS = 8;          // logit sets compared per path (incl. post-prefill)
        const int BENCH_TOKENS = 64;
        const int BENCH_REPS = 2;             // interleaved legacy/flash pairs per depth
        static readonly int[] BENCH_PREFILL = { 120, 1900 };
        const int CHUNK = 8;                  // mirrors the models' ForwardPromptChunked

        LLM llm;
        int vocab;
        Action<Tensor> fwd;                   // Forward(ids, useCache: true, lastPosOnly: true)
        Func<int, Tensor> readLogits;
        Func<int> sampleGreedy;
        Action resetCache;
        Action<bool> setFlash;
        Func<string, int, float[]> tokenize;

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
            Debug.Log("[FlashAttnProbe] " + s);
        }

        void Start()
        {
            Application.runInBackground = true;
            StartCoroutine(Guarded());
        }

        // Coroutines swallow exceptions per-frame; this pumps Run() manually so any throw is
        // captured into the status file (the bridge watcher's only window into play mode).
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

        void BindModel()
        {
            if (lmKind == LMKind.Gemma3)
            {
                var g = new Gemma3ForCausalLM();
                var m = g.model;
                llm = g;
                vocab = Gemma3Modeling.Gemma3Config.VOCAB_SIZE;
                fwd = ids => m.Forward(ids, useCache: true, lastPosOnly: true);
                readLogits = n => m.ReadLogits(n);
                sampleGreedy = () => m.SampleGreedy();
                resetCache = () => m.ResetCache();
                setFlash = f => Gemma3Modeling.Gemma3Model.UseFlashAttention = f;
                tokenize = (text, maxTok) =>
                {
                    (Tensor ids, Tensor _) = g.tokenizer.Encode(text, add_special_tokens: false, truncation: true, max_length: maxTok);
                    return TensorToIds(ids, maxTok);
                };
            }
            else
            {
                var q = new Qwen3_5ForCausalLM();
                var m = q.model;
                llm = q;
                vocab = Qwen3_5Modeling.Qwen3_5Config.VOCAB_SIZE;
                fwd = ids => m.Forward(ids, useCache: true, lastPosOnly: true);
                readLogits = n => m.ReadLogits(n);
                sampleGreedy = () => m.SampleGreedy();
                resetCache = () => m.ResetCache();
                setFlash = f => Qwen3_5Modeling.Qwen3_5Model.UseFlashAttention = f;
                tokenize = (text, maxTok) =>
                {
                    (Tensor ids, Tensor _) = q.tokenizer.Encode(text, add_special_tokens: false, truncation: true, max_length: maxTok);
                    return TensorToIds(ids, maxTok);
                };
            }
        }

        static float[] TensorToIds(Tensor ids, int maxTokens)
        {
            int n = Math.Min(ids.Size(-1), maxTokens);
            var arr = new float[n];
            for (int i = 0; i < n; i++) arr[i] = ids[i];
            return arr;
        }

        // Deterministic pseudo-random token ids away from the special-token range
        // (1000..200999 is valid for both vocabs).
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

        void Prefill(float[] ids)
        {
            resetCache();
            for (int s = 0; s < ids.Length; s += CHUNK)
            {
                int len = Math.Min(CHUNK, ids.Length - s);
                float[] part = new float[len];
                Array.Copy(ids, s, part, 0, len);
                fwd(Tensor.Constant(part));
            }
        }

        IEnumerator Run()
        {
            Status($"constructing {lmKind}");
            BindModel();
            var w = llm.Warmup();
            while (w.MoveNext()) yield return w.Current;
            while (!llm.IsReady) yield return null;

            float[] prompt = SyntheticIds(PREFILL_TOKENS);

            report.AppendLine($"# FlashAttention A/B probe ({lmKind})");
            report.AppendLine();
            report.AppendLine($"- prefill: {PREFILL_TOKENS} tokens (chunked {CHUNK})");
            report.AppendLine($"- GPU: {SystemInfo.graphicsDeviceName} | {SystemInfo.graphicsDeviceType}");
            report.AppendLine();

            // ---------------- correctness ----------------
            Status("correctness: legacy reference run");
            setFlash(false);
            Prefill(prompt);
            var refLogits = new Tensor[COMPARE_STEPS];
            var refTok = new int[COMPARE_STEPS];
            for (int k = 0; k < COMPARE_STEPS; k++)
            {
                refLogits[k] = readLogits(1);
                int best = 0; float bv = float.NegativeInfinity;
                for (int i = 0; i < vocab; i++) { float v = refLogits[k][i]; if (v > bv) { bv = v; best = i; } }
                refTok[k] = best;
                if (k < COMPARE_STEPS - 1)
                    fwd(Tensor.Constant((float)best));
                yield return null;
            }

            Status("correctness: flash run");
            setFlash(true);
            Prefill(prompt);
            report.AppendLine("## Correctness (flash vs legacy, identical token feed)");
            report.AppendLine();
            report.AppendLine("| step | kv len | max abs diff | mean abs diff | argmax match |");
            report.AppendLine("|---|---|---|---|---|");
            float worst = 0f;
            for (int k = 0; k < COMPARE_STEPS; k++)
            {
                Tensor fl = readLogits(1);
                Tensor rl = refLogits[k];
                float maxd = 0f; double sumd = 0;
                int best = 0; float bv = float.NegativeInfinity;
                for (int i = 0; i < vocab; i++)
                {
                    float fv = fl[i];
                    float d = Math.Abs(fv - rl[i]);
                    if (d > maxd) maxd = d;
                    sumd += d;
                    if (fv > bv) { bv = fv; best = i; }
                }
                bool match = best == refTok[k];
                if (!match) pass = false;
                if (maxd > worst) worst = maxd;
                report.AppendLine($"| {k} | {PREFILL_TOKENS + k} | {maxd:0.000000} | {sumd / vocab:0.00000000} | {(match ? "yes" : $"NO ({best} vs {refTok[k]})")} |");
                if (k < COMPARE_STEPS - 1)
                    fwd(Tensor.Constant((float)refTok[k]));
                Status($"correctness: flash step {k + 1}/{COMPARE_STEPS} maxdiff {maxd:0.0000}");
                yield return null;
            }
            if (worst > 0.05f) pass = false;   // logits are O(10) — beyond this isn't fp reordering
            report.AppendLine();

            // ---------------- sync decode benchmark ----------------
            report.AppendLine("## Decode benchmark (greedy, Forward+Sample per token, ms/token)");
            report.AppendLine();
            report.AppendLine("| cache depth | path | rep | mean ms | median ms | tok/s (median) |");
            report.AppendLine("|---|---|---|---|---|---|");
            var medians = new System.Collections.Generic.Dictionary<string, double>();
            foreach (int depth in BENCH_PREFILL)
            {
                float[] ctx = SyntheticIds(depth);
                for (int rep = 0; rep < BENCH_REPS; rep++)
                {
                    foreach (bool flash in new[] { false, true })
                    {
                        setFlash(flash);
                        string label = flash ? "flash" : "legacy";
                        Status($"benchmark: depth {depth} rep {rep} {label} — prefill");
                        Prefill(ctx);
                        yield return null;

                        Status($"benchmark: depth {depth} rep {rep} {label} — {BENCH_TOKENS} tokens");
                        var ms = new double[BENCH_TOKENS];
                        int tok = 2000;
                        var sw = new System.Diagnostics.Stopwatch();
                        for (int t = 0; t < BENCH_TOKENS; t++)
                        {
                            sw.Restart();
                            fwd(Tensor.Constant((float)tok));
                            tok = sampleGreedy();   // blocking 4-byte readback = full pipeline flush
                            ms[t] = sw.Elapsed.TotalMilliseconds;
                        }
                        var sorted = ms.OrderBy(x => x).ToArray();
                        double median = sorted[BENCH_TOKENS / 2], mean = ms.Average();
                        medians[$"{depth}/{label}/{rep}"] = median;
                        report.AppendLine($"| {depth} | {label} | {rep} | {mean:0.00} | {median:0.00} | {1000.0 / median:0.0} |");
                        yield return null;
                    }
                }
            }

            // ---------------- end-to-end generation (production Generate path) ----------------
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;

            Status("generation: tokenizing prompts");
            string story = "Once upon a time in a small village by the sea, there lived an old fisherman " +
                           "who knew the tides better than anyone. Every morning he walked the shore and " +
                           "watched the gulls argue over scraps while the waves rolled in. ";
            var shortIds = tokenize(story, 48);
            var longIds = tokenize(string.Concat(Enumerable.Repeat(story, 60)), 1800);

            report.AppendLine();
            report.AppendLine("## End-to-end Generate() (production path, greedy, 64 new tokens, uncapped fps)");
            report.AppendLine();
            report.AppendLine("| prompt tokens | path | rep | prefill ms | decode tok/s | tokens out |");
            report.AppendLine("|---|---|---|---|---|---|");
            var genRates = new System.Collections.Generic.Dictionary<string, double>();
            foreach (var (label, ids) in new[] { ("short", shortIds), ("long", longIds) })
            {
                for (int rep = 0; rep < BENCH_REPS; rep++)
                {
                    foreach (bool flash in new[] { false, true })
                    {
                        setFlash(flash);
                        string path = flash ? "flash" : "legacy";
                        Status($"generation: {label} ({ids.Length} tok) rep {rep} {path}");

                        int count = 0;
                        double tFirst = 0, tLast = 0;
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        var gen = llm.Generate(Tensor.Constant(ids), tk =>
                        {
                            count++;
                            double now = sw.Elapsed.TotalMilliseconds;
                            if (count == 1) tFirst = now;
                            tLast = now;
                        }, max_new_tokens: BENCH_TOKENS, temperature: 0f);
                        while (gen.MoveNext()) yield return gen.Current;

                        double rate = count > 1 ? (count - 1) * 1000.0 / (tLast - tFirst) : 0;
                        genRates[$"{label}/{path}/{rep}"] = rate;
                        report.AppendLine($"| {ids.Length} ({label}) | {path} | {rep} | {tFirst:0} | {rate:0.0} | {count} |");
                        yield return null;
                    }
                }
            }

            report.AppendLine();
            report.AppendLine("## Summary");
            foreach (int depth in BENCH_PREFILL)
            {
                double leg = Enumerable.Range(0, BENCH_REPS).Min(r => medians[$"{depth}/legacy/{r}"]);
                double fla = Enumerable.Range(0, BENCH_REPS).Min(r => medians[$"{depth}/flash/{r}"]);
                report.AppendLine($"- sync decode, cache {depth}: legacy {leg:0.00} ms/tok ({1000 / leg:0.0} tok/s) -> flash {fla:0.00} ms/tok ({1000 / fla:0.0} tok/s), **{leg / fla:0.00}x**");
            }
            foreach (string label in new[] { "short", "long" })
            {
                double leg = Enumerable.Range(0, BENCH_REPS).Max(r => genRates[$"{label}/legacy/{r}"]);
                double fla = Enumerable.Range(0, BENCH_REPS).Max(r => genRates[$"{label}/flash/{r}"]);
                report.AppendLine($"- Generate() {label} prompt: legacy {leg:0.0} tok/s -> flash {fla:0.0} tok/s, **{(leg > 0 ? fla / leg : 0):0.00}x**");
            }
            report.AppendLine($"- correctness: worst logit diff {worst:0.000000} -> {(pass ? "PASS" : "FAIL")}");

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
                Debug.Log($"[FlashAttnProbe] report written to {reportDirectory}");
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }
    }
}
