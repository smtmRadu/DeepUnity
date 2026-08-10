#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using DeepUnity.Gemma3Modeling;
using DeepUnity.PocketTTSModeling;
using DeepUnity.Qwen3_5Modeling;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // ============================================================================================
    // ModuleLatencyProbe — where one decode step's time goes, per MECHANISM, for the four models
    // the evaluation chapter ships. Fills fig:eval-pertoken-1650 in the dissertation, whose
    // per-module ms are currently dummy.
    //
    // EDIT MODE, no play mode, no scene: batch-callable directly.
    //   Unity.exe -projectPath <proj> -batchmode -executeMethod DeepUnity.ModuleLatencyProbe.RunAll
    //   (NO -nographics — compute shaders need a graphics device.)
    // Writes ProbeLogs/module_latency_<tag>_<ts>/latency.json, one file per arm, consumed by the
    // thesis-side tools/fill_latency_figure.py.
    //
    // METHOD — two measurements per arm, because neither alone is usable:
    //   1. CLEAN pass: unprofiled decode steps -> honest ms/token (this is the number that must
    //      agree with the published decode rates).
    //   2. PROFILED pass: Qwen3_5Model/Gemma3Model.StageProfile on. Mark() drains the GPU queue
    //      after every dispatch group, so the TOTAL is badly inflated by the syncs — but the
    //      SHARES are sound. Per-mechanism ms = share x clean ms/token, then / the layer count of
    //      that mechanism. So the split is measured and the total is measured; nothing is invented.
    //
    // PASSES passes of both, and the LAST pass is what gets reported: the first pass still carries
    // shader compilation and driver warmup even after the explicit warmup steps. Every pass is kept
    // in the json so drift between them is visible rather than hidden.
    //
    // Attribution granularity is deliberately one bucket per FIGURE GLYPH, not per kernel:
    //   Qwen3.5   fa:*  -> full attention (6 layers)   lin:* -> DeltaNet (18)   mlp:* -> SwiGLU (24)
    //   Gemma3    swa:* -> sliding window (15)         fa:*  -> full attention (3)  mlp:* -> GeGLU (18)
    //   both      final+lmhead -> LM head (1)          embed / glue / sample -> overhead
    // A finer split would buy nothing for the figure and would add sync distortion per mark.
    // ============================================================================================
    public static class ModuleLatencyProbe
    {
        // Context the step is timed under. 2048 matches the protocol point every other number in
        // the chapter is quoted at (fig:eval-prefill, fig:eval-decode, the engine comparison), so
        // the totals here are comparable to them. Lower it if you only want a quick sanity run.
        const int PREFILL_LEN = 2048;
        const int CACHE_CAP = 4096;
        const int WARMUP = 6;      // discarded decode steps before any timing
        const int TIMED = 24;      // clean decode steps per pass
        const int PROFILED = 4;    // serialized decode steps per pass (each is ~100+ GPU syncs)
        const int PASSES = 3;      // the LAST one is reported

        // Weights live in EITHER of two trees and the flat one shadows the other, exactly as
        // LLM.ResolveParamsDir decides it (that method is protected, so the order is mirrored here
        // rather than called). Hardcoding the flat path is what made this probe report Gemma's
        // weights "missing": Qwen sits in both trees, but Gemma only ever existed in the
        // per-architecture one.
        static string Weights(string arch, string dir)
        {
            string cwd = Directory.GetCurrentDirectory();
            string flat = $"Assets/Resources/Weights/{dir}";
            if (Directory.Exists(Path.Combine(cwd, flat))) return flat;
            string organized = $"Assets/Resources/DeepUnity/LLM/{arch}/{dir}";
            if (Directory.Exists(Path.Combine(cwd, organized))) return organized;
            return $"Assets/DeepUnity/InferenceEngine/LLM/{arch}/{dir}";
        }

        static string QWEN_08 => Weights("Qwen3_5", "weights_qwen3.5_0.8B_int8");
        static string QWEN_2B => Weights("Qwen3_5", "weights_qwen3.5_2B_int8");
        static string GEMMA => Weights("Gemma3", "weights_gemma3_270M_int8");

        // ---------------- menu / batch entry points ----------------
        [MenuItem("DeepUnity/Benchmarks/Module Latency/Qwen3.5 0.8B (int8)")]
        public static void RunQwen08() => Guard(() => Qwen(Qwen3_5Size.B0_8, QWEN_08, "qwen3.5-0.8B"));

        [MenuItem("DeepUnity/Benchmarks/Module Latency/Qwen3.5 2B (int8)")]
        public static void RunQwen2B() => Guard(() => Qwen(Qwen3_5Size.B2, QWEN_2B, "qwen3.5-2B"));

        [MenuItem("DeepUnity/Benchmarks/Module Latency/Gemma3 270M (int8)")]
        public static void RunGemma() => Guard(Gemma);

        [MenuItem("DeepUnity/Benchmarks/Module Latency/pocket-tts (int8)")]
        public static void RunPocketTts() => Guard(PocketTts);

        // One Unity launch, all four arms. Cheaper than four launches (no editor restart per arm)
        // and the arms do not share mutable state beyond Qwen3_5Config, which Qwen() re-applies.
        [MenuItem("DeepUnity/Benchmarks/Module Latency/ALL FOUR")]
        public static void RunAll() => Guard(() =>
        {
            Qwen(Qwen3_5Size.B0_8, QWEN_08, "qwen3.5-0.8B");
            Qwen(Qwen3_5Size.B2, QWEN_2B, "qwen3.5-2B");
            Gemma();
            PocketTts();
        });

        static void Guard(Action body)
        {
            int code = 0;
            try { body(); }
            catch (Exception e) { Debug.LogException(e); code = 1; }
            finally { EditorUtility.ClearProgressBar(); }
            if (Application.isBatchMode) EditorApplication.Exit(code);
        }

        // ---------------- Qwen3.5 (0.8B and 2B share the layer layout; only widths differ) --------
        static void Qwen(Qwen3_5Size size, string weights, string tag)
        {
            RequireWeights(weights, tag);
            // MUST precede construction: the ctor copies HIDDEN_SIZE / MLP_INTERMEDIATE_SIZE out of
            // the static config. Without this the 2B silently builds 0.8B dims over 2B weights.
            Qwen3_5Config.ApplySize(size);

            Qwen3_5Model m = null;
            try
            {
                Progress(tag, "loading weights", 0.1f);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                m = new Qwen3_5Model(weights, CACHE_CAP, LLMQuant.INT8, KVQuant.FP16);
                m.LoadBlockingForProbe();
                float loadMs = (float)sw.Elapsed.TotalMilliseconds;

                int nFull = 0, nLin = 0;
                foreach (var t in Qwen3_5Config.layer_types)
                    if (t == Qwen3_5LayerType.FullAttention) nFull++; else nLin++;
                var counts = new Dictionary<string, int> {
                    { "full_attention", nFull }, { "deltanet", nLin },
                    { "mlp", Qwen3_5Config.NUM_LAYERS }, { "lm_head", 1 },
                };

                int tok = Prefill(tag, out _, ids =>
                {
                    m.Forward(Tensor.Constant(ids), useCache: true, lastPosOnly: true);
                    return m.SampleGreedy();
                });
                for (int i = 0; i < WARMUP; i++) tok = Step(m, tok);

                var clean = new List<float>();
                var profiles = new List<Dictionary<string, float>>();
                for (int p = 0; p < PASSES; p++)
                {
                    Progress(tag, $"pass {p + 1}/{PASSES}", 0.4f + 0.5f * p / PASSES);
                    sw.Restart();
                    for (int i = 0; i < TIMED; i++) tok = Step(m, tok);
                    clean.Add((float)sw.Elapsed.TotalMilliseconds / TIMED);

                    Qwen3_5Model.StageProfile = new Dictionary<string, float>();
                    for (int i = 0; i < PROFILED; i++) tok = Step(m, tok);
                    profiles.Add(Per(Qwen3_5Model.StageProfile, PROFILED));
                    Qwen3_5Model.StageProfile = null;
                }

                Emit(tag, "int8", loadMs, counts, clean, profiles,
                     new[] { ("fa:", "full_attention"), ("lin:", "deltanet"), ("mlp:", "mlp") });
            }
            finally { Qwen3_5Model.StageProfile = null; m?.Dispose(); }
        }

        static int Step(Qwen3_5Model m, int tok)
        {
            m.Forward(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
            return m.SampleGreedy();
        }

        // ---------------- Gemma3 270M (5:1 sliding-window : full attention) ----------------
        static void Gemma()
        {
            const string tag = "gemma3-270M";
            RequireWeights(GEMMA, tag);

            Gemma3Model m = null;
            try
            {
                Progress(tag, "loading weights", 0.1f);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                m = new Gemma3Model(GEMMA, CACHE_CAP, LLMQuant.INT8, KVQuant.FP16);
                m.LoadBlockingForProbe();
                float loadMs = (float)sw.Elapsed.TotalMilliseconds;

                int nSW = 0, nFull = 0;
                foreach (var t in Gemma3Config.layer_types)
                    if (t == GemmaLayerType.SlidingWindowAttention) nSW++; else nFull++;
                var counts = new Dictionary<string, int> {
                    { "sliding_window_attention", nSW }, { "full_attention", nFull },
                    { "mlp", Gemma3Config.NUM_LAYERS }, { "lm_head", 1 },
                };

                int tok = Prefill(tag, out _, ids =>
                {
                    m.Forward(Tensor.Constant(ids), useCache: true, lastPosOnly: true);
                    return m.SampleGreedy();
                });
                for (int i = 0; i < WARMUP; i++) tok = Step(m, tok);

                var clean = new List<float>();
                var profiles = new List<Dictionary<string, float>>();
                for (int p = 0; p < PASSES; p++)
                {
                    Progress(tag, $"pass {p + 1}/{PASSES}", 0.4f + 0.5f * p / PASSES);
                    sw.Restart();
                    for (int i = 0; i < TIMED; i++) tok = Step(m, tok);
                    clean.Add((float)sw.Elapsed.TotalMilliseconds / TIMED);

                    Gemma3Model.StageProfile = new Dictionary<string, float>();
                    for (int i = 0; i < PROFILED; i++) tok = Step(m, tok);
                    profiles.Add(Per(Gemma3Model.StageProfile, PROFILED));
                    Gemma3Model.StageProfile = null;
                }

                Emit(tag, "int8", loadMs, counts, clean, profiles,
                     new[] { ("swa:", "sliding_window_attention"), ("fa:", "full_attention"), ("mlp:", "mlp") });
            }
            finally { Gemma3Model.StageProfile = null; m?.Dispose(); }
        }

        static int Step(Gemma3Model m, int tok)
        {
            m.Forward(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
            return m.SampleGreedy();
        }

        static int Prefill(string tag, out float ms, Func<float[], int> forward)
        {
            Progress(tag, $"prefill {PREFILL_LEN} tokens", 0.3f);
            var ids = new float[PREFILL_LEN];
            for (int i = 0; i < PREFILL_LEN; i++) ids[i] = 1000 + (i % 20000);   // fixed dummy prompt
            var sw = System.Diagnostics.Stopwatch.StartNew();
            int tok = forward(ids);
            ms = (float)sw.Elapsed.TotalMilliseconds;
            return tok;
        }

        // ---------------- pocket-tts (backbone / flow head / Mimi decoder) ----------------
        static void PocketTts()
        {
            const string tag = "pocket-tts";
            const string TEXT =
                "The old lighthouse keeper climbed the spiral stairs every evening at dusk. " +
                "He lit the great lamp and watched the beam sweep across the darkening waves; " +
                "ships far at sea counted on that light to find their way home safely.";
            RequireWeights(PocketTTSConfig.WEIGHTS_DIR_INT8, tag);

            PocketTTS tts = null;
            bool overlapWas = PocketTTS.OverlapMimi;
            try
            {
                Progress(tag, "loading weights", 0.1f);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                tts = new PocketTTS(PocketTTSConfig.WEIGHTS_DIR_INT8);
                tts.LoadBlocking();
                float loadMs = (float)sw.Elapsed.TotalMilliseconds;

                int[] ids = tts.Tokenize(TEXT);
                Progress(tag, "warmup (shader compiles)", 0.3f);
                tts.GenerateOffline(ids, null, useKvCache: true);   // untimed

                // (1) CLEAN, production settings: the honest per-chunk total.
                var cleanMsPerChunk = new List<float>();
                var cleanRtf = new List<float>();
                int frames = 0;
                for (int p = 0; p < PASSES; p++)
                {
                    Progress(tag, $"clean pass {p + 1}/{PASSES}", 0.4f + 0.2f * p / PASSES);
                    float[] wav = tts.GenerateOffline(ids, null, useKvCache: true);
                    frames = tts.LastFrames;
                    float total = tts.GenMs + tts.DecodeMs;
                    float sec = wav.Length / (float)PocketTTSConfig.SAMPLE_RATE;
                    cleanMsPerChunk.Add(total / Math.Max(1, frames));
                    cleanRtf.Add(total / 1000f / Math.Max(sec, 1e-6f));
                }

                // (2) SPLIT, legacy per-frame loop with overlap off — the only configuration in
                // which the backbone step, the flow head and Mimi are separable at all.
                PocketTTS.ForceLegacyArLoop = true;
                PocketTTS.OverlapMimi = false;
                PocketTTS.PerfCounting = true;
                var splits = new List<Dictionary<string, float>>();
                for (int p = 0; p < PASSES; p++)
                {
                    Progress(tag, $"split pass {p + 1}/{PASSES}", 0.6f + 0.3f * p / PASSES);
                    PocketTTS.StatReset();
                    tts.GenerateOffline(ids, null, useKvCache: true);
                    int f = Math.Max(1, tts.LastFrames);
                    splits.Add(new Dictionary<string, float> {
                        { "backbone",     (float)(PocketTTS.StatDecodeCallMs / f) },
                        { "flow_head",    (float)(PocketTTS.StatFlowCallMs / f) },
                        { "mimi_decoder", tts.DecodeMs / f },
                        { "input_linear", (float)(PocketTTS.StatTokenCpuMs / f) },
                    });
                }

                var last = splits[splits.Count - 1];
                float sum = 0f; foreach (var kv in last) sum += kv.Value;
                float chunkMs = cleanMsPerChunk[cleanMsPerChunk.Count - 1];
                var counts = new Dictionary<string, int> {
                    { "backbone", PocketTTSConfig.TF_LAYERS },
                    { "flow_head", PocketTTSConfig.FLOW_DEPTH },
                    { "mimi_decoder", 1 },
                    { "input_linear", 1 },
                };

                var sb = new StringBuilder();
                sb.Append("{\n");
                sb.Append($"  \"tag\": \"{tag}\",\n  \"quant\": \"int8\",\n");
                sb.Append($"  \"gpu\": \"{Esc(SystemInfo.graphicsDeviceName)}\",\n");
                sb.Append($"  \"unit\": \"one 80 ms audio chunk (1 Mimi frame = {PocketTTSConfig.SAMPLES_PER_LATENT} samples @ {PocketTTSConfig.SAMPLE_RATE} Hz)\",\n");
                sb.Append($"  \"load_ms\": {loadMs:F0},\n  \"frames\": {frames},\n  \"passes\": {PASSES},\n");
                sb.Append($"  \"clean_ms_per_chunk_per_pass\": {Arr(cleanMsPerChunk)},\n");
                sb.Append($"  \"clean_ms_per_chunk\": {chunkMs:F3},\n");
                sb.Append($"  \"rtf_per_pass\": {Arr(cleanRtf)},\n");
                sb.Append($"  \"split_legacy_ms_per_chunk\": {Obj(last)},\n");
                sb.Append("  \"counts\": {"); int n = 0;
                foreach (var kv in counts) sb.Append((n++ > 0 ? ", " : "") + $"\"{kv.Key}\": {kv.Value}");
                sb.Append("},\n");
                var shares = new Dictionary<string, float>();
                var perChunk = new Dictionary<string, float>();
                var perModule = new Dictionary<string, float>();
                foreach (var kv in last)
                {
                    float s = sum > 0 ? kv.Value / sum : 0f;
                    shares[kv.Key] = s;
                    perChunk[kv.Key] = s * chunkMs;
                    perModule[kv.Key] = s * chunkMs / Math.Max(1, counts[kv.Key]);
                }
                sb.Append($"  \"shares\": {Obj(shares, 4)},\n");
                sb.Append($"  \"ms_per_chunk_by_module\": {Obj(perChunk)},\n");
                sb.Append($"  \"ms_per_module\": {Obj(perModule)},\n");
                sb.Append("  \"note\": \"shares come from the legacy per-frame loop (the only separable path); the total they are scaled onto is the clean production run\"\n}\n");
                Write(tag, sb.ToString());
            }
            finally
            {
                PocketTTS.ForceLegacyArLoop = false;
                PocketTTS.OverlapMimi = overlapWas;
                PocketTTS.PerfCounting = false;
                tts?.Dispose();
            }
        }

        // ---------------- shared reporting ----------------
        static Dictionary<string, float> Per(Dictionary<string, float> raw, int steps)
        {
            var d = new Dictionary<string, float>();
            foreach (var kv in raw) d[kv.Key] = kv.Value / steps;
            return d;
        }

        static void Emit(string tag, string quant, float loadMs, Dictionary<string, int> counts,
                         List<float> clean, List<Dictionary<string, float>> profiles,
                         (string prefix, string bucket)[] map)
        {
            var raw = profiles[profiles.Count - 1];              // LAST pass — the warm one
            float cleanMs = clean[clean.Count - 1];

            var buckets = new Dictionary<string, float>();
            foreach (var kv in raw)
            {
                string b = "overhead";
                foreach (var (prefix, bucket) in map)
                    if (kv.Key.StartsWith(prefix, StringComparison.Ordinal)) { b = bucket; break; }
                if (kv.Key.StartsWith("final+lmhead", StringComparison.Ordinal)) b = "lm_head";
                buckets.TryGetValue(b, out float v);
                buckets[b] = v + kv.Value;
            }
            float total = 0f; foreach (var kv in buckets) total += kv.Value;

            var shares = new Dictionary<string, float>();
            var perTok = new Dictionary<string, float>();
            var perModule = new Dictionary<string, float>();
            foreach (var kv in buckets)
            {
                float s = total > 0 ? kv.Value / total : 0f;
                shares[kv.Key] = s;
                perTok[kv.Key] = s * cleanMs;
                perModule[kv.Key] = s * cleanMs / Math.Max(1, counts.TryGetValue(kv.Key, out int c) ? c : 1);
            }

            var sb = new StringBuilder();
            sb.Append("{\n");
            sb.Append($"  \"tag\": \"{tag}\",\n  \"quant\": \"{quant}\",\n");
            sb.Append($"  \"gpu\": \"{Esc(SystemInfo.graphicsDeviceName)}\",\n");
            sb.Append($"  \"context\": {PREFILL_LEN},\n  \"load_ms\": {loadMs:F0},\n");
            sb.Append($"  \"passes\": {PASSES},\n  \"timed_steps_per_pass\": {TIMED},\n  \"profiled_steps_per_pass\": {PROFILED},\n");
            sb.Append($"  \"clean_ms_per_token_per_pass\": {Arr(clean)},\n");
            sb.Append($"  \"clean_ms_per_token\": {cleanMs:F3},\n");
            sb.Append($"  \"decode_tok_s\": {(cleanMs > 0 ? 1000f / cleanMs : 0f):F2},\n");
            sb.Append("  \"layer_counts\": {"); int n = 0;
            foreach (var kv in counts) sb.Append((n++ > 0 ? ", " : "") + $"\"{kv.Key}\": {kv.Value}");
            sb.Append("},\n");
            sb.Append($"  \"profiled_raw_ms_per_token\": {Obj(raw)},\n");
            sb.Append($"  \"serialized_total_ms_per_token\": {total:F2},\n");
            sb.Append($"  \"shares\": {Obj(shares, 4)},\n");
            sb.Append($"  \"ms_per_token_by_mechanism\": {Obj(perTok)},\n");
            sb.Append($"  \"ms_per_module\": {Obj(perModule)},\n");
            sb.Append("  \"note\": \"shares from the serialized profile, scaled onto the clean ms/token; serialized_total is inflated by the per-mark GPU syncs and is NOT a latency\"\n}\n");
            Write(tag, sb.ToString());

            var log = new StringBuilder();
            log.AppendLine($"[ModuleLatency] {tag} {quant} @ctx{PREFILL_LEN}: {cleanMs:F2} ms/token " +
                           $"= {1000f / cleanMs:F1} tok/s (clean, last of {PASSES} passes)");
            foreach (var kv in perModule)
                log.AppendLine($"    {kv.Key,-26} {kv.Value,7:F2} ms/module x{(counts.TryGetValue(kv.Key, out int c) ? c : 1),-3}" +
                               $"  = {perTok[kv.Key],6:F2} ms/token  ({100f * shares[kv.Key],4:F1}%)");
            Debug.Log(log.ToString());
        }

        // ---------------- plumbing ----------------
        static void RequireWeights(string rel, string tag)
        {
            string abs = Path.Combine(Directory.GetCurrentDirectory(), rel);
            if (!Directory.Exists(abs) && !Directory.Exists(rel))
                throw new DirectoryNotFoundException(
                    $"{tag}: weights not found at {rel}. They are gitignored — regenerate with " +
                    $"import_params.py (see benchmarking/BENCHMARK.md) before running this probe.");
        }

        static void Write(string tag, string json)
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs",
                                      $"module_latency_{tag}_{DateTime.UtcNow:yyyyMMdd_HHmmss}".ToLowerInvariant());
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, "latency.json");
            File.WriteAllText(path, json);
            Debug.Log($"[ModuleLatency] wrote {path}");
        }

        static void Progress(string tag, string what, float f)
            => EditorUtility.DisplayProgressBar($"module latency — {tag}", what, f);

        static string Esc(string s) => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");

        static string Arr(List<float> v)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < v.Count; i++) sb.Append((i > 0 ? ", " : "") + v[i].ToString("F3"));
            return sb.Append("]").ToString();
        }

        static string Obj(Dictionary<string, float> d, int dp = 3)
        {
            var sb = new StringBuilder("{");
            int n = 0;
            foreach (var kv in d) sb.Append((n++ > 0 ? ", " : "") + $"\"{Esc(kv.Key)}\": {kv.Value.ToString("F" + dp)}");
            return sb.Append("}").ToString();
        }
    }
}
#endif
