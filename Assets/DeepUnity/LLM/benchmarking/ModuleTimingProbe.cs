using System;
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // ============================================================================================
    // ModuleTimingProbe  —  per-module compute-time breakdown of an LLM forward pass.
    //
    //   STATUS: SCAFFOLD ONLY. The lifecycle, reporting and batch/headless plumbing are in place
    //   and compile; the actual per-module timing in Run() is intentionally NOT implemented yet
    //   (see the TODO block). Fill it in when we want the numbers for the paper.
    //
    // GOAL (for the evaluation chapter): given a model + quantization, report where the GPU time
    // goes inside ONE decode step (and, separately, inside prefill), broken down by module:
    //
    //     embedding lookup | input RMSNorm | Q/K/V projection | RoPE | attention scores |
    //     softmax | attention·V | output projection | post-attn RMSNorm | pre-FFN RMSNorm |
    //     MLP gate+up | activation | MLP down | post-FFN RMSNorm | final RMSNorm |
    //     LM head | sampling
    //
    // INTENDED METHOD (when implemented):
    //   - Construct the model at `quant`, Warmup(), wait for IsReady (same as the other probes).
    //   - Wrap each kernel dispatch group so its GPU cost can be attributed to a module. Two
    //     options, decide at implementation time:
    //       (a) CPU stopwatch around dispatch + an explicit GPU sync (AsyncGPUReadback / a tiny
    //           readback) per module — simple, coarse, includes sync overhead; or
    //       (b) GraphicsFence / GPU timestamp queries straddling each module — accurate GPU-only
    //           time, more wiring. (b) is preferred for paper-grade numbers.
    //   - PREFILL pass: time the modules once over an N-token synthetic prompt (reuse the
    //     SyntheticIds pattern from the other probes) -> first-token latency contribution per module.
    //   - DECODE pass: time the modules over M cached single-token steps; report MEDIAN ms/module
    //     (medians, not means — first steps include driver warmup spikes).
    //   - Emit: per-module ms, % of step, and totals; plus the headline first-token-latency and
    //     decode tok/s so this probe alone covers the "prefill timing / decode speed by quant" rows.
    //   - Run across quant ∈ {FP16, INT8, INT4} (one process each, like the quant probes) so the
    //     report can be diffed by quantization.
    //
    // Driven (once implemented) via ModuleTimingProbeRunner.Run* -> ProbeLogs/module_timing_*/report.md.
    // ============================================================================================
    public class ModuleTimingProbe : MonoBehaviour
    {
        public FlashAttnProbe.LMKind lmKind = FlashAttnProbe.LMKind.Gemma3;
        public LLMQuant quant = LLMQuant.FP16;
        public string reportDirectory;

        // Measurement budget (used once the timing body is implemented).
        public int prefillTokens = 256;   // first-token-latency prompt length
        public int warmupSteps   = 16;    // discarded decode steps (driver/cache warmup)
        public int measureSteps  = 64;    // timed decode steps -> median ms/module

        // The modules we attribute GPU time to. Keep in forward-pass order so the report reads
        // top-to-bottom like the network. (Implementation maps each to its kernel dispatch group.)
        public enum Module
        {
            EmbeddingLookup,
            InputRMSNorm,
            QKVProjection,
            RoPE,
            AttentionScores,
            Softmax,
            AttentionValue,
            OutputProjection,
            PostAttnRMSNorm,
            PreFFNRMSNorm,
            MLPGateUp,
            Activation,
            MLPDown,
            PostFFNRMSNorm,
            FinalRMSNorm,
            LMHead,
            Sampling,
        }

        bool finished;
        const float BATCH_TIMEOUT_SECONDS = 600f;
        readonly System.Diagnostics.Stopwatch lifeSw = System.Diagnostics.Stopwatch.StartNew();

        static string StatusPath => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "module_timing_status.txt");

        void Status(string s)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(StatusPath));
                File.WriteAllText(StatusPath, $"[{DateTime.Now:HH:mm:ss}] {s}");
            }
            catch { }
            Debug.Log("[ModuleTimingProbe] " + s);
        }

        void Start()
        {
            Application.runInBackground = true;
            StartCoroutine(Guarded());
        }

        void Update()
        {
            if (finished || !Application.isBatchMode) return;
            if (lifeSw.Elapsed.TotalSeconds > BATCH_TIMEOUT_SECONDS)
            {
                Status("ERROR: batch timeout");
                WriteReport(false, "(timed out)");
                BatchExit(2);
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
                    WriteReport(false, ex.Message);
                    BatchExit(1);
                    yield break;
                }
                yield return cur;
            }
        }

        IEnumerator Run()
        {
            // TODO(benchmarking): implement the per-module timing described in the header block.
            // Intentionally left unimplemented for now — this scaffold only wires up the lifecycle,
            // headless batch exit and report file so the probe is ready to flesh out.
            Status($"NOT IMPLEMENTED — {lmKind} {quant} module-timing probe is a stub");
            WriteReport(false, "ModuleTimingProbe is a scaffold; per-module timing not implemented yet.");
            BatchExit(0);
            yield break;
        }

        void WriteReport(bool success, string note)
        {
            try
            {
                Directory.CreateDirectory(reportDirectory);
                var sb = new StringBuilder();
                sb.AppendLine($"<!-- success: {success} -->");
                sb.AppendLine($"# Per-module timing ({lmKind}, {quant})");
                sb.AppendLine();
                sb.AppendLine(note);
                File.WriteAllText(Path.Combine(reportDirectory, "report.md"), sb.ToString());
                Debug.Log($"[ModuleTimingProbe] report written to {reportDirectory}");
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

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
    }
}
