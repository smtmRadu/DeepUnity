using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Boot-vs-framedrop tradeoff probe. Boots ONE model (kind + quant set on the component) several
    // times, once per weight-streaming budget in `budgetsMB` (the LLM.UploadBudgetBytes knob), and
    // for each setting records total boot wall time against the frame-drop it caused while the
    // weights streamed in. Lower budget = smoother frames, longer boot; higher = faster boot, fatter
    // hitches — this probe quantifies that curve so you can pick a budget per target device.
    //
    // Kernels are prewarmed ONCE up front (LMProbeCommon.Prewarm) so the one-time first-dispatch
    // compile spikes don't contaminate the per-config streaming measurement. Between configs the
    // model is released and a few idle frames let the driver/GC settle.
    //
    // Drive it via LMBootKnobProbeRunner (batch or interactive). One model kind per editor run.
    public class LMBootKnobProbe : MonoBehaviour
    {
        [Header("What to boot (one model per editor run)")]
        public ProbeModelKind model = ProbeModelKind.Qwen3_5_0_8B;
        public LLMQuant quant = LLMQuant.FP16;

        [Header("Knob sweep: weight-streaming budget, MB per frame")]
        public int[] budgetsMB = { 2, 4, 8, 16, 32 };

        [Header("Output / safety")]
        public string reportDirectory;
        public int settleFrames = 30;              // idle frames between configs (driver/GC settle)
        public float perConfigTimeoutSeconds = 120f;
        public float totalTimeoutSeconds = 1200f;

        struct Row { public int budgetMB; public int frame; public double ms; public string phase; }
        struct Result
        {
            public int budgetMB;
            public double ctorMs, readyMs, warmupMs, bootMs;
            public int frames, dropped60, dropped30;
            public double meanMs, p95Ms, worstMs;
            public int gcDelta;
        }

        readonly List<Row> rows = new List<Row>(16384);
        readonly List<Result> results = new List<Result>();
        readonly List<string> logLines = new List<string>(256);
        readonly System.Diagnostics.Stopwatch frameSw = new System.Diagnostics.Stopwatch();
        readonly System.Diagnostics.Stopwatch lifeSw = System.Diagnostics.Stopwatch.StartNew();

        LLM lm;
        bool capturing;
        int curBudgetMB;
        string phase = "idle";
        int globalFrame;
        bool finished;

        void Awake() { Application.logMessageReceived += OnLog; }
        void OnDestroy() { Application.logMessageReceived -= OnLog; }
        void OnLog(string condition, string stack, LogType type) { logLines.Add($"[{type}] {condition}"); }

        void Start()
        {
            Application.runInBackground = true;
            StartCoroutine(Guarded());
        }

        void Update()
        {
            if (finished) return;

            double ms = frameSw.Elapsed.TotalMilliseconds;
            frameSw.Restart();
            if (capturing)
                rows.Add(new Row { budgetMB = curBudgetMB, frame = globalFrame, ms = ms, phase = phase });
            globalFrame++;

            if (lifeSw.Elapsed.TotalSeconds > totalTimeoutSeconds)
            {
                Status("ERROR: total timeout");
                WriteReport(false);
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
                    WriteReport(false);
                    Exit(1);
                    yield break;
                }
                yield return cur;
            }
        }

        IEnumerator Run()
        {
            Status($"prewarming kernels for {LMProbeCommon.ModelLabel(model)} (once)");
            yield return LMProbeCommon.Prewarm(model);

            foreach (int mb in budgetsMB)
            {
                Status($"boot @ {mb} MB/frame budget");
                LLM.UploadBudgetBytes = mb * 1024 * 1024;
                curBudgetMB = mb;

                int gc0Start = GC.CollectionCount(0);
                var configSw = System.Diagnostics.Stopwatch.StartNew();

                // Start capturing one frame before the ctor so the blocking ctor frame lands in "load".
                phase = "load";
                frameSw.Restart();
                capturing = true;

                var ctorSw = System.Diagnostics.Stopwatch.StartNew();
                lm = LMProbeCommon.Build(model, quant);
                double ctorMs = ctorSw.Elapsed.TotalMilliseconds;

                bool timedOut = false;
                var perCfg = System.Diagnostics.Stopwatch.StartNew();
                while (!lm.IsReady)
                {
                    if (perCfg.Elapsed.TotalSeconds > perConfigTimeoutSeconds) { timedOut = true; break; }
                    yield return null;
                }
                double readyMs = configSw.Elapsed.TotalMilliseconds;

                double warmupMs = 0;
                if (!timedOut)
                {
                    phase = "warmup";
                    var wSw = System.Diagnostics.Stopwatch.StartNew();
                    yield return lm.Warmup();
                    warmupMs = wSw.Elapsed.TotalMilliseconds;
                }

                double bootMs = configSw.Elapsed.TotalMilliseconds;
                capturing = false;
                phase = "idle";

                var f = rows.Where(r => r.budgetMB == mb).Select(r => r.ms).OrderBy(x => x).ToArray();
                results.Add(new Result
                {
                    budgetMB = mb,
                    ctorMs = ctorMs,
                    readyMs = readyMs,
                    warmupMs = warmupMs,
                    bootMs = bootMs,
                    frames = f.Length,
                    dropped60 = f.Count(x => x > 16.7),
                    dropped30 = f.Count(x => x > 33.0),
                    meanMs = f.Length > 0 ? f.Average() : 0,
                    p95Ms = f.Length > 0 ? f[Math.Min(f.Length - 1, (int)(f.Length * 0.95))] : 0,
                    worstMs = f.Length > 0 ? f[f.Length - 1] : 0,
                    gcDelta = GC.CollectionCount(0) - gc0Start,
                });

                if (timedOut) Status($"WARNING: {mb} MB config timed out before ready");

                lm.Release();
                lm = null;
                for (int i = 0; i < settleFrames; i++) yield return null;
            }

            WriteReport(true);
            Status($"DONE — report at {reportDirectory}");
            Exit(0);
        }

        void WriteReport(bool success)
        {
            finished = true;
            // Force invariant culture so decimals are '.' (a comma-decimal locale would break JSON/CSV).
            var prevCulture = System.Threading.Thread.CurrentThread.CurrentCulture;
            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            try
            {
                Directory.CreateDirectory(reportDirectory);

                var csv = new StringBuilder("budget_mb,frame,ms,phase\n");
                foreach (var r in rows)
                    csv.Append(r.budgetMB).Append(',').Append(r.frame).Append(',')
                       .Append(r.ms.ToString("0.000")).Append(',').Append(r.phase).Append('\n');
                File.WriteAllText(Path.Combine(reportDirectory, "frames.csv"), csv.ToString());

                File.WriteAllText(Path.Combine(reportDirectory, "log.txt"), string.Join("\n", logLines));

                var md = new StringBuilder();
                md.AppendLine($"# Boot vs frame-drop tradeoff — {LMProbeCommon.ModelLabel(model)} {quant}");
                md.AppendLine();
                md.AppendLine($"- success: {success}");
                md.AppendLine($"- knob: LLM.UploadBudgetBytes (weight-streaming GPU budget per frame)");
                md.AppendLine($"- kernels prewarmed once up front (excluded from per-config frame stats)");
                md.AppendLine();
                md.AppendLine(LMProbeCommon.SystemInfoBlock());
                md.AppendLine("## Tradeoff (one boot per budget)");
                md.AppendLine();
                md.AppendLine("| budget MB | boot ms | ctor ms | →ready ms | warmup ms | frames | >16.7ms | >33ms | mean ms | p95 ms | worst ms | GC0 |");
                md.AppendLine("|---|---|---|---|---|---|---|---|---|---|---|---|");
                foreach (var r in results.OrderBy(r => r.budgetMB))
                    md.AppendLine($"| {r.budgetMB} | {r.bootMs:0} | {r.ctorMs:0} | {r.readyMs:0} | {r.warmupMs:0} | " +
                                  $"{r.frames} | {r.dropped60} | {r.dropped30} | {r.meanMs:0.00} | {r.p95Ms:0.00} | {r.worstMs:0.00} | {r.gcDelta} |");
                md.AppendLine();
                md.AppendLine("Lower budget → smoother frames (fewer >16.7/>33 ms, smaller worst slice) but longer boot. " +
                              "Pick the largest budget whose worst-frame is still under your device's frame target.");
                File.WriteAllText(Path.Combine(reportDirectory, "report.md"), md.ToString());

                // summary.json — machine + per-config metrics, for aggregating the 6 runs.
                var js = new StringBuilder();
                js.Append("{\n");
                js.Append("  \"probe\": \"boot_framedrop\",\n");
                js.Append("  \"model\": ").Append(LMProbeCommon.JsonStr(LMProbeCommon.ModelLabel(model))).Append(",\n");
                js.Append("  \"quant\": ").Append(LMProbeCommon.JsonStr(quant.ToString())).Append(",\n");
                js.Append("  \"success\": ").Append(success ? "true" : "false").Append(",\n");
                js.Append("  \"knob\": \"LLM.UploadBudgetBytes\",\n");
                js.Append("  \"machine\": ").Append(LMProbeCommon.MachineJson()).Append(",\n");
                js.Append("  \"configs\": [\n");
                var ordered = results.OrderBy(r => r.budgetMB).ToArray();
                for (int i = 0; i < ordered.Length; i++)
                {
                    var r = ordered[i];
                    js.Append("    {")
                      .Append($"\"budget_mb\": {r.budgetMB}, ")
                      .Append($"\"boot_ms\": {r.bootMs:0.0}, \"ctor_ms\": {r.ctorMs:0.0}, ")
                      .Append($"\"ready_ms\": {r.readyMs:0.0}, \"warmup_ms\": {r.warmupMs:0.0}, ")
                      .Append($"\"frames\": {r.frames}, \"dropped_60fps\": {r.dropped60}, \"dropped_30fps\": {r.dropped30}, ")
                      .Append($"\"mean_ms\": {r.meanMs:0.000}, \"p95_ms\": {r.p95Ms:0.000}, \"worst_ms\": {r.worstMs:0.000}, ")
                      .Append($"\"gc_gen0\": {r.gcDelta}")
                      .Append(i < ordered.Length - 1 ? "},\n" : "}\n");
                }
                js.Append("  ]\n}\n");
                File.WriteAllText(Path.Combine(reportDirectory, "summary.json"), js.ToString());

                Debug.Log($"[LMBootKnobProbe] report written to {reportDirectory}");
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

        static string StatusPath => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "boot_knob_probe_status.txt");
        void Status(string s)
        {
            try { Directory.CreateDirectory(Path.GetDirectoryName(StatusPath)); File.WriteAllText(StatusPath, $"[{DateTime.Now:HH:mm:ss}] {s}"); }
            catch { }
            Debug.Log("[LMBootKnobProbe] " + s);
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
