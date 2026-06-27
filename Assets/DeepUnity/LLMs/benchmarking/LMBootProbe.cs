using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Batch-mode boot/load probe that mirrors the PRODUCTION boot sequence so the numbers are real:
    //
    //   prewarm  -> Prewarm(): compiles every compute kernel (one per frame; the big DeltaNet one is
    //               ~hundreds of ms) AND kicks off the async tokenizer parse. This is the LOADING
    //               SCREEN cost — frame spikes here are expected and hidden from the player.
    //   ctor     -> new model: cheap, builds the weight manifest + starts the background stream.
    //   load     -> stream weights to the GPU under the per-frame upload budget. THIS is the in-game
    //               streaming phase the upload knobs are tuned for — it should stay <16.7 ms/frame.
    //   warmup   -> Warmup(): kernels already prewarmed, so this is fast.
    //   initchat -> system-prompt prefill (first real forward). Model is now READY FOR INPUT.
    //   chat     -> first greedy reply (correctness check; NOT counted in total boot).
    //
    // Skipping Prewarm (the old probe did) dumped the tokenizer-parse GC + kernel compiles into the
    // load phase, faking a ~270 ms streaming spike and undercounting total boot. We measure tokenizer
    // ready-time separately (TokenizerReady) and report total_boot_s = start -> input-ready.
    //
    // Drive via LMBenchmarkProbeRunner.RunBootProbe (-model/-quant). One config per editor run.
    public class LMBootProbe : MonoBehaviour
    {
        [Header("What to boot (one model per editor run)")]
        public ProbeModelKind model = ProbeModelKind.Qwen3_5_0_8B;
        public LLMQuant quant = LLMQuant.FP16;
        public KVQuant kvQuant = KVQuant.FP16;

        [Header("Boot")]
        public int chatTokens = 24;
        public float timeoutSeconds = 900f;

        [Header("Output")]
        public string reportDirectory;

        LLM lm;

        struct Row { public int frame; public double ms; public string phase; public int gc0; }
        readonly List<Row> rows = new List<Row>(16384);
        readonly List<string> logLines = new List<string>(256);
        readonly System.Diagnostics.Stopwatch frameSw = new System.Diagnostics.Stopwatch();
        readonly System.Diagnostics.Stopwatch totalSw = new System.Diagnostics.Stopwatch();
        string phase = "prewarm";
        double ctorMs, prewarmMs, tokenizerReadyMs = -1, readyMs, initchatMs, inputReadyMs;
        int frameIdx;
        bool finished;

        void Awake() => Application.logMessageReceived += OnLog;
        void OnDestroy() => Application.logMessageReceived -= OnLog;
        void OnLog(string condition, string stackTrace, LogType type) => logLines.Add($"[{type}] {condition}");

        void Start()
        {
            Application.runInBackground = true;
            totalSw.Start();
            frameSw.Restart();
            StartCoroutine(Guarded());
        }

        void Update()
        {
            if (finished) return;
            double ms = frameSw.Elapsed.TotalMilliseconds;
            frameSw.Restart();
            rows.Add(new Row { frame = frameIdx++, ms = ms, phase = phase, gc0 = GC.CollectionCount(0) });
            if (totalSw.Elapsed.TotalSeconds > timeoutSeconds)
            {
                Status("ERROR: timeout");
                WriteReport("(timed out)", success: false);
                Exit(2);
            }
        }

        IEnumerator Guarded()
        {
            var e = Run();
            while (true)
            {
                object cur;
                try { if (!e.MoveNext()) break; cur = e.Current; }
                catch (Exception ex)
                {
                    Status("ERROR: " + ex.Message + "\n" + ex.StackTrace);
                    WriteReport("(" + ex.Message + ")", success: false);
                    Exit(1);
                    yield break;
                }
                yield return cur;
            }
        }

        IEnumerator Run()
        {
            // ---- prewarm: kernels (1/frame) + async tokenizer kickoff (loading-screen cost) ----
            phase = "prewarm";
            Status($"prewarm {LMProbeCommon.ModelLabel(model)} {quant} (kernels + tokenizer)");
            var pw = LMProbeCommon.Prewarm(model);
            while (pw.MoveNext()) yield return pw.Current;
            prewarmMs = totalSw.Elapsed.TotalMilliseconds;

            // ---- ctor: cheap; gets the cached (still-parsing) tokenizer + starts weight stream ----
            phase = "ctor";
            var sw = System.Diagnostics.Stopwatch.StartNew();
            lm = LMProbeCommon.Build(model, quant, kvQuant);
            ctorMs = sw.Elapsed.TotalMilliseconds;
            if (lm.TokenizerReady) tokenizerReadyMs = totalSw.Elapsed.TotalMilliseconds; // finished during prewarm

            // ---- load: stream weights to GPU (the 60fps-relevant phase) ----
            phase = "load";
            Status("weight stream");
            while (!lm.IsReady)
            {
                if (tokenizerReadyMs < 0 && lm.TokenizerReady) tokenizerReadyMs = totalSw.Elapsed.TotalMilliseconds;
                yield return null;
            }
            if (tokenizerReadyMs < 0) tokenizerReadyMs = totalSw.Elapsed.TotalMilliseconds;
            readyMs = totalSw.Elapsed.TotalMilliseconds;

            // ---- warmup: kernels already compiled -> fast ----
            phase = "warmup";
            Status("warmup");
            yield return lm.Warmup();

            // ---- initchat: system-prompt prefill -> model is READY FOR INPUT here ----
            phase = "initchat";
            Status("system-prompt prefill");
            double t0init = totalSw.Elapsed.TotalMilliseconds;
            yield return lm.InitializeChat("You are a grumpy tavern owner in a medieval fantasy town. Stay in character.");
            initchatMs = totalSw.Elapsed.TotalMilliseconds - t0init;
            inputReadyMs = totalSw.Elapsed.TotalMilliseconds;   // <-- TOTAL FULL BOOT until input

            // ---- chat: first reply (correctness only; not part of boot total) ----
            phase = "chat";
            Status("first chat (correctness)");
            var reply = new StringBuilder();
            yield return lm.Chat("Hello! What's on the menu today?", t => reply.Append(t),
                max_new_tokens: chatTokens, temperature: 0f,
                presence_penalty: lm.Config.DefaultPresencePenalty);
            phase = "done";
            lm.Release();
            WriteReport(reply.ToString(), success: true);
            Status($"DONE — report at {reportDirectory}");
            Exit(0);
        }

        struct PhaseStat
        {
            public string phase; public int frames; public double sumMs, meanMs, p95Ms, worstMs;
            public int over16, over33, gc;
        }

        List<PhaseStat> PhaseStats()
        {
            var order = new[] { "prewarm", "ctor", "load", "warmup", "initchat", "chat" };
            var stats = new List<PhaseStat>();
            foreach (var ph in order)
            {
                var g = rows.Where(r => r.phase == ph).ToArray();
                if (g.Length == 0) continue;
                var ms = g.Select(r => r.ms).OrderBy(x => x).ToArray();
                int p95i = Math.Min(ms.Length - 1, (int)(ms.Length * 0.95));
                stats.Add(new PhaseStat
                {
                    phase = ph, frames = ms.Length, sumMs = ms.Sum(), meanMs = ms.Average(),
                    p95Ms = ms[p95i], worstMs = ms[ms.Length - 1],
                    over16 = g.Count(r => r.ms > 16.7), over33 = g.Count(r => r.ms > 33.0),
                    gc = g.Last().gc0 - g.First().gc0,
                });
            }
            return stats;
        }

        void WriteReport(string chatReply, bool success)
        {
            finished = true;
            var prevCulture = System.Threading.Thread.CurrentThread.CurrentCulture;
            System.Threading.Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            try
            {
                Directory.CreateDirectory(reportDirectory);

                var csv = new StringBuilder("frame,ms,phase,gc0\n");
                foreach (var r in rows)
                    csv.Append(r.frame).Append(',').Append(r.ms.ToString("0.000")).Append(',')
                       .Append(r.phase).Append(',').Append(r.gc0).Append('\n');
                File.WriteAllText(Path.Combine(reportDirectory, "frames.csv"), csv.ToString());
                File.WriteAllText(Path.Combine(reportDirectory, "log.txt"), string.Join("\n", logLines));

                var stats = PhaseStats();
                PhaseStat? loadStat = stats.Any(s => s.phase == "load") ? stats.First(s => s.phase == "load") : (PhaseStat?)null;
                double weightStreamS = (loadStat?.sumMs ?? 0) / 1000.0;
                double loadWorst = loadStat?.worstMs ?? 0;
                int loadOver16 = loadStat?.over16 ?? 0;
                int loadOver33 = loadStat?.over33 ?? 0;
                double totalBootS = inputReadyMs / 1000.0;
                double totalWallS = totalSw.Elapsed.TotalSeconds;
                int gcGen0 = rows.Count > 0 ? rows[rows.Count - 1].gc0 - rows[0].gc0 : 0;

                // ---- report.md ----
                var md = new StringBuilder();
                md.AppendLine($"# Boot / load — {LMProbeCommon.ModelLabel(model)} {quant}");
                md.AppendLine();
                md.AppendLine($"- success: {success}");
                md.AppendLine($"- **total boot until input-ready: {totalBootS:0.00} s** (prewarm + ctor + stream + warmup + system-prompt prefill)");
                md.AppendLine($"- prewarm (kernels+tokenizer kickoff): {prewarmMs:0.0} ms | ctor: {ctorMs:0.0} ms");
                md.AppendLine($"- tokenizer ready at: {tokenizerReadyMs:0.0} ms | weights+tokenizer ready at: {readyMs:0.0} ms | system-prompt prefill: {initchatMs:0.0} ms");
                md.AppendLine($"- weight STREAM (in-game phase): {weightStreamS:0.00} s, worst frame {loadWorst:0.00} ms, frames >16.7ms: {loadOver16}, >33ms: {loadOver33}");
                md.AppendLine($"- total wall incl. first chat: {totalWallS:0.0} s | GC gen0 total: {gcGen0}");
                md.AppendLine();
                md.AppendLine(LMProbeCommon.SystemInfoBlock());
                md.AppendLine("| phase | frames | mean ms | p95 ms | worst ms | >16.7ms | >33ms | GC gen0 |");
                md.AppendLine("|---|---|---|---|---|---|---|---|");
                foreach (var s in stats)
                    md.AppendLine($"| {s.phase} | {s.frames} | {s.meanMs:0.00} | {s.p95Ms:0.00} | {s.worstMs:0.00} | {s.over16} | {s.over33} | {s.gc} |");
                md.AppendLine();
                md.AppendLine("## Worst 10 frames");
                md.AppendLine("| frame | phase | ms |");
                md.AppendLine("|---|---|---|");
                foreach (var r in rows.OrderByDescending(r => r.ms).Take(10))
                    md.AppendLine($"| {r.frame} | {r.phase} | {r.ms:0.00} |");
                md.AppendLine();
                md.AppendLine("## Greedy chat reply (correctness check)");
                md.AppendLine("```");
                md.AppendLine(chatReply);
                md.AppendLine("```");
                File.WriteAllText(Path.Combine(reportDirectory, "report.md"), md.ToString());

                // ---- summary.json ----
                var js = new StringBuilder();
                js.Append("{\n");
                js.Append("  \"probe\": \"boot_load\",\n");
                js.Append("  \"model\": ").Append(LMProbeCommon.JsonStr(LMProbeCommon.ModelLabel(model))).Append(",\n");
                js.Append("  \"quant\": ").Append(LMProbeCommon.JsonStr(quant.ToString())).Append(",\n");
                js.Append("  \"kv\": ").Append(LMProbeCommon.JsonStr(kvQuant.ToString())).Append(",\n");
                js.Append("  \"success\": ").Append(success ? "true" : "false").Append(",\n");
                js.Append($"  \"total_boot_s\": {totalBootS:0.00},\n");
                js.Append($"  \"prewarm_ms\": {prewarmMs:0.0},\n");
                js.Append($"  \"ctor_ms\": {ctorMs:0.0},\n");
                js.Append($"  \"tokenizer_ready_ms\": {tokenizerReadyMs:0.0},\n");
                js.Append($"  \"ready_ms\": {readyMs:0.0},\n");
                js.Append($"  \"initchat_ms\": {initchatMs:0.0},\n");
                js.Append($"  \"weight_stream_s\": {weightStreamS:0.00},\n");
                js.Append($"  \"load_worst_frame_ms\": {loadWorst:0.00},\n");
                js.Append($"  \"load_frames_over_16ms\": {loadOver16},\n");
                js.Append($"  \"load_frames_over_33ms\": {loadOver33},\n");
                js.Append($"  \"total_wall_s\": {totalWallS:0.0},\n");
                js.Append($"  \"gc_gen0\": {gcGen0},\n");
                js.Append("  \"chat_reply\": ").Append(LMProbeCommon.JsonStr(chatReply)).Append(",\n");
                js.Append("  \"machine\": ").Append(LMProbeCommon.MachineJson()).Append(",\n");
                js.Append("  \"phases\": [\n");
                for (int i = 0; i < stats.Count; i++)
                {
                    var s = stats[i];
                    js.Append($"    {{\"phase\": {LMProbeCommon.JsonStr(s.phase)}, \"frames\": {s.frames}, " +
                              $"\"mean_ms\": {s.meanMs:0.00}, \"p95_ms\": {s.p95Ms:0.00}, \"worst_ms\": {s.worstMs:0.00}, " +
                              $"\"over_16ms\": {s.over16}, \"over_33ms\": {s.over33}, \"gc\": {s.gc}}}")
                      .Append(i < stats.Count - 1 ? ",\n" : "\n");
                }
                js.Append("  ]\n}\n");
                File.WriteAllText(Path.Combine(reportDirectory, "summary.json"), js.ToString());

                Debug.Log($"[LMBootProbe] report written to {reportDirectory}");
            }
            catch (Exception e) { Debug.LogException(e); }
            finally { System.Threading.Thread.CurrentThread.CurrentCulture = prevCulture; }
        }

        static string StatusPath => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "boot_probe_status.txt");
        void Status(string s)
        {
            try { Directory.CreateDirectory(Path.GetDirectoryName(StatusPath)); File.WriteAllText(StatusPath, $"[{DateTime.Now:HH:mm:ss}] {s}"); }
            catch { }
            Debug.Log("[LMBootProbe] " + s);
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
