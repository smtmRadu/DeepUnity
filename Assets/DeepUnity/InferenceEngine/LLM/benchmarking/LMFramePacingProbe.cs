using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Frame-pacing probe for task #20 (Qwen3.5 decode smoothness). Measures PER-FRAME wall times
    // while streaming tokens through the REAL interactive path (ForwardYielding + SampleYielding,
    // pumped one MoveNext per frame — exactly how NPCChatBase drives a Talk coroutine), and A/Bs
    // the shipped one-burst + async-readback decode against the pre-#20 behaviors:
    //   spread_sync  — yield-per-layer decode + blocking GetData sample (the old spiky path)
    //   burst_sync   — one-burst decode + blocking GetData sample      (isolates the burst change)
    //   burst_async  — one-burst decode + AsyncGPUReadback sample      (SHIPPED, task #20)
    // Every arm re-prefills the same ids and decodes greedily, so all three walk the identical
    // token chain — the GPU work is the same, only the pacing strategy differs. Reports spike
    // counts (>20/>33/>50 ms), mean/p95/max frame, tok/s and frames/token per arm.
    // Drive via LMFramePacingRunner (menu or ClaudeBridge); the runner restores the editor scene.
    public class LMFramePacingProbe : MonoBehaviour
    {
        [Header("What to boot (Qwen only — the #20 fix is Qwen-side)")]
        public ProbeModelKind model = ProbeModelKind.Qwen3_5_0_8B;
        public LLMQuant quant = LLMQuant.INT8;
        public KVQuant kvQuant = KVQuant.INT8;

        [Header("Workload")]
        public int prefillTokens = 192;  // conversation-sized prompt before decoding
        public int decodeTokens = 128;   // timed tokens per arm
        public int warmupTokens = 16;    // untimed decode steps per arm (stabilize dispatch)

        [Header("Output / safety")]
        public string reportDirectory;
        public float timeoutSeconds = 900f;

        // ---- per-frame sampler (Update delta, like FrameSpikeLogger but recording) ----
        readonly System.Diagnostics.Stopwatch frameSw = new System.Diagnostics.Stopwatch();
        List<double> rec;                 // active recording target (null = off)
        bool firstFrame = true;

        readonly System.Diagnostics.Stopwatch lifeSw = System.Diagnostics.Stopwatch.StartNew();
        bool finished;
        int prevVsync; int prevTarget;

        struct ArmResult
        {
            public string name;
            public double[] frames;       // per-frame ms during the timed decode
            public double wallMs;
            public int tokens;
        }
        readonly List<ArmResult> results = new List<ArmResult>();

        void Start()
        {
            Application.runInBackground = true;   // keep pumping while the editor is unfocused
            prevVsync = QualitySettings.vSyncCount;
            prevTarget = Application.targetFrameRate;
            QualitySettings.vSyncCount = 0;       // uncapped: raw frame cost, not vsync quanta
            Application.targetFrameRate = -1;
            StartCoroutine(Guarded());
        }

        void OnDestroy()
        {
            QualitySettings.vSyncCount = prevVsync;
            Application.targetFrameRate = prevTarget;
            Qwen3_5Modeling.Qwen3_5Model.DebugSpreadDecode = false;  // never leak the diagnostic flag
        }

        void Update()
        {
            double ms = frameSw.Elapsed.TotalMilliseconds;
            frameSw.Restart();
            if (!firstFrame && rec != null) rec.Add(ms);
            firstFrame = false;

            if (!finished && lifeSw.Elapsed.TotalSeconds > timeoutSeconds)
            {
                Status("ERROR: timeout");
                WriteReport(false, "(timed out)");
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
                    WriteReport(false, ex.Message);
                    Exit(1);
                    yield break;
                }
                yield return cur;
            }
        }

        IEnumerator Run()
        {
            Status($"constructing {LMProbeCommon.ModelLabel(model)} {quant} kv{kvQuant}");
            LLM lm = LMProbeCommon.Build(model, quant, kvQuant);
            while (!lm.IsReady) yield return null;

            if (!(lm is Qwen3_5ForCausalLM qwen))
            {
                Status("ERROR: frame-pacing probe is Qwen-only");
                WriteReport(false, "non-Qwen model kind");
                Exit(1);
                yield break;
            }

            Status("warmup (kernel compile)");
            yield return lm.Warmup();
            var m = qwen.model;

            // Deterministic prompt ids — every arm re-prefills the SAME prompt, so the greedy
            // decode chain (and therefore the GPU work) is identical across arms.
            var ids = new float[prefillTokens];
            for (int i = 0; i < prefillTokens; i++) ids[i] = 1000 + (i * 7919) % 30000;

            string[] armNames = { "spread_sync", "burst_sync", "burst_async" };
            for (int a = 0; a < armNames.Length; a++)
            {
                bool spread = a == 0;
                bool asyncSample = a == 2;
                Qwen3_5Modeling.Qwen3_5Model.DebugSpreadDecode = spread;

                Status($"arm {armNames[a]}: prefill x{prefillTokens}");
                m.ResetCache();
                var pe = m.ForwardYielding(Tensor.Constant(ids), useCache: true, lastPosOnly: true);
                while (pe.MoveNext()) yield return pe.Current;
                int tok = m.Sample(0f, 0, 1f, 0f);   // consume prefill logits identically per arm

                Status($"arm {armNames[a]}: warmup decode x{warmupTokens}");
                for (int i = 0; i < warmupTokens; i++)
                {
                    var st = Step(m, tok, asyncSample);
                    while (st.MoveNext()) yield return st.Current;
                    tok = lastTok;
                }

                Status($"arm {armNames[a]}: timed decode x{decodeTokens}");
                rec = new List<double>(8192);
                var wall = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < decodeTokens; i++)
                {
                    var st = Step(m, tok, asyncSample);
                    while (st.MoveNext()) yield return st.Current;
                    tok = lastTok;
                }
                wall.Stop();
                var frames = rec.ToArray();
                rec = null;
                results.Add(new ArmResult { name = armNames[a], frames = frames, wallMs = wall.Elapsed.TotalMilliseconds, tokens = decodeTokens });
            }
            Qwen3_5Modeling.Qwen3_5Model.DebugSpreadDecode = false;

            lm.Release();
            WriteReport(true, null);
            Status("DONE");
            Exit(0);
        }

        int lastTok;
        readonly int[] sampled = new int[1];

        // One decode step through the real streaming machinery (mirrors Qwen3_5.cs's generate loop).
        IEnumerator Step(Qwen3_5Modeling.Qwen3_5Model m, int tok, bool asyncSample)
        {
            var e = m.ForwardYielding(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;
            if (asyncSample)
            {
                var s = m.SampleYielding(0f, 0, 1f, 0f, 0f, 1f, sampled);
                while (s.MoveNext()) yield return s.Current;
                lastTok = sampled[0];
            }
            else
            {
                lastTok = m.Sample(0f, 0, 1f, 0f);   // blocking GetData (the pre-#20 stall)
            }
        }

        // ---------------- metrics + report ----------------

        struct ArmStats
        {
            public string name;
            public int frames, over20, over33, over50;
            public double meanMs, p95Ms, maxMs, tokS, msPerTok, framesPerTok;
        }

        ArmStats Stats(ArmResult r)
        {
            var s = new ArmStats { name = r.name, frames = r.frames.Length };
            if (r.frames.Length > 0)
            {
                var sorted = (double[])r.frames.Clone();
                Array.Sort(sorted);
                double sum = 0;
                foreach (double f in r.frames)
                {
                    sum += f;
                    if (f > 20.0) s.over20++;
                    if (f > 33.4) s.over33++;
                    if (f > 50.0) s.over50++;
                }
                s.meanMs = sum / r.frames.Length;
                s.p95Ms = sorted[(int)(0.95 * (sorted.Length - 1))];
                s.maxMs = sorted[sorted.Length - 1];
            }
            s.tokS = r.wallMs > 0 ? r.tokens / (r.wallMs / 1000.0) : 0;
            s.msPerTok = r.tokens > 0 ? r.wallMs / r.tokens : 0;
            s.framesPerTok = r.tokens > 0 ? (double)r.frames.Length / r.tokens : 0;
            return s;
        }

        void WriteReport(bool success, string note)
        {
            finished = true;
            var prevCulture = System.Threading.Thread.CurrentThread.CurrentCulture;
            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            try
            {
                Directory.CreateDirectory(reportDirectory);

                var stats = new List<ArmStats>();
                foreach (var r in results)
                {
                    stats.Add(Stats(r));
                    var csv = new StringBuilder("frame_index,ms\n");
                    for (int i = 0; i < r.frames.Length; i++)
                        csv.Append(i).Append(',').Append(r.frames[i].ToString("0.000")).Append('\n');
                    File.WriteAllText(Path.Combine(reportDirectory, $"frames_{r.name}.csv"), csv.ToString());
                }

                // Verdict on the SHIPPED arm only (the old arms are the documented baseline):
                // no frame above 33 ms and at most ~2% above 20 ms during steady decode.
                ArmStats shipped = stats.Count > 0 ? stats[stats.Count - 1] : default;
                bool pass = success && stats.Count == 3 &&
                            shipped.over33 == 0 &&
                            shipped.over20 <= Math.Max(1, shipped.frames / 50) &&
                            shipped.tokS > 5;

                var md = new StringBuilder();
                md.AppendLine($"# Decode frame-pacing (#20) — {LMProbeCommon.ModelLabel(model)} {quant} kv{kvQuant}");
                md.AppendLine();
                md.AppendLine($"- success: {success}{(note != null ? $" | note: {note}" : "")}");
                md.AppendLine($"- workload: prefill {prefillTokens} → {decodeTokens} greedy tokens per arm (identical chain, vsync OFF)");
                md.AppendLine($"- pump: one MoveNext per frame (same as NPCChatBase Talk coroutine)");
                md.AppendLine($"- verdict (shipped burst_async): **{(pass ? "PASS" : "FAIL")}** — gate: 0 frames >33 ms, ≤2% >20 ms");
                md.AppendLine();
                md.AppendLine(LMProbeCommon.SystemInfoBlock());
                md.AppendLine("## Arms");
                md.AppendLine();
                md.AppendLine("| arm | frames | >20ms | >33ms | >50ms | mean ms | p95 ms | max ms | tok/s | ms/tok | frames/tok |");
                md.AppendLine("|---|---|---|---|---|---|---|---|---|---|---|");
                foreach (var s in stats)
                    md.AppendLine($"| {s.name} | {s.frames} | {s.over20} | {s.over33} | {s.over50} | {s.meanMs:0.00} | {s.p95Ms:0.00} | {s.maxMs:0.0} | {s.tokS:0.0} | {s.msPerTok:0.00} | {s.framesPerTok:0.0} |");
                md.AppendLine();
                md.AppendLine("frames/tok drives the in-game feel: at 60 fps a decode-bound NPC streams ≈ 60 / frames_per_tok tok/s;");
                md.AppendLine("spikes (>33 ms) are the visible hitches #20 set out to kill.");
                File.WriteAllText(Path.Combine(reportDirectory, "report.md"), md.ToString());

                var js = new StringBuilder();
                js.Append("{\n");
                js.Append("  \"probe\": \"frame_pacing\",\n");
                js.Append("  \"model\": ").Append(LMProbeCommon.JsonStr(LMProbeCommon.ModelLabel(model))).Append(",\n");
                js.Append("  \"quant\": ").Append(LMProbeCommon.JsonStr(quant.ToString())).Append(",\n");
                js.Append("  \"kv\": ").Append(LMProbeCommon.JsonStr(kvQuant.ToString())).Append(",\n");
                js.Append("  \"success\": ").Append(success ? "true" : "false").Append(",\n");
                js.Append("  \"pass\": ").Append(pass ? "true" : "false").Append(",\n");
                if (note != null) js.Append("  \"note\": ").Append(LMProbeCommon.JsonStr(note)).Append(",\n");
                js.Append($"  \"prefill_tokens\": {prefillTokens}, \"decode_tokens\": {decodeTokens},\n");
                js.Append("  \"machine\": ").Append(LMProbeCommon.MachineJson()).Append(",\n");
                js.Append("  \"arms\": [\n");
                for (int i = 0; i < stats.Count; i++)
                {
                    var s = stats[i];
                    js.Append($"    {{\"name\": {LMProbeCommon.JsonStr(s.name)}, \"frames\": {s.frames}, \"over20\": {s.over20}, \"over33\": {s.over33}, \"over50\": {s.over50}, ")
                      .Append($"\"mean_ms\": {s.meanMs:0.000}, \"p95_ms\": {s.p95Ms:0.000}, \"max_ms\": {s.maxMs:0.0}, \"tok_s\": {s.tokS:0.0}, \"ms_per_tok\": {s.msPerTok:0.00}, \"frames_per_tok\": {s.framesPerTok:0.00}}}")
                      .Append(i < stats.Count - 1 ? ",\n" : "\n");
                }
                js.Append("  ]\n}\n");
                File.WriteAllText(Path.Combine(reportDirectory, "summary.json"), js.ToString());

                File.WriteAllText(Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "framepacing_done.txt"),
                                  (pass ? "PASS" : "FAIL") + " | " + reportDirectory);
                Debug.Log($"[LMFramePacingProbe] report written to {reportDirectory}");
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

        static string StatusPath => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge", "framepacing_status.txt");
        void Status(string s)
        {
            try { Directory.CreateDirectory(Path.GetDirectoryName(StatusPath)); File.WriteAllText(StatusPath, $"[{DateTime.Now:HH:mm:ss}] {s}"); }
            catch { }
            Debug.Log("[LMFramePacingProbe] " + s);
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
