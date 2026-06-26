using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Batch-mode probe: boots Qwen3.5 and records per-frame main-thread time + GC collections
    // through ctor / weight load / warmup / system-prompt / first chat, then writes a report and
    // exits the editor. Verifies the streaming weight loader causes no frame hitches, and the
    // greedy chat reply doubles as a correctness check of the upload path.
    public class LMBootProbe : MonoBehaviour
    {
        public string reportDirectory;
        public int chatTokens = 24;
        public float timeoutSeconds = 900f;

        Qwen3_5ForCausalLM model;

        struct Row { public int frame; public double ms; public string phase; public int gc0; }
        readonly List<Row> rows = new List<Row>(16384);
        readonly List<string> logLines = new List<string>(256);
        readonly System.Diagnostics.Stopwatch frameSw = new System.Diagnostics.Stopwatch();
        readonly System.Diagnostics.Stopwatch totalSw = new System.Diagnostics.Stopwatch();
        string phase = "ctor";
        double ctorMs;
        int frameIdx;
        bool finished;

        void Awake()
        {
            Application.logMessageReceived += OnLog;
        }

        void OnDestroy()
        {
            Application.logMessageReceived -= OnLog;
        }

        void OnLog(string condition, string stackTrace, LogType type)
        {
            logLines.Add($"[{type}] {condition}");
        }

        void Start()
        {
            totalSw.Start();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            model = new Qwen3_5ForCausalLM();
            ctorMs = sw.Elapsed.TotalMilliseconds;
            phase = "load";
            frameSw.Restart();
            StartCoroutine(Run());
        }

        void Update()
        {
            if (finished) return;
            double ms = frameSw.Elapsed.TotalMilliseconds;
            frameSw.Restart();
            rows.Add(new Row { frame = frameIdx++, ms = ms, phase = phase, gc0 = GC.CollectionCount(0) });

            if (totalSw.Elapsed.TotalSeconds > timeoutSeconds)
            {
                WriteReport("(timed out)", success: false);
                Exit(2);
            }
        }

        IEnumerator Run()
        {
            while (!model.IsReady) yield return null;
            phase = "warmup";
            yield return model.Warmup();
            phase = "initchat";
            yield return model.InitializeChat("You are a grumpy tavern owner in a medieval fantasy town. Stay in character.");
            phase = "chat";
            var reply = new StringBuilder();
            // temperature 0 = greedy, but penalties apply before the argmax — keep the presence
            // penalty this probe always ran with so its reply stays comparable across runs
            yield return model.Chat("Hello! What's on the menu today?", t => reply.Append(t),
                max_new_tokens: chatTokens, temperature: 0f,
                presence_penalty: model.Config.DefaultPresencePenalty);
            phase = "done";
            WriteReport(reply.ToString(), success: true);
            Exit(0);
        }

        void WriteReport(string chatReply, bool success)
        {
            finished = true;
            try
            {
                Directory.CreateDirectory(reportDirectory);

                var csv = new StringBuilder("frame,ms,phase,gc0\n");
                foreach (var r in rows)
                    csv.Append(r.frame).Append(',').Append(r.ms.ToString("0.000")).Append(',')
                       .Append(r.phase).Append(',').Append(r.gc0).Append('\n');
                File.WriteAllText(Path.Combine(reportDirectory, "frames.csv"), csv.ToString());

                File.WriteAllText(Path.Combine(reportDirectory, "log.txt"), string.Join("\n", logLines));

                var md = new StringBuilder();
                md.AppendLine("# Qwen3.5 boot smoothness probe");
                md.AppendLine();
                md.AppendLine($"- success: {success}");
                md.AppendLine($"- constructor (blocking frame): {ctorMs:0.0} ms");
                md.AppendLine($"- total wall time: {totalSw.Elapsed.TotalSeconds:0.0} s");
                md.AppendLine();
                md.AppendLine("| phase | frames | mean ms | p95 ms | worst ms | frames >16.7ms | frames >33ms | GC gen0 |");
                md.AppendLine("|---|---|---|---|---|---|---|---|");
                foreach (var g in rows.GroupBy(r => r.phase))
                {
                    var ms = g.Select(r => r.ms).OrderBy(x => x).ToArray();
                    if (ms.Length == 0) continue;
                    // First load frame includes ctor leftovers; keep it, the table shows the truth.
                    double mean = ms.Average();
                    double p95 = ms[(int)(ms.Length * 0.95) >= ms.Length ? ms.Length - 1 : (int)(ms.Length * 0.95)];
                    int gcDelta = g.Last().gc0 - g.First().gc0;
                    md.AppendLine($"| {g.Key} | {ms.Length} | {mean:0.00} | {p95:0.00} | {ms.Last():0.00} | " +
                                  $"{g.Count(r => r.ms > 16.7)} | {g.Count(r => r.ms > 33.0)} | {gcDelta} |");
                }
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

                Debug.Log($"[LMBootProbe] Report written to {reportDirectory}");
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
    }
}
