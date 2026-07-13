using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Reproduces EXACTLY what an NPC prefetch-zone entry does and measures where the reported
    // "instant freeze" comes from, across the three real-flow scenarios:
    //   A. COLD  Acquire — no Prewarm ran (worst case: player enters the zone at scene start,
    //            racing the Start() prewarm coroutine).
    //   B. WARM  Acquire — after Qwen3_5ForCausalLM.Prewarm() completed (the intended flow).
    //   C. RE-ENTRY Acquire — walk out (pool Release) and back in with everything cached.
    // For each: synchronous ctor wall ms + boot-phase split + per-frame times while the
    // governor-replica slow window streams the weights.
    // Output: ProbeLogs/zone_entry_probe.md + .done marker (bridge-orchestrated via ZoneEntryRunner).
    public class ZoneEntryProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/zone_entry_probe.md";
        const string MarkerPath = "ProbeLogs/zone_entry_probe.done";

        readonly StringBuilder sb = new StringBuilder();
        LLM llm;
        int phase = -1;          // -1 waiting for prewarm, 0 settle, 1 load window, 2 done
        int settleFrames;
        double acquireMs;
        int fullBudget;
        float slowDeadline, loadStart;
        readonly List<float> load = new List<float>();
        int scenario;            // 0 = warm (post-prewarm), 1 = re-entry
        bool coldDone;

        void Start()
        {
            sb.AppendLine("# Zone-entry freeze attribution (Qwen3.5-0.8B int8, governor replica)");
            // Scenario A first: a COLD Acquire+Release before any prewarm — replicates entering
            // the zone at scene start, racing the prewarm coroutine. Release immediately after
            // measuring the ctor (we only want the synchronous cost here).
            int g0 = System.GC.CollectionCount(0), g1 = System.GC.CollectionCount(1), g2 = System.GC.CollectionCount(2);
            var swCold = System.Diagnostics.Stopwatch.StartNew();
            var cold = LLMPool.Acquire("Qwen3.5-0.8B", LLMQuant.INT8, KVQuant.INT8);
            double coldMs = swCold.Elapsed.TotalMilliseconds;
            AppendCtor("A. COLD Acquire (prewarm NOT run — zone entered at scene start)", coldMs, cold);
            AppendGc(g0, g1, g2);
            LLMPool.Release(cold);
            coldDone = true;

            StartCoroutine(PrewarmThenGo());
        }

        IEnumerator PrewarmThenGo()
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            yield return Qwen3_5ForCausalLM.Prewarm();
            sb.AppendLine($"\n- Prewarm coroutine total: {sw.Elapsed.TotalMilliseconds:0} ms (spread over frames)");
            for (int i = 0; i < 30; i++) yield return null;   // let the tokenizer task settle
            phase = 0;
        }

        void AppendCtor(string title, double ms, LLM m)
        {
            sb.AppendLine($"\n## {title}");
            sb.AppendLine($"- **synchronous Acquire (ctor): {ms:0.0} ms**");
            if (m is Qwen3_5ForCausalLM q && q.model?.weights != null)
            {
                var w = q.model.weights;
                sb.AppendLine($"- split: tokenizer {w.bootTokenizerMs:0.0} | kernels {w.bootKernelsMs:0.0} | " +
                              $"weights-alloc {w.allocMs:0.0} | kv-cache {w.bootCacheMs:0.0} | " +
                              $"rope-kick {w.bootRopeMs:0.0} | scratch {w.bootScratchMs:0.0} ms");
            }
            sb.AppendLine($"- boot trace: {Qwen3_5ForCausalLM.BootTrace}");
        }

        void AppendGc(int g0, int g1, int g2)
        {
            sb.AppendLine($"- GC collections during Acquire: gen0 +{System.GC.CollectionCount(0) - g0}, " +
                          $"gen1 +{System.GC.CollectionCount(1) - g1}, gen2 +{System.GC.CollectionCount(2) - g2}");
        }

        void Update()
        {
            if (!coldDone) return;
            float dt = Time.unscaledDeltaTime * 1000f;
            switch (phase)
            {
                case 0:
                    if (++settleFrames >= 60)
                    {
                        int g0 = System.GC.CollectionCount(0), g1 = System.GC.CollectionCount(1), g2 = System.GC.CollectionCount(2);
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        llm = LLMPool.Acquire("Qwen3.5-0.8B", LLMQuant.INT8, KVQuant.INT8);
                        acquireMs = sw.Elapsed.TotalMilliseconds;
                        AppendCtor(scenario == 0
                            ? "B. WARM Acquire (after Prewarm — the intended zone-entry flow)"
                            : "C. RE-ENTRY Acquire (walked out and back in, everything cached)", acquireMs, llm);
                        AppendGc(g0, g1, g2);
                        fullBudget = LLM.UploadBudgetBytes;
                        slowDeadline = Time.unscaledTime + 3f;
                        loadStart = Time.unscaledTime;
                        load.Clear();
                        phase = 1;
                    }
                    break;

                case 1:
                    load.Add(dt);
                    if (Time.unscaledTime < slowDeadline && !llm.IsReady)
                    {
                        long remaining = llm.TotalWeightBytes - llm.UploadedWeightBytes;
                        if (remaining > 0)
                        {
                            long frames = (long)Mathf.Max(1f, (slowDeadline - Time.unscaledTime) * 60f);
                            LLM.UploadBudgetBytes = (int)System.Math.Min(fullBudget,
                                System.Math.Max(64 * 1024, remaining / frames));
                        }
                    }
                    else
                        LLM.UploadBudgetBytes = fullBudget;

                    if (llm.IsReady || Time.unscaledTime - loadStart > 90f)
                    {
                        LLM.UploadBudgetBytes = fullBudget;
                        AppendLoadStats();
                        if (scenario == 0)
                        {
                            // walk out of the zone: release, settle, then re-enter (scenario C)
                            LLMPool.Release(llm); llm = null;
                            scenario = 1; settleFrames = 0; phase = 0;
                        }
                        else
                        {
                            // scenario D: the FULL LLM+TTS zone-entry bundle, exactly what
                            // NPCChatBase.Update does for Velmire — Kokoro slow prefetch +
                            // kernel prewarm on the SAME frame as the (lingering, so instant)
                            // LLM acquire. This is the part the earlier scenarios never covered.
                            LLMPool.Release(llm); llm = null;   // lingers — like walking out
                            kk = gameObject.AddComponent<KokoroVoice>();
                            kk.loadOnStart = false;
                            kk.streaming = true;
                            settleFrames = 0; phase = 3;
                        }
                    }
                    break;

                case 3:
                    if (++settleFrames >= 60)
                    {
                        load.Clear();
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        kk.SlowPrefetchNow(3f);
                        kk.PrewarmKernels();
                        llm = LLMPool.Acquire("Qwen3.5-0.8B", LLMQuant.INT8, KVQuant.INT8);
                        double bundleMs = sw.Elapsed.TotalMilliseconds;
                        sb.AppendLine("\n## D. FULL zone entry (Kokoro slow-prefetch + prewarm + LLM acquire, same frame)");
                        sb.AppendLine($"- **synchronous bundle cost: {bundleMs:0.0} ms** (llm was lingering — its share ~0)");
                        loadStart = Time.unscaledTime;
                        phase = 4;
                    }
                    break;

                case 4:
                    load.Add(dt);
                    if (Time.unscaledTime - loadStart > 10f)   // 10 s window: covers prewarm + stream
                    {
                        AppendLoadStats();
                        Directory.CreateDirectory("ProbeLogs");
                        File.WriteAllText(ReportPath, sb.ToString());
                        File.WriteAllText(MarkerPath, "done");
                        Debug.Log($"[ZoneEntryProbe] report -> {ReportPath}");
                        phase = 2;
                    }
                    break;
            }
        }

        KokoroVoice kk;

        void AppendLoadStats()
        {
            int over17 = 0, over33 = 0, worstIdx = -1; float worst = 0;
            for (int i = 0; i < load.Count; i++)
            {
                if (load[i] > 16.7f) over17++;
                if (load[i] > 33.4f) over33++;
                if (load[i] > worst) { worst = load[i]; worstIdx = i; }
            }
            if (llm is Qwen3_5ForCausalLM q && q.model?.weights != null)
                sb.AppendLine($"- upload: {q.model.weights.uploadMs:0} ms over {q.model.weights.uploadFrames} frames, " +
                              $"worst single slice {q.model.weights.worstUploadMs:0.0} ms");
            sb.AppendLine($"- load window: {load.Count} frames, >16.7 ms: {over17}, >33.4 ms: {over33}, " +
                          $"worst {worst:0.0} ms @ #{worstIdx} (frame 0 = frame AFTER Acquire)");
            var idx = new List<int>(); for (int i = 0; i < load.Count; i++) idx.Add(i);
            idx.Sort((a, b) => load[b].CompareTo(load[a]));
            var top = new StringBuilder("- top frames: ");
            for (int i = 0; i < Mathf.Min(6, idx.Count); i++) top.Append($"{load[idx[i]]:0.0}@#{idx[i]}  ");
            sb.AppendLine(top.ToString());
        }
    }
}
