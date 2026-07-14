using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // #31 ResumeFromCompact v2 — MECHANICAL end-to-end probe in the real ChatDemo3D scene.
    // Verifies the machinery only (per user spec, NOT the compact's text quality — whatever the
    // model answers is accepted as the compact):
    //   A1  closing a long-enough chat starts CompactConversationRoutine
    //   A2  reopening MID-compaction waits: the dialogue reaches WaitingInInteraction only AFTER
    //       the compaction finished, and sends (AskNPCSilent) no-op while it runs
    //   A3  after compaction: compactSummary non-empty, transcript CLEARED, chat still live,
    //       post-compact conversation-KV save observed (when cacheKVCache)
    //   A4  a follow-up turn on the compacted chat completes normally
    //   A5  zone-exit during compaction #2: llm stays non-null for the whole life of
    //       compactRoutine (model never leaves the GPU mid-compact), release happens only after
    // Every frame is recorded tagged with the current phase (vsync OFF) — the report buckets
    // spikes per phase so compaction-behind-gameplay frame drops are visible at a glance.
    public class NpcCompactProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_compact_probe.md";
        const string MarkerPath = "ProbeLogs/npc_compact_probe.done";

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<string> fails = new List<string>();

        struct Frame { public float ms; public string phase; }
        readonly List<Frame> frames = new List<Frame>(65536);
        string phase = "boot";

        Transform player;
        CharacterController playerCC;
        NPCChatBase npc;
        INPCChatWindow window;
        int prevVsync; int prevTarget;

        // ---- reflection surface (probe-only; the runtime API stays minimal) ----
        static readonly BindingFlags BF = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        static FieldInfo fHistoryMode, fMaxTokens, fCacheKV, fUseZone, fCompactRoutine, fSummary,
                         fTranscript, fChatLive, fLlm, fCompactingNpc, fKvSaveInFlight;
        static MethodInfo mReleaseAfterSave;

        static void Bind()
        {
            var t = typeof(NPCChatBase);
            fHistoryMode    = t.GetField("historyMode", BF);
            fMaxTokens      = t.GetField("maxContextLength", BF);   // compaction now fires at the context limit
            fCacheKV        = t.GetField("cacheKVCache", BF);
            fUseZone        = t.GetField("usePrefetchZone", BF);
            fCompactRoutine = t.GetField("compactRoutine", BF);
            fSummary        = t.GetField("compactSummary", BF);
            fTranscript     = t.GetField("transcript", BF);
            fChatLive       = t.GetField("chatLive", BF);
            fLlm            = t.GetField("llm", BF);
            fCompactingNpc  = t.GetField("compactingNpc", BindingFlags.Static | BindingFlags.NonPublic);
            fKvSaveInFlight = t.GetField("kvSaveInFlight", BindingFlags.Static | BindingFlags.NonPublic);
            mReleaseAfterSave = t.GetMethod("ReleaseLlmAfterKvSave", BF);
        }

        bool Compacting => fCompactRoutine.GetValue(npc) != null;
        bool KvSaving => (bool)fKvSaveInFlight.GetValue(null);
        object Llm => fLlm.GetValue(npc);
        string Summary => (string)fSummary.GetValue(npc);
        int TranscriptCount => ((System.Collections.ICollection)fTranscript.GetValue(npc)).Count;
        bool ChatLive => (bool)fChatLive.GetValue(npc);

        void Check(bool ok, string what)
        {
            sb.AppendLine($"- {(ok ? "PASS" : "**FAIL**")} {what}");
            if (!ok) fails.Add(what);
        }

        void Awake()
        {
            Application.logMessageReceived += OnLog;
            Application.runInBackground = true;
            prevVsync = QualitySettings.vSyncCount;
            prevTarget = Application.targetFrameRate;
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
        }

        void OnDestroy()
        {
            Application.logMessageReceived -= OnLog;
            QualitySettings.vSyncCount = prevVsync;
            Application.targetFrameRate = prevTarget;
        }

        void OnLog(string msg, string stack, LogType type)
        {
            if (type == LogType.Exception || type == LogType.Error)
                if (errors.Count < 40) errors.Add($"{type}: {msg.Substring(0, Mathf.Min(160, msg.Length))}");
        }

        void Update() => frames.Add(new Frame { ms = Time.unscaledDeltaTime * 1000f, phase = phase });

        IEnumerator Start()
        {
            sb.AppendLine("# ResumeFromCompact v2 mechanical probe (#31) — real ChatDemo3D, vsync OFF — " +
                          System.DateTime.Now.ToString("yyyy-MM-dd HH:mm"));
            Bind();
            yield return null;

            var playerGO = GameObject.FindWithTag("Player");
            foreach (var n in FindObjectsOfType<NPCChatBase>(true))
                if (n.gameObject.name.Contains("Velmire")) npc = n;
            window = FindObjectOfType<Tutorials.ChatDemo3D.SoulsChatWindow>(true);
            player = playerGO != null ? playerGO.transform : null;
            playerCC = playerGO != null ? playerGO.GetComponent<CharacterController>() : null;
            if (player == null || npc == null || window == null)
            {
                Check(false, $"scene wiring (player={player != null} npc={npc != null} window={window != null})");
                Finish(); yield break;
            }

            // force the mode under test on the play-mode instance (not persisted)
            fHistoryMode.SetValue(npc, NPCChatBase.HistoryMode.ResumeFromCompact);
            fMaxTokens.SetValue(npc, 1);   // context limit = 1 token → the first reply compacts immediately
            bool cacheKV = (bool)fCacheKV.GetValue(npc);
            bool useZone = (bool)fUseZone.GetValue(npc);
            sb.AppendLine($"- npc=Velmire historyMode→ResumeFromCompact maxContextLength→1 cacheKVCache={cacheKV} usePrefetchZone={useZone}");

            // ---------- open + one turn ----------
            phase = "load";
            Teleport(ZonePoint(7.5f));
            yield return WaitFor(90f, () => npc.LlmReady);
            Check(npc.LlmReady, "LLM ready after zone approach");
            Teleport(ZonePoint(1.6f));
            yield return new WaitForSecondsRealtime(1f);

            phase = "open1";
            npc.StartInteraction();
            yield return WaitFor(60f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(npc.State == NPCChatBase.NPCState.WaitingInInteraction, "dialogue #1 open");

            phase = "turn1";
            window.InputField.text = "Tell me about this castle in two sentences.";
            npc.AskNPC();
            yield return WaitFor(120f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(npc.State == NPCChatBase.NPCState.WaitingInInteraction, "turn 1 reply completed");
            int transcriptBefore = TranscriptCount;
            sb.AppendLine($"- transcript before compaction: {transcriptBefore} turn(s)");

            // ---------- A1: close starts compaction ----------
            phase = "compact1";
            npc.CloseInteraction();
            yield return WaitFor(5f, () => Compacting);
            Check(Compacting, "A1 compaction started after close");

            // wait until CompactConversationRoutine is PAST its pre-touch guard (actively driving
            // the model) — reopening earlier legitimately aborts the not-yet-started routine
            // instead of waiting. Passed-guard signal: the routine drops llm.DiskKVCache for the
            // one-shot re-seed right after the guard (restored when Compact ends).
            yield return WaitFor(90f, () => !Compacting || (Llm is LLM la && !la.DiskKVCache));
            bool driving = Compacting && Llm is LLM lb && !lb.DiskKVCache;
            Check(driving, "compaction reached the model (past pre-touch guard)");
            if (!driving) { Finish(); yield break; }

            // ---------- A2: reopen mid-compaction WAITS, sends blocked ----------
            phase = "reopen-during-compact";
            npc.StartInteraction();
            yield return null; yield return null;
            bool blockedDuring = true, sawPostCompactSave = false;
            float tCompactDone = -1f;
            npc.AskNPCSilent("ping");   // must no-op while compacting
            float start = Time.unscaledTime;
            while (Time.unscaledTime - start < 180f)
            {
                bool compacting = Compacting;
                if (compacting && npc.State == NPCChatBase.NPCState.WaitingInInteraction) blockedDuring = false;
                if (compacting && npc.State == NPCChatBase.NPCState.TalkingInInteraction) blockedDuring = false;
                if (!compacting)
                {
                    if (tCompactDone < 0) tCompactDone = Time.unscaledTime;
                    if (KvSaving) sawPostCompactSave = true;
                    if (npc.State == NPCChatBase.NPCState.WaitingInInteraction) break;
                }
                yield return null;
            }
            float tOpenReady = Time.unscaledTime;
            Check(tCompactDone > 0, "compaction #1 finished");
            Check(npc.State == NPCChatBase.NPCState.WaitingInInteraction, "dialogue #2 opened after compaction");
            Check(blockedDuring, "A2 input blocked while compacting (never Waiting/Talking mid-compact)");
            Check(tOpenReady >= tCompactDone, $"A2 open completed only after compact done (+{tOpenReady - tCompactDone:0.00}s)");

            // ---------- A3: compacted state (mechanics only — text accepted as-is) ----------
            string summary = Summary;
            Check(!string.IsNullOrEmpty(summary), "A3 compactSummary non-empty");
            Check(TranscriptCount == 0, $"A3 transcript cleared (count={TranscriptCount})");
            Check(ChatLive, "A3 chat still live on the compacted prefix");
            Check(Llm != null, "A3 model still resident");
            if (cacheKV) Check(sawPostCompactSave, "A3 post-compact conversation-KV save observed");
            sb.AppendLine($"\n> compact (verbatim, {summary?.Length ?? 0} chars):\n> " +
                          (summary ?? "").Replace("\n", "\n> ") + "\n");

            // ---------- A4: the compacted chat still talks ----------
            phase = "turn2";
            window.InputField.text = "Say one short sentence.";
            npc.AskNPC();
            yield return WaitFor(120f, () => npc.State == NPCChatBase.NPCState.TalkingInInteraction);
            yield return WaitFor(120f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(npc.State == NPCChatBase.NPCState.WaitingInInteraction, "A4 turn on compacted chat completed");
            Check(TranscriptCount == 1, $"A4 transcript grows from zero (count={TranscriptCount})");

            // ---------- A5: zone-exit during compaction #2 — model stays until the compact lands ----------
            phase = "compact2-zone-exit";
            npc.CloseInteraction();
            yield return WaitFor(5f, () => Compacting);
            Check(Compacting, "compaction #2 started");
            Teleport(ZonePoint(60f));                                    // real Update zone-exit branch
            if (!useZone) StartCoroutine((IEnumerator)mReleaseAfterSave.Invoke(npc, null));
            bool heldThroughout = true;
            float tCompact2Done = -1f, tReleased = -1f;
            start = Time.unscaledTime;
            while (Time.unscaledTime - start < 240f)
            {
                if (Compacting && Llm == null) heldThroughout = false;
                if (!Compacting && tCompact2Done < 0) tCompact2Done = Time.unscaledTime;
                if (Llm == null) { tReleased = Time.unscaledTime; break; }
                yield return null;
            }
            Check(heldThroughout, "A5 llm stayed resident for the whole compaction");
            Check(tCompact2Done > 0, "compaction #2 finished");
            Check(tReleased > 0, "A5 llm released after compaction + KV save");
            if (tReleased > 0 && tCompact2Done > 0)
                Check(tReleased >= tCompact2Done, $"A5 release strictly after compact (+{tReleased - tCompact2Done:0.00}s)");

            phase = "done";
            FrameReport();
            Finish();
        }

        // ---------------- frame report ----------------

        void FrameReport()
        {
            sb.AppendLine("\n## Frames per phase (vsync OFF)");
            sb.AppendLine("| phase | frames | mean ms | p95 ms | max ms | >16.7 | >22.2 | >33.4 |");
            sb.AppendLine("|---|---|---|---|---|---|---|---|");
            var order = new List<string>(); var byPhase = new Dictionary<string, List<float>>();
            foreach (var f in frames)
            {
                if (!byPhase.TryGetValue(f.phase, out var l)) { byPhase[f.phase] = l = new List<float>(); order.Add(f.phase); }
                l.Add(f.ms);
            }
            foreach (var p in order)
            {
                var l = byPhase[p]; l.Sort();
                double mean = 0; int o17 = 0, o22 = 0, o33 = 0;
                foreach (float v in l) { mean += v; if (v > 16.7f) o17++; if (v > 22.2f) o22++; if (v > 33.4f) o33++; }
                sb.AppendLine($"| {p} | {l.Count} | {mean / l.Count:0.00} | {l[(int)(0.95f * (l.Count - 1))]:0.00} " +
                              $"| {l[l.Count - 1]:0.0} | {o17} | {o22} | {o33} |");
            }
            if (errors.Count > 0)
            {
                sb.AppendLine("\n## Console errors during the run");
                foreach (var e in errors) sb.AppendLine("- " + e);
            }
        }

        // ---------------- helpers ----------------

        IEnumerator WaitFor(float timeout, System.Func<bool> until)
        {
            float t0 = Time.unscaledTime;
            while (Time.unscaledTime - t0 < timeout)
            {
                if (until != null && until()) break;
                yield return null;
            }
        }

        void Teleport(Vector3 pos)
        {
            if (playerCC != null) playerCC.enabled = false;
            player.position = pos;
            if (playerCC != null) playerCC.enabled = true;
        }

        Vector3 ZonePoint(float dist)
        {
            Vector3 d = player.position - npc.transform.position; d.y = 0;
            Vector3 dir = d.sqrMagnitude < 0.01f ? Vector3.back : d.normalized;
            Vector3 p = npc.transform.position + dir * dist;
            p.y = player.position.y;
            return p;
        }

        void Finish()
        {
            sb.AppendLine(fails.Count == 0 ? "\n## RESULT: ALL PASS" : $"\n## RESULT: {fails.Count} FAIL(S)");
            Directory.CreateDirectory("ProbeLogs");
            File.WriteAllText(ReportPath, sb.ToString());
            File.WriteAllText(MarkerPath, fails.Count == 0 ? "PASS" : "FAIL");
            Debug.Log($"[NpcCompactProbe] report -> {ReportPath}");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }
}
