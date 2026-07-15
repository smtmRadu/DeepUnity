using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Mid-reply interruption — MECHANICAL end-to-end probe in the real ChatDemo3D scene.
    // Verifies the cooperative-cancel machinery (send-while-talking / leave-while-talking):
    //   T1  ask WHILE the reply is still generating: the old reply cancels at a token boundary
    //       (cancel console line seen, new turn starts within seconds — not after the full
    //       reply), the voice FADES (never a hard cut: quiet arrives >=0.3s after the ask),
    //       the truncated turn is recorded, and the NEW reply completes normally
    //   T2  the conversation stays consistent after the truncation: a follow-up turn completes
    //       with a non-empty reply (KV was left exactly as after a natural stop token)
    //   T3  Escape/close mid-generation: state drops to Idle instantly, the reply unwinds
    //       cooperatively (dialogue coroutine null within seconds, no StopCoroutine fallback),
    //       chatLive STAYS true (clean close) and the conversation-KV save fires; reopening
    //       resumes the same conversation and a further turn completes
    //   T0/global: zero console errors/exceptions through the whole run
    public class NpcInterruptProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_interrupt_probe.md";
        const string MarkerPath = "ProbeLogs/npc_interrupt_probe.done";

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
        bool sawCancelLine;   // model-side "canceled at a token boundary" console line
        int cancelLines;

        // ---- reflection surface (probe-only; the runtime API stays minimal) ----
        static readonly BindingFlags BF = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        static FieldInfo fHistoryMode, fCacheKV, fTranscript, fChatLive, fLlm, fKvSaveInFlight,
                         fDialogue, fActiveResponse, fPkVoice, fInterruptPending, fMaxTokens;

        static void Bind()
        {
            var t = typeof(NPCChatBase);
            fHistoryMode      = t.GetField("historyMode", BF);
            fCacheKV          = t.GetField("cacheKVCache", BF);
            fTranscript       = t.GetField("transcript", BF);
            fChatLive         = t.GetField("chatLive", BF);
            fLlm              = t.GetField("llm", BF);
            fKvSaveInFlight   = t.GetField("kvSavesInFlight", BindingFlags.Static | BindingFlags.NonPublic);
            fDialogue         = t.GetField("dialogueCoroutine", BF);
            fActiveResponse   = t.GetField("activeResponse", BF);
            fPkVoice          = t.GetField("pkVoice", BF);
            fInterruptPending = t.GetField("interruptPending", BF);
            fMaxTokens        = t.GetField("maxContextLength", BF);
        }

        // gate is a per-instance Dictionary<LLM, NPCChatBase> now — "any save in flight"
        // matches the old global-bool read (single-NPC probe)
        bool KvSaving => ((System.Collections.IDictionary)fKvSaveInFlight.GetValue(null)).Count > 0;
        bool ChatLive => (bool)fChatLive.GetValue(npc);
        bool ReplyInFlight => fDialogue.GetValue(npc) != null;
        int GenChars => (fActiveResponse.GetValue(npc) as StringBuilder)?.Length ?? 0;
        PocketTTSModeling.PocketTTSVoice Pk => fPkVoice.GetValue(npc) as PocketTTSModeling.PocketTTSVoice;
        bool VoiceSpeaking => Pk != null && Pk.IsSpeaking;
        bool InterruptPendingNow => (bool)fInterruptPending.GetValue(npc);
        // audible = synthesis in flight OR ring/tail still playing (mirrors NPCChatBase.VoicesAudible)
        bool VoiceAudible => Pk != null && (Pk.IsSpeaking || Pk.IsAudioPlaying);
        // fully settled: no reply, no queued interrupt-ask, audio fully quiet — the next AskNPC
        // takes the direct path (or, when deliberately NOT awaited, the interrupt path)
        bool Settled => !ReplyInFlight && !InterruptPendingNow && !VoiceAudible;
        System.Collections.IList Transcript => (System.Collections.IList)fTranscript.GetValue(npc);
        int TranscriptCount => Transcript.Count;
        string TurnNpcText(int i)
        {
            object turn = Transcript[i];
            return (string)turn.GetType().GetField("npc", BF).GetValue(turn);
        }

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
            if (msg.Contains("canceled at a token boundary")) { sawCancelLine = true; cancelLines++; }
            if (type == LogType.Exception || type == LogType.Error)
                if (errors.Count < 40) errors.Add($"{type}: {msg.Substring(0, Mathf.Min(160, msg.Length))}");
        }

        void Update() => frames.Add(new Frame { ms = Time.unscaledDeltaTime * 1000f, phase = phase });

        IEnumerator Start()
        {
            sb.AppendLine("# Mid-reply interruption mechanical probe — real ChatDemo3D, vsync OFF — " +
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

            // continue mode with a huge context limit — the scene value can be tiny (compaction
            // testing) and a mid-probe auto-compact would block the asks under test
            fHistoryMode.SetValue(npc, NPCChatBase.HistoryMode.ResumeFromCompact);
            fMaxTokens.SetValue(npc, 1000000);
            bool cacheKV = (bool)fCacheKV.GetValue(npc);
            sb.AppendLine($"- npc=Velmire historyMode→ResumeFromCompact cacheKVCache={cacheKV}");

            // ---------- T0: open + one clean baseline turn ----------
            phase = "load";
            Teleport(ZonePoint(7.5f));
            yield return WaitFor(90f, () => npc.LlmReady);
            Check(npc.LlmReady, "T0 LLM ready after zone approach");
            Teleport(ZonePoint(1.6f));
            yield return new WaitForSecondsRealtime(1f);

            phase = "open1";
            npc.StartInteraction();
            yield return WaitFor(60f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(npc.State == NPCChatBase.NPCState.WaitingInInteraction, "T0 dialogue open");
            int baseTurns = TranscriptCount;   // disk-restored history may pre-populate it
            sb.AppendLine($"- transcript at open: {baseTurns} turn(s)");

            phase = "turn-baseline";
            window.InputField.text = "Greet me in one short sentence.";
            npc.AskNPC();
            yield return WaitFor(120f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction && !ReplyInFlight);
            Check(TranscriptCount == baseTurns + 1 && !string.IsNullOrEmpty(TurnNpcText(baseTurns)),
                  "T0 baseline turn completed");

            // ---------- T1: ask WHILE the reply is generating ----------
            phase = "t1-longask";
            yield return WaitFor(30f, () => Settled);   // deterministic start: baseline voice done
            window.InputField.text = "Describe this castle and its whole history in great detail, at least eight long sentences.";
            npc.AskNPC();
            // mid-generation: reply coroutine alive and a decent chunk of text already produced
            yield return WaitFor(60f, () => ReplyInFlight && GenChars > 80);
            bool midGen = ReplyInFlight && GenChars > 80;
            Check(midGen, $"T1 reached mid-generation (chars={GenChars})");
            if (!midGen) { Finish(); yield break; }
            int longTurnIdx = TranscriptCount - 1;

            phase = "t1-interrupt";
            bool speakingAtAsk = VoiceSpeaking;
            float tAsk = Time.unscaledTime;
            window.InputField.text = "Stop. Just say hello.";
            npc.AskNPC();
            bool pendingSeen = (bool)fInterruptPending.GetValue(npc);
            float tQuiet = -1f, tNewTurn = -1f;
            while (Time.unscaledTime - tAsk < 30f)
            {
                if (tQuiet < 0 && !VoiceAudible) tQuiet = Time.unscaledTime;
                if (TranscriptCount == longTurnIdx + 2) { tNewTurn = Time.unscaledTime; break; }
                yield return null;
            }
            Check(tQuiet > 0 && (tNewTurn < 0 || tQuiet <= tNewTurn),
                  "T1 old audio FULLY quiet (incl. ring/tail) before the new turn landed");
            Check(pendingSeen, "T1 interrupt path taken (interruptPending latched)");
            Check(tNewTurn > 0, "T1 new turn started after the interrupt");
            Check(tNewTurn > 0 && tNewTurn - tAsk < 10f,
                  $"T1 old reply canceled quickly (+{(tNewTurn > 0 ? tNewTurn - tAsk : -1):0.00}s — a full 8-sentence reply would take far longer)");
            Check(sawCancelLine, "T1 model logged the token-boundary cancel");
            if (speakingAtAsk)
                Check(tQuiet > 0 && tQuiet - tAsk >= 0.3f && tQuiet - tAsk <= 5f,
                      $"T1 voice FADED out, not hard-cut (+{(tQuiet > 0 ? tQuiet - tAsk : -1):0.00}s to silence)");
            else sb.AppendLine("- (voice was already quiet at the interrupt — fade timing not applicable)");
            string truncated = TurnNpcText(longTurnIdx);
            Check(!string.IsNullOrEmpty(truncated), $"T1 truncated turn recorded ({truncated?.Length ?? 0} chars)");
            sb.AppendLine($"\n> truncated reply (verbatim): {truncated}\n");

            phase = "t1-newreply";
            yield return WaitFor(120f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction && !ReplyInFlight);
            Check(!string.IsNullOrEmpty(TurnNpcText(longTurnIdx + 1)), "T1 new reply completed after the interrupt");

            // ---------- T2: conversation still consistent after the truncation ----------
            // asked WITHOUT waiting for the voice: when the tail is still speaking this also
            // exercises the speak-tail interrupt (fade first, question lands after)
            phase = "t2-followup";
            bool audibleAtAsk2 = VoiceAudible;
            window.InputField.text = "What did I first ask you about?";
            float tAsk2 = Time.unscaledTime;
            npc.AskNPC();
            bool fullyQuietSeen2 = false, quietBeforeTurn2 = false;
            while (Time.unscaledTime - tAsk2 < 30f)
            {
                if (!fullyQuietSeen2 && !VoiceAudible) fullyQuietSeen2 = true;
                if (TranscriptCount == longTurnIdx + 3) { quietBeforeTurn2 = fullyQuietSeen2; break; }
                yield return null;
            }
            Check(TranscriptCount == longTurnIdx + 3,
                  $"T2 follow-up turn started (audio audible at ask={audibleAtAsk2} — fade path when true)");
            Check(quietBeforeTurn2, "T2 old audio FULLY stopped (incl. ring/tail) before the question landed");
            yield return WaitFor(120f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction && !ReplyInFlight);
            Check(TranscriptCount == longTurnIdx + 3 && !string.IsNullOrEmpty(TurnNpcText(longTurnIdx + 2)),
                  "T2 follow-up turn on the truncated history completed");

            // ---------- T3: close mid-generation ----------
            phase = "t3-longask";
            yield return WaitFor(30f, () => Settled);   // let T2's voice finish (no stray interrupt latch)
            window.InputField.text = "Tell me the full legend of the pale king, at least eight long sentences.";
            npc.AskNPC();
            yield return WaitFor(30f, () => TranscriptCount == longTurnIdx + 4);   // turn started
            yield return WaitFor(60f, () => ReplyInFlight && GenChars > 80);
            Check(ReplyInFlight && GenChars > 80, $"T3 reached mid-generation (chars={GenChars})");

            phase = "t3-close";
            int cancelsBefore = cancelLines;
            float tClose = Time.unscaledTime;
            npc.CloseInteraction();
            Check(npc.State == NPCChatBase.NPCState.Idle, "T3 state drops to Idle immediately on close");
            bool sawSave = false;
            float tUnwound = -1f;
            while (Time.unscaledTime - tClose < 30f)
            {
                if (!ReplyInFlight && tUnwound < 0) tUnwound = Time.unscaledTime;
                if (KvSaving) sawSave = true;
                if (tUnwound > 0 && !KvSaving && Time.unscaledTime - tUnwound > 2f) break;
                yield return null;
            }
            Check(tUnwound > 0, $"T3 reply unwound cooperatively (+{(tUnwound > 0 ? tUnwound - tClose : -1):0.00}s after close)");
            Check(cancelLines > cancelsBefore, "T3 model logged the token-boundary cancel");
            Check(ChatLive, "T3 KV stays LIVE after leave-mid-reply (clean close, no dead-KV fallback)");
            if (cacheKV) Check(sawSave, "T3 conversation-KV save fired on the clean close");

            // ---------- T3b: reopen resumes the same conversation ----------
            phase = "t3-reopen";
            int turnsBeforeReopen = TranscriptCount;
            npc.StartInteraction();
            yield return WaitFor(90f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(npc.State == NPCChatBase.NPCState.WaitingInInteraction, "T3b dialogue reopened");
            Check(TranscriptCount == turnsBeforeReopen, "T3b transcript survived the close");

            phase = "t3-finalturn";
            yield return WaitFor(30f, () => Settled);
            window.InputField.text = "Say goodbye in one short sentence.";
            npc.AskNPC();
            yield return WaitFor(120f, () => npc.State == NPCChatBase.NPCState.WaitingInInteraction && !ReplyInFlight);
            Check(TranscriptCount == turnsBeforeReopen + 1 && !string.IsNullOrEmpty(TurnNpcText(TranscriptCount - 1)),
                  "T3b turn after reopen completed (resumed KV coherent)");

            npc.CloseInteraction();
            yield return new WaitForSecondsRealtime(2f);

            Check(errors.Count == 0, $"zero console errors/exceptions through the run ({errors.Count})");
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
            Debug.Log($"[NpcInterruptProbe] report -> {ReportPath}");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }
}
