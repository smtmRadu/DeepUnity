using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Post-audit hardening — MECHANICAL end-to-end probe in the real ChatDemo3D scene.
    // Covers the four audit fixes the older probes don't reach (two NPCs share the window):
    //   A1  idle-sibling Leave (audit #1): CloseInteraction() on the IDLE Morwenna while
    //       Velmire's dialogue is open must be a pure no-op — her state stays Idle, Velmire's
    //       state/window/input untouched, no conversation-KV save latched for her, her voice
    //       volume untouched
    //   A2  AskNPCSilent gates (audit #3): a silent ask mid-generation is ignored (state gate)
    //       and one during the voice tail is dropped WITH the warning line — transcript never
    //       grows from a gated silent ask
    //   A3  Busy backstop (F1): llm.Busy reads TRUE at some point while a reply generates and
    //       FALSE once it completes; the backstop refusal warning never fires in the whole run
    //       (the NPC-side choreography must make it unreachable)
    //   A4  cancel-during-prefill (audit #14): Escape-close within ~3 frames of an ask unwinds
    //       cooperatively (no 10 s StopCoroutine fallback), chatLive stays true, a cancel line
    //       (prefill or token-boundary) is logged, and reopen + one further turn work
    //   T0/global: zero console errors/exceptions through the whole run
    public class NpcAuditProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_audit_probe.md";
        const string MarkerPath = "ProbeLogs/npc_audit_probe.done";

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<string> fails = new List<string>();

        Transform player;
        CharacterController playerCC;
        NPCChatBase velmire, morwenna;
        Tutorials.ChatDemo3D.SoulsChatWindow window;
        int prevVsync; int prevTarget;
        int silentDropLines;      // "AskNPCSilent dropped" warnings
        int busyRefusedLines;     // LLM Busy-backstop refusals (must stay 0)
        int cancelLines;          // prefill OR token-boundary cancel lines

        // ---- reflection surface (probe-only; the runtime API stays minimal) ----
        static readonly BindingFlags BF = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        static FieldInfo fHistoryMode, fMaxTokens, fTranscript, fChatLive, fLlm, fDialogue,
                         fActiveResponse, fInterruptPending, fPkVoice, fKvSavesInFlight;
        static System.Type turnType;

        static void Bind()
        {
            var t = typeof(NPCChatBase);
            fHistoryMode      = t.GetField("historyMode", BF);
            fMaxTokens        = t.GetField("maxContextLength", BF);
            fTranscript       = t.GetField("transcript", BF);
            fChatLive         = t.GetField("chatLive", BF);
            fLlm              = t.GetField("llm", BF);
            fDialogue         = t.GetField("dialogueCoroutine", BF);
            fActiveResponse   = t.GetField("activeResponse", BF);
            fInterruptPending = t.GetField("interruptPending", BF);
            fPkVoice          = t.GetField("pkVoice", BF);
            fKvSavesInFlight  = t.GetField("kvSavesInFlight", BindingFlags.Static | BindingFlags.NonPublic);
            turnType          = t.GetNestedType("Turn", BindingFlags.NonPublic);
        }

        bool ChatLive => (bool)fChatLive.GetValue(velmire);
        bool ReplyInFlight => fDialogue.GetValue(velmire) != null;
        int GenChars => (fActiveResponse.GetValue(velmire) as StringBuilder)?.Length ?? 0;
        bool InterruptPendingNow => (bool)fInterruptPending.GetValue(velmire);
        PocketTTSModeling.PocketTTSVoice Pk => fPkVoice.GetValue(velmire) as PocketTTSModeling.PocketTTSVoice;
        bool VoiceAudible => Pk != null && (Pk.IsSpeaking || Pk.IsAudioPlaying);
        bool Settled => !ReplyInFlight && !InterruptPendingNow && !VoiceAudible;
        bool VelmireBusy => fLlm.GetValue(velmire) is LLM m && m.Busy;
        System.Collections.IList Transcript => (System.Collections.IList)fTranscript.GetValue(velmire);
        int TranscriptCount => Transcript.Count;
        string TurnNpcText(int i)
        {
            object turn = Transcript[i];
            return (string)turnType.GetField("npc", BF).GetValue(turn);
        }

        bool AnySaveInFlight => ((System.Collections.IDictionary)fKvSavesInFlight.GetValue(null)).Count > 0;
        bool SaveOwnedBy(NPCChatBase npc)
        {
            var d = (System.Collections.IDictionary)fKvSavesInFlight.GetValue(null);
            foreach (System.Collections.DictionaryEntry e in d)
                if (ReferenceEquals(e.Value, npc)) return true;
            return false;
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
            if (msg.Contains("AskNPCSilent dropped")) silentDropLines++;
            if (msg.Contains("refused — another operation")) busyRefusedLines++;
            if (msg.Contains("canceled during prefill") || msg.Contains("canceled at a token boundary")) cancelLines++;
            if (type == LogType.Exception || type == LogType.Error)
                if (errors.Count < 40) errors.Add($"{type}: {msg.Substring(0, Mathf.Min(160, msg.Length))}");
        }

        IEnumerator Start()
        {
            sb.AppendLine("# NPC audit-fix mechanical probe — real ChatDemo3D, vsync OFF — " +
                          System.DateTime.Now.ToString("yyyy-MM-dd HH:mm"));
            Bind();
            yield return null;

            var playerGO = GameObject.FindWithTag("Player");
            foreach (var n in FindObjectsOfType<NPCChatBase>(true))
            {
                if (n.gameObject.name.Contains("Velmire")) velmire = n;
                if (n.gameObject.name.Contains("Morwenna")) morwenna = n;
            }
            window = FindObjectOfType<Tutorials.ChatDemo3D.SoulsChatWindow>(true);
            player = playerGO != null ? playerGO.transform : null;
            playerCC = playerGO != null ? playerGO.GetComponent<CharacterController>() : null;
            if (player == null || velmire == null || morwenna == null || window == null)
            {
                Check(false, $"scene wiring (player={player != null} velmire={velmire != null} " +
                             $"morwenna={morwenna != null} window={window != null})");
                Finish(); yield break;
            }

            // huge context limit so no compaction interferes (scene value can be tiny); continue
            // mode so the transcript/chatLive asserts are meaningful
            fHistoryMode.SetValue(velmire, NPCChatBase.HistoryMode.ResumeFromCompact);
            fMaxTokens.SetValue(velmire, 1000000);
            // Morwenna is staged like an NPC with a PAST conversation (chatLive + one turn):
            // exactly the state in which the pre-fix idle-sibling close corrupted things
            fHistoryMode.SetValue(morwenna, NPCChatBase.HistoryMode.ResumeFromCompact);
            fChatLive.SetValue(morwenna, true);
            object fakeTurn = System.Activator.CreateInstance(turnType);
            turnType.GetField("user", BF).SetValue(fakeTurn, "old question");
            turnType.GetField("npc", BF).SetValue(fakeTurn, "old answer");
            ((System.Collections.IList)fTranscript.GetValue(morwenna)).Add(fakeTurn);
            sb.AppendLine("- velmire→ResumeFromCompact maxContextLength→1000000; morwenna staged " +
                          "ResumeFromCompact + chatLive + 1 fake turn (idle-sibling bait)");

            // ---------- load + open Velmire's dialogue ----------
            Teleport(ZonePoint(7.5f));
            yield return WaitFor(90f, () => velmire.LlmReady);
            Check(velmire.LlmReady, "T0 LLM ready after zone approach");
            Teleport(ZonePoint(1.6f));
            yield return new WaitForSecondsRealtime(1f);
            velmire.StartInteraction();
            yield return WaitFor(60f, () => velmire.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(velmire.State == NPCChatBase.NPCState.WaitingInInteraction, "T0 dialogue open (Velmire)");

            // ---------- A1: idle-sibling CloseInteraction is a pure no-op ----------
            window.InputField.text = "sentinel question";
            float morwennaVol = morwenna.GetComponent<AudioSource>() != null
                ? morwenna.GetComponent<AudioSource>().volume : -1f;
            morwenna.CloseInteraction();
            bool morwennaSaveSeen = false;
            float t0 = Time.unscaledTime;
            while (Time.unscaledTime - t0 < 1f)
            {
                if (SaveOwnedBy(morwenna)) morwennaSaveSeen = true;
                yield return null;
            }
            Check(morwenna.State == NPCChatBase.NPCState.Idle, "A1 Morwenna stays Idle after her CloseInteraction");
            Check(velmire.State == NPCChatBase.NPCState.WaitingInInteraction, "A1 Velmire's dialogue state untouched");
            Check(window.IsOpen, "A1 shared window still open");
            Check(window.InputField.text == "sentinel question", "A1 typed input preserved");
            Check(!morwennaSaveSeen, "A1 no conversation-KV save latched for Morwenna");
            float morwennaVolAfter = morwenna.GetComponent<AudioSource>() != null
                ? morwenna.GetComponent<AudioSource>().volume : -1f;
            Check(Mathf.Approximately(morwennaVol, morwennaVolAfter), "A1 Morwenna's voice volume untouched (no fade)");

            // ---------- A2 + A3: silent-ask gates and the Busy flag across one long reply ----------
            window.InputField.text = "Describe this castle and its keepers in great detail, at least six long sentences.";
            velmire.AskNPC();
            bool sawBusy = false;
            yield return WaitFor(60f, () => { if (ReplyInFlight && VelmireBusy) sawBusy = true;
                                              return ReplyInFlight && GenChars > 40; });
            Check(ReplyInFlight && GenChars > 40, $"A2 reached mid-generation (chars={GenChars})");

            int turnsMidGen = TranscriptCount;
            velmire.AskNPCSilent("ping");   // state gate (TalkingInInteraction) must ignore it
            yield return new WaitForSecondsRealtime(0.5f);
            Check(TranscriptCount == turnsMidGen, "A2a silent ask mid-generation ignored (transcript unchanged)");

            yield return WaitFor(120f, () => { if (ReplyInFlight && VelmireBusy) sawBusy = true;
                                               return velmire.State == NPCChatBase.NPCState.WaitingInInteraction
                                                      && !ReplyInFlight; });
            Check(sawBusy, "A3 llm.Busy observed TRUE while the reply was generating");
            Check(!VelmireBusy, "A3 llm.Busy back to FALSE once the reply completed");

            // voice tail (pocket keeps speaking for seconds after a 6-sentence reply)
            if (VoiceAudible)
            {
                int turnsTail = TranscriptCount;
                int dropsBefore = silentDropLines;
                velmire.AskNPCSilent("ping");   // audible-gate must drop it WITH the warning
                yield return new WaitForSecondsRealtime(0.5f);
                Check(silentDropLines > dropsBefore, "A2b silent ask during the voice tail dropped (warning line observed)");
                Check(TranscriptCount == turnsTail, "A2b transcript unchanged after the tail silent-ask");
            }
            else sb.AppendLine("- (voice tail already over at the tail silent-ask — A2b not applicable this run)");
            yield return WaitFor(60f, () => Settled);

            // ---------- A4: close within ~3 frames of an ask (cancel lands in the prefill) ----------
            window.InputField.text = "Tell me the full legend of the pale king and the drowned queen, at least eight long sentences.";
            int cancelsBefore = cancelLines;
            int turnsBeforeCancel = TranscriptCount;
            velmire.AskNPC();
            yield return null; yield return null; yield return null;   // ~3 frames: inside the prefill
            float tClose = Time.unscaledTime;
            velmire.CloseInteraction();
            Check(velmire.State == NPCChatBase.NPCState.Idle, "A4 state drops to Idle immediately on close");
            float tUnwound = -1f;
            while (Time.unscaledTime - tClose < 15f)
            {
                if (!ReplyInFlight) { tUnwound = Time.unscaledTime; break; }
                yield return null;
            }
            Check(tUnwound > 0 && tUnwound - tClose < 9f,
                  $"A4 reply unwound cooperatively (+{(tUnwound > 0 ? tUnwound - tClose : -1):0.00}s — no 10 s StopCoroutine fallback)");
            Check(cancelLines > cancelsBefore, "A4 cancel line logged (prefill or token boundary)");
            Check(ChatLive, "A4 chatLive stays TRUE (clean close, KV trusted)");
            yield return WaitFor(30f, () => !AnySaveInFlight);   // let the clean-close snapshot land

            // reopen + one further turn on the same conversation
            velmire.StartInteraction();
            yield return WaitFor(90f, () => velmire.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(velmire.State == NPCChatBase.NPCState.WaitingInInteraction, "A4 dialogue reopened after the prefill cancel");
            Check(TranscriptCount == turnsBeforeCancel + 1, "A4 canceled (empty/partial) turn recorded once");
            yield return WaitFor(30f, () => Settled);
            window.InputField.text = "Say one short sentence.";
            int turnsFinal = TranscriptCount;
            velmire.AskNPC();
            yield return WaitFor(120f, () => velmire.State == NPCChatBase.NPCState.WaitingInInteraction && !ReplyInFlight);
            Check(TranscriptCount == turnsFinal + 1 && !string.IsNullOrEmpty(TurnNpcText(TranscriptCount - 1)),
                  "A4 turn after reopen completed (KV coherent after the prefill cancel)");

            velmire.CloseInteraction();
            yield return new WaitForSecondsRealtime(2f);

            // ---------- global ----------
            Check(busyRefusedLines == 0, $"F1 Busy backstop never fired ({busyRefusedLines} refusals — choreography holds)");
            Check(errors.Count == 0, $"zero console errors/exceptions through the run ({errors.Count})");
            Finish();
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
            Vector3 d = player.position - velmire.transform.position; d.y = 0;
            Vector3 dir = d.sqrMagnitude < 0.01f ? Vector3.back : d.normalized;
            Vector3 p = velmire.transform.position + dir * dist;
            p.y = player.position.y;
            return p;
        }

        void Finish()
        {
            sb.AppendLine(fails.Count == 0 ? "\n## RESULT: ALL PASS" : $"\n## RESULT: {fails.Count} FAIL(S)");
            if (errors.Count > 0)
            {
                sb.AppendLine("\n## Console errors during the run");
                foreach (var e in errors) sb.AppendLine("- " + e);
            }
            Directory.CreateDirectory("ProbeLogs");
            File.WriteAllText(ReportPath, sb.ToString());
            File.WriteAllText(MarkerPath, fails.Count == 0 ? "PASS" : "FAIL");
            Debug.Log($"[NpcAuditProbe] report -> {ReportPath}");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }
}
