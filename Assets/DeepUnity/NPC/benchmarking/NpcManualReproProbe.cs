using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Manual-play repro probe: runs Velmire EXACTLY as serialized in the scene (no historyMode /
    // maxContextLength overrides — currently ResumeFromCompact @ 400 tokens, pooled Qwen shared
    // with Morwenna) and sends messages through the REAL UI event path (InputField.onSubmit
    // Invoke — the same persistent listeners the keyboard fires), not npc.AskNPC() directly.
    // Chases the user report: "sending a message in the middle while he is talking not working".
    //   R1  send mid-GENERATION via UI → the reply cancels and the new turn lands
    //   R2  send mid-SPEECH (generation done, voice talking) via UI → fade, then the turn lands
    //   R3  send DURING "Compacting…" → correctly blocked, typed text preserved; works after
    public class NpcManualReproProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_manual_repro_probe.md";
        const string MarkerPath = "ProbeLogs/npc_manual_repro_probe.done";

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<string> fails = new List<string>();
        string phase = "boot";

        Transform player;
        CharacterController playerCC;
        NPCChatBase npc;
        Tutorials.ChatDemo3D.SoulsChatWindow window;
        int prevVsync; int prevTarget;

        static readonly BindingFlags BF = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        static FieldInfo fTranscript, fLlm, fDialogue, fActiveResponse, fPkVoice, fInterruptPending,
                         fCompactRoutine, fHistoryMode, fMaxTokens;

        static void Bind()
        {
            var t = typeof(NPCChatBase);
            fTranscript       = t.GetField("transcript", BF);
            fLlm              = t.GetField("llm", BF);
            fDialogue         = t.GetField("dialogueCoroutine", BF);
            fActiveResponse   = t.GetField("activeResponse", BF);
            fPkVoice          = t.GetField("pkVoice", BF);
            fInterruptPending = t.GetField("interruptPending", BF);
            fCompactRoutine   = t.GetField("compactRoutine", BF);
            fHistoryMode      = t.GetField("historyMode", BF);
            fMaxTokens        = t.GetField("maxContextLength", BF);
        }

        bool ReplyInFlight => fDialogue.GetValue(npc) != null;
        int GenChars => (fActiveResponse.GetValue(npc) as StringBuilder)?.Length ?? 0;
        PocketTTSModeling.PocketTTSVoice Pk => fPkVoice.GetValue(npc) as PocketTTSModeling.PocketTTSVoice;
        bool VoiceAudible => Pk != null && (Pk.IsSpeaking || Pk.IsAudioPlaying);
        bool InterruptPendingNow => (bool)fInterruptPending.GetValue(npc);
        bool Compacting => fCompactRoutine.GetValue(npc) != null;
        int TranscriptCount => ((System.Collections.ICollection)fTranscript.GetValue(npc)).Count;
        bool FullySettled => !ReplyInFlight && !InterruptPendingNow && !VoiceAudible && !Compacting
                             && npc.State == NPCChatBase.NPCState.WaitingInInteraction;

        // the REAL send path: what Enter does (persistent onSubmit listeners → AskNPC + click sfx)
        void UiSend(string text)
        {
            window.InputField.text = text;
            window.InputField.onSubmit.Invoke(text);
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
            if (type == LogType.Exception || type == LogType.Error)
                if (errors.Count < 40) errors.Add($"{type}: {msg.Substring(0, Mathf.Min(200, msg.Length))}");
        }

        IEnumerator Start()
        {
            sb.AppendLine("# Manual-play repro probe (scene-true settings, UI send path) — " +
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
            sb.AppendLine($"- SCENE-TRUE: historyMode={(NPCChatBase.HistoryMode)(int)fHistoryMode.GetValue(npc)} " +
                          $"maxContextLength={fMaxTokens.GetValue(npc)}");

            // ---------- open (may include the on-open crash-recovery compaction) ----------
            phase = "open";
            Teleport(ZonePoint(7.5f));
            yield return WaitFor(90f, () => npc.LlmReady);
            Check(npc.LlmReady, "LLM ready after zone approach");
            Teleport(ZonePoint(1.6f));
            yield return new WaitForSecondsRealtime(1f);
            npc.StartInteraction();
            bool sawOpenCompact = false;
            float t0 = Time.unscaledTime;
            while (Time.unscaledTime - t0 < 180f)
            {
                if (Compacting) sawOpenCompact = true;
                if (npc.State == NPCChatBase.NPCState.WaitingInInteraction && !Compacting) break;
                yield return null;
            }
            Check(npc.State == NPCChatBase.NPCState.WaitingInInteraction, $"dialogue open (on-open compact={sawOpenCompact})");

            // ---------- R1: send mid-GENERATION through the UI ----------
            phase = "r1-longask";
            UiSend("Tell me about this castle and its history in several long sentences.");
            yield return WaitFor(60f, () => ReplyInFlight && GenChars > 40);
            Check(ReplyInFlight && GenChars > 40, $"R1 mid-generation reached (chars={GenChars})");
            int before = TranscriptCount;

            phase = "r1-send";
            float tAsk = Time.unscaledTime;
            UiSend("Stop. Just say hello.");
            bool landed = false;
            while (Time.unscaledTime - tAsk < 30f)
            {
                if (TranscriptCount == before + 1) { landed = true; break; }
                yield return null;
            }
            Check(landed, $"R1 UI send mid-generation landed the new turn (+{(landed ? Time.unscaledTime - tAsk : -1):0.00}s)");
            yield return WaitFor(120f, () => !ReplyInFlight && npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(TranscriptCount == before + 1, "R1 new reply completed");

            // ---------- settle through any post-reply compaction (400-token limit!) ----------
            phase = "settle1";
            yield return WaitFor(240f, () => FullySettled);
            Check(FullySettled, "settled after R1 (incl. post-reply compaction at the 400 limit)");

            // ---------- R2: send mid-SPEECH through the UI ----------
            phase = "r2-longask";
            UiSend("Describe the mist gate in several long sentences.");
            // generation DONE but the voice still audibly speaking — the exact manual window
            yield return WaitFor(120f, () => !ReplyInFlight && VoiceAudible);
            bool inTail = !ReplyInFlight && VoiceAudible;
            Check(inTail, "R2 reached the speech-tail window (generation done, voice audible)");
            before = TranscriptCount;

            phase = "r2-send";
            tAsk = Time.unscaledTime;
            UiSend("Enough. One word answer: yes or no?");
            bool quietSeen = false, quietBefore = false; landed = false;
            while (Time.unscaledTime - tAsk < 30f)
            {
                if (!quietSeen && !VoiceAudible) quietSeen = true;
                if (TranscriptCount == before + 1) { landed = true; quietBefore = quietSeen; break; }
                yield return null;
            }
            Check(landed, $"R2 UI send mid-speech landed the new turn (+{(landed ? Time.unscaledTime - tAsk : -1):0.00}s)");
            Check(!inTail || quietBefore, "R2 old audio fully stopped before the question landed");
            yield return WaitFor(120f, () => !ReplyInFlight && npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(TranscriptCount == before + 1, "R2 new reply completed");

            // ---------- R3: send during "Compacting…" is blocked, then works ----------
            phase = "r3-compact";
            float tw = Time.unscaledTime; bool sawCompact = false;
            while (Time.unscaledTime - tw < 240f)
            {
                if (Compacting) { sawCompact = true; break; }
                if (FullySettled) break;   // settled without compacting (limit not hit) — also fine
                yield return null;
            }
            sb.AppendLine($"- post-R2 compaction observed: {sawCompact}");
            if (sawCompact)
            {
                before = TranscriptCount;
                UiSend("Are you still there?");
                yield return new WaitForSecondsRealtime(1.5f);
                Check(TranscriptCount == before && Compacting || !Compacting,
                      "R3 send during Compacting did not start a turn while compacting");
                Check(window.InputField.text.Length > 0 || TranscriptCount == before,
                      "R3 typed text not swallowed while blocked");
                yield return WaitFor(240f, () => FullySettled);
                before = TranscriptCount;
                UiSend("Say goodbye briefly.");
                yield return WaitFor(30f, () => TranscriptCount == before + 1);
                Check(TranscriptCount == before + 1, "R3 send works again after the compaction");
                yield return WaitFor(120f, () => !ReplyInFlight && npc.State == NPCChatBase.NPCState.WaitingInInteraction);
            }

            npc.CloseInteraction();
            yield return new WaitForSecondsRealtime(2f);
            Check(errors.Count == 0, $"zero console errors/exceptions through the run ({errors.Count})");
            phase = "done";
            Finish();
        }

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
            if (errors.Count > 0)
            {
                sb.AppendLine("\n## Console errors");
                foreach (var e in errors) sb.AppendLine("- " + e);
            }
            Directory.CreateDirectory("ProbeLogs");
            File.WriteAllText(ReportPath, sb.ToString());
            File.WriteAllText(MarkerPath, fails.Count == 0 ? "PASS" : "FAIL");
            Debug.Log($"[NpcManualReproProbe] report -> {ReportPath}");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }
}
