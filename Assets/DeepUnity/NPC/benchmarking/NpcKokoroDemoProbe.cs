using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // Velmire-on-Kokoro smoke test inside the REAL ChatDemo3D scene: the scene ships Velmire on
    // pocket-tts, so this probe flips the NPC's serialized ttsModel/ttsVoice to Kokoro/af_heart
    // via reflection BEFORE the prefetch zone runs (the NpcInterruptProbe settings-forcing
    // pattern), then walks the scripted player through zone entry -> dialogue -> ask -> voiced
    // reply -> close, recording frames + console errors like NpcE2EProbe. The voice gate latches
    // KokoroVoice.IsSpeaking / IsAudioPlaying (synthesis/ring driven — valid headless, where the
    // batch audio device never drains the ring). Report: ProbeLogs/npc_kokoro_demo.md + .done.
    public class NpcKokoroDemoProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_kokoro_demo.md";
        const string MarkerPath = "ProbeLogs/npc_kokoro_demo.done";

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<float> frames = new List<float>();
        bool recording;
        int failures;

        Transform player;
        CharacterController playerCC;
        NPCChatBase velmire;
        INPCChatWindow window;

        void Awake() => Application.logMessageReceived += OnLog;
        void OnDestroy() => Application.logMessageReceived -= OnLog;

        void OnLog(string msg, string stack, LogType type)
        {
            if (type == LogType.Exception || type == LogType.Error)
                if (errors.Count < 40) errors.Add($"{type}: {msg.Substring(0, Mathf.Min(160, msg.Length))}");
        }

        void Update()
        {
            if (recording) frames.Add(Time.unscaledDeltaTime * 1000f);
        }

        IEnumerator Start()
        {
            sb.AppendLine("# Velmire-on-Kokoro demo probe (real ChatDemo3D scene) — " + System.DateTime.Now.ToString("yyyy-MM-dd HH:mm"));
            yield return null;   // let the scene's own Start()s run

            var playerGO = GameObject.FindWithTag("Player");
            foreach (var npc in FindObjectsOfType<NPCChatBase>(true))
                if (npc.gameObject.name.Contains("Velmire")) velmire = npc;
            var winGO = FindObjectOfType<Tutorials.ChatDemo3D.SoulsChatWindow>(true);
            window = winGO;
            player = playerGO != null ? playerGO.transform : null;
            playerCC = playerGO != null ? playerGO.GetComponent<CharacterController>() : null;

            if (player == null || velmire == null || window == null)
            {
                Fail($"scene wiring: player={(player != null)} velmire={(velmire != null)} window={(window != null)}");
                Finish(); yield break;
            }

            // K0 — flip the NPC to Kokoro. NPCChatBase.Start() already ran EnsureVoice() (it
            // attached PocketTTSVoice per the scene setting), so after flipping the fields we
            // re-invoke EnsureVoice via reflection — it attaches KokoroVoice and the speak path
            // (switching on EffectiveTtsModel) uses it from now on; the idle pocket component
            // just sits there.
            var flags = BindingFlags.NonPublic | BindingFlags.Instance;
            var fModel = typeof(NPCChatBase).GetField("ttsModel", flags);
            var fVoice = typeof(NPCChatBase).GetField("ttsVoice", flags);
            var mEnsure = typeof(NPCChatBase).GetMethod("EnsureVoice", flags);
            if (fModel == null || fVoice == null || mEnsure == null)
            {
                Fail("reflection: ttsModel/ttsVoice/EnsureVoice not found on NPCChatBase");
                Finish(); yield break;
            }
            fModel.SetValue(velmire, NPCChatBase.TtsModel.Kokoro);
            fVoice.SetValue(velmire, "af_heart");   // exported voicepack ("jean" is pocket-only)
            mEnsure.Invoke(velmire, null);
            Check(velmire.GetComponent<KokoroVoice>() != null,
                  "K0 Velmire forced to TtsModel.Kokoro (voice af_heart), KokoroVoice attached");

            // K1 — walk into the prefetch zone: LLM + Kokoro weights must land. Start OUTSIDE
            // the zone so entry (OnPlayerContact) drives the prefetch of the NEW voice.
            Teleport(velmire.transform.position + ZoneDir(velmire.transform) * 25f);
            yield return Phase("K0 boot settle (prewarm window)", 6f, null);
            Teleport(ZonePoint(velmire.transform, 7.5f));
            float t0 = Time.unscaledTime;
            yield return Phase("K1 Velmire zone entry -> models land", 60f, () => velmire.LlmReady);
            Check(velmire.LlmReady, $"K1 Velmire LLM ready in {Time.unscaledTime - t0:0.0} s");

            // K2 — open the dialogue, ask, and require ACTUAL Kokoro synthesis + a finished reply
            Teleport(ZonePoint(velmire.transform, 1.6f));
            yield return new WaitForSecondsRealtime(1.5f);   // let the talk trigger register the player
            velmire.StartInteraction();
            yield return Phase("K2a dialogue open (KV restore/prefill)", 90f,
                               () => velmire.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(velmire.State == NPCChatBase.NPCState.WaitingInInteraction, "K2a dialogue reached Waiting");

            var kk = velmire.GetComponent<KokoroVoice>();
            Check(kk != null, "K2a KokoroVoice component attached (EnsureVoice took the Kokoro path)");

            window.InputField.text = "Greet me in one short sentence.";
            float askT = Time.unscaledTime;
            velmire.AskNPC();
            bool spoke = false; float ttVoice = -1f;
            yield return Phase("K2b reply generation + Kokoro voice", 120f, () =>
            {
                if (!spoke && kk != null && (kk.IsSpeaking || kk.IsAudioPlaying))
                { spoke = true; ttVoice = Time.unscaledTime - askT; }
                return velmire.State == NPCChatBase.NPCState.WaitingInInteraction
                       && (kk == null || !kk.IsSpeaking) && spoke;
            });
            Check(spoke, $"K2b Kokoro synthesis produced audio (time-to-voice {ttVoice:0.00} s after Ask)");
            Check(velmire.State == NPCChatBase.NPCState.WaitingInInteraction, "K2b reply finished (state Waiting)");

            // K3 — second turn on the same session (KV continuity with the Kokoro voice attached)
            window.InputField.text = "Say goodbye in one short sentence.";
            spoke = false;
            velmire.AskNPC();
            yield return Phase("K3 second turn", 120f, () =>
            {
                if (!spoke && kk != null && (kk.IsSpeaking || kk.IsAudioPlaying)) spoke = true;
                return velmire.State == NPCChatBase.NPCState.WaitingInInteraction
                       && (kk == null || !kk.IsSpeaking) && spoke;
            });
            Check(spoke, "K3 second turn voiced");

            velmire.CloseInteraction();
            yield return new WaitForSecondsRealtime(0.5f);
            Check(velmire.State == NPCChatBase.NPCState.Idle, "K3 interaction closed clean");

            Finish();
        }

        IEnumerator Phase(string name, float timeout, System.Func<bool> until)
        {
            frames.Clear(); recording = true;
            float start = Time.unscaledTime;
            while (Time.unscaledTime - start < timeout)
            {
                if (until != null && until()) break;
                yield return null;
            }
            recording = false;
            bool timedOut = until != null && !until();
            int over17 = 0, over33 = 0; float worst = 0;
            foreach (float f in frames)
            {
                if (f > 16.7f) over17++;
                if (f > 33.4f) over33++;
                if (f > worst) worst = f;
            }
            sb.AppendLine($"\n## {name}");
            sb.AppendLine($"- {Time.unscaledTime - start:0.00} s, {frames.Count} frames | >16.7 ms: {over17} | >33.4 ms: {over33} | worst {worst:0.0} ms{(timedOut ? "  **TIMEOUT**" : "")}");
            if (timedOut) failures++;
        }

        void Check(bool ok, string what)
        {
            sb.AppendLine($"- {(ok ? "PASS" : "**FAIL**")}: {what}");
            if (!ok) failures++;
        }

        void Fail(string what) { sb.AppendLine($"- **FAIL**: {what}"); failures++; }

        void Teleport(Vector3 pos)
        {
            if (playerCC != null) playerCC.enabled = false;
            player.position = pos;
            if (playerCC != null) playerCC.enabled = true;
        }

        // a point at `dist` from the NPC toward the current player position, at player height
        Vector3 ZonePoint(Transform npc, float dist)
        {
            Vector3 dir = ZoneDir(npc);
            Vector3 p = npc.position + dir * dist;
            p.y = player.position.y;
            return p;
        }

        Vector3 ZoneDir(Transform npc)
        {
            Vector3 d = player.position - npc.position; d.y = 0;
            return d.sqrMagnitude < 0.01f ? Vector3.back : d.normalized;
        }

        void Finish()
        {
            sb.AppendLine($"\n## Console errors/exceptions during the run: {errors.Count}");
            foreach (var e in errors) sb.AppendLine($"- {e}");
            sb.AppendLine($"\n## VERDICT: {(failures == 0 && errors.Count == 0 ? "ALL PASS" : $"{failures} failures, {errors.Count} errors")}");
            Directory.CreateDirectory("ProbeLogs");
            File.WriteAllText(ReportPath, sb.ToString());
            File.WriteAllText(MarkerPath, failures == 0 && errors.Count == 0 ? "PASS" : "FAIL");
            Debug.Log($"[NpcKokoroDemoProbe] report -> {ReportPath}");
        }
    }
}
