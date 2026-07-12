using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // End-to-end play-mode test of the WHOLE NPC stack inside the REAL ChatDemo3D scene:
    // a scripted "player" walks into prefetch zones, opens dialogues, asks questions, listens
    // for voice, walks out — while every phase records frame times and every console
    // error/exception is captured. Scenarios:
    //   S1  enter Velmire's zone        -> slow prefetch runs, LLM+Kokoro land, frames stay clean
    //   S2  talk to Velmire (LLM+TTS)   -> reply streams, Kokoro audio actually PLAYS
    //   S3  leave the zone              -> IMMEDIATE unload (no grace) of LLM (+ TTS defetch)
    //   S4  Velmire zone -> witch zone  -> pooled Qwen is SHARED (witch ready instantly, no 2nd load)
    //   S5  talk to the witch (LlmOnly) -> text reply works on the shared instance (KV ownership)
    //   S6  CosyVoice smoke test        -> WRONG voice name on purpose: engine must fall back
    //                                      (not silence), stream weights, and produce audio
    // Report: ProbeLogs/npc_e2e.md + .done marker (bridge-orchestrated via NpcE2ERunner).
    public class NpcE2EProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_e2e.md";
        const string MarkerPath = "ProbeLogs/npc_e2e.done";

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<float> frames = new List<float>();
        bool recording;
        int failures;

        Transform player;
        CharacterController playerCC;
        NPCChatBase velmire, witch;
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
            sb.AppendLine("# NPC end-to-end probe (real ChatDemo3D scene) — " + System.DateTime.Now.ToString("yyyy-MM-dd HH:mm"));
            yield return null;   // let the scene's own Start()s run

            var playerGO = GameObject.FindWithTag("Player");
            foreach (var npc in FindObjectsOfType<NPCChatBase>(true))
            {
                if (npc.gameObject.name.Contains("Velmire")) velmire = npc;
                if (npc.gameObject.name.Contains("Morwenna")) witch = npc;
            }
            var winGO = FindObjectOfType<Tutorials.ChatDemo3D.SoulsChatWindow>(true);
            window = winGO;
            player = playerGO != null ? playerGO.transform : null;
            playerCC = playerGO != null ? playerGO.GetComponent<CharacterController>() : null;

            if (player == null || velmire == null || witch == null || window == null)
            {
                Fail($"scene wiring: player={(player != null)} velmire={(velmire != null)} witch={(witch != null)} window={(window != null)}");
                Finish(); yield break;
            }

            // S0 — settle: scene-start prewarm runs (kernels + tokenizer + GC sweep)
            yield return Phase("S0 boot settle (prewarm window)", 6f, null);

            // S1 — walk into Velmire's prefetch zone, models must land without frame drops
            Teleport(ZonePoint(velmire.transform, 7.5f));
            float t0 = Time.unscaledTime;
            yield return Phase("S1 Velmire zone entry -> models land", 45f, () => velmire.LlmReady);
            Check(velmire.LlmReady, $"S1 Velmire LLM ready in {Time.unscaledTime - t0:0.0} s");

            // S2 — open the dialogue, ask, and require ACTUAL audio (Kokoro) + a finished reply
            Teleport(ZonePoint(velmire.transform, 1.6f));
            yield return new WaitForSecondsRealtime(1.5f);   // let the talk trigger register the player
            velmire.StartInteraction();
            yield return Phase("S2a dialogue open (KV restore/prefill)", 60f,
                               () => velmire.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(velmire.State == NPCChatBase.NPCState.WaitingInInteraction, "S2a dialogue reached Waiting");

            var kk = velmire.GetComponent<KokoroVoice>();
            window.InputField.text = "Greet me in one short sentence.";
            float askT = Time.unscaledTime;
            velmire.AskNPC();
            bool heard = false; float ttAudio = -1f;
            yield return Phase("S2b reply generation + voice", 90f, () =>
            {
                if (!heard && kk != null && kk.IsAudioPlaying) { heard = true; ttAudio = Time.unscaledTime - askT; }
                return velmire.State == NPCChatBase.NPCState.WaitingInInteraction
                       && (kk == null || !kk.IsSpeaking) && heard;
            });
            Check(heard, $"S2b Kokoro audio audible (time-to-audio {ttAudio:0.00} s after Ask)");
            Check(velmire.State == NPCChatBase.NPCState.WaitingInInteraction, "S2b reply finished (state Waiting)");

            velmire.CloseInteraction();
            yield return new WaitForSecondsRealtime(0.5f);

            // S3 — leave the zone: unload must start IMMEDIATELY (no grace period)
            Teleport(velmire.transform.position + ZoneDir(velmire.transform) * 25f);
            yield return Phase("S3 zone exit -> immediate unload", 12f, () => !velmire.LlmLoaded);
            Check(!velmire.LlmLoaded, "S3 LLM released after leaving the zone (immediate, no linger)");

            // S4 — pooled sharing: load in Velmire's zone, then walk STRAIGHT into the witch's
            // zone while still holding Velmire's model — the witch must reuse the same instance.
            Teleport(ZonePoint(velmire.transform, 7.5f));
            yield return Phase("S4a re-enter Velmire zone (fresh load)", 45f, () => velmire.LlmReady);
            Teleport(ZonePoint(witch.transform, 7.5f));   // inside witch zone; may still be in Velmire's
            float shareT = Time.unscaledTime;
            yield return Phase("S4b step into witch zone (pool share)", 10f, () => witch.LlmReady);
            Check(witch.LlmReady, $"S4b witch LLM ready in {Time.unscaledTime - shareT:0.00} s (pooled — must be ~instant)");

            // S5 — text-only dialogue with the witch on the shared instance
            Teleport(ZonePoint(witch.transform, 1.6f));
            yield return new WaitForSecondsRealtime(1.5f);   // let the talk trigger register the player
            witch.StartInteraction();
            yield return Phase("S5a witch dialogue open", 60f,
                               () => witch.State == NPCChatBase.NPCState.WaitingInInteraction);
            window.InputField.text = "What do you brew, in one sentence?";
            witch.AskNPC();
            yield return Phase("S5b witch reply (LlmOnly)", 90f,
                               () => witch.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(witch.State == NPCChatBase.NPCState.WaitingInInteraction, "S5b witch reply finished");
            witch.CloseInteraction();
            yield return new WaitForSecondsRealtime(0.5f);
            Teleport(witch.transform.position + ZoneDir(witch.transform) * 30f);
            yield return Phase("S5c exit all zones -> full unload", 12f,
                               () => !velmire.LlmLoaded && !witch.LlmLoaded);
            Check(!velmire.LlmLoaded && !witch.LlmLoaded, "S5c pooled LLM fully released outside every zone");

            // S6 — CosyVoice smoke: WRONG voice name on purpose (the exact user repro: a Kokoro
            // pack name left set) — the engine must warn+fall back and still produce audio.
            var cvGO = new GameObject("CosyVoiceSmoke");
            cvGO.transform.position = player.position;
            var cvSrc = cvGO.AddComponent<AudioSource>(); cvSrc.spatialBlend = 0f;
            var cv = cvGO.AddComponent<CosyVoiceModeling.CosyVoiceVoice>();
            cv.loadOnStart = false;
            cv.voiceName = "velmire_elder";   // does NOT exist for CosyVoice — fallback path
            cv.PrefetchNow();
            yield return Phase("S6a CosyVoice weights stream (full speed)", 120f, () => cv.IsReady);
            Check(cv.IsReady, "S6a CosyVoice ready despite unknown voice name (fallback)");
            bool spoke = false;
            if (cv.IsReady)
            {
                var say = cv.SayRoutine("The old mill by the river is turning again.");
                float sayT = Time.unscaledTime;
                while (say.MoveNext())
                {
                    if (cv.IsSpeaking) spoke = true;
                    if (Time.unscaledTime - sayT > 120f) break;
                    yield return say.Current;
                }
            }
            Check(spoke, "S6b CosyVoice produced audible speech (ring played)");
            cv.DefetchNow();

            Finish();
        }

        // -------------------------------------------------------------- plumbing

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
            File.WriteAllText(MarkerPath, "done");
            Debug.Log($"[NpcE2EProbe] report -> {ReportPath}");
        }
    }
}
