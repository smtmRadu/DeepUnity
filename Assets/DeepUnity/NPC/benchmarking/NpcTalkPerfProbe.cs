using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // #29 — talk-time frame-pacing diagnostic in the REAL ChatDemo3D scene. The user sees dips
    // to ~45 FPS while an NPC speaks (pocket-tts streaming + Qwen decode concurrently); this
    // probe walks up to Velmire, holds a MULTI-TURN conversation, and records EVERY frame's
    // duration tagged with what the pipeline was doing that frame:
    //   GEN    LLM reply in flight (state == TalkingInInteraction)
    //   SPK    pocket-tts synthesis in flight (pkVoice.IsSpeaking)
    //   AUD    speech audible (pkVoice.IsAudioPlaying)
    //   FLUSH  a Mimi chunk landed in the ring THIS frame (BufferedSamples jumped up)
    // The report buckets frames by flag combo (mean/p95/max/spike counts per combo) and lists
    // the worst frames — pinpointing whether the dips ride on chunk flushes, generation frames,
    // both pumps colliding, etc. vsync is DISABLED during the run (raw frame cost; a 22 ms
    // frame = the 45 FPS dip the user sees). Diagnostic — writes report + done marker, self-exits.
    public class NpcTalkPerfProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_talkperf.md";
        const string MarkerPath = "ProbeLogs/npc_talkperf.done";

        static readonly string[] QUESTIONS =
        {
            "Greet me in one short sentence.",
            "Tell me the story of this castle in three or four sentences.",
            "What do you think about the witch across the courtyard? Two sentences.",
            "Describe tonight's weather and what it means for travelers, in several sentences.",
        };

        // tag = PocketTTS.LastHeavyTick read-and-cleared each frame (#29 iteration 2): which TTS
        // pipeline stage worked in the measured window. ±1 frame skew possible (Update order vs
        // the voice's pump is undefined) — fine for attributing spike RUNS to a stage.
        struct Frame { public float ms; public byte flags; public int ringDelta; public string tag; }
        const byte GEN = 1, SPK = 2, AUD = 4, FLUSH = 8;

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<Frame> turnFrames = new List<Frame>(32768);
        readonly List<List<Frame>> turns = new List<List<Frame>>();
        bool recording;
        int lastBuffered;

        Transform player;
        CharacterController playerCC;
        NPCChatBase velmire;
        INPCChatWindow window;
        PocketTTSModeling.PocketTTSVoice pk;
        int prevVsync; int prevTarget;

        void Awake()
        {
            Application.logMessageReceived += OnLog;
            Application.runInBackground = true;
            prevVsync = QualitySettings.vSyncCount;
            prevTarget = Application.targetFrameRate;
            QualitySettings.vSyncCount = 0;      // raw frame cost — a 22 ms frame IS the 45 FPS dip
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

        void Update()
        {
            if (!recording) return;
            if (pk == null && velmire != null) pk = velmire.GetComponent<PocketTTSModeling.PocketTTSVoice>();

            byte flags = 0;
            int ringDelta = 0;
            if (velmire != null && velmire.State == NPCChatBase.NPCState.TalkingInInteraction) flags |= GEN;
            if (pk != null)
            {
                if (pk.IsSpeaking) flags |= SPK;
                if (pk.IsAudioPlaying) flags |= AUD;
                int buffered = pk.BufferedSamples;
                ringDelta = buffered - lastBuffered;
                if (ringDelta > 0) flags |= FLUSH;   // a Mimi chunk was pushed this frame
                lastBuffered = buffered;
            }
            string tag = PocketTTSModeling.PocketTTS.LastHeavyTick;
            PocketTTSModeling.PocketTTS.LastHeavyTick = null;
            turnFrames.Add(new Frame { ms = Time.unscaledDeltaTime * 1000f, flags = flags, ringDelta = ringDelta, tag = tag });
        }

        IEnumerator Start()
        {
            sb.AppendLine("# NPC talk-time frame-pacing (#29) — real ChatDemo3D, vsync OFF — " +
                          System.DateTime.Now.ToString("yyyy-MM-dd HH:mm"));
            yield return null;

            var playerGO = GameObject.FindWithTag("Player");
            foreach (var npc in FindObjectsOfType<NPCChatBase>(true))
                if (npc.gameObject.name.Contains("Velmire")) velmire = npc;
            var winGO = FindObjectOfType<Tutorials.ChatDemo3D.SoulsChatWindow>(true);
            window = winGO;
            player = playerGO != null ? playerGO.transform : null;
            playerCC = playerGO != null ? playerGO.GetComponent<CharacterController>() : null;
            if (player == null || velmire == null || window == null)
            {
                sb.AppendLine($"- **FAIL** scene wiring: player={player != null} velmire={velmire != null} window={window != null}");
                Finish(); yield break;
            }

            // settle (scene prewarm) then walk up — models land during the approach
            yield return WaitFor(6f, null);
            Teleport(ZonePoint(velmire.transform, 7.5f));
            yield return WaitFor(60f, () => velmire.LlmReady);
            sb.AppendLine($"- Velmire LLM ready: {velmire.LlmReady}");

            Teleport(ZonePoint(velmire.transform, 1.6f));
            yield return new WaitForSecondsRealtime(1.5f);
            velmire.StartInteraction();
            yield return WaitFor(60f, () => velmire.State == NPCChatBase.NPCState.WaitingInInteraction);
            sb.AppendLine($"- dialogue open: {velmire.State}");

            // ---- the monitored multi-turn conversation ----
            for (int t = 0; t < QUESTIONS.Length; t++)
            {
                window.InputField.text = QUESTIONS[t];
                pk = velmire.GetComponent<PocketTTSModeling.PocketTTSVoice>();
                lastBuffered = pk != null ? pk.BufferedSamples : 0;
                long defer0 = FramePacing.TtsDeferrals;
                turnFrames.Clear();
                recording = true;
                velmire.AskNPC();
                float start = Time.unscaledTime;
                // record until the reply is fully generated AND fully spoken (audible tail included)
                while (Time.unscaledTime - start < 150f)
                {
                    bool done = velmire.State == NPCChatBase.NPCState.WaitingInInteraction &&
                                (pk == null || (!pk.IsSpeaking && !pk.IsAudioPlaying));
                    if (done && Time.unscaledTime - start > 2f) break;
                    yield return null;
                }
                recording = false;
                turns.Add(new List<Frame>(turnFrames));
                sb.AppendLine($"- turn {t + 1} recorded: {turnFrames.Count} frames over {Time.unscaledTime - start:0.0} s " +
                              $"(tts pump ceded {FramePacing.TtsDeferrals - defer0} LLM frames)");
                yield return new WaitForSecondsRealtime(0.75f);
            }

            velmire.CloseInteraction();
            Analyze();
            Finish();
        }

        // ---------------- analysis ----------------

        static string FlagName(byte f)
        {
            if (f == 0) return "idle";
            var p = new List<string>(4);
            if ((f & GEN) != 0) p.Add("GEN");
            if ((f & SPK) != 0) p.Add("SPK");
            if ((f & AUD) != 0) p.Add("AUD");
            if ((f & FLUSH) != 0) p.Add("FLUSH");
            return string.Join("+", p);
        }

        void Analyze()
        {
            var all = new List<Frame>();
            foreach (var t in turns) all.AddRange(t);
            sb.AppendLine($"\n## Aggregate ({all.Count} frames across {turns.Count} turns)");
            Buckets(all);

            for (int t = 0; t < turns.Count; t++)
            {
                sb.AppendLine($"\n## Turn {t + 1} — \"{QUESTIONS[t]}\"");
                Buckets(turns[t]);
            }

            // spike attribution: which TTS pipeline stage the >22.2 ms frames rode on (#29 it.2)
            var byTag = new Dictionary<string, List<float>>();
            foreach (var f in all)
            {
                if (f.ms <= 22.2f) continue;
                string key = f.tag ?? (((f.flags & GEN) != 0) ? "(no tts tick; GEN)" : "(no tts tick)");
                if (!byTag.TryGetValue(key, out var l)) byTag[key] = l = new List<float>();
                l.Add(f.ms);
            }
            sb.AppendLine("\n## Heavy-tick attribution — frames > 22.2 ms");
            sb.AppendLine("| stage tag | frames | mean ms | max ms |");
            sb.AppendLine("|---|---|---|---|");
            foreach (var kv in byTag)
            {
                double mean = 0; float mx = 0;
                foreach (float v in kv.Value) { mean += v; mx = Mathf.Max(mx, v); }
                sb.AppendLine($"| {kv.Key} | {kv.Value.Count} | {mean / kv.Value.Count:0.00} | {mx:0.0} |");
            }

            // worst frames with context
            all.Sort((a, b) => b.ms.CompareTo(a.ms));
            sb.AppendLine("\n## Worst 15 frames (aggregate)");
            sb.AppendLine("| ms | flags | ring delta (samples) | tag |");
            sb.AppendLine("|---|---|---|---|");
            for (int i = 0; i < Mathf.Min(15, all.Count); i++)
                sb.AppendLine($"| {all[i].ms:0.0} | {FlagName(all[i].flags)} | {all[i].ringDelta} | {all[i].tag ?? "-"} |");

            if (errors.Count > 0)
            {
                sb.AppendLine("\n## Console errors during the run");
                foreach (var e in errors) sb.AppendLine("- " + e);
            }
        }

        void Buckets(List<Frame> frames)
        {
            var byFlag = new Dictionary<byte, List<float>>();
            int over17 = 0, over22 = 0, over33 = 0;
            foreach (var f in frames)
            {
                if (!byFlag.TryGetValue(f.flags, out var l)) byFlag[f.flags] = l = new List<float>();
                l.Add(f.ms);
                if (f.ms > 16.7f) over17++;
                if (f.ms > 22.2f) over22++;   // 45 FPS — the user-visible dip
                if (f.ms > 33.4f) over33++;
            }
            sb.AppendLine($"- spikes: >16.7 ms (60fps miss): **{over17}** | >22.2 ms (45fps dip): **{over22}** | >33.4 ms: **{over33}**");
            sb.AppendLine("| flags | frames | mean ms | p95 ms | max ms | >16.7 | >22.2 |");
            sb.AppendLine("|---|---|---|---|---|---|---|");
            foreach (var kv in byFlag)
            {
                var l = kv.Value; l.Sort();
                double mean = 0; int o17 = 0, o22 = 0;
                foreach (float v in l) { mean += v; if (v > 16.7f) o17++; if (v > 22.2f) o22++; }
                mean /= l.Count;
                float p95 = l[(int)(0.95f * (l.Count - 1))];
                sb.AppendLine($"| {FlagName(kv.Key)} | {l.Count} | {mean:0.00} | {p95:0.00} | {l[l.Count - 1]:0.0} | {o17} | {o22} |");
            }
        }

        // ---------------- helpers (NpcE2EProbe patterns) ----------------

        IEnumerator WaitFor(float timeout, System.Func<bool> until)
        {
            float start = Time.unscaledTime;
            while (Time.unscaledTime - start < timeout)
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

        Vector3 ZonePoint(Transform npc, float dist)
        {
            Vector3 d = player.position - npc.position; d.y = 0;
            Vector3 dir = d.sqrMagnitude < 0.01f ? Vector3.back : d.normalized;
            Vector3 p = npc.position + dir * dist;
            p.y = player.position.y;
            return p;
        }

        void Finish()
        {
            Directory.CreateDirectory("ProbeLogs");
            File.WriteAllText(ReportPath, sb.ToString());
            File.WriteAllText(MarkerPath, "DONE");
            Debug.Log($"[NpcTalkPerfProbe] report -> {ReportPath}");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }
}
