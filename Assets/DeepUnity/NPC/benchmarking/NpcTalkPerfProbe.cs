using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // #29 — talk-time frame-pacing diagnostic in the REAL ChatDemo3D scene. Records EVERY frame
    // from scene settle to the end of the conversation, split into named PHASES, each frame
    // tagged with what the pipeline was doing:
    //   GEN    LLM reply in flight (state == TalkingInInteraction)
    //   SPK    pocket-tts synthesis in flight (pkVoice.IsSpeaking)
    //   AUD    speech audible (pkVoice.IsAudioPlaying)
    //   FLUSH  a Mimi chunk landed in the ring THIS frame (BufferedSamples jumped up)
    // Protocol (2026-08-02, matches how the author actually plays — the old one waited for
    // LlmReady at the zone edge, so the BOOSTED-loading window the user reports dips in was
    // never exercised, let alone captured): enter the zone, give the slow prefetch only ~3.5 s,
    // then open the dialogue with the weights still streaming — StartInteraction is the boost
    // edge — and record straight through the open. Then a multi-turn conversation as before.
    // The report buckets frames by flag combo per phase, attributes >22.2 ms frames to the TTS
    // stage + pump state that ran them, and lists the worst frames with the LLM phase attached.
    // vsync is DISABLED during the run (raw frame cost; a 22 ms frame = the 45 FPS dip the user
    // sees). Diagnostic — writes report + done marker, self-exits.
    public class NpcTalkPerfProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_talkperf.md";
        const string MarkerPath = "ProbeLogs/npc_talkperf.done";

        // ONE reply (2026-08-03, author's call): every open problem lives in the load-up window
        // — settle / walk-up / boosted open / first clause — and turns 2-4 only re-measured a
        // conversation that has been clean for days while tripling the cycle time. One multi-
        // clause reply still exercises clause starts, streaming and cruise; ProbeLogs history
        // holds the 4-question runs if a full-conversation measurement is ever needed again.
        static readonly string[] QUESTIONS =
        {
            "Tell me the story of this castle in three or four sentences.",
        };

        // tag = PocketTTS.LastHeavyTick read-and-cleared each frame (#29 iteration 2): which TTS
        // pipeline stage worked in the measured window. ±1 frame skew possible (Update order vs
        // the voice's pump is undefined) — fine for attributing spike RUNS to a stage.
        struct Frame
        {
            public float ms; public byte flags; public int ringDelta;
            public string tag; public string pump; public float ringS; public string llm;
        }
        const byte GEN = 1, SPK = 2, AUD = 4, FLUSH = 8;

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<(string name, List<Frame> frames)> phases = new List<(string, List<Frame>)>();
        List<Frame> current;    // the phase being recorded, null = not recording
        int lastBuffered;

        Transform player;
        CharacterController playerCC;
        NPCChatBase velmire;
        INPCChatWindow window;
        PocketTTSModeling.PocketTTSVoice pk;
        int prevVsync; int prevTarget;

        // Exactly ONE probe may drive the protocol (2026-08-03): the scene can carry an armed
        // copy AND NpcTalkPerfRunner spawns its own — two probes both teleport the player and
        // both AskNPC, each question cutting the other's reply (the "turn 1: 2.1 s" runs).
        // First Awake wins; later instances self-destruct loudly. The static survives
        // domain-reload-off replays, but a destroyed Unity object compares == null, so the
        // guard self-heals each session.
        static NpcTalkPerfProbe s_driver;

        void Awake()
        {
            if (s_driver != null && s_driver != this)
            {
                Debug.LogWarning("[NpcTalkPerfProbe] duplicate probe detected — self-destructing. " +
                                 "One copy drives the protocol; remove the scene-armed one or don't, " +
                                 "this guard handles it either way.");
                Destroy(gameObject);
                return;
            }
            s_driver = this;
            Application.logMessageReceived += OnLog;
            Application.runInBackground = true;
            prevVsync = QualitySettings.vSyncCount;
            prevTarget = Application.targetFrameRate;
            QualitySettings.vSyncCount = 0;      // raw frame cost — a 22 ms frame IS the 45 FPS dip
            Application.targetFrameRate = -1;
        }

        void OnDestroy()
        {
            if (s_driver != this) return;   // a self-destructed duplicate armed nothing —
                                            // restoring here would write its default-0 fields
                                            // over the driver's real vsync/targetFrameRate
            s_driver = null;
            Application.logMessageReceived -= OnLog;
            QualitySettings.vSyncCount = prevVsync;
            Application.targetFrameRate = prevTarget;
        }

        void OnLog(string msg, string stack, LogType type)
        {
            if (type == LogType.Exception || type == LogType.Error)
                if (errors.Count < 40) errors.Add($"{type}: {msg.Substring(0, Mathf.Min(160, msg.Length))}");
        }

        void StartPhase(string name)
        {
            current = new List<Frame>(16384);
            phases.Add((name, current));
            lastBuffered = pk != null ? pk.BufferedSamples : 0;
        }

        void Update()
        {
            if (current == null) return;
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
            // NON-clearing frame-stamped read (2026-08-03): this used to read-and-clear, which
            // races every other consumer — with a second probe in the scene the tags were split
            // between the two CSVs and an entire attribution run was garbage. Same discipline as
            // FrameSpikeProbe: a tag is only evidence about the frame it was written in.
            int tickAge = Time.frameCount - PocketTTSModeling.PocketTTS.LastHeavyTickFrame;
            string tag = tickAge <= 1 ? PocketTTSModeling.PocketTTS.LastHeavyTick : null;
            // pump snapshot (2026-08-02): what the voice pump says it did this frame — state
            // (speaking/cruise/low-ring/prefill/cede-llm) and the ring level it decided on.
            bool pumpFresh = Time.frameCount - PocketTTSModeling.PocketTTSVoice.PumpFrame <= 1;
            current.Add(new Frame
            {
                ms = Time.unscaledDeltaTime * 1000f, flags = flags, ringDelta = ringDelta, tag = tag,
                pump = pumpFresh ? PocketTTSModeling.PocketTTSVoice.PumpState : null,
                ringS = pumpFresh ? PocketTTSModeling.PocketTTSVoice.PumpRingSeconds : -1f,
                llm = LLM.CurrentPhase,
            });
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

            // scene prewarm tail — the GC-burst window the fps timeline attributes to asset load
            StartPhase("settle");
            yield return WaitFor(6f, null);

            // walk-up: zone entry arms slow prefetch + kernel prewarm + voice prepare, and gets
            // only ~3.5 s of it — the author does not stand at the zone edge waiting for a bar.
            StartPhase("walk-up (slow prefetch)");
            Teleport(ZonePoint(velmire.transform, 7.5f));
            yield return WaitFor(3.5f, null);
            Teleport(ZonePoint(velmire.transform, 1.6f));
            yield return WaitFor(1.0f, null);

            // the boost edge: dialogue opens with the weights still streaming. Everything the
            // user calls "boosted model loading" happens inside this phase — full-budget upload,
            // LLM boot + warmup, voice prepare, system-prompt prefill, camera + UI open.
            StartPhase("boosted open");
            bool llmWasReady = velmire.LlmReady;
            velmire.StartInteraction();
            yield return WaitFor(150f, () => velmire.State == NPCChatBase.NPCState.WaitingInInteraction);
            sb.AppendLine($"- dialogue open: {velmire.State} (LlmReady at open: {llmWasReady}, " +
                          $"open phase {current.Count} frames)");

            // ---- the monitored multi-turn conversation ----
            for (int t = 0; t < QUESTIONS.Length; t++)
            {
                window.InputField.text = QUESTIONS[t];
                pk = velmire.GetComponent<PocketTTSModeling.PocketTTSVoice>();
                long defer0 = FramePacing.TtsDeferrals;
                StartPhase($"turn {t + 1}");
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
                sb.AppendLine($"- turn {t + 1} recorded: {current.Count} frames over {Time.unscaledTime - start:0.0} s " +
                              $"(tts pump ceded {FramePacing.TtsDeferrals - defer0} LLM frames)");
                yield return new WaitForSecondsRealtime(0.75f);
            }
            current = null;

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
            foreach (var (_, f) in phases) all.AddRange(f);
            sb.AppendLine($"\n## Aggregate ({all.Count} frames across {phases.Count} phases — includes load-up, not only turns)");
            Buckets(all);

            foreach (var (name, f) in phases)
            {
                sb.AppendLine($"\n## Phase: {name} ({f.Count} frames)");
                Buckets(f);
            }

            // spike attribution: which TTS pipeline stage + pump state the >22.2 ms frames rode on,
            // with the LLM boot/prefill/decode phases separated (the load-up window lives there).
            var byTag = new Dictionary<string, List<float>>();
            foreach (var f in all)
            {
                if (f.ms <= 22.2f) continue;
                string key = f.tag ?? "(no tts tick)";
                if (f.pump != null) key += " @" + f.pump;
                if (!string.IsNullOrEmpty(f.llm) && f.llm != "idle") key += " [" + f.llm + "]";
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
            sb.AppendLine("\n## Worst 20 frames (aggregate)");
            sb.AppendLine("| ms | flags | llm | ring delta | tag | pump | ring s |");
            sb.AppendLine("|---|---|---|---|---|---|---|");
            for (int i = 0; i < Mathf.Min(20, all.Count); i++)
                sb.AppendLine($"| {all[i].ms:0.0} | {FlagName(all[i].flags)} | {all[i].llm} | {all[i].ringDelta} | {all[i].tag ?? "-"} " +
                              $"| {all[i].pump ?? "-"} | {(all[i].ringS < 0f ? "-" : all[i].ringS.ToString("0.00"))} |");

            if (errors.Count > 0)
            {
                sb.AppendLine("\n## Console errors during the run");
                foreach (var e in errors) sb.AppendLine("- " + e);
            }
        }

        void Buckets(List<Frame> frames)
        {
            if (frames.Count == 0) { sb.AppendLine("- (no frames)"); return; }
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
