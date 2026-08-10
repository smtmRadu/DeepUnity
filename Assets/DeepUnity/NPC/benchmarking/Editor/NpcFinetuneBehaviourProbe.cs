#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        /// <summary>
        /// BEHAVIOURAL probe for the roleplay-finetuned Qwen3.5-0.8B — does the model that came back
        /// from the LoRA run actually play the NPC, on the engine that will ship it?
        ///
        /// WHY IT EXISTS AND WHY IT LIVES IN UNITY. Every other check on this adapter is mechanical:
        /// the merge applied 186/186 modules, the int8 export reconstructs to 2.3e-3, the wire probes
        /// pass. None of that says the model stays in character, and none of it can be run under HF
        /// transformers on this box — the checkpoint's config declares mtp_num_hidden_layers=1, so
        /// transformers instantiates Multi-Token-Prediction modules that are not in the checkpoint and
        /// preallocates ~2.8 GiB before a single activation, on a 3.9 GB card. DeepUnity's importer
        /// drops MTP entirely, so the ONLY place these weights can be exercised on this machine is
        /// here — which is also the only place whose numbers mean anything, because it is the runtime
        /// that will serve them.
        ///
        /// THE PROMPT IS NOT WRITTEN HERE. It is read out of the shipped ChatDemo3D scene
        /// (NPCChatBase.EffectivePromptPreview on the Velmire object): the assembled ## NAME + persona
        /// + # Tools text the game seeds, not a paraphrase of it. A probe that invents its own prompt
        /// measures the prompt it invented. If the scene or the NPC cannot be found the run FAILS
        /// rather than quietly substituting one.
        ///
        /// EDIT MODE, NO PLAY MODE — same reason as Qwen3_5ResetProbe, and pumped the same way:
        /// -batchmode play mode freezes pre-play on this box, and the code under test waits on
        /// AsyncGPUReadback, which nothing drains inside a tight while(MoveNext) loop.
        ///
        /// THE DISK KV CACHE IS OFF FOR THE WHOLE RUN. A cached system-prompt KV blob is keyed by
        /// owner + prompt hash and knows NOTHING about which weights computed it, so a snapshot left
        /// behind by the BASE model would be restored on top of the finetuned ones and the probe would
        /// grade a prefix the finetune never produced.
        ///
        ///   menu:  DeepUnity/Qwen3.5/NPC Finetune Behaviour
        ///   batch: Unity.exe -batchmode -projectPath &lt;repo&gt; ^
        ///            -logFile ProbeLogs/npc_finetune_behaviour.log ^
        ///            -executeMethod DeepUnity.Qwen3_5Modeling.NpcFinetuneBehaviourProbe.Run
        /// No -quit (the run exits itself) and NO -nographics: real compute shaders.
        /// </summary>
        public static class NpcFinetuneBehaviourProbe
        {
            // One report per sampling mode — a preset run must never overwrite the greedy baseline
            // it is meant to be compared against.
            static string REPORT => greedy ? "ProbeLogs/npc_finetune_behaviour.md"
                                           : "ProbeLogs/npc_finetune_behaviour_preset.md";
            static string DONE => greedy ? "ProbeLogs/npc_finetune_behaviour.done"
                                         : "ProbeLogs/npc_finetune_behaviour_preset.done";
            const string SCENE = "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity";
            const string NPC_NAME = "Velmire";

            const int CACHE_CAPACITY = 4096;   // Velmire's prompt is ~1.2k tokens; leave room for the chat
            const int REPLY_TOKENS = 160;
            const int COMPACT_TOKENS = 256;

            // Sampling for the non-greedy mode. The Qwen3.5 CARD preset is t=1.0/top_p=1.0, and
            // Config still serves that to shipped NPCs — but a card preset is tuned for the stock
            // instruct model, not for a LoRA that has been pulled hard toward one persona and one
            // tool schema. t=1.0 with repetition_penalty=1.0 gave the run-3 checkpoint enough rope
            // to loop (user 2026-08-07), so the probe grades at the tighter pair below and passes
            // the SAME pair to compaction, which until today ran greedy with no penalty at all.
            const float PROBE_TEMPERATURE = 0.7f;
            const float PROBE_REPETITION_PENALTY = 1.1f;

            static readonly StringBuilder report = new StringBuilder();
            static readonly List<string> fails = new List<string>();
            static readonly List<string> consoleErrors = new List<string>();

            static void Log(string s) { report.AppendLine(s); Debug.Log("[NpcFT] " + s); }
            static void Fail(string s) { fails.Add(s); report.AppendLine("**FAIL** " + s); Debug.LogWarning("[NpcFT] FAIL " + s); }
            static void Check(bool ok, string what)
            {
                if (ok) { report.AppendLine("- PASS  " + what); Debug.Log("[NpcFT] PASS " + what); }
                else Fail(what);
            }

            // THE DEFAULT IS THE SHIPPED SAMPLER, NOT GREEDY (user 2026-08-06). A model is only worth
            // what it does at the settings it will actually serve: NPCChatBase forwards Config.Default*
            // (Qwen3.5's non-thinking preset) for every inspector field left at -1, which is how every
            // shipped NPC is configured. Grading at t=0 flatters the model — the 2026-08-06 adapter
            // scored 23/24 greedy and 19/24 at the preset, and the greedy number would have been read
            // as "it works" when the sale silently fails in game.
            // Greedy stays reachable for ONE purpose: it is deterministic, so an A/B between two
            // adapters is attributable rather than sampling noise. It is never the headline result.
            static bool greedy = false;

            [MenuItem("DeepUnity/Qwen3.5/NPC Finetune Behaviour")]
            public static void RunInteractive() { greedy = false; Execute(exitWhenDone: false); }

            [MenuItem("DeepUnity/Qwen3.5/NPC Finetune Behaviour (greedy A-B only)")]
            public static void RunInteractiveGreedy() { greedy = true; Execute(exitWhenDone: false); }

            /// <summary>Batch entry (-executeMethod) at the sampling preset NPCChatBase serves.</summary>
            public static void Run() { greedy = false; Execute(exitWhenDone: true); }

            /// <summary>Batch entry, greedy — for adapter-vs-adapter comparison ONLY, not for grading.</summary>
            public static void RunGreedy() { greedy = true; Execute(exitWhenDone: true); }

            // ------------------------------------------------------------------ run lifecycle

            const double STEP_BUDGET_MS = 8.0;
            const int MAX_STEPS = 4_000_000;
            const double MAX_SECONDS = 3_000.0;

            static readonly Stack<IEnumerator> stack = new Stack<IEnumerator>();
            static bool exitAtEnd;
            static int steps;
            static double startedAt;

            static void Execute(bool exitWhenDone)
            {
                report.Clear();
                fails.Clear();
                consoleErrors.Clear();
                everySpokenLine.Clear();
                steps = 0;
                exitAtEnd = exitWhenDone;
                startedAt = EditorApplication.timeSinceStartup;
                Directory.CreateDirectory("ProbeLogs");
                try { if (File.Exists(DONE)) File.Delete(DONE); } catch { }

                Application.logMessageReceived -= OnLog;
                Application.logMessageReceived += OnLog;

                Log($"# NPC finetune behaviour — {DateTime.Now:yyyy-MM-dd HH:mm}");
                Log("");
                Log(LMProbeCommon.SystemInfoBlock());

                stack.Clear();
                stack.Push(Steps());
                EditorApplication.update -= Pump;
                EditorApplication.update += Pump;
            }

            static void OnLog(string msg, string stack_, LogType type)
            {
                if ((type == LogType.Error || type == LogType.Exception) && !msg.StartsWith("[NpcFT]"))
                    consoleErrors.Add(msg);
            }

            static void Pump()
            {
                var budget = System.Diagnostics.Stopwatch.StartNew();
                while (stack.Count > 0 && budget.Elapsed.TotalMilliseconds < STEP_BUDGET_MS)
                {
                    if (++steps > MAX_STEPS || EditorApplication.timeSinceStartup - startedAt > MAX_SECONDS)
                    {
                        Fail($"**ABORTED**: the probe did not finish in {MAX_SECONDS:0}s / {MAX_STEPS} steps.");
                        stack.Clear();
                        break;
                    }
                    AsyncGPUReadback.WaitAllRequests();   // see Qwen3_5ResetProbe.Pump

                    var top = stack.Peek();
                    bool moved;
                    try { moved = top.MoveNext(); }
                    catch (Exception e)
                    {
                        Fail($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                        stack.Clear();
                        break;
                    }
                    if (!moved) { stack.Pop(); continue; }
                    if (top.Current is IEnumerator nested) stack.Push(nested);
                }
                if (stack.Count == 0) Finish();
            }

            static void Finish()
            {
                EditorApplication.update -= Pump;
                Application.logMessageReceived -= OnLog;

                report.AppendLine();
                report.AppendLine("## Console");
                report.AppendLine(consoleErrors.Count == 0
                    ? "- PASS  no errors or exceptions during the run"
                    : $"- **FAIL** {consoleErrors.Count} console error(s): {string.Join(" | ", consoleErrors.ToArray())}");
                if (consoleErrors.Count > 0) fails.Add($"{consoleErrors.Count} console error(s)");

                report.AppendLine();
                report.AppendLine(fails.Count == 0
                    ? "## RESULT: PASS"
                    : $"## RESULT: FAIL ({fails.Count} failure(s))");
                foreach (string f in fails) report.AppendLine($"  - {f}");

                File.WriteAllText(REPORT, report.ToString());
                File.WriteAllText(DONE, fails.Count == 0 ? "PASS" : "FAIL");
                Debug.Log($"[NpcFT] done — {fails.Count} failure(s). Report: {REPORT}");
                if (exitAtEnd && Application.isBatchMode)
                    EditorApplication.Exit(fails.Count == 0 ? 0 : 1);
            }

            // ------------------------------------------------------------------ the run

            static string Esc(string s) =>
                s == null ? "(null)" : s.Replace("\r", "").Replace("\n", "\\n");

            /// <summary>Markdown block for one exchange: what was asked, what came back verbatim.</summary>
            static void Transcript(string who, string said, string reply, float seconds)
            {
                report.AppendLine();
                report.AppendLine($"**{who}** ({seconds:F1}s)");
                report.AppendLine();
                report.AppendLine("```");
                report.AppendLine("> " + said);
                report.AppendLine(reply.Trim().Length == 0 ? "(empty reply)" : reply.Trim());
                report.AppendLine("```");
            }

            // The reply channel split the game does (NPCChatBase.SplitToolCall) reduced to what a
            // behavioural grade needs: the tool-call body, and the function name + parameters inside it.
            static readonly Regex RxCall = new Regex(@"<tool_call>(.*?)</tool_call>", RegexOptions.Singleline);
            static readonly Regex RxFunc = new Regex(@"<function=([A-Za-z0-9_]+)>");
            static readonly Regex RxParam = new Regex(@"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>", RegexOptions.Singleline);
            static readonly Regex RxAsterisk = new Regex(@"\*[^*\n]{3,}\*");

            static string CallName(string reply)
            {
                var c = RxCall.Match(reply);
                if (!c.Success) return null;
                var f = RxFunc.Match(c.Groups[1].Value);
                return f.Success ? f.Groups[1].Value : null;
            }

            static Dictionary<string, string> CallParams(string reply)
            {
                var d = new Dictionary<string, string>();
                var c = RxCall.Match(reply);
                if (!c.Success) return d;
                foreach (Match m in RxParam.Matches(c.Groups[1].Value))
                    d[m.Groups[1].Value] = m.Groups[2].Value;
                return d;
            }

            /// <summary>Visible speech = the reply minus any tool-call block (what the player hears).</summary>
            static string Spoken(string reply) => RxCall.Replace(reply ?? "", "").Trim();

            static Qwen3_5ForCausalLM llm;
            static readonly List<string> everySpokenLine = new List<string>();

            /// <summary>The turn as the game issues it: greedy for reproducibility, or the model's
            /// own Config preset — the exact knobs NPCChatBase forwards when its inspector fields
            /// are left at -1, which is how every shipped NPC is configured.</summary>
            static IEnumerator Turn(string user, StringBuilder sink) =>
                greedy
                ? llm.Chat(user, t => sink.Append(t), max_new_tokens: REPLY_TOKENS, temperature: 0f)
                : llm.Chat(user, t => sink.Append(t), max_new_tokens: REPLY_TOKENS,
                           temperature: PROBE_TEMPERATURE,
                           top_k: llm.Config.DefaultTopK, top_p: llm.Config.DefaultTopP,
                           min_p: llm.Config.DefaultMinP,
                           presence_penalty: llm.Config.DefaultPresencePenalty,
                           repetition_penalty: PROBE_REPETITION_PENALTY);

            static IEnumerator TurnTool(string json, StringBuilder sink) =>
                greedy
                ? llm.ChatToolResult(json, t => sink.Append(t), max_new_tokens: REPLY_TOKENS, temperature: 0f)
                : llm.ChatToolResult(json, t => sink.Append(t), max_new_tokens: REPLY_TOKENS,
                           temperature: PROBE_TEMPERATURE,
                           top_k: llm.Config.DefaultTopK, top_p: llm.Config.DefaultTopP,
                           min_p: llm.Config.DefaultMinP,
                           presence_penalty: llm.Config.DefaultPresencePenalty,
                           repetition_penalty: PROBE_REPETITION_PENALTY);

            static IEnumerator Say(string user, string label, Action<string> onReply)
            {
                var sb = new StringBuilder();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                yield return Turn(user, sb);
                string reply = sb.ToString();
                Transcript(label, user, reply, (float)sw.Elapsed.TotalSeconds);
                everySpokenLine.Add(Spoken(reply));
                onReply(reply);
            }

            static IEnumerator ToolResult(string json, string label, Action<string> onReply)
            {
                var sb = new StringBuilder();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                yield return TurnTool(json, sb);
                string reply = sb.ToString();
                Transcript(label, "<tool_response> " + json, reply, (float)sw.Elapsed.TotalSeconds);
                everySpokenLine.Add(Spoken(reply));
                onReply(reply);
            }

            static IEnumerator Steps()
            {
                // ---- 1. the prompt, out of the shipped scene ------------------------------------
                string sys = null;
                try
                {
                    EditorSceneManager.OpenScene(SCENE, OpenSceneMode.Single);
                    foreach (var npc in UnityEngine.Object.FindObjectsOfType<NPCChatBase>(true))
                    {
                        string n = npc.name;
                        if (n.IndexOf(NPC_NAME, StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            sys = npc.EffectivePromptPreview;
                            Log($"prompt source : `{SCENE}` -> GameObject `{n}` " +
                                $"({npc.GetType().Name}.EffectivePromptPreview)");
                            break;
                        }
                    }
                }
                catch (Exception e) { Fail($"opening `{SCENE}` threw {e.GetType().Name}: {e.Message}"); }

                if (string.IsNullOrEmpty(sys))
                {
                    Fail($"no NPC matching `{NPC_NAME}` in `{SCENE}` — the probe will NOT substitute a " +
                         "prompt of its own, because then it would be grading a prompt the game never sends.");
                    yield break;
                }

                bool hasTools = sys.Contains("<tools>");
                Log($"prompt        : {sys.Length} chars, # Tools block {(hasTools ? "PRESENT" : "**ABSENT**")}");
                Check(hasTools, "the scene prompt carries a # Tools block (tool checks below are meaningful)");
                report.AppendLine();
                report.AppendLine("<details><summary>system prompt as sent</summary>");
                report.AppendLine();
                report.AppendLine("```");
                report.AppendLine(sys);
                report.AppendLine("```");
                report.AppendLine();
                report.AppendLine("</details>");

                // ---- 2. the model, exactly as NPCChatBase.EnsureLlm builds it -------------------
                // int8 weights + INT8 KV (NPCChatBase pairs them that way for anything but FP16).
                Qwen3_5ForCausalLM.SystemPromptDiskCache = false;   // see the class docs
                try
                {
                    llm = new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, LLMQuant.INT8,
                                                 maxModelLength: CACHE_CAPACITY, kv_quant: KVQuant.INT8);
                    llm.DiskKVCache = false;
                    llm.CacheOwnerKey = "__NpcFinetuneBehaviourProbe";
                    llm.model.LoadBlockingForProbe();
                }
                catch (Exception e)
                {
                    Fail($"building the int8 model threw {e.GetType().Name}: {e.Message}");
                    yield break;
                }
                while (!llm.tokenizer.IsReady) yield return null;
                while (!llm.IsReady) yield return null;
                Log($"weights       : `{llm.WeightsLabel}` int8 / KV INT8 / capacity {CACHE_CAPACITY}");

                var swInit = System.Diagnostics.Stopwatch.StartNew();
                yield return llm.InitializeChat(sys);
                Log($"prefill       : {llm.CurrentContextTokens} tokens in {swInit.Elapsed.TotalSeconds:F1}s");
                report.AppendLine();
                report.AppendLine(greedy
                    ? "## Conversation (greedy, temperature 0 — reproducible, NOT what ships)"
                    : $"## Conversation (t={PROBE_TEMPERATURE}, top_k={llm.Config.DefaultTopK}, "
                      + $"top_p={llm.Config.DefaultTopP}, min_p={llm.Config.DefaultMinP}, "
                      + $"presence={llm.Config.DefaultPresencePenalty}, "
                      + $"repetition={PROBE_REPETITION_PENALTY} — compaction uses the same pair)");

                // ---- 3. the conversation --------------------------------------------------------
                string rGreet = "", rLore = "", rOff = "", rGear = "", rAfterGear = "",
                       rLowball = "", rDeal = "";

                yield return Say("hello there", "1. greeting", r => rGreet = r);
                yield return Say("what waits beyond that golden mist?", "2. grounded lore", r => rLore = r);
                yield return Say("who is the current president of France?", "3. out-of-knowledge", r => rOff = r);
                yield return Say("do you have a weapon you could spare me?", "4. weapon request", r => rGear = r);

                // Whatever it called, answer the gear read the way the game would, so the sale can proceed.
                if (CallName(rGear) == "CheckMyGear")
                    yield return ToolResult(
                        "{\"you_have_sword\": true, \"you_have_shield\": false, " +
                        "\"already_given_away\": false, \"player_has_weapon\": false}",
                        "5. gear result fed back", r => rAfterGear = r);
                else
                    Log("_(no CheckMyGear call to answer — skipping the tool-result turn)_");

                yield return Say("I only have 25 souls. please, take 25 for the sword.",
                                 "6. lowball below the floor", r => rLowball = r);
                yield return Say("fine. sixty souls for the sword. deal.",
                                 "7. agreed price", r => rDeal = r);

                // ---- 4. the grades --------------------------------------------------------------
                report.AppendLine();
                report.AppendLine("## Behaviour");

                string greetLow = Spoken(rGreet).ToLowerInvariant();
                string loreLow = Spoken(rLore).ToLowerInvariant();
                string offLow = Spoken(rOff).ToLowerInvariant();

                Check(Spoken(rGreet).Trim().Length > 0, "it answers a greeting at all");
                Check(greetLow.Contains("wanderer"),
                      "register: it calls the player 'wanderer' on the greeting");
                Check(loreLow.Contains("sentinel") || loreLow.Contains("halberd") || loreLow.Contains("knight"),
                      "grounding: the mist answer names the Sentinel / halberd / hollow knight");
                Check(!offLow.Contains("macron") && !offLow.Contains("president of france"),
                      "deflection: it does NOT answer the French-politics question with world knowledge");
                Check(offLow.Length > 0 && (offLow.Contains("wanderer") || offLow.Contains("know")
                      || offLow.Contains("gate") || offLow.Contains("mist") || offLow.Contains("not")),
                      "deflection stays in character (it deflects as Velmire, not as an assistant)");

                string gearCall = CallName(rGear);
                Check(rGear.Contains("<tool_call>") && rGear.Contains("<function="),
                      "tool format: the weapon request produces a well-formed <tool_call><function=...>");
                Check(gearCall == "CheckMyGear",
                      $"tool choice: it looks before it offers — CheckMyGear (called: {gearCall ?? "none"})");

                string lowLow = Spoken(rLowball).ToLowerInvariant();
                bool holdsFloor = (lowLow.Contains("60") || lowLow.Contains("sixty"))
                                  && !CallHasPriceBelow(rLowball, 60);
                Check(holdsFloor,
                      "price floor: 25 souls is refused and the sixty-soul floor is restated");
                Check(!CallHasPriceBelow(rLowball, 60),
                      "price floor: no GiveItem is emitted below 60 souls");

                string dealCall = CallName(rDeal);
                var dealArgs = CallParams(rDeal);
                Check(dealCall == "GiveItem",
                      $"the agreed price produces a GiveItem call (called: {dealCall ?? "none"})");
                Check(dealArgs.ContainsKey("item") && dealArgs["item"].Trim().ToLowerInvariant().Contains("sword"),
                      $"GiveItem item is the sword (got: {(dealArgs.ContainsKey("item") ? dealArgs["item"] : "missing")})");
                Check(dealArgs.ContainsKey("price") && dealArgs["price"].Trim() == "60",
                      $"GiveItem price is exactly 60 (got: {(dealArgs.ContainsKey("price") ? dealArgs["price"] : "missing")})");

                bool narrates = false;
                foreach (string line in everySpokenLine)
                    if (RxAsterisk.IsMatch(line)) narrates = true;
                Check(!narrates, "no *action narration* asterisks anywhere in the spoken channel");

                // ---- 5. compaction — the engine's own path --------------------------------------
                // LLM.Compact sends the bare COMPACT_PROMPT and re-seeds [sys]\n\n## MEMORY\n[reply].
                report.AppendLine();
                report.AppendLine("## Compaction");
                int beforeTokens = llm.CurrentContextTokens;
                string summary = null;
                var swC = System.Diagnostics.Stopwatch.StartNew();
                yield return llm.Compact(sys, s => summary = s, max_summary_tokens: COMPACT_TOKENS,
                                         temperature: greedy ? 0f : PROBE_TEMPERATURE,
                                         repetition_penalty: greedy ? 1f : PROBE_REPETITION_PENALTY);
                float compactSeconds = (float)swC.Elapsed.TotalSeconds;
                summary ??= "";

                report.AppendLine();
                report.AppendLine($"`{LLM.COMPACT_PROMPT}` -> {compactSeconds:F1}s, " +
                                  $"context {beforeTokens} -> {llm.CurrentContextTokens} tokens");
                report.AppendLine();
                report.AppendLine("```");
                report.AppendLine(summary.Trim().Length == 0 ? "(empty compact)" : summary.Trim());
                report.AppendLine("```");
                report.AppendLine();

                string sLow = summary.ToLowerInvariant();
                Check(summary.Trim().Length > 0, "compaction produces a non-empty summary");
                Check(llm.CurrentContextTokens < beforeTokens,
                      $"compaction RECLAIMS context ({beforeTokens} -> {llm.CurrentContextTokens} tokens)");
                Check(sLow.Contains("sword"), "the compact keeps the sword deal");
                Check(summary.Contains("60") || sLow.Contains("sixty"), "the compact keeps the sixty-soul price");
                Check(sLow.Contains("sentinel") || sLow.Contains("mist"),
                      "the compact keeps the Sentinel / the mist");
                Check(!RxAsterisk.IsMatch(summary), "the compact carries no *action narration*");
                Check(!summary.Contains("<tool_call>"), "the compact does not emit a tool call");

                // A compact that is longer than the conversation it replaces reclaims nothing.
                Check(summary.Length < 1600,
                      $"the compact is bounded ({summary.Length} chars, cap {COMPACT_TOKENS} tokens)");

                // The NPC must still be able to talk on the re-seeded prefix.
                string rAfter = "";
                yield return Say("what did we agree on?", "8. after compaction", r => rAfter = r);
                string afterLow = Spoken(rAfter).ToLowerInvariant();
                Check(Spoken(rAfter).Trim().Length > 0, "a turn on the compacted chat still answers");
                Check(afterLow.Contains("sword") || afterLow.Contains("60") || afterLow.Contains("sixty"),
                      "the NPC still remembers the deal THROUGH the compact (the ## MEMORY re-seed works)");

                // ---- 6. teardown ----------------------------------------------------------------
                try { llm.Release(); } catch { }
                llm = null;
                GC.Collect();
            }

            /// <summary>True when the reply's tool call carries a numeric price under
            /// <paramref name="floor"/> — the thing the haggling rule forbids.</summary>
            static bool CallHasPriceBelow(string reply, int floor)
            {
                var p = CallParams(reply);
                return p.ContainsKey("price")
                       && int.TryParse(p["price"].Trim(), out int v)
                       && v < floor;
            }
        }
    }
}
#endif
