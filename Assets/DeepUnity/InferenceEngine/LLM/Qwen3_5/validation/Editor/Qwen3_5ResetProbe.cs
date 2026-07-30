#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        /// <summary>
        /// Gate for "Reset Conversation" — <c>NPCChatBase.ResetConversation</c> /
        /// <c>ResetConversationRoutine</c>, whose model-side step is a re-initialize on the bare
        /// system prompt.
        ///
        /// WHY IT EXISTS. On 2026-07-28 Reset Conversation changed bookkeeping only: it cleared the
        /// transcript, the compact, the pool claim and the disk snapshot, and never touched the LIVE
        /// KV cache. The window emptied, the context bar did not move, and the model answered the rest
        /// of that session with the whole pre-reset conversation still in its context. Nothing threw.
        /// The tell was a UI bar — which is exactly why this file measures the CACHE and not the UI.
        ///
        /// WHY GATE 2 IS THE IMPORTANT ONE, and why it must never be weakened into a token count:
        /// Qwen3.5 is a HYBRID model — 18 Gated DeltaNet layers next to 6 full-attention ones.
        /// <c>Qwen3_5Cache.CachedTokenCount</c> has a setter and the K/V layout is token-major, so
        /// "reset" looks like one assignment away. It is not. The DeltaNet layers hold
        /// <c>conv_state</c>/<c>recurrent_state</c>, running state that is NOT indexed by token
        /// position, so a cursor rewind truncates the attention layers correctly and leaves eighteen
        /// layers remembering the entire conversation. No error, no exception, plausible output.
        /// Gate 2 reads those buffers back and compares them byte for byte against a fresh
        /// InitializeChat, and it carries its own NEGATIVE CONTROL: it first performs the tempting
        /// cursor-only rewind and asserts that the state DIFFERS. If that control ever stopped
        /// differing, the gate would be vacuous and this probe fails instead of passing quietly.
        /// The current implementation (full re-initialize) passes gate 2 by construction — that is the
        /// point: the gate is here to fail the day somebody "optimises" it back into a cursor
        /// assignment.
        ///
        /// EVERY GATE IS MEASURED MID-CONVERSATION, IN ONE SESSION. There is no close-and-reopen
        /// anywhere below, because "it is fine after you leave and come back" was never the
        /// requirement: the reset has to be in force the instant it is pressed (user 2026-07-28).
        ///
        ///   menu:  DeepUnity/Qwen3.5/Reset Conversation Gate
        ///   batch: Unity.exe -batchmode -projectPath &lt;repo&gt; ^
        ///            -logFile ProbeLogs/qwen35_reset.log ^
        ///            -executeMethod DeepUnity.Qwen3_5Modeling.Qwen3_5ResetProbe.Run
        /// No -quit (the run exits itself: 0 on PASS, 1 on FAIL) and NO -nographics: this probe runs
        /// real compute shaders, and kernel lookup fails without a graphics device
        /// ("Kernel 'CopyBuffer' not found").
        /// </summary>
        public static class Qwen3_5ResetProbe
        {
            const string REPORT = "ProbeLogs/qwen35_reset.md";
            const string DONE = "ProbeLogs/qwen35_reset.done";

            const int CACHE_CAPACITY = 2048;
            const int REPLY_TOKENS = 24;      // enough for a real reply, short enough to keep the run brisk

            // A persona nothing else in the project uses, so this probe's own KV cache files
            // (qwen35_prompt___ResetProbe_*.kv, keyed by CacheOwnerKey below) can never collide with
            // — or clobber — a real NPC's.
            const string SYS = "## NAME\n__ResetProbe\n\nYou are a stone gate warden. "
                             + "Answer in one short sentence and stay in character.";
            // Exactly the shape NPCChatBase.BuildResumePrompt / LLM.CompactCore produce:
            // [system prompt]\n\n## MEMORY\n[compact]. This is the ResumeFromCompact prefix a reset
            // has to get the model OFF of.
            const string MEMORY = "The traveller asked about the north gate and was told it is barred at dusk. "
                                + "They promised to return with the warden's seal.";

            const string Q_PARITY = "What is your name, warden?";
            static readonly string[] FILLER =
            {
                "Is the north gate open?",
                "Who holds the seal?",
                "Tell me about the watchman.",
            };

            static readonly StringBuilder report = new StringBuilder();
            static int failures;

            static void Log(string s)
            {
                report.AppendLine(s);
                Debug.Log("[QwenReset] " + s);
            }

            static void Fail(string s)
            {
                failures++;
                report.AppendLine(s);
                Debug.LogError("[QwenReset] " + s);
            }

            static void Gate(bool ok, string what)
            {
                if (ok) Log("OK    " + what);
                else Fail("FAIL  " + what);
            }

            // ONE MODEL PER RUN — do not "save time" by sweeping both KV precisions in one invocation.
            // The first attempt did exactly that and took the editor down with "Could not allocate
            // memory: System out of memory!" the same SECOND the first instance was released
            // (2026-07-28, 3.9 GB GTX 1650 / 24 GB RAM, ChatDemo3D loaded): Release() frees the GPU
            // buffers, but the ~1 GB of managed byte arrays the weight loader read the files into is
            // only reclaimed by the GC, and the second load asks for another ~1 GB immediately. The
            // KV precision is fixed at construction, so covering both means two instances, which
            // means two invocations.
            [MenuItem("DeepUnity/Qwen3.5/Reset Conversation Gate")]
            public static void RunInteractive() => Execute(KVQuant.INT8, exitWhenDone: false);

            /// <summary>The FP16-KV variant of gates 1-3. Its own entry point (its own editor session)
            /// — see the note above. The shipped NPC configuration is int8 weights + INT8 KV
            /// (NPCChatBase.EnsureLlm), which is what the default run covers; this one exists because
            /// the K/V half of the state is packed differently per precision (INT8 additionally carries
            /// the per-(token,head) scale/zero-point planes) and a reset has to be indifferent to
            /// that.</summary>
            [MenuItem("DeepUnity/Qwen3.5/Reset Conversation Gate (FP16 KV)")]
            public static void RunInteractiveFp16() => Execute(KVQuant.FP16, exitWhenDone: false);

            /// <summary>Batch entry (-executeMethod). Exits 0 on PASS, 1 on FAIL.</summary>
            public static void Run() => Execute(KVQuant.INT8, exitWhenDone: true);

            /// <summary>Batch entry for the FP16-KV variant. Exits 0 on PASS, 1 on FAIL.</summary>
            public static void RunFp16Kv() => Execute(KVQuant.FP16, exitWhenDone: true);

            // ------------------------------------------------------------------ run lifecycle
            //
            // Pumped from EditorApplication.update, NOT drained in a tight while(MoveNext) loop like
            // the pure-text probes: the code under test waits on AsyncGPUReadback (SampleYielding's
            // token id, Qwen3_5Cache.SaveYielding's prefix readbacks), nothing pumps Unity's readback
            // queue inside a tight loop, and a tight loop would also freeze the editor for the whole
            // run — unacceptable while the user has the project open. A tick runs as many steps as
            // fit in STEP_BUDGET_MS and flushes pending readbacks between them, so an unfocused
            // editor (which ticks a handful of times a second) still finishes in reasonable time.

            const double STEP_BUDGET_MS = 8.0;
            const int MAX_STEPS = 2_000_000;      // runaway guard: fail loudly instead of hanging batch mode
            const double MAX_SECONDS = 1_200.0;

            static readonly Stack<IEnumerator> stack = new Stack<IEnumerator>();
            static bool exitAtEnd;
            static int steps;
            static double startedAt;

            static void Execute(KVQuant kv, bool exitWhenDone)
            {
                report.Clear();
                failures = 0;
                steps = 0;
                exitAtEnd = exitWhenDone;
                startedAt = EditorApplication.timeSinceStartup;
                Directory.CreateDirectory("ProbeLogs");
                try { if (File.Exists(DONE)) File.Delete(DONE); } catch { }

                Log($"# Qwen3.5 Reset-Conversation gate — {DateTime.Now:yyyy-MM-dd HH:mm}");
                Log("");
                Log(LMProbeCommon.SystemInfoBlock());

                stack.Clear();
                stack.Push(Steps(kv));
                EditorApplication.update -= Pump;   // a previous aborted run must not double-pump
                EditorApplication.update += Pump;
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
                    // Completes anything the step below is about to wait on (see the note above).
                    // A readback's data is only valid in the frame it completed, and we MoveNext in
                    // this same frame, so this is the one place it can be flushed from.
                    AsyncGPUReadback.WaitAllRequests();

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
                    // Unity's own nesting semantics: `yield return <IEnumerator>` runs it to
                    // completion before resuming. Everything else (null, WaitForSeconds) means
                    // "next step" — the probe has no real-time behaviour to honour.
                    if (top.Current is IEnumerator nested) stack.Push(nested);
                }
                if (stack.Count == 0) Finish();
            }

            static void Finish()
            {
                EditorApplication.update -= Pump;
                Log("");
                Log(failures == 0 ? "## RESULT: PASS" : $"## RESULT: FAIL ({failures} failure(s))");
                File.WriteAllText(REPORT, report.ToString());
                File.WriteAllText(DONE, failures == 0 ? "PASS" : "FAIL");
                if (failures != 0)
                    Debug.LogError($"[QwenReset] {failures} failure(s) — Reset Conversation does NOT put the model " +
                                   $"back on its system prompt. See {REPORT}.");
                else
                    Debug.Log($"[QwenReset] PASS — see {REPORT}");
                if (exitAtEnd && Application.isBatchMode)
                    EditorApplication.Exit(failures == 0 ? 0 : 1);
            }

            // ------------------------------------------------------------------ the run

            static IEnumerator Steps(KVQuant kv)
            {
                // Gates 1-4 against a real model at this KV precision; INT8 (the default entry point)
                // is the configuration the NPCs actually run — NPCChatBase.EnsureLlm pairs int8
                // weights with INT8 KV. Gate 4 (the ResumeFromCompact / ## MEMORY case) only runs on
                // that default pass: it is a property of the PROMPT, not of the KV packing.
                yield return ModelPass(kv, fullSuite: kv == KVQuant.INT8);
                // Gate 5: the no-model, edit-mode path — the inspector's right-click reset.
                yield return EditModePass();
            }

            static IEnumerator ModelPass(KVQuant kv, bool fullSuite)
            {
                Log("");
                Log($"## KV = {kv} (weights int8, capacity {CACHE_CAPACITY})");
                Log("");

                string dir = Path.Combine(Application.persistentDataPath, "DeepUnity");
                string owner = "__ResetProbe_" + kv;
                string promptCache = Path.Combine(dir, $"qwen35_prompt_{owner}.kv");
                try { if (File.Exists(promptCache)) File.Delete(promptCache); } catch { }

                Qwen3_5ForCausalLM llm = null;
                try
                {
                    llm = new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, LLMQuant.INT8,
                                                 maxModelLength: CACHE_CAPACITY, kv_quant: kv);
                    // One cache file for this probe, deterministic and ours alone — so it can be
                    // deleted at both ends of the run without going near a real NPC's cache.
                    llm.CacheOwnerKey = owner;

                    // Weights: the dispatcher coroutine that normally streams them never ticks
                    // outside play mode, so drive the pump synchronously (the #31 probe entry point).
                    llm.model.LoadBlockingForProbe();
                    // Tokenizer: 13 MB of JSON parsed on the thread pool; IsReady is set from inside
                    // that worker, so polling it here is enough.
                    while (!llm.tokenizer.IsReady) yield return null;
                    while (!llm.IsReady) yield return null;

                    int want = ExpectedPromptTokens(llm, SYS);
                    Log($"system prompt : {want} tokens by independent tokenization "
                        + "(im_start + role line + trimmed text + im_end + turn tail)");

                    // ---- fresh InitializeChat: the reference state, and gate 1's yardstick --------
                    yield return llm.InitializeChat(SYS);
                    Gate(llm.CurrentContextTokens == want,
                         $"gate 1a  a fresh InitializeChat leaves {want} tokens cached "
                         + $"(got {llm.CurrentContextTokens})");
                    var reference = Snapshot(llm.model, "fresh-init");
                    Log($"          reference state: {reference.Describe()}");

                    // The reply this same prompt produces from this same state — gate 3's reference.
                    var refReply = new StringBuilder();
                    yield return llm.Chat(Q_PARITY, t => refReply.Append(t),
                                          max_new_tokens: REPLY_TOKENS, temperature: 0f);
                    Log($"          reference reply (greedy): {Esc(refReply.ToString().Trim())}");

                    // ---- talk, so the live state is unmistakably a conversation ------------------
                    foreach (string q in FILLER)
                    {
                        var sink = new StringBuilder();
                        yield return llm.Chat(q, t => sink.Append(t),
                                              max_new_tokens: REPLY_TOKENS, temperature: 0f);
                    }
                    int conversationTokens = llm.CurrentContextTokens;
                    Log($"conversation  : {conversationTokens} tokens cached after "
                        + $"{FILLER.Length + 1} exchanges");
                    Gate(conversationTokens > want,
                         $"the conversation really did grow the cache ({conversationTokens} > {want})");

                    // ---- gate 2's NEGATIVE CONTROL: the tempting cursor-only rewind --------------
                    // Do the wrong thing on purpose and prove the gate can see it. If this ever
                    // reports "no difference", gate 2 has stopped testing anything.
                    llm.model.cache.CachedTokenCount = want;
                    var rewound = Snapshot(llm.model, "cursor-rewind");
                    var ctrl = Compare(reference, rewound);
                    rewound = null;   // ~20 MB of managed uint[]; only `reference` is needed past its comparison
                    Gate(ctrl.deltaNetDiffering > 0,
                         $"gate 2-control  a cursor-only rewind is DETECTABLY wrong: "
                         + $"{ctrl.deltaNetDiffering}/{ctrl.deltaNetBlobs} DeltaNet buffers still hold the "
                         + "conversation");
                    Log($"          (its attention prefix matches — {ctrl.attentionEqual}/{ctrl.attentionBlobs} "
                        + "K/V blobs equal — which is exactly why a token count cannot be the gate)");

                    // ---- THE RESET, mid-conversation, in this session ---------------------------
                    // Same call NPCChatBase.ResetConversationRoutine makes on the live model.
                    yield return llm.InitializeChat(SYS);

                    Gate(llm.CurrentContextTokens == want,
                         $"gate 1b  after the reset the live cache is back to {want} tokens "
                         + $"(got {llm.CurrentContextTokens}) — the context bar reads the system prompt");

                    var afterReset = Snapshot(llm.model, "after-reset");
                    var cmp = Compare(reference, afterReset);
                    afterReset = null;
                    Gate(cmp.AllEqual,
                         $"gate 2   the post-reset state is BIT-IDENTICAL to a fresh InitializeChat "
                         + $"({cmp.Summary})");
                    if (!cmp.AllEqual) Log("          " + cmp.detail);

                    var resetReply = new StringBuilder();
                    yield return llm.Chat(Q_PARITY, t => resetReply.Append(t),
                                          max_new_tokens: REPLY_TOKENS, temperature: 0f);
                    Gate(resetReply.ToString() == refReply.ToString(),
                         "gate 3   greedy generation after the reset is identical to generation from a "
                         + "fresh init");
                    if (resetReply.ToString() != refReply.ToString())
                    {
                        Log($"          fresh : {Esc(refReply.ToString())}");
                        Log($"          reset : {Esc(resetReply.ToString())}");
                    }

                    if (!fullSuite) yield break;

                    // ---- gate 4: ResumeFromCompact — the memory goes too -------------------------
                    // The prefix a ResumeFromCompact NPC actually runs on, built exactly as
                    // NPCChatBase.BuildResumePrompt builds it. DiskKVCache off for the call, same as
                    // OpenConversation does for a resume prefix (a one-shot prefix must not litter
                    // the prompt cache).
                    string withMemory = SYS + "\n\n" + LLM.HISTORY_HEADING + "\n" + MEMORY;
                    int wantWithMemory = ExpectedPromptTokens(llm, withMemory);
                    llm.DiskKVCache = false;
                    yield return llm.InitializeChat(withMemory);
                    llm.DiskKVCache = true;
                    Gate(llm.CurrentContextTokens == wantWithMemory && wantWithMemory > want,
                         $"gate 4-setup  the memory-bearing prefix is live and longer: "
                         + $"{llm.CurrentContextTokens} tokens vs {want} bare (## MEMORY present)");
                    var sinkM = new StringBuilder();
                    yield return llm.Chat(FILLER[0], t => sinkM.Append(t),
                                          max_new_tokens: REPLY_TOKENS, temperature: 0f);

                    yield return llm.InitializeChat(SYS);
                    Gate(!SYS.Contains(LLM.HISTORY_HEADING),
                         $"gate 4a  the prompt the reset re-initializes on carries no "
                         + $"'{LLM.HISTORY_HEADING}' block");
                    Gate(llm.CurrentContextTokens == want,
                         $"gate 4b  the live cache matches the SHORTER, memory-free prompt: "
                         + $"{llm.CurrentContextTokens} == {want} (was {wantWithMemory} with memory)");
                    var cmpM = Compare(reference, Snapshot(llm.model, "after-reset-from-memory"));
                    Gate(cmpM.AllEqual,
                         "gate 4c  ...and it is the SAME state as a fresh init — a reset lands on state 0 "
                         + $"whatever prefix it came from ({cmpM.Summary})");
                    if (!cmpM.AllEqual) Log("          " + cmpM.detail);
                }
                finally
                {
                    llm?.Release();
                    try { if (File.Exists(promptCache)) File.Delete(promptCache); } catch { }
                }
            }

            // ------------------------------------------------------------------ gate 5: no model

            /// <summary>
            /// The inspector's right-click reset in EDIT mode: there is no <c>llm</c> at all, so the
            /// whole live branch is unreachable and the reset is exactly "clear the fields, drop the
            /// memory, delete the on-disk conversation". That path is what silently did nothing before
            /// 2026-07-28 (it left the snapshot on disk and the next play restored it verbatim), and it
            /// is now the ONLY way to reach the reset while stopped — the inspector BUTTON is gated to
            /// play mode. So it is gated here.
            /// <para>The fixture NPC is added to an INACTIVE GameObject on purpose: no Awake, no
            /// OnEnable, so nothing prewarms a model or reads a sidecar behind the probe's back.</para>
            /// </summary>
            static IEnumerator EditModePass()
            {
                Log("");
                Log("## edit mode, no model (the inspector's right-click reset)");
                Log("");

                // NpcName is already sanitizer-clean (letters/digits/underscores), so
                // ConversationKvKey() — which is private — is the name verbatim.
                const string NAME = "__ResetProbeNPC";
                string dir = Path.Combine(Application.persistentDataPath, "DeepUnity");
                Directory.CreateDirectory(dir);
                string snapshot = Path.Combine(dir, $"qwen35_conv_{NAME}.kv");
                string sidecar = Path.Combine(dir, $"npc_compact_{NAME}.txt");

                var go = new GameObject("__ResetProbeFixture");
                go.SetActive(false);
                try
                {
                    var npc = go.AddComponent<ResetFixtureNpc>();
                    npc.Configure(NAME, "You are a stone gate warden.", MEMORY);
                    File.WriteAllText(snapshot, "not a real cache file — only its existence is under test");
                    File.WriteAllText(sidecar, MEMORY);

                    Gate(npc.EffectivePromptPreview.Contains(LLM.HISTORY_HEADING),
                         $"gate 5-setup  the fixture's prompt carries a '{LLM.HISTORY_HEADING}' block, "
                         + "a conversation snapshot and a compact sidecar");

                    npc.ResetConversation();

                    Gate(!npc.EffectivePromptPreview.Contains(LLM.HISTORY_HEADING),
                         $"gate 5a  the memory is gone from the prompt ('{LLM.HISTORY_HEADING}' absent)");
                    Gate(!File.Exists(snapshot),
                         "gate 5b  the on-disk conversation snapshot is deleted (no model needed)");
                    Gate(!File.Exists(sidecar),
                         "gate 5c  the compact sidecar is deleted, so the inspector stops showing a "
                         + "memory that no longer exists");
                    Gate(!npc.HasPlayerMessage,
                         "gate 5d  HasPlayerMessage is false afterwards — the inspector button correctly "
                         + "reads as 'nothing to reset'");
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(go);
                    try { if (File.Exists(snapshot)) File.Delete(snapshot); } catch { }
                    try { if (File.Exists(sidecar)) File.Delete(sidecar); } catch { }
                }
                yield break;
            }

            // ------------------------------------------------------------------ prompt length

            /// <summary>
            /// The system prompt's length in tokens, computed INDEPENDENTLY of the cache — the same id
            /// sequence <c>Qwen3_5ForCausalLM.InitializeChatCore</c> builds: <c>&lt;|im_start|&gt;</c>,
            /// the system role line, the TRIMMED text (the template's <c>| trim</c>), <c>&lt;|im_end|&gt;</c>,
            /// the turn tail. Asserting the cache against this rather than against "whatever the first
            /// init happened to leave" is what makes gate 1 a measurement instead of a tautology.
            /// (InitializeChatCore also truncates the text encode at 2048 tokens; this probe's prompts
            /// are nowhere near it.)
            /// </summary>
            static int ExpectedPromptTokens(Qwen3_5ForCausalLM llm, string sys)
                => 1                                                       // <|im_start|>
                 + TokenCount(llm, Qwen3_5ChatTemplate.SystemRoleLine)
                 + TokenCount(llm, (sys ?? "").Trim())
                 + 1                                                       // <|im_end|>
                 + TokenCount(llm, Qwen3_5ChatTemplate.TurnEndTail);

            static int TokenCount(Qwen3_5ForCausalLM llm, string text)
            {
                if (string.IsNullOrEmpty(text)) return 0;
                (Tensor t, _) = llm.tokenizer.Encode(text, add_special_tokens: false);
                return t.Size(-1);
            }

            // ------------------------------------------------------------------ cache snapshots

            /// <summary>
            /// Everything the cache holds that a reset has to deal with, read back to managed memory:
            /// the DeltaNet <c>conv_state</c>/<c>recurrent_state</c> of every linear layer IN FULL
            /// (they are not indexed by position, so there is no prefix to take), and the first
            /// <c>CachedTokenCount</c> rows of every full-attention layer's K/V — plus, under INT8 KV,
            /// the matching per-(token, head) scale/zero-point rows. Only the PREFIX rows of K/V: the
            /// tail beyond the token count is stale conversation data that the cache legitimately
            /// leaves behind (CachedTokenCount masks it), so comparing it would fail a correct reset.
            /// <para>Synchronous <c>GetData</c>, not AsyncGPUReadback: a probe can afford the pipeline
            /// stall, and it keeps the comparison free of any frame-timing subtlety.</para>
            /// </summary>
            sealed class CacheState
            {
                public string label;
                public int tokens;
                public readonly List<string> names = new List<string>();
                public readonly List<uint[]> blobs = new List<uint[]>();

                public void Add(string name, uint[] data) { names.Add(name); blobs.Add(data); }

                public string Describe()
                {
                    long words = 0;
                    foreach (var b in blobs) words += b.Length;
                    return $"{tokens} tokens, {blobs.Count} buffers, {words * 4 / 1024} KB read back";
                }
            }

            static CacheState Snapshot(Qwen3_5Model model, string label)
            {
                var c = model.cache;
                var st = new CacheState { label = label, tokens = c.CachedTokenCount };
                int cap = c.Capacity;
                for (int i = 0; i < c.convStates.Length; i++)
                {
                    if (c.kCaches[i] != null)
                    {
                        int rowUints = c.kCaches[i].count / cap;   // exact at any packing (see Qwen3_5Cache)
                        st.Add($"L{i}.K", Read(c.kCaches[i], st.tokens * rowUints));
                        st.Add($"L{i}.V", Read(c.vCaches[i], st.tokens * rowUints));
                        if (c.kScaleZp != null && c.kScaleZp[i] != null)
                        {
                            int szUints = c.kScaleZp[i].count / cap;
                            st.Add($"L{i}.kScaleZp", Read(c.kScaleZp[i], st.tokens * szUints));
                            st.Add($"L{i}.vScaleZp", Read(c.vScaleZp[i], st.tokens * szUints));
                        }
                    }
                    else
                    {
                        st.Add($"L{i}.conv_state", Read(c.convStates[i], c.convStates[i].count));
                        st.Add($"L{i}.recurrent_state", Read(c.recurrentStates[i], c.recurrentStates[i].count));
                    }
                }
                return st;
            }

            static uint[] Read(ComputeBuffer buf, int count)
            {
                if (count <= 0) return Array.Empty<uint>();
                var a = new uint[count];
                buf.GetData(a, 0, 0, count);
                return a;
            }

            sealed class Diff
            {
                public int deltaNetBlobs, deltaNetDiffering;
                public int attentionBlobs, attentionEqual;
                public string detail = "";
                public bool AllEqual => deltaNetDiffering == 0 && attentionEqual == attentionBlobs;
                public string Summary =>
                    $"{deltaNetBlobs - deltaNetDiffering}/{deltaNetBlobs} DeltaNet + "
                    + $"{attentionEqual}/{attentionBlobs} attention buffers equal";
            }

            /// <summary>Byte-compares two snapshots, keeping the DeltaNet and attention halves apart —
            /// which half disagrees is the whole diagnosis. A cursor rewind shows attention EQUAL and
            /// DeltaNet DIFFERING; a broken re-init shows both differing.</summary>
            static Diff Compare(CacheState a, CacheState b)
            {
                var d = new Diff();
                var sb = new StringBuilder();
                if (a.tokens != b.tokens)
                    sb.Append($"token counts differ ({a.label} {a.tokens} vs {b.label} {b.tokens}); ");
                for (int i = 0; i < a.names.Count; i++)
                {
                    bool ssm = a.names[i].EndsWith("conv_state", StringComparison.Ordinal)
                            || a.names[i].EndsWith("recurrent_state", StringComparison.Ordinal);
                    if (ssm) d.deltaNetBlobs++; else d.attentionBlobs++;

                    bool equal = i < b.names.Count && a.names[i] == b.names[i] && Equal(a.blobs[i], b.blobs[i]);
                    if (equal) { if (!ssm) d.attentionEqual++; continue; }
                    if (ssm) d.deltaNetDiffering++;
                    if (sb.Length < 400) sb.Append(a.names[i]).Append(' ');
                }
                d.detail = sb.Length == 0 ? "" : "differing: " + sb.ToString().TrimEnd();
                return d;
            }

            static bool Equal(uint[] x, uint[] y)
            {
                if (x.Length != y.Length) return false;
                for (int i = 0; i < x.Length; i++) if (x[i] != y[i]) return false;
                return true;
            }

            static string Esc(string s) => (s ?? "").Replace("\\", "\\\\").Replace("\n", "\\n")
                                                    .Replace("\r", "\\r").Replace("\t", "\\t");
        }

        /// <summary>Minimal concrete <see cref="NPCChatBase"/> so the model-free half of Reset
        /// Conversation can be exercised without a scene, a chat window or a GPU. TOP-LEVEL, not
        /// nested inside the probe class, because Unity's AddComponent wants a plain MonoBehaviour
        /// type; and it lives in the EDITOR assembly deliberately — it is a probe fixture, not
        /// something that should ever be droppable onto a real GameObject.</summary>
        internal class ResetFixtureNpc : NPCChatBase
        {
            protected override KeyCode InteractKey => KeyCode.E;
            protected override bool PlayerReady => false;
            protected override float DialogueOpenDelay => 0f;
            protected override void OnInteractionStarted() { }
            protected override void OnInteractionClosed(bool interrupted) { }

            public void Configure(string name, string rules, string memory)
            {
                NpcName = name;
                descriptionAndRules = rules;
                compactSummary = memory;
                historyMode = HistoryMode.ResumeFromCompact;   // the mode that HAS a memory to drop
            }
        }
    }
}
#endif
