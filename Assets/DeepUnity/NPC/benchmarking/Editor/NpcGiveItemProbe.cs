#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using DeepUnity.Tutorials.ChatDemo3D;

namespace DeepUnity
{
    /// <summary>A per-binding accept-gate that answers whatever the probe tells it to and counts how
    /// often it was asked — the seam for "the RESOLVED binding's gate decides, and only that one".
    /// Top-level (not nested in the static probe class) so <c>AddComponent&lt;T&gt;</c> can find its
    /// MonoScript; it lives in the Editor assembly and is never saved into a scene.</summary>
    internal class NpcDecisionProbeGate : MonoBehaviour, INPCDecisionGate
    {
        public bool answer = true;
        public int asked;
        public NPCDecisionResult lastPending;

        public bool CanAccept(NPCDecisionResult pending)
        {
            asked++;
            lastPending = pending;
            return answer;
        }
    }

    /// <summary>
    /// Guard for the NPC's SECOND interactive tool, <b>GiveItem</b>, and for the decision-binding
    /// table both interactive tools resolve through. Six things, none of which any other probe reaches:
    /// <list type="number">
    /// <item><b>schema drift</b> — <see cref="NPCChatBase.GiveItemSchema"/> must equal
    /// dataset_creation/wire_format.py's GIVE_ITEM_SCHEMA byte for byte (and AskUserQuestionSchema its
    /// ASK_SCHEMA). Same reasoning as Qwen3_5ChatTemplateProbe: the prompt the finetune trains on and
    /// the prompt the engine serves are the same bytes or the finetune is worth nothing, and a drifted
    /// schema produces perfectly plausible output, so nothing else notices.
    /// <para>GIVE_ITEM_SCHEMA <b>only</b>. wire_format.py also defines GIVE_TOOL_SCHEMA — the same
    /// bytes under the corpus's name <c>GiveTool</c> — and that divergence is DELIBERATE (author,
    /// 2026-08-01): a model that meets one interactive give-tool under one name in every sample
    /// memorizes the name instead of reading the declared schema, so the corpus teaches name variance
    /// and the engine ships <c>GiveItem</c>. Pinning the engine to the corpus's name would destroy the
    /// thing being tested. Do not "fix" it.</para></item>
    /// <item><b>the wire round trip</b> — a synthetic call in the XML shape Qwen3.5's own template
    /// declares, read by the SAME parser the dialogue uses
    /// (<see cref="NPCChatBase.TryReadGiveItemCall"/>), shown on a REAL window
    /// (<see cref="SoulsChatWindow"/>, which is what Velmire talks through), clicked, and mapped back
    /// through <see cref="NPCChatBase.GiveItemResult"/> — which must be exactly
    /// <c>{"accepted": true}</c> / <c>{"accepted": false}</c> and nothing else.</item>
    /// <item><b>decision resolution</b> — <see cref="NPCDecisionTable"/>'s exact / alias / substring
    /// tiers, the article-and-punctuation normalization, first-declared-wins on a tie, and unmatched
    /// falling through with a warning instead of throwing.</item>
    /// <item><b>two bindings on one NPC</b> — a sword offer and a shield offer on the SAME NPC must
    /// resolve to DIFFERENT ids. This is the case the whole refactor exists for: with the old
    /// one-gate-one-event API there was no key at all, so the game had to string-match text the model
    /// rewords every run.</item>
    /// <item><b>the accept-gate</b> — per-binding first, NPC-wide second. 50 souls against an 80-soul
    /// price renders Accept DEAD (no listener, not interactable) while Decline still answers, so the
    /// exchange cannot dead-end; and a binding's onResolved fires exactly once on Accept and not at all
    /// on Decline.</item>
    /// <item><b>the transaction</b> — accepting at 80 takes the player from 100 souls to 20, through the
    /// demo's own gate and hand-over methods (<see cref="NPCGearOffer"/>), not a re-implementation.</item>
    /// </list>
    /// Edit mode, no play mode, no GPU, no model — the panel is built at runtime and tears itself down
    /// with DestroyImmediate off the play loop, which is exactly what makes this checkable headless.
    /// What is NOT covered: the coroutine around the panel (GiveItemRoutine's epoch/state guards) and
    /// the model turn that consumes the result, both of which need a live dialogue and an LLM. The
    /// panel's LOOK is checked separately by ChatDemo3DBuilder.UiProbeBatch, which screenshots it.
    ///
    ///   menu:  DeepUnity/NPC/GiveItem Guard
    ///   batch: Unity.exe -batchmode -projectPath &lt;repo&gt; ^
    ///            -logFile ProbeLogs/giveitem.log ^
    ///            -executeMethod DeepUnity.NpcGiveItemProbe.Run
    /// No -quit (the method exits itself: 0 on PASS, 1 on FAIL). Never -nographics: it builds uGUI.
    /// </summary>
    public static class NpcGiveItemProbe
    {
        const string REPORT = "ProbeLogs/npc_giveitem.md";

        // The pin lives in the dissertation repo, outside this project — the corpus is authored there
        // and the C# const is the copy. Override with DEEPUNITY_WIRE_FORMAT on a machine that keeps it
        // somewhere else.
        static readonly string[] WireFormatCandidates =
        {
            "E:/Development/Dissertation/dataset_creation/wire_format.py",
            "../Dissertation/dataset_creation/wire_format.py",
            "../../Dissertation/dataset_creation/wire_format.py",
        };

        static readonly StringBuilder report = new StringBuilder();
        static int failures;

        static void Log(string s)
        {
            report.AppendLine(s);
            Debug.Log("[GiveItem] " + s);
        }

        static void Fail(string s)
        {
            failures++;
            report.AppendLine("**FAIL** " + s);
            Debug.LogError("[GiveItem] FAIL: " + s);
        }

        static void Check(bool ok, string what, string detail = null)
        {
            if (ok) Log($"PASS  {what}");
            else Fail(what + (detail == null ? "" : " — " + detail));
        }

        [MenuItem("DeepUnity/NPC/GiveItem Guard")]
        public static void RunInteractive() => Execute(exitWhenDone: false);

        /// <summary>Batch entry (-executeMethod). Exits 0 on PASS, 1 on FAIL.</summary>
        public static void Run() => Execute(exitWhenDone: true);

        static void Execute(bool exitWhenDone)
        {
            report.Clear();
            failures = 0;
            Directory.CreateDirectory("ProbeLogs");
            try
            {
                Log($"# GiveItem guard — {DateTime.Now:yyyy-MM-dd HH:mm}");
                Log("");
                CheckSchemas();
                Log("");
                CheckWireRoundTrip();
                Log("");
                CheckDecisionResolution();
                Log("");
                CheckDecisionBindings();
                Log("");
                CheckDemoWiring();
            }
            catch (Exception e)
            {
                Fail($"EXCEPTION: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
            }

            Log("");
            Log(failures == 0 ? "## RESULT: PASS" : $"## RESULT: FAIL ({failures} failure(s))");
            File.WriteAllText(REPORT, report.ToString());
            if (exitWhenDone) EditorApplication.Exit(failures == 0 ? 0 : 1);
        }

        // ------------------------------------------------------------------ 1. schema drift

        static void CheckSchemas()
        {
            Log("## 1. schemas vs dataset_creation/wire_format.py");
            string path = null;
            string env = Environment.GetEnvironmentVariable("DEEPUNITY_WIRE_FORMAT");
            if (!string.IsNullOrEmpty(env) && File.Exists(env)) path = env;
            if (path == null)
                foreach (string c in WireFormatCandidates)
                    if (File.Exists(c)) { path = c; break; }

            if (path == null)
            {
                Fail("wire_format.py not found (looked at DEEPUNITY_WIRE_FORMAT and " +
                     string.Join(", ", WireFormatCandidates) + ") — the schemas cannot be verified " +
                     "against their pin, so this is a FAILURE, not a skip.");
                return;
            }
            Log($"pin: `{Path.GetFullPath(path)}`");
            string py = File.ReadAllText(path);

            // GIVE_ITEM_SCHEMA is the engine's pin. GIVE_TOOL_SCHEMA is NOT compared to anything in
            // C# — see the class doc: the corpus deliberately ships this tool under the older name so
            // the model learns to read the declared schema instead of memorizing one name.
            CompareToPin(py, "GIVE_ITEM_SCHEMA", NPCChatBase.GiveItemSchema, nameof(NPCChatBase.GiveItemSchema));
            CompareToPin(py, "ASK_SCHEMA", NPCChatBase.AskUserQuestionSchema,
                         nameof(NPCChatBase.AskUserQuestionSchema));

            // ...and where the corpus's own name IS still there, the divergence must be EXACTLY the
            // name, or "same schema under another name" has quietly become "a different schema".
            // Deliberately soft on absence: a later batch is SUPPOSED to invent a third name, and this
            // probe must not turn that into a red build.
            if (py.IndexOf("\nGIVE_TOOL_SCHEMA", StringComparison.Ordinal) < 0)
            {
                Log("note  wire_format.py declares no GIVE_TOOL_SCHEMA any more — the corpus's give-tool " +
                    "name has moved on again, which is the intended direction. Nothing to cross-check.");
                return;
            }
            string giveTool = PythonStringConst(py, "GIVE_TOOL_SCHEMA");
            if (giveTool == null)
            {
                Log("note  GIVE_TOOL_SCHEMA is DERIVED from GIVE_ITEM_SCHEMA in wire_format.py, not " +
                    "written out as a literal — so the two shapes cannot drift apart and there is " +
                    "nothing to byte-compare. The name difference is the point; see the class doc.");
                return;
            }
            Check(giveTool == NPCChatBase.GiveItemSchema.Replace("\"name\": \"GiveItem\"",
                                                                 "\"name\": \"GiveTool\""),
                  "the corpus's GIVE_TOOL_SCHEMA differs from the engine's by the NAME only " +
                  "(a DELIBERATE divergence — the corpus teaches name variance on purpose)",
                  giveTool);
        }

        static void CompareToPin(string py, string pyName, string csValue, string csName)
        {
            string pinned = PythonStringConst(py, pyName);
            if (pinned == null)
            {
                Fail($"could not read {pyName} out of wire_format.py");
                return;
            }
            if (pinned == csValue)
            {
                Log($"PASS  {csName} == {pyName} ({csValue.Length} bytes, byte-identical)");
                return;
            }
            Fail($"{csName} != {pyName}\n" +
                 $"  pin ({pinned.Length} B): {pinned}\n" +
                 $"  C#  ({csValue.Length} B): {csValue}\n" +
                 $"  first difference at index {FirstDiff(pinned, csValue)}");
        }

        static int FirstDiff(string a, string b)
        {
            int n = Mathf.Min(a.Length, b.Length);
            for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
            return n;
        }

        /// <summary>The value of a Python module-level constant written as one or more adjacent
        /// single-quoted string literals (optionally inside parentheses) — which is how wire_format.py
        /// writes every pinned wire string. Text extraction only: no Python is interpreted, exactly as
        /// in Qwen3_5ChatTemplateProbe.</summary>
        static string PythonStringConst(string py, string name)
        {
            int at = py.IndexOf("\n" + name, StringComparison.Ordinal);
            if (at < 0) return null;
            int eq = py.IndexOf('=', at);
            if (eq < 0) return null;

            var sb = new StringBuilder();
            bool any = false;
            for (int i = eq + 1; i < py.Length; i++)
            {
                char c = py[i];
                if (c == '\'')
                {
                    for (i++; i < py.Length; i++)
                    {
                        if (py[i] == '\\' && i + 1 < py.Length) { sb.Append(py[++i]); continue; }
                        if (py[i] == '\'') break;
                        sb.Append(py[i]);
                    }
                    any = true;
                    continue;
                }
                if (c == '#')   // a trailing comment cannot contain more of the value
                {
                    while (i < py.Length && py[i] != '\n') i++;
                    continue;
                }
                // whitespace, newlines and the wrapping parens are the only other things allowed
                // between literals; anything else means the constant has ended
                if (c == '(' || c == ')' || char.IsWhiteSpace(c)) continue;
                break;
            }
            return any ? sb.ToString() : null;
        }

        // ------------------------------------------------------------------ 2. the wire round trip

        // EXACTLY the shape Qwen3.5's chat template renders tool calls in (and what an un-finetuned
        // 0.8B emits), minus the <tool_call> tags the streaming splitter already strips.
        const string SyntheticCall =
            "\n<function=GiveItem>\n" +
            "<parameter=item>\nVelmire's sword\n</parameter>\n" +
            "<parameter=price>\n80\n</parameter>\n" +
            "</function>\n";

        static void CheckWireRoundTrip()
        {
            Log("## 2. synthetic GiveItem call → panel → tool result");

            Check(NPCChatBase.GiveItemResult(true) == "{\"accepted\": true}",
                  "GiveItemResult(true) is exactly {\"accepted\": true}", NPCChatBase.GiveItemResult(true));
            Check(NPCChatBase.GiveItemResult(false) == "{\"accepted\": false}",
                  "GiveItemResult(false) is exactly {\"accepted\": false}", NPCChatBase.GiveItemResult(false));

            if (!NPCChatBase.TryReadGiveItemCall(SyntheticCall, out GiveItemOffer offer))
            {
                Fail("the synthetic XML call did not parse as a GiveItem offer");
                return;
            }
            Check(offer.item == "Velmire's sword", "parsed item", offer.item ?? "<null>");
            Check(offer.price == 80, "parsed price", offer.price?.ToString() ?? "<null>");
            Check(!offer.quantity.HasValue, "no quantity → stays null (not 0)",
                  offer.quantity?.ToString() ?? "<null>");

            // an item-only call is a GIFT: no price, no quantity, still a valid offer
            Check(NPCChatBase.TryReadGiveItemCall("<function=GiveItem><parameter=item>estus flask</parameter>" +
                                                 "</function>", out GiveItemOffer gift)
                  && gift.item == "estus flask" && !gift.price.HasValue,
                  "an item-only call parses as a priceless gift");
            // ...and a call with no item at all is not an offer (the schema's one required parameter)
            Check(!NPCChatBase.TryReadGiveItemCall("<function=GiveItem><parameter=price>80</parameter>" +
                                                  "</function>", out _),
                  "a call with no item is rejected");
            // quantity rides through, and a scruffy price still lands
            Check(NPCChatBase.TryReadGiveItemCall("<function=GiveItem><parameter=item>arrow</parameter>" +
                                                 "<parameter=quantity>12</parameter>" +
                                                 "<parameter=price>30 souls</parameter></function>",
                                                 out GiveItemOffer stack)
                  && stack.quantity == 12 && stack.price == 30,
                  "quantity parses and a price written \"30 souls\" still reads as 30");
            // the Hermes/JSON shape the SFT corpus uses must parse identically
            Check(NPCChatBase.TryReadGiveItemCall(
                      "{\"name\": \"GiveItem\", \"arguments\": {\"item\": \"sword\", \"price\": 80}}",
                      out GiveItemOffer json) && json.item == "sword" && json.price == 80,
                  "the JSON wire shape parses to the same offer");

            // --- the panel, on the REAL window class Velmire talks through
            var go = new GameObject("GiveItemProbeWindow", typeof(RectTransform));
            try
            {
                var win = go.AddComponent<SoulsChatWindow>();

                string accepted = Decide(win, offer, canAccept: true, clickIndex: 0);
                Check(accepted == "{\"accepted\": true}", "Accept → {\"accepted\": true}",
                      accepted ?? "<no decision>");

                string declined = Decide(win, offer, canAccept: true, clickIndex: 1);
                Check(declined == "{\"accepted\": false}", "Decline → {\"accepted\": false}",
                      declined ?? "<no decision>");

                // the two buttons and nothing else, in this order
                win.ShowGiveItem("Velmire", offer, true, _ => { });
                List<Button> btns = Buttons(win);
                Check(btns.Count == 2, "the offer panel has EXACTLY two buttons", btns.Count.ToString());
                if (btns.Count == 2)
                {
                    Check(Label(btns[0]) == "Accept" && Label(btns[1]) == "Decline",
                          "buttons are Accept | Decline", $"{Label(btns[0])} | {Label(btns[1])}");
                }
                // ...and the item line the player reads, currency included (SoulsChatWindow's override)
                string line = PanelPrompt(win);
                Check(line == "Velmire's sword  -  80 souls", "item line renders the price in souls",
                      line ?? "<none>");
                win.HideGiveItem();

                NPCChatBase.TryReadGiveItemCall("<function=GiveItem><parameter=item>arrow</parameter>" +
                                                "<parameter=quantity>12</parameter>" +
                                                "<parameter=price>30</parameter></function>",
                                                out GiveItemOffer arrows);
                win.ShowGiveItem("Velmire", arrows, true, _ => { });
                string stackLine = PanelPrompt(win);
                Check(stackLine == "arrow x12  -  30 souls", "quantity renders as \"x12\"",
                      stackLine ?? "<none>");
                win.HideGiveItem();
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        /// <summary>Open the offer panel, click one button, return the tool result the dialogue would
        /// send — null when the click produced no decision (a gated-off button).</summary>
        static string Decide(SoulsChatWindow win, GiveItemOffer offer, bool canAccept, int clickIndex)
        {
            string result = null;
            // this lambda is the routine's own tail: the window reports a bool, GiveItemResult turns it
            // into the bytes, and Talk() sends those bytes as the <tool_response>
            win.ShowGiveItem("Velmire", offer, canAccept, ok => result = NPCChatBase.GiveItemResult(ok));
            List<Button> btns = Buttons(win);
            if (btns.Count <= clickIndex)
            {
                Fail($"the offer panel built {btns.Count} button(s) — cannot click index {clickIndex}");
                win.HideGiveItem();
                return null;
            }
            btns[clickIndex].onClick.Invoke();   // the click, exactly as uGUI would raise it
            win.HideGiveItem();                  // no-op after a real pick; idempotent
            return result;
        }

        static List<Button> Buttons(Component win)
        {
            var found = new List<Button>();
            foreach (var b in win.GetComponentsInChildren<Button>(true)) found.Add(b);
            return found;
        }

        static string Label(Component button)
        {
            var t = button.GetComponentInChildren<TMP_Text>(true);
            return t != null ? t.text : null;
        }

        /// <summary>The panel's own prompt line — the first TMP text under the panel root that is not a
        /// button label.</summary>
        static string PanelPrompt(Component win)
        {
            foreach (var t in win.GetComponentsInChildren<TMP_Text>(true))
                if (t.GetComponentInParent<Button>() == null) return t.text;
            return null;
        }

        // ------------------------------------------------------------------ 3. decision resolution
        // Pure NPCDecisionTable: no NPC, no window, no scene. These rules are what BOTH interactive
        // tools bind through, so they are asserted once, here, instead of twice through two dispatches.

        static NPCDecision Row(string id, params string[] aliases)
            => new NPCDecision { id = id, aliases = aliases, onResolved = new NPCDecisionEvent() };

        static void CheckDecisionResolution()
        {
            Log("## 3. decision resolution (NPCDecisionTable)");

            Check(NPCDecisionTable.Normalize("  The   Arming Sword. ") == "arming sword",
                  "normalize: case, runs of whitespace, trailing punctuation and a leading article all go",
                  NPCDecisionTable.Normalize("  The   Arming Sword. "));
            Check(NPCDecisionTable.Normalize("Velmire's sword") == "velmire's sword",
                  "normalize: an apostrophe INSIDE a word survives",
                  NPCDecisionTable.Normalize("Velmire's sword"));
            Check(NPCDecisionTable.Normalize("theatre ticket") == "theatre ticket",
                  "normalize: \"theatre\" does not lose a leading \"the\"",
                  NPCDecisionTable.Normalize("theatre ticket"));

            var table = new List<NPCDecision>
            {
                Row("sell_sword", "sword", "arming sword", "blade"),
                Row("sell_shield", "shield", "kite shield"),
            };

            Check(Resolved(table, "sell_sword") == "sell_sword", "tier 1: the id itself matches exactly");
            Check(Resolved(table, "Arming Sword") == "sell_sword", "tier 2: an exact alias, case-insensitively");
            Check(Resolved(table, "this old blade of mine") == "sell_sword",
                  "tier 3: an alias CONTAINED in the subject", Resolved(table, "this old blade of mine"));
            Check(Resolved(table, "kite shield") == "sell_shield",
                  "a shield subject does NOT fall into the sword binding");
            Check(Resolved(table, "a wheel of cheese") == null,
                  "an unknown subject resolves to NOTHING (not to the first row)",
                  Resolved(table, "a wheel of cheese") ?? "<null>");
            Check(Resolved(table, "") == null && Resolved(table, null) == null,
                  "an empty subject matches nothing");

            // subject CONTAINED IN an alias — the other half of tier 3
            var verbose = new List<NPCDecision> { Row("sell_sword", "velmire's own arming sword") };
            Check(Resolved(verbose, "arming sword") == "sell_sword",
                  "tier 3 the other way round: the SUBJECT contained in an alias");

            // an all-blank row must not swallow everything
            var blank = new List<NPCDecision> { Row("", "", "   "), Row("sell_sword", "sword") };
            Check(Resolved(blank, "sword") == "sell_sword",
                  "a row with a blank id and blank aliases matches nothing",
                  Resolved(blank, "sword") ?? "<null>");

            // determinism: two rows that both match → the FIRST DECLARED wins, and the clash is reported
            var clash = new List<NPCDecision>
            {
                Row("first_sword", "sword"),
                Row("second_sword", "sword"),
            };
            NPCDecisionTable.Resolve(clash, "sword", out string ambiguity);
            Check(Resolved(clash, "sword") == "first_sword",
                  "ambiguous subject: the first DECLARED row wins", Resolved(clash, "sword") ?? "<null>");
            Check(ambiguity != null && ambiguity.Contains("second_sword"),
                  "ambiguous subject: the clash is reported, naming the row that lost",
                  ambiguity ?? "<no warning>");
            // ...and reversing the declaration order reverses the winner, i.e. it is ORDER, not luck
            clash.Reverse();
            Check(Resolved(clash, "sword") == "second_sword",
                  "reversing the table reverses the winner — the tie-break is declaration order");

            // a STRONGER tier always beats a weaker one, whatever the declaration order
            var tiers = new List<NPCDecision>
            {
                Row("substring_row", "old sword of the herald"),   // would match by containment
                Row("sword", "nothing"),                           // matches by ID, tier 1
            };
            Check(Resolved(tiers, "sword") == "sword",
                  "an id match outranks a containment match declared before it",
                  Resolved(tiers, "sword") ?? "<null>");
        }

        static string Resolved(List<NPCDecision> table, string subject)
            => NPCDecisionTable.Resolve(table, subject, out _)?.id;

        // ------------------------------------------------------------------ 4. bindings on a real NPC
        // THE case this refactor exists for: two offers on ONE NPC routing to two different ids, with
        // the per-binding gate deciding the Accept button and the per-binding event carrying the
        // payload. Driven through the very seams NPCChatBase's dispatch coroutines call
        // (PrepareGiveItemDecision / SettleGiveItemDecision / SettleQuestionDecision), so the probe
        // cannot pass on code the dialogue does not run.

        static void CheckDecisionBindings()
        {
            Log("## 4. two bindings on one NPC (routing, gating, payload)");
            var npcGO = new GameObject("GiveItemProbeBindingNpc");
            var winGO = new GameObject("GiveItemProbeBindingWindow", typeof(RectTransform));
            try
            {
                var npc = npcGO.AddComponent<NPCInteractor3D>();
                var swordGate = npcGO.AddComponent<NpcDecisionProbeGate>();
                var shieldGate = npcGO.AddComponent<NpcDecisionProbeGate>();

                var sword = Row("sell_sword", "sword", "arming sword", "blade");
                var shield = Row("sell_shield", "shield", "kite shield");
                sword.gate = swordGate;
                shield.gate = shieldGate;
                npc.Decisions.Clear();
                npc.Decisions.Add(sword);
                npc.Decisions.Add(shield);

                int swordFired = 0, shieldFired = 0;
                NPCDecisionResult swordPayload = default;
                sword.onResolved.AddListener(r => { swordFired++; swordPayload = r; });
                shield.onResolved.AddListener(_ => shieldFired++);

                // the NPC-wide hooks stay wired at the same time — they are ADDITIVE, not replaced
                int globalAccepted = 0;
                GiveItemOffer globalOffer = default;
                npc.GiveItemAccepted += o => { globalAccepted++; globalOffer = o; };
                int globalGateAsked = 0;
                npc.GiveItemAcceptGate = _ => { globalGateAsked++; return true; };

                // --- (c)/(d) routing: two offers on ONE NPC reach two DIFFERENT ids
                var swordOffer = new GiveItemOffer { item = "Velmire's sword", price = 80 };
                npc.PrepareGiveItemDecision(swordOffer, out NPCDecisionResult swordPending,
                                            out NPCDecision swordHit);
                var shieldOffer = new GiveItemOffer { item = "kite shield", price = 40 };
                npc.PrepareGiveItemDecision(shieldOffer, out _, out NPCDecision shieldHit);
                Check(swordHit != null && swordHit.id == "sell_sword"
                      && shieldHit != null && shieldHit.id == "sell_shield",
                      "TWO bindings on one NPC: sword and shield route to DIFFERENT ids",
                      $"{swordHit?.id ?? "<null>"} / {shieldHit?.id ?? "<null>"}");
                Check(swordPending.decisionId == "sell_sword" && swordPending.price == 80
                      && swordPending.quantity == null && !swordPending.accepted
                      && swordPending.kind == NPCDecisionKind.GiveItem
                      && swordPending.subject == "Velmire's sword",
                      "the pending decision handed to the gate carries id, subject, price and accepted=false",
                      swordPending.ToString());

                // substring routing, and only the binding's OWN gate is consulted
                swordGate.asked = shieldGate.asked = 0;
                npc.PrepareGiveItemDecision(new GiveItemOffer { item = "this old blade of mine" },
                                            out _, out NPCDecision blade);
                Check(blade != null && blade.id == "sell_sword",
                      "\"this old blade of mine\" routes to sell_sword by containment",
                      blade?.id ?? "<null>");
                Check(swordGate.asked == 1 && shieldGate.asked == 0,
                      "only the RESOLVED binding's gate is asked",
                      $"sword={swordGate.asked} shield={shieldGate.asked}");
                Check(globalGateAsked == 0,
                      "a binding gate SHADOWS the NPC-wide gate (it is not asked as well)",
                      globalGateAsked.ToString());

                // --- (c) unmatched: falls through to the global gate + a warning, and is not fatal
                globalGateAsked = 0;
                var logs = new List<string>();
                Application.logMessageReceived += Collect(logs);
                bool unmatchedCanAccept;
                NPCDecision none;
                try
                {
                    unmatchedCanAccept = npc.PrepareGiveItemDecision(
                        new GiveItemOffer { item = "a wheel of cheese" }, out _, out none);
                }
                finally { Application.logMessageReceived -= Collect(logs); }
                Check(none == null, "an unknown item resolves to NO binding", none?.id ?? "<null>");
                Check(unmatchedCanAccept && globalGateAsked == 1,
                      "unmatched falls through to the NPC-wide gate (today's behaviour, kept)",
                      $"canAccept={unmatchedCanAccept} globalGateAsked={globalGateAsked}");
                Check(logs.Exists(l => l.Contains("wheel of cheese") && l.Contains("sell_sword")
                                       && l.Contains("sell_shield")),
                      "unmatched logs ONE warning naming the subject and listing the declared ids",
                      logs.Count == 0 ? "<no warning logged>" : string.Join(" | ", logs));

                // the global ACCEPTED hook still fires for an unbound decision
                globalAccepted = 0;
                var cheese = new GiveItemOffer { item = "a wheel of cheese" };
                npc.SettleGiveItemDecision(cheese, accepted: true, binding: null);
                Check(globalAccepted == 1 && globalOffer.item == "a wheel of cheese",
                      "an UNBOUND decision still fires the global GiveItemAccepted hook (additive, not replaced)",
                      globalAccepted.ToString());

                // --- (f) the binding's event: once on accept, never on decline
                swordFired = shieldFired = globalAccepted = 0;
                npc.SettleGiveItemDecision(swordOffer, accepted: true, binding: sword);
                Check(swordFired == 1 && shieldFired == 0,
                      "accepted → THAT binding's onResolved fires exactly once, the other one never",
                      $"sword={swordFired} shield={shieldFired}");
                Check(swordPayload.decisionId == "sell_sword"
                      && swordPayload.kind == NPCDecisionKind.GiveItem
                      && swordPayload.subject == "Velmire's sword"
                      && swordPayload.picked == null
                      && swordPayload.accepted
                      && swordPayload.price == 80,
                      "the onResolved payload carries id, kind, subject, accepted and the model's price",
                      swordPayload.ToString());
                Check(globalAccepted == 1,
                      "...and the global hook fired alongside it, exactly once", globalAccepted.ToString());

                swordFired = globalAccepted = 0;
                npc.SettleGiveItemDecision(swordOffer, accepted: false, binding: sword);
                Check(swordFired == 0 && globalAccepted == 0,
                      "declined → the binding's onResolved is NOT invoked (nor the global hook)",
                      $"binding={swordFired} global={globalAccepted}");

                // --- (e) a binding gate that says no
                swordGate.answer = false;
                bool canAccept = npc.PrepareGiveItemDecision(swordOffer, out _, out _);
                Check(!canAccept, "a binding gate returning false gates Accept off");
                var win = winGO.AddComponent<SoulsChatWindow>();
                win.ShowGiveItem("Velmire", swordOffer, canAccept, _ => { });
                List<Button> btns = Buttons(win);
                Check(btns.Count == 2 && !btns[0].interactable && btns[1].interactable,
                      "binding gate false: Accept is not interactable, Decline is",
                      btns.Count == 2 ? $"accept={btns[0].interactable} decline={btns[1].interactable}"
                                      : $"{btns.Count} buttons");
                win.HideGiveItem();
                Check(Decide(win, swordOffer, canAccept, 0) == null,
                      "binding gate false: clicking Accept produces NO decision");
                Check(Decide(win, swordOffer, canAccept, 1) == "{\"accepted\": false}",
                      "binding gate false: Decline still answers");
                swordGate.answer = true;

                // --- (h) a QUESTION decision resolves on the question text and carries the pick
                var pathQ = Row("mist_gate", "walk beneath the golden mist", "golden mist");
                npc.Decisions.Add(pathQ);
                int qFired = 0;
                NPCDecisionResult qPayload = default;
                pathQ.onResolved.AddListener(r => { qFired++; qPayload = r; });
                string globalQuestion = null, globalPick = null;
                npc.ToolQuestionAnswered += (q, p) => { globalQuestion = q; globalPick = p; };

                const string question = "Will you walk beneath the golden mist, or turn back?";
                NPCDecision qHit = npc.SettleQuestionDecision(question, "I will walk through");
                Check(qHit != null && qHit.id == "mist_gate",
                      "a QUESTION resolves on the QUESTION text", qHit?.id ?? "<null>");
                Check(qFired == 1 && qPayload.kind == NPCDecisionKind.Question
                      && qPayload.picked == "I will walk through"
                      && qPayload.subject == question
                      && qPayload.accepted && qPayload.price == null,
                      "the question payload carries the picked option (the game branches on it)",
                      qPayload.ToString());
                Check(globalQuestion == question && globalPick == "I will walk through",
                      "...and ToolQuestionAnswered still fired with the same (question, pick)",
                      $"{globalQuestion} / {globalPick}");
                Check(shieldFired == 0, "no unrelated binding was invoked anywhere in this section",
                      shieldFired.ToString());
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(winGO);
                UnityEngine.Object.DestroyImmediate(npcGO);
            }
        }

        // One stable delegate instance per list, so -= actually unsubscribes the handler that += added.
        static readonly Dictionary<List<string>, Application.LogCallback> collectors =
            new Dictionary<List<string>, Application.LogCallback>();

        static Application.LogCallback Collect(List<string> into)
        {
            if (!collectors.TryGetValue(into, out var cb))
            {
                cb = (msg, stack, type) => { if (type == LogType.Warning) into.Add(msg); };
                collectors[into] = cb;
            }
            return cb;
        }

        // ------------------------------------------------------------------ 5. the demo's wiring

        static void CheckDemoWiring()
        {
            Log("## 5. Velmire end-to-end: the sell_sword binding on a real NPC");
            var playerGO = new GameObject("GiveItemProbePlayer");
            var npcGO = new GameObject("GiveItemProbeNpc");
            try
            {
                var souls = playerGO.AddComponent<PlayerSouls>();
                var gear = playerGO.AddComponent<PlayerGear>();
                var npc = npcGO.AddComponent<NPCInteractor3D>();
                var offerComp = npcGO.AddComponent<NPCGearOffer>();
                SetPrivate(offerComp, "playerSouls", souls);
                SetPrivate(offerComp, "playerGear", gear);
                // PlayerGear only flips HasSword/HasShield for objects it actually holds, so the
                // hand-over needs something to activate — stand-ins under the probe's player.
                var swordT = new GameObject("ProbeSword").transform;
                var shieldT = new GameObject("ProbeShield").transform;
                swordT.SetParent(playerGO.transform, false);
                shieldT.SetParent(playerGO.transform, false);
                SetPrivate(gear, "sword", swordT);
                SetPrivate(gear, "shield", shieldT);

                // Velmire's table, wired exactly as ChatDemo3DBuilder.BindVelmireSwordSale does: the
                // gear component is BOTH the binding's gate and its onResolved target. (The builder
                // registers the listener persistently so it shows in the inspector; a persistent
                // listener cannot be added to an in-memory object outside a serialized scene, so the
                // probe adds the same call as a runtime listener — same method, same payload.)
                var sale = new NPCDecision
                {
                    id = NPCGearOffer.SellSwordDecisionId,
                    aliases = new[] { "sword", "arming sword", "blade" },
                    gate = offerComp,
                    onResolved = new NPCDecisionEvent(),
                };
                sale.onResolved.AddListener(offerComp.OnDecisionResolved);
                npc.Decisions.Clear();
                npc.Decisions.Add(sale);

                var eighty = new GiveItemOffer { item = "sword", price = 80 };

                // the item Velmire's persona is told to write must hit his binding EXACTLY
                npc.PrepareGiveItemDecision(eighty, out _, out NPCDecision hit);
                Check(hit != null && hit.id == NPCGearOffer.SellSwordDecisionId,
                      "the bare item \"sword\" his prompt asks for resolves to sell_sword",
                      hit?.id ?? "<null>");
                npc.PrepareGiveItemDecision(new GiveItemOffer { item = "Velmire's arming sword", price = 80 },
                                            out _, out NPCDecision embroidered);
                Check(embroidered != null && embroidered.id == NPCGearOffer.SellSwordDecisionId,
                      "...and so does \"Velmire's arming sword\" when the model embroiders",
                      embroidered?.id ?? "<null>");

                // (c) 50 souls against an 80-soul price: the BINDING's gate says no...
                SetPrivate(souls, "souls", 50);
                bool gated = npc.PrepareGiveItemDecision(eighty, out _, out _);
                Check(!gated, "gate: 50 souls cannot afford 80 (through the binding's own gate slot)");

                // ...and the panel then draws Accept dead while Decline still answers
                var winGO = new GameObject("GiveItemProbeWindow2", typeof(RectTransform));
                try
                {
                    var win = winGO.AddComponent<SoulsChatWindow>();
                    win.ShowGiveItem("Velmire", eighty, canAccept: false, _ => { });
                    List<Button> btns = Buttons(win);
                    Check(btns.Count == 2 && !btns[0].interactable && btns[1].interactable,
                          "gate false: Accept is not interactable, Decline is",
                          btns.Count == 2 ? $"accept={btns[0].interactable} decline={btns[1].interactable}"
                                          : $"{btns.Count} buttons");
                    win.HideGiveItem();

                    string blocked = Decide(win, eighty, canAccept: false, clickIndex: 0);
                    Check(blocked == null, "gate false: clicking Accept produces NO decision",
                          blocked ?? "<null>");
                    string stillDeclines = Decide(win, eighty, canAccept: false, clickIndex: 1);
                    Check(stillDeclines == "{\"accepted\": false}",
                          "gate false: Decline still returns {\"accepted\": false}",
                          stillDeclines ?? "<no decision>");
                }
                finally { UnityEngine.Object.DestroyImmediate(winGO); }

                // 100 souls, accepted at 80 → 20 left, through the binding's own event
                SetPrivate(souls, "souls", 100);
                Check(npc.PrepareGiveItemDecision(eighty, out _, out _), "gate: 100 souls can afford 80");
                npc.SettleGiveItemDecision(eighty, accepted: true, binding: sale);
                Check(souls.Souls == 20, "accepting at 80 leaves 20 souls of 100",
                      souls.Souls.ToString());
                Check(gear.HasSword && gear.HasShield, "...and the gear actually changed hands",
                      $"sword={gear.HasSword} shield={gear.HasShield}");

                // declining takes nothing, even at a price the player can pay
                SetPrivate(souls, "souls", 100);
                npc.SettleGiveItemDecision(eighty, accepted: false, binding: sale);
                Check(souls.Souls == 100, "declining spends nothing", souls.Souls.ToString());

                // a priceless gift is free and takes nothing
                SetPrivate(souls, "souls", 7);
                var gift = new GiveItemOffer { item = "sword" };
                Check(npc.PrepareGiveItemDecision(gift, out _, out _),
                      "gate: a priceless gift is always acceptable");
                npc.SettleGiveItemDecision(gift, accepted: true, binding: sale);
                Check(souls.Souls == 7, "accepting a priceless gift spends nothing", souls.Souls.ToString());
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(npcGO);
                UnityEngine.Object.DestroyImmediate(playerGO);
            }
        }

        static void SetPrivate(object target, string field, object value)
        {
            FieldInfo f = target.GetType().GetField(field,
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (f == null) throw new Exception($"{target.GetType().Name} has no field '{field}'");
            f.SetValue(target, value);
        }
    }
}
#endif
