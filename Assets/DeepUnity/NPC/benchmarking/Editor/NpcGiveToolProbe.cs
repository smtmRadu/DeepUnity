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
    /// <summary>
    /// Guard for the NPC's SECOND interactive tool, <b>GiveTool</b>. Four things, none of which any
    /// other probe reaches:
    /// <list type="number">
    /// <item><b>schema drift</b> — <see cref="NPCChatBase.GiveToolSchema"/> must equal
    /// dataset_creation/wire_format.py's GIVE_TOOL_SCHEMA byte for byte (and AskUserQuestionSchema its
    /// ASK_SCHEMA). Same reasoning as Qwen3_5ChatTemplateProbe: the prompt the finetune trains on and
    /// the prompt the engine serves are the same bytes or the finetune is worth nothing, and a drifted
    /// schema produces perfectly plausible output, so nothing else notices.</item>
    /// <item><b>the wire round trip</b> — a synthetic call in the XML shape Qwen3.5's own template
    /// declares, read by the SAME parser the dialogue uses
    /// (<see cref="NPCChatBase.TryReadGiveToolCall"/>), shown on a REAL window
    /// (<see cref="SoulsChatWindow"/>, which is what Velmire talks through), clicked, and mapped back
    /// through <see cref="NPCChatBase.ToolGiveResult"/> — which must be exactly
    /// <c>{"accepted": true}</c> / <c>{"accepted": false}</c> and nothing else.</item>
    /// <item><b>the accept-gate</b> — 50 souls against an 80-soul price renders Accept DEAD (no
    /// listener, not interactable) while Decline still answers, so the exchange cannot dead-end.</item>
    /// <item><b>the transaction</b> — accepting at 80 takes the player from 100 souls to 20, through the
    /// demo's own gate and hand-over methods (<see cref="NPCGearOffer"/>), not a re-implementation.</item>
    /// </list>
    /// Edit mode, no play mode, no GPU, no model — the panel is built at runtime and tears itself down
    /// with DestroyImmediate off the play loop, which is exactly what makes this checkable headless.
    /// What is NOT covered: the coroutine around the panel (GiveToolRoutine's epoch/state guards) and
    /// the model turn that consumes the result, both of which need a live dialogue and an LLM. The
    /// panel's LOOK is checked separately by ChatDemo3DBuilder.UiProbeBatch, which screenshots it.
    ///
    ///   menu:  DeepUnity/NPC/GiveTool Guard
    ///   batch: Unity.exe -batchmode -projectPath &lt;repo&gt; ^
    ///            -logFile ProbeLogs/givetool.log ^
    ///            -executeMethod DeepUnity.NpcGiveToolProbe.Run
    /// No -quit (the method exits itself: 0 on PASS, 1 on FAIL). Never -nographics: it builds uGUI.
    /// </summary>
    public static class NpcGiveToolProbe
    {
        const string REPORT = "ProbeLogs/npc_givetool.md";

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
            Debug.Log("[GiveTool] " + s);
        }

        static void Fail(string s)
        {
            failures++;
            report.AppendLine("**FAIL** " + s);
            Debug.LogError("[GiveTool] FAIL: " + s);
        }

        static void Check(bool ok, string what, string detail = null)
        {
            if (ok) Log($"PASS  {what}");
            else Fail(what + (detail == null ? "" : " — " + detail));
        }

        [MenuItem("DeepUnity/NPC/GiveTool Guard")]
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
                Log($"# GiveTool guard — {DateTime.Now:yyyy-MM-dd HH:mm}");
                Log("");
                CheckSchemas();
                Log("");
                CheckWireRoundTrip();
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

            CompareToPin(py, "GIVE_TOOL_SCHEMA", NPCChatBase.GiveToolSchema, nameof(NPCChatBase.GiveToolSchema));
            CompareToPin(py, "ASK_SCHEMA", NPCChatBase.AskUserQuestionSchema,
                         nameof(NPCChatBase.AskUserQuestionSchema));
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
            "\n<function=GiveTool>\n" +
            "<parameter=item>\nVelmire's sword\n</parameter>\n" +
            "<parameter=price>\n80\n</parameter>\n" +
            "</function>\n";

        static void CheckWireRoundTrip()
        {
            Log("## 2. synthetic GiveTool call → panel → tool result");

            Check(NPCChatBase.ToolGiveResult(true) == "{\"accepted\": true}",
                  "ToolGiveResult(true) is exactly {\"accepted\": true}", NPCChatBase.ToolGiveResult(true));
            Check(NPCChatBase.ToolGiveResult(false) == "{\"accepted\": false}",
                  "ToolGiveResult(false) is exactly {\"accepted\": false}", NPCChatBase.ToolGiveResult(false));

            if (!NPCChatBase.TryReadGiveToolCall(SyntheticCall, out ToolGiveOffer offer))
            {
                Fail("the synthetic XML call did not parse as a GiveTool offer");
                return;
            }
            Check(offer.item == "Velmire's sword", "parsed item", offer.item ?? "<null>");
            Check(offer.price == 80, "parsed price", offer.price?.ToString() ?? "<null>");
            Check(!offer.quantity.HasValue, "no quantity → stays null (not 0)",
                  offer.quantity?.ToString() ?? "<null>");

            // an item-only call is a GIFT: no price, no quantity, still a valid offer
            Check(NPCChatBase.TryReadGiveToolCall("<function=GiveTool><parameter=item>estus flask</parameter>" +
                                                 "</function>", out ToolGiveOffer gift)
                  && gift.item == "estus flask" && !gift.price.HasValue,
                  "an item-only call parses as a priceless gift");
            // ...and a call with no item at all is not an offer (the schema's one required parameter)
            Check(!NPCChatBase.TryReadGiveToolCall("<function=GiveTool><parameter=price>80</parameter>" +
                                                  "</function>", out _),
                  "a call with no item is rejected");
            // quantity rides through, and a scruffy price still lands
            Check(NPCChatBase.TryReadGiveToolCall("<function=GiveTool><parameter=item>arrow</parameter>" +
                                                 "<parameter=quantity>12</parameter>" +
                                                 "<parameter=price>30 souls</parameter></function>",
                                                 out ToolGiveOffer stack)
                  && stack.quantity == 12 && stack.price == 30,
                  "quantity parses and a price written \"30 souls\" still reads as 30");
            // the Hermes/JSON shape the SFT corpus uses must parse identically
            Check(NPCChatBase.TryReadGiveToolCall(
                      "{\"name\": \"GiveTool\", \"arguments\": {\"item\": \"sword\", \"price\": 80}}",
                      out ToolGiveOffer json) && json.item == "sword" && json.price == 80,
                  "the JSON wire shape parses to the same offer");

            // --- the panel, on the REAL window class Velmire talks through
            var go = new GameObject("GiveToolProbeWindow", typeof(RectTransform));
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
                win.ShowToolGive("Velmire", offer, true, _ => { });
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
                win.HideToolGive();

                NPCChatBase.TryReadGiveToolCall("<function=GiveTool><parameter=item>arrow</parameter>" +
                                                "<parameter=quantity>12</parameter>" +
                                                "<parameter=price>30</parameter></function>",
                                                out ToolGiveOffer arrows);
                win.ShowToolGive("Velmire", arrows, true, _ => { });
                string stackLine = PanelPrompt(win);
                Check(stackLine == "arrow x12  -  30 souls", "quantity renders as \"x12\"",
                      stackLine ?? "<none>");
                win.HideToolGive();
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        /// <summary>Open the offer panel, click one button, return the tool result the dialogue would
        /// send — null when the click produced no decision (a gated-off button).</summary>
        static string Decide(SoulsChatWindow win, ToolGiveOffer offer, bool canAccept, int clickIndex)
        {
            string result = null;
            // this lambda is the routine's own tail: the window reports a bool, ToolGiveResult turns it
            // into the bytes, and Talk() sends those bytes as the <tool_response>
            win.ShowToolGive("Velmire", offer, canAccept, ok => result = NPCChatBase.ToolGiveResult(ok));
            List<Button> btns = Buttons(win);
            if (btns.Count <= clickIndex)
            {
                Fail($"the offer panel built {btns.Count} button(s) — cannot click index {clickIndex}");
                win.HideToolGive();
                return null;
            }
            btns[clickIndex].onClick.Invoke();   // the click, exactly as uGUI would raise it
            win.HideToolGive();                  // no-op after a real pick; idempotent
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

        // ------------------------------------------------------------------ 3. the demo's wiring

        static void CheckDemoWiring()
        {
            Log("## 3. ChatDemo3D accept-gate and transaction (NPCGearOffer + PlayerSouls)");
            var playerGO = new GameObject("GiveToolProbePlayer");
            var npcGO = new GameObject("GiveToolProbeNpc");
            try
            {
                var souls = playerGO.AddComponent<PlayerSouls>();
                var gear = playerGO.AddComponent<PlayerGear>();
                var offerComp = npcGO.AddComponent<NPCGearOffer>();
                SetPrivate(offerComp, "playerSouls", souls);
                SetPrivate(offerComp, "playerGear", gear);

                var eighty = new ToolGiveOffer { item = "Velmire's sword", price = 80 };

                // (c) 50 souls against an 80-soul price: the gate says no...
                SetPrivate(souls, "souls", 50);
                bool gated = offerComp.CanAcceptOffer(eighty);
                Check(!gated, "gate: 50 souls cannot afford 80");

                // ...and the panel then draws Accept dead while Decline still answers
                var winGO = new GameObject("GiveToolProbeWindow2", typeof(RectTransform));
                try
                {
                    var win = winGO.AddComponent<SoulsChatWindow>();
                    win.ShowToolGive("Velmire", eighty, canAccept: false, _ => { });
                    List<Button> btns = Buttons(win);
                    Check(btns.Count == 2 && !btns[0].interactable && btns[1].interactable,
                          "gate false: Accept is not interactable, Decline is",
                          btns.Count == 2 ? $"accept={btns[0].interactable} decline={btns[1].interactable}"
                                          : $"{btns.Count} buttons");
                    win.HideToolGive();

                    string blocked = Decide(win, eighty, canAccept: false, clickIndex: 0);
                    Check(blocked == null, "gate false: clicking Accept produces NO decision",
                          blocked ?? "<null>");
                    string stillDeclines = Decide(win, eighty, canAccept: false, clickIndex: 1);
                    Check(stillDeclines == "{\"accepted\": false}",
                          "gate false: Decline still returns {\"accepted\": false}",
                          stillDeclines ?? "<no decision>");
                }
                finally { UnityEngine.Object.DestroyImmediate(winGO); }

                // (d) 100 souls, accepted at 80 → 20 left, through the demo's own hand-over
                SetPrivate(souls, "souls", 100);
                Check(offerComp.CanAcceptOffer(eighty), "gate: 100 souls can afford 80");
                offerComp.OnOfferAccepted(eighty);
                Check(souls.Souls == 20, "accepting at 80 leaves 20 souls of 100",
                      souls.Souls.ToString());

                // a priceless gift is free and takes nothing
                SetPrivate(souls, "souls", 7);
                var gift = new ToolGiveOffer { item = "estus flask" };
                Check(offerComp.CanAcceptOffer(gift), "gate: a priceless gift is always acceptable");
                offerComp.OnOfferAccepted(gift);
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
