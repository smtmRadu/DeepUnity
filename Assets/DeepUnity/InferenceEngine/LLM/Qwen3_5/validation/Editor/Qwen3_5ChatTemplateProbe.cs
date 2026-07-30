#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // Drift guard for Qwen3_5ChatTemplate. Loads the VENDORED chat_template.jinja, extracts the
        // string literals the template emits, and asserts every const in the class still equals the
        // one it was transcribed from — then asserts that the fragments the engine feeds
        // AppendTextTokens reassemble, byte for byte, into the turns the template writes.
        //
        // WHY this exists, and why it is loud: the prompt the finetune is trained on and the prompt
        // the engine serves are the same bytes or the finetune is worth nothing. On 2026-07-26 three
        // separate prompt divergences were caught BY EYE — and every one of them had already
        // contaminated hundreds of training samples before anybody noticed, because a wrong prompt
        // produces perfectly plausible output. Nothing in a Unity build fails when a string drifts;
        // this does.
        //
        // Text extraction only. The tool block is ONE Jinja string literal, and every other emitted
        // shape is a literal too, so pulling the quoted spans out of the {{- ... }} expressions is
        // enough — there is deliberately no Jinja interpreter in here.
        //
        //   menu:  DeepUnity/Qwen3.5/Chat-Template Drift Guard
        //   batch: Unity.exe -batchmode -nographics -projectPath <repo> ^
        //            -logFile ProbeLogs/qwen35_chat_template.log ^
        //            -executeMethod DeepUnity.Qwen3_5Modeling.Qwen3_5ChatTemplateProbe.Run
        // No -quit (the method exits itself: 0 on PASS, 1 on FAIL), and -nographics IS safe here,
        // unlike every other probe in this repo — this one is pure text, no play mode, no GPU.
        public static class Qwen3_5ChatTemplateProbe
        {
            const string REPORT = "ProbeLogs/qwen35_chat_template.md";
            const string DONE = "ProbeLogs/qwen35_chat_template.done";

            static readonly StringBuilder report = new StringBuilder();
            static int failures;

            static void Log(string s)
            {
                report.AppendLine(s);
                Debug.Log("[QwenTemplate] " + s);
            }

            static void Fail(string s)
            {
                failures++;
                report.AppendLine(s);
                Debug.LogError("[QwenTemplate] " + s);
            }

            [MenuItem("DeepUnity/Qwen3.5/Chat-Template Drift Guard")]
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
                    Log($"# Qwen3.5 chat-template drift guard — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log("");
                    Check();
                }
                catch (Exception e)
                {
                    Fail($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }

                Log("");
                Log(failures == 0 ? "## RESULT: PASS" : $"## RESULT: FAIL ({failures} divergence(s))");
                File.WriteAllText(REPORT, report.ToString());
                File.WriteAllText(DONE, failures == 0 ? "PASS" : "FAIL");
                if (failures != 0)
                    Debug.LogError($"[QwenTemplate] {failures} divergence(s) between Qwen3_5ChatTemplate and " +
                                   $"{Qwen3_5ChatTemplate.VendoredTemplatePath}. See {REPORT}. " +
                                   "Do NOT ship or train on this prompt until they agree.");
                if (exitWhenDone && Application.isBatchMode)
                    EditorApplication.Exit(failures == 0 ? 0 : 1);
            }

            // ------------------------------------------------------------------ the checks

            static void Check()
            {
                string path = Qwen3_5ChatTemplate.VendoredTemplatePath;
                if (!File.Exists(path))
                {
                    Fail($"vendored template MISSING at `{path}` — the const table has nothing to be checked against.");
                    return;
                }

                // Read with line endings normalized to LF. core.autocrlf=true and no .gitattributes
                // means a fresh clone on Windows gets this file as CRLF; the emitted literals write
                // their newlines as the two characters \ and n INSIDE the quotes, so normalizing
                // changes no literal — it only keeps the hash and the line count reproducible.
                string jinja = File.ReadAllText(path).Replace("\r\n", "\n").Replace("\r", "\n");
                Log($"template : `{path}`");
                Log($"snapshot : {Qwen3_5ChatTemplate.TemplateSnapshot}");
                Log($"lines    : {jinja.Split('\n').Length}");

                string sha = Sha256Lf(jinja);
                if (sha == Qwen3_5ChatTemplate.TemplateSha256Lf)
                    Log($"sha256   : {sha} (LF-normalized) — matches TemplateSha256Lf");
                else
                    Fail($"sha256 (LF-normalized) is `{sha}` but TemplateSha256Lf says " +
                         $"`{Qwen3_5ChatTemplate.TemplateSha256Lf}`. The vendored template was replaced. " +
                         "If that was intentional, update TemplateSnapshot AND TemplateSha256Lf in the " +
                         "same commit — the project must record which template it was built against.");

                List<string> lits = EmittedLiterals(jinja);
                Log($"literals : {lits.Count} emitted string literals extracted");
                Log("");

                // A guard that passes because it found nothing is worse than no guard.
                if (lits.Count < 30)
                {
                    Fail($"only {lits.Count} literals extracted — the extractor did not understand this file. " +
                         "Refusing to report PASS on an empty comparison.");
                    return;
                }

                // ---- 1. every verbatim const must BE one of the template's emitted literals -----
                Log("## verbatim consts vs emitted literals");
                var set = new HashSet<string>(lits);
                Verbatim(set, "SystemTurnOpen", Qwen3_5ChatTemplate.SystemTurnOpen);
                Verbatim(set, "TurnEnd", Qwen3_5ChatTemplate.TurnEnd);
                Verbatim(set, "SystemContentSeparator", Qwen3_5ChatTemplate.SystemContentSeparator);
                Verbatim(set, "ImStart", Qwen3_5ChatTemplate.ImStart);
                Verbatim(set, "ImEnd", Qwen3_5ChatTemplate.ImEnd);
                Verbatim(set, "ToolsHeader", Qwen3_5ChatTemplate.ToolsHeader);
                Verbatim(set, "ToolSchemaSeparator", Qwen3_5ChatTemplate.ToolSchemaSeparator);
                Verbatim(set, "ToolsClose", Qwen3_5ChatTemplate.ToolsClose);
                Verbatim(set, "ToolsSpec", Qwen3_5ChatTemplate.ToolsSpec);
                Verbatim(set, "AssistantTurnOpen", Qwen3_5ChatTemplate.AssistantTurnOpen);
                Verbatim(set, "ThinkBlockOpen", Qwen3_5ChatTemplate.ThinkBlockOpen);
                Verbatim(set, "ThinkBlockClose", Qwen3_5ChatTemplate.ThinkBlockClose);
                Verbatim(set, "ThinkPrefill", Qwen3_5ChatTemplate.ThinkPrefill);
                Verbatim(set, "EmptyThinkBlock", Qwen3_5ChatTemplate.EmptyThinkBlock);
                Verbatim(set, "ToolCallOpenAfterContent", Qwen3_5ChatTemplate.ToolCallOpenAfterContent);
                Verbatim(set, "ToolCallOpenFirst", Qwen3_5ChatTemplate.ToolCallOpenFirst);
                Verbatim(set, "ToolCallOpenSubsequent", Qwen3_5ChatTemplate.ToolCallOpenSubsequent);
                Verbatim(set, "TagNameClose", Qwen3_5ChatTemplate.TagNameClose);
                Verbatim(set, "ToolCallParamOpen", Qwen3_5ChatTemplate.ToolCallParamOpen);
                Verbatim(set, "ToolCallParamClose", Qwen3_5ChatTemplate.ToolCallParamClose);
                Verbatim(set, "ToolCallClose", Qwen3_5ChatTemplate.ToolCallClose);
                Verbatim(set, "ToolResponseTurnOpen", Qwen3_5ChatTemplate.ToolResponseTurnOpen);
                Verbatim(set, "ToolResponseOpen", Qwen3_5ChatTemplate.ToolResponseOpen);
                Verbatim(set, "ToolResponseClose", Qwen3_5ChatTemplate.ToolResponseClose);
                // Never emitted by this text-only port; checked anyway so the whole file is covered.
                Verbatim(set, "ImagePlaceholder", Qwen3_5ChatTemplate.ImagePlaceholder);
                Verbatim(set, "VideoPlaceholder", Qwen3_5ChatTemplate.VideoPlaceholder);
                Verbatim(set, "PictureIdPrefix", Qwen3_5ChatTemplate.PictureIdPrefix);
                Verbatim(set, "VideoIdPrefix", Qwen3_5ChatTemplate.VideoIdPrefix);
                Verbatim(set, "VisionIdSuffix", Qwen3_5ChatTemplate.VisionIdSuffix);

                // ---- 2. the tool block is ONE literal, and it is the one we hold ----------------
                Log("");
                Log("## the # Tools block");
                int headers = lits.FindAll(s => s.Contains("You have access to the following functions")).Count;
                Gate(headers == 1, $"the tools header appears in exactly ONE emitted literal (found {headers})");
                int specs = lits.FindAll(s => s.Contains("ONLY reply in the following format")).Count;
                Gate(specs == 1, $"the call-format spec + <IMPORTANT> reminder is ONE literal (found {specs})");
                // If the spec literal exists but differs, say HOW — this is the diff that matters most.
                string liveSpec = lits.Find(s => s.Contains("ONLY reply in the following format"));
                if (liveSpec != null && liveSpec != Qwen3_5ChatTemplate.ToolsSpec)
                    Diff("ToolsSpec", Qwen3_5ChatTemplate.ToolsSpec, "const", liveSpec, ".jinja");

                // ---- 3. relationships the sliced/derived members depend on ---------------------
                Log("");
                Log("## slice relationships (the derived members are substrings, not transcriptions)");
                Gate(Qwen3_5ChatTemplate.ToolsHeader.StartsWith(Qwen3_5ChatTemplate.ToolsHeading, StringComparison.Ordinal),
                     "ToolsHeader starts with ToolsHeading");
                Gate(Qwen3_5ChatTemplate.ToolsSpec.EndsWith(Qwen3_5ChatTemplate.ReminderTerminator, StringComparison.Ordinal),
                     "ToolsSpec ends with ReminderTerminator");
                Gate(Qwen3_5ChatTemplate.ToolsSpec.StartsWith(Qwen3_5ChatTemplate.ToolsSpecOpen, StringComparison.Ordinal)
                     && Qwen3_5ChatTemplate.ToolsSpecOpen.Length
                        == Qwen3_5ChatTemplate.ToolsSpec.Length - Qwen3_5ChatTemplate.ReminderTerminator.Length,
                     "ToolsSpecOpen is ToolsSpec minus its terminator");
                Gate(Qwen3_5ChatTemplate.SystemTurnOpen.StartsWith(Qwen3_5ChatTemplate.ImStart, StringComparison.Ordinal),
                     "SystemTurnOpen starts with ImStart");
                Gate(Qwen3_5ChatTemplate.AssistantTurnOpen.StartsWith(Qwen3_5ChatTemplate.ImStart, StringComparison.Ordinal),
                     "AssistantTurnOpen starts with ImStart");
                Gate(Qwen3_5ChatTemplate.ToolResponseTurnOpen.StartsWith(Qwen3_5ChatTemplate.ImStart, StringComparison.Ordinal),
                     "ToolResponseTurnOpen starts with ImStart");
                Gate(Qwen3_5ChatTemplate.TurnEnd.StartsWith(Qwen3_5ChatTemplate.ImEnd, StringComparison.Ordinal),
                     "TurnEnd starts with ImEnd");
                Gate(Qwen3_5ChatTemplate.ToolResponseOpen.Contains(Qwen3_5ChatTemplate.ToolResponseTag),
                     "ToolResponseOpen contains ToolResponseTag");
                Gate(Qwen3_5ChatTemplate.ToolResponseClose.Contains(Qwen3_5ChatTemplate.ToolResponseEndTag),
                     "ToolResponseClose contains ToolResponseEndTag");
                Gate(Qwen3_5ChatTemplate.ThinkPrefill.Contains(Qwen3_5ChatTemplate.ThinkTag),
                     "ThinkPrefill contains ThinkTag");
                Gate(Qwen3_5ChatTemplate.EmptyThinkBlock.Contains(Qwen3_5ChatTemplate.ThinkTag)
                     && Qwen3_5ChatTemplate.EmptyThinkBlock.Contains(Qwen3_5ChatTemplate.ThinkEndTag),
                     "EmptyThinkBlock contains both think tags");
                Gate(Qwen3_5ChatTemplate.ToolCallOpenFirst.Contains(Qwen3_5ChatTemplate.ToolCallTag),
                     "ToolCallOpenFirst contains ToolCallTag");
                Gate(Qwen3_5ChatTemplate.ToolCallClose.Contains(Qwen3_5ChatTemplate.ToolCallEndTag),
                     "ToolCallClose contains ToolCallEndTag");
                Gate(Qwen3_5ChatTemplate.ToolsSpec.Contains(Qwen3_5ChatTemplate.ToolCallTag)
                     && Qwen3_5ChatTemplate.ToolsSpec.Contains(Qwen3_5ChatTemplate.ToolCallEndTag),
                     "ToolsSpec quotes both tool_call tags (the parser and the spec agree)");

                // ---- 4. tokenized render == template render, byte for byte ---------------------
                // The engine appends the tags as token IDS and text-encodes only what sits between
                // them. Rebuild each turn the way Qwen3_5.cs does — special tokens spelled out as
                // their text, exactly what the tokenizer will decode them back to — and compare
                // against the template's own assembly of the same turn.
                Log("");
                Log("## tokenized render vs template render (the seams differ; the bytes must not)");
                const string PERSONA = "## NAME\nVelmire\n\nYou are a knight.";
                const string LINE = "I need a weapon.";
                const string RESULT = "{\"selected\": \"Take the sword\"}";

                Bytes("system turn",
                      Qwen3_5ChatTemplate.ImStart + Qwen3_5ChatTemplate.SystemRoleLine + PERSONA
                          + Qwen3_5ChatTemplate.ImEnd + Qwen3_5ChatTemplate.TurnEndTail,
                      Qwen3_5ChatTemplate.RenderSystemTurn(PERSONA));

                Bytes("user turn",
                      Qwen3_5ChatTemplate.ImStart + Qwen3_5ChatTemplate.UserRoleLine + LINE
                          + Qwen3_5ChatTemplate.ImEnd + Qwen3_5ChatTemplate.TurnEndTail,
                      Qwen3_5ChatTemplate.RenderUserTurn(LINE));

                Bytes("tool-result turn",
                      Qwen3_5ChatTemplate.ImStart + Qwen3_5ChatTemplate.UserRoleLine
                          + Qwen3_5ChatTemplate.ToolResponseTag + Qwen3_5ChatTemplate.ToolResponseOpenTail
                          + RESULT
                          + Qwen3_5ChatTemplate.ToolResponseCloseHead + Qwen3_5ChatTemplate.ToolResponseEndTag
                          + Qwen3_5ChatTemplate.ImEnd + Qwen3_5ChatTemplate.TurnEndTail,
                      Qwen3_5ChatTemplate.RenderToolResponseTurn(RESULT));

                Bytes("generation prompt (thinking ON)",
                      Qwen3_5ChatTemplate.ImStart + Qwen3_5ChatTemplate.AssistantRoleLine
                          + Qwen3_5ChatTemplate.ThinkTag + Qwen3_5ChatTemplate.ThinkPrefillTail,
                      Qwen3_5ChatTemplate.RenderGenerationPrompt(true));

                Bytes("generation prompt (thinking OFF)",
                      Qwen3_5ChatTemplate.ImStart + Qwen3_5ChatTemplate.AssistantRoleLine
                          + Qwen3_5ChatTemplate.ThinkTag + Qwen3_5ChatTemplate.EmptyThinkMid
                          + Qwen3_5ChatTemplate.ThinkEndTag + Qwen3_5ChatTemplate.EmptyThinkTail,
                      Qwen3_5ChatTemplate.RenderGenerationPrompt(false));

                // ---- 4b. | trim parity (L55/L63/L71/L82) ---------------------------------------
                // The renderers and the tokenized path both Trim(). Whitespace-padded content must
                // therefore be indistinguishable from clean content — this is the gate that would
                // have caught the untrimmed encode, and it stays cheap to keep.
                Log("");
                Log("## | trim on message content");
                Gate(Qwen3_5ChatTemplate.RenderSystemTurn("  \n" + PERSONA + "\n\n  ")
                         == Qwen3_5ChatTemplate.RenderSystemTurn(PERSONA),
                     "a padded persona renders identically to a clean one");
                Gate(Qwen3_5ChatTemplate.RenderUserTurn(" " + LINE + "\n")
                         == Qwen3_5ChatTemplate.RenderUserTurn(LINE),
                     "a padded player line renders identically to a clean one");
                Gate(Qwen3_5ChatTemplate.RenderToolResponseTurn("\n" + RESULT + "\n")
                         == Qwen3_5ChatTemplate.RenderToolResponseTurn(RESULT),
                     "a padded tool result renders identically to a clean one");
                // Compare RENDERS, not substrings: SystemContentSeparator is "\n\n", which the tools
                // block already contains several times, so Contains() can only ever say yes.
                string blk = Qwen3_5ChatTemplate.RenderToolsBlock(new[] { "{\"name\": \"T\"}" });
                Gate(Qwen3_5ChatTemplate.RenderSystemTurn("   ", blk)
                         == Qwen3_5ChatTemplate.RenderSystemTurn("", blk),
                     "whitespace-only content drops the tools/persona separator (it is not content)");

                // ---- 5. RenderToolsBlock: canonical, and the host-bullet splice ----------------
                Log("");
                Log("## RenderToolsBlock");
                var one = new[] { "{\"type\": \"function\", \"function\": {\"name\": \"AskUserQuestion\"}}" };
                string canonical = Qwen3_5ChatTemplate.RenderToolsBlock(one);
                Bytes("canonical block == the template's own emission for the same tools list",
                      Qwen3_5ChatTemplate.ToolsHeader + Qwen3_5ChatTemplate.ToolSchemaSeparator + one[0]
                          + Qwen3_5ChatTemplate.ToolsClose + Qwen3_5ChatTemplate.ToolsSpec,
                      canonical);

                const string OURS = "- a host rule\n- a second host rule\n";
                string spliced = Qwen3_5ChatTemplate.RenderToolsBlock(one, OURS);
                Bytes("host bullets land immediately before </IMPORTANT>, moving none of Qwen's bytes",
                      canonical.Substring(0, canonical.Length - Qwen3_5ChatTemplate.ReminderTerminator.Length)
                          + OURS + Qwen3_5ChatTemplate.ReminderTerminator,
                      spliced);
                Gate(spliced.EndsWith(OURS + Qwen3_5ChatTemplate.ReminderTerminator, StringComparison.Ordinal),
                     "the host's bullets are the LAST thing in the reminder list");

                // Two tools must be separated by exactly the template's separator, nothing else.
                var two = new[] { "{\"a\": 1}", "{\"b\": 2}" };
                Gate(Qwen3_5ChatTemplate.RenderToolsBlock(two).Contains(
                         two[0] + Qwen3_5ChatTemplate.ToolSchemaSeparator + two[1]),
                     "consecutive schemas are joined by ToolSchemaSeparator only");
            }

            // ------------------------------------------------------------------ assertions

            static void Verbatim(HashSet<string> emitted, string name, string value)
            {
                if (emitted.Contains(value)) { Log($"OK    {name}"); return; }
                Fail($"FAIL  {name} is not among the template's emitted literals.");
                // Offer the nearest candidate so the reader sees WHAT moved, not just that it did.
                string near = Nearest(emitted, value);
                if (near != null) Diff(name, value, "const", near, ".jinja");
                else
                {
                    report.AppendLine($"        const : {Esc(value)}");
                    Debug.LogError($"[QwenTemplate] {name} const : {Esc(value)} (no similar literal in the file at all)");
                }
            }

            static void Gate(bool ok, string what)
            {
                if (ok) Log("OK    " + what);
                else Fail("FAIL  " + what);
            }

            /// <summary>Byte gate between the same string assembled two ways: <paramref name="byParts"/>
            /// from the individual fragments the engine feeds the tokenizer, and
            /// <paramref name="byHelper"/> from the class's own render helper. They must agree, or one
            /// of the two paths has a fragment sliced at the wrong place.</summary>
            static void Bytes(string what, string byParts, string byHelper)
            {
                if (byParts == byHelper) { Log($"OK    {what} ({byHelper.Length} chars)"); return; }
                Fail($"FAIL  {what}");
                Diff(what, byParts, "by-parts", byHelper, "by-helper");
            }

            /// <summary>The loud part: both lengths, the index of the first divergence, and a window
            /// around it from each side with every control character escaped so it is visible. A
            /// silent boolean would be useless here — the point is that a human reading the console
            /// can see WHICH byte moved and in which direction.</summary>
            static void Diff(string what, string a, string aLabel, string b, string bLabel)
            {
                int n = Math.Min(a.Length, b.Length);
                int at = n;
                for (int i = 0; i < n; i++)
                    if (a[i] != b[i]) { at = i; break; }

                int from = Math.Max(0, at - 48);
                var sb = new StringBuilder();
                sb.AppendLine($"      DIFF {what}: {aLabel} is {a.Length} chars, {bLabel} is {b.Length}; " +
                              $"first divergence at index {at}");
                sb.AppendLine($"        {aLabel,-9} …{Esc(Window(a, from, 96))}…");
                sb.AppendLine($"        {bLabel,-9} …{Esc(Window(b, from, 96))}…");
                report.Append(sb.ToString());
                Debug.LogError("[QwenTemplate] " + sb.ToString().TrimEnd());
            }

            static string Window(string s, int from, int len)
            {
                if (from >= s.Length) return "";
                return s.Substring(from, Math.Min(len, s.Length - from));
            }

            /// <summary>Closest emitted literal by shared prefix — good enough to point at the one
            /// that drifted without dragging an edit-distance implementation in here.</summary>
            static string Nearest(HashSet<string> emitted, string value)
            {
                string best = null;
                int bestShared = 0;
                foreach (string cand in emitted)
                {
                    int n = Math.Min(cand.Length, value.Length), shared = 0;
                    while (shared < n && cand[shared] == value[shared]) shared++;
                    if (shared > bestShared) { bestShared = shared; best = cand; }
                }
                return bestShared >= 4 ? best : null;
            }

            static string Esc(string s) => s.Replace("\\", "\\\\").Replace("\n", "\\n")
                                            .Replace("\r", "\\r").Replace("\t", "\\t");

            static string Sha256Lf(string lfText)
            {
                using (var sha = SHA256.Create())
                {
                    byte[] h = sha.ComputeHash(new UTF8Encoding(false).GetBytes(lfText));
                    var sb = new StringBuilder(h.Length * 2);
                    foreach (byte b in h) sb.Append(b.ToString("x2"));
                    return sb.ToString();
                }
            }

            // ------------------------------------------------------------------ extraction
            // Every string the template EMITS sits inside a {{- ... }} output expression. Pull the
            // quoted spans out of those and decode the Jinja escapes; that covers the whole file
            // because none of the emitted shapes are built by anything but concatenation of
            // literals and interpolated data. {%- ... %} statement blocks are skipped on purpose:
            // their literals are set/compare values (role names, raise_exception messages), not
            // output.

            static List<string> EmittedLiterals(string jinja)
            {
                var found = new List<string>();
                int i = 0;
                while (true)
                {
                    int open = jinja.IndexOf("{{-", i, StringComparison.Ordinal);
                    if (open < 0) break;
                    i = ScanExpression(jinja, open + 3, found);
                }
                return found;
            }

            /// <summary>Walks one output expression from <paramref name="p"/>, collecting decoded
            /// literals, and returns the index just past its closing <c>}}</c>. Quote-aware, so a
            /// literal that happened to contain <c>}}</c> could not end the expression early.</summary>
            static int ScanExpression(string s, int p, List<string> found)
            {
                while (p < s.Length)
                {
                    char c = s[p];
                    if (c == '\'' || c == '"')
                    {
                        var lit = new StringBuilder();
                        char quote = c;
                        p++;
                        while (p < s.Length && s[p] != quote)
                        {
                            if (s[p] == '\\' && p + 1 < s.Length)
                            {
                                char e = s[p + 1];
                                switch (e)
                                {
                                    case 'n': lit.Append('\n'); break;
                                    case 't': lit.Append('\t'); break;
                                    case 'r': lit.Append('\r'); break;
                                    case '\\': lit.Append('\\'); break;
                                    case '\'': lit.Append('\''); break;
                                    case '"': lit.Append('"'); break;
                                    default: lit.Append('\\').Append(e); break;   // unknown escape: keep it visible
                                }
                                p += 2;
                                continue;
                            }
                            lit.Append(s[p]);
                            p++;
                        }
                        p++;   // closing quote
                        found.Add(lit.ToString());
                        continue;
                    }
                    if (c == '}' && p + 1 < s.Length && s[p + 1] == '}') return p + 2;
                    p++;
                }
                return p;
            }
        }
    }
}
#endif
