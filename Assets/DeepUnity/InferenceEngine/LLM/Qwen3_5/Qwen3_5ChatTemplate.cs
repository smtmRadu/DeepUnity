using System;
using System.Collections.Generic;
using System.Text;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        /// <summary>
        /// Qwen3.5's chat template, as the ONE place this project spells its wire format out.
        /// Every literal below is transcribed verbatim from the vendored
        /// <see cref="VendoredTemplatePath"/> (<see cref="TemplateSnapshot"/>), and
        /// Qwen3_5ChatTemplateProbe re-extracts the same literals out of that file and fails the
        /// build-time check when a const drifts from it.
        /// <para>Why this class exists: until 2026-07-28 the tools preamble lived as a hand-typed
        /// string inside <c>NPCChatBase</c> — a gameplay behaviour class owning a tokenizer-level
        /// contract. The prompt the finetune is trained on and the prompt the engine serves are the
        /// same bytes or the finetune is worth nothing, and three separate divergences were caught
        /// BY EYE on 2026-07-26, each after it had already contaminated hundreds of training
        /// samples. One source, one probe.</para>
        /// <para><b>Deliberately not inspector-exposed</b> — a static class, not a ScriptableObject
        /// and not a serialized struct, so nothing here can ever appear as a field or a dropdown.
        /// A wire format is not a setting: a designer who could edit "&lt;tool_call&gt;" from the
        /// inspector could break train/serve parity for an entire scene without a compile error,
        /// and the drift guard (which reads the .jinja, not a scene asset) would still pass.</para>
        /// </summary>
        public static class Qwen3_5ChatTemplate
        {
            // ---------------------------------------------------------------- provenance

            /// <summary>HF snapshot the vendored template came from
            /// (<c>Qwen/Qwen3.5-0.8B</c> → <c>chat_template.jinja</c>, 153 lines, 7755 bytes).</summary>
            public const string TemplateSnapshot = "2fc06364715b967f1860aea9cf38778875588b17";

            /// <summary>sha256 of the vendored file with LINE ENDINGS NORMALIZED TO LF — not of the
            /// bytes on disk. This repo runs <c>core.autocrlf=true</c> with no .gitattributes, so a
            /// fresh clone on Windows gets the .jinja as CRLF and a raw-byte hash would fail for
            /// everyone but whoever produced it. Asserted by Qwen3_5ChatTemplateProbe so that
            /// re-vendoring a NEW template cannot silently leave <see cref="TemplateSnapshot"/>
            /// claiming the old one: a legitimate bump updates both, in the same commit.</summary>
            public const string TemplateSha256Lf =
                "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80";

            /// <summary>The template is VENDORED into the repo rather than read from a machine's
            /// ~/.cache/huggingface: the project must record which template it was built against,
            /// and a checkout on a machine that never pulled the weights must still be able to
            /// verify itself. Kept BYTE-IDENTICAL to the snapshot above (LF endings, no added
            /// header) so `diff` against a fresh HF download is empty — hence the provenance note
            /// lives here in C# and not as a Jinja comment at the top of the file.</summary>
            public const string VendoredTemplatePath =
                "Assets/DeepUnity/InferenceEngine/LLM/Qwen3_5/Qwen3_5ChatTemplate.jinja";

            // ---------------------------------------------------------------- tag spellings
            // The plain-text spelling of the special tokens whose ids live in Qwen3_5Config. The
            // template writes these as TEXT (it produces a string; tokenization happens after);
            // the engine emits the ids directly instead — see the "tokenized render" section.
            // Each one is asserted to be a substring of the emitted literal it belongs to, so
            // these are not a second transcription that could drift on its own.

            public const string ImStart = "<|im_start|>";              // L88, L101, L103
            public const string ImEnd = "<|im_end|>";                  // L88
            public const string ThinkTag = "<think>";
            public const string ThinkEndTag = "</think>";
            public const string ToolCallTag = "<tool_call>";
            public const string ToolCallEndTag = "</tool_call>";
            public const string ToolResponseTag = "<tool_response>";
            public const string ToolResponseEndTag = "</tool_response>";

            // ---------------------------------------------------------------- turn delimiters

            /// <summary>Opens the system message — L46 (tools branch) and L64 (no-tools branch).</summary>
            public const string SystemTurnOpen = "<|im_start|>system\n";

            /// <summary>Closes EVERY turn: system (L60/L64), user (L88), assistant (L130), tool
            /// group (L139/L141). The trailing newline is part of it.</summary>
            public const string TurnEnd = "<|im_end|>\n";

            /// <summary>What joins the tools block to the system content inside the ONE system
            /// message (L57: <c>'\n\n' + content</c>). Also the shape L88's user turn has no
            /// equivalent of — a user message is content only.</summary>
            public const string SystemContentSeparator = "\n\n";

            // ---------------------------------------------------------------- # Tools block
            // Written into the SYSTEM message and BEFORE the persona (L46-60): the template opens
            // <|im_start|>system, writes the block, and only then appends the system content after
            // '\n\n'. That order is not cosmetic — it is the order every finetuning sample is
            // written in, so it is the order that must ship.

            /// <summary>The heading the block starts with. Used to FIND a block inside authored
            /// prompt text (NPCChatBase.StripToolsBlock); a prefix of <see cref="ToolsHeader"/>.</summary>
            public const string ToolsHeading = "# Tools";

            /// <summary>L47, verbatim. Ends at the open <c>&lt;tools&gt;</c> — the schemas follow.</summary>
            public const string ToolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

            /// <summary>L49: what precedes each schema inside the <c>&lt;tools&gt;</c> list. The
            /// schema itself is <c>tool | tojson</c> — a compact one-line JSON object.</summary>
            public const string ToolSchemaSeparator = "\n";

            /// <summary>L52, verbatim (closes the schema list; no trailing newline of its own —
            /// <see cref="ToolsSpec"/> opens with the blank line).</summary>
            public const string ToolsClose = "\n</tools>";

            /// <summary>
            /// L53, verbatim: the whole call-format spec plus the <c>&lt;IMPORTANT&gt;</c> reminder
            /// list, ONE Jinja string literal from the leading blank line to the closing
            /// <c>&lt;/IMPORTANT&gt;</c> (817 chars).
            /// <para>Kept verbatim rather than paraphrased because it is what the model was trained
            /// on. MEASURED 2026-07-25, un-finetuned Qwen/Qwen3.5-0.8B: a compact block with no
            /// format spec stops the model converting a prose question into a call at all — it just
            /// repeats the prose line. It costs ~300 tokens of every tool-bearing NPC's context on
            /// every turn; that is the price of the offer actually reaching the player. Do not
            /// "optimize" it without re-running the elicitation check.</para>
            /// </summary>
            public const string ToolsSpec =
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
                "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n" +
                "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n" +
                "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n" +
                "<IMPORTANT>\nReminder:\n" +
                "- Function calls MUST follow the specified format: an inner <function=...></function> block must be " +
                "nested within <tool_call></tool_call> XML tags\n" +
                "- Required parameters MUST be specified\n" +
                "- You may provide optional reasoning for your function call in natural language BEFORE the function " +
                "call, but NOT after\n" +
                "- If there is no function call available, answer the question like normal with your current knowledge " +
                "and do not tell the user about function calls\n" +
                "</IMPORTANT>";

            /// <summary>Closes the reminder list, and therefore the whole # Tools block. Also the
            /// marker that says where a block ENDS inside authored prompt text, so a stale block
            /// can be replaced instead of stacked (NPCChatBase.StripToolsBlock).</summary>
            public const string ReminderTerminator = "</IMPORTANT>";

            /// <summary><see cref="ToolsSpec"/> up to but NOT including
            /// <see cref="ReminderTerminator"/>, i.e. ending on the last Qwen bullet's newline.
            /// This is the seam a host appends its OWN reminder bullets at: everything before it is
            /// Qwen's, everything after it until the terminator is the host's. Sliced, never
            /// re-typed — the spec text exists exactly once in this file.</summary>
            public static readonly string ToolsSpecOpen =
                ToolsSpec.Substring(0, ToolsSpec.Length - ReminderTerminator.Length);

            // ---------------------------------------------------------------- assistant turn

            /// <summary>L148, verbatim — the generation prompt's turn open (identical to what L101/
            /// L103 build for an assistant message, since the role there is <c>"assistant"</c>).</summary>
            public const string AssistantTurnOpen = "<|im_start|>assistant\n";

            /// <summary>L101 head: the thinking block of the assistant turn that FOLLOWS the last
            /// user query, i.e. the turn being generated. Reasoning goes between this and
            /// <see cref="ThinkBlockClose"/>.</summary>
            public const string ThinkBlockOpen = "\n<think>\n";

            /// <summary>L101 tail: closes the reasoning and separates it from the visible content.</summary>
            public const string ThinkBlockClose = "\n</think>\n\n";

            /// <summary>L150: the generation prompt with <c>enable_thinking</c> true — the model is
            /// left INSIDE an open think block and closes it itself.</summary>
            public const string ThinkPrefill = "<think>\n";

            /// <summary>L152: the generation prompt with thinking off (the default — the template
            /// takes this branch whenever <c>enable_thinking</c> is undefined). An EMPTY think block
            /// is still emitted, so the model starts its visible answer past a closed
            /// <c>&lt;/think&gt;</c> instead of never seeing one.</summary>
            public const string EmptyThinkBlock = "<think>\n\n</think>\n\n";

            // A <tool_call> in the assistant turn (L110-127). DeepUnity never RENDERS one — the
            // model emits it and the NPC parses it — but the shape belongs here so the parser and
            // any dataset exporter agree with the template. Three opens because the template
            // spaces the call differently depending on what precedes it.
            /// <summary>L112: first call, with visible content before it (blank line between).</summary>
            public const string ToolCallOpenAfterContent = "\n\n<tool_call>\n<function=";
            /// <summary>L114: first call, no visible content before it.</summary>
            public const string ToolCallOpenFirst = "<tool_call>\n<function=";
            /// <summary>L117: a SECOND or later call in the same assistant turn. DeepUnity never
            /// produces this — decoding is stopped the moment <c>&lt;/tool_call&gt;</c> lands, so a
            /// turn carries at most one call.</summary>
            public const string ToolCallOpenSubsequent = "\n<tool_call>\n<function=";
            /// <summary>L112/L114/L117 and L121 tail: closes a <c>&lt;function=</c> or
            /// <c>&lt;parameter=</c> opening tag.</summary>
            public const string TagNameClose = ">\n";
            /// <summary>L121 head: opens one argument. Values that are objects or arrays are
            /// rendered as JSON (L122), scalars as their string form.</summary>
            public const string ToolCallParamOpen = "<parameter=";
            /// <summary>L124: closes one argument.</summary>
            public const string ToolCallParamClose = "\n</parameter>\n";
            /// <summary>L127: closes the function and the call. No newline after it — L130's
            /// <see cref="TurnEnd"/> follows.</summary>
            public const string ToolCallClose = "</function>\n</tool_call>";

            // ---------------------------------------------------------------- tool result turn
            // A tool result is a USER turn (L131-142): role:"tool" messages are folded into a user
            // message wrapping them in <tool_response>. Note the template GROUPS CONSECUTIVE tool
            // results into a single user turn (L132/L138 look at previtem/nextitem). DeepUnity can
            // never reach that branch — it stops decoding at </tool_call>, so a turn carries one
            // call and therefore one result. Documented, not implemented.

            /// <summary>L133, verbatim — opens the user turn a tool result rides in. NOTE: no
            /// trailing newline; the newline comes from <see cref="ToolResponseOpen"/>'s head. The
            /// engine splits the same bytes the other way (role line first, then the tag), which is
            /// why this is the only place the template spells the role name "user" out.</summary>
            public const string ToolResponseTurnOpen = "<|im_start|>user";

            /// <summary>L135, verbatim.</summary>
            public const string ToolResponseOpen = "\n<tool_response>\n";

            /// <summary>L137, verbatim. No newline after it — <see cref="TurnEnd"/> follows directly,
            /// so a result turn ends <c>&lt;/tool_response&gt;&lt;|im_end|&gt;</c> with nothing between.</summary>
            public const string ToolResponseClose = "\n</tool_response>";

            // ---------------------------------------------------------------- vision placeholders
            // The template is the multimodal one (Qwen3.5 VL shares it). This port is TEXT-ONLY:
            // it never emits anything below, and Qwen3_5Config's IMAGE/VIDEO/VISION token ids are
            // declared but unused. Transcribed anyway so the drift guard covers the whole file and
            // a future VL port starts from verified strings rather than a fresh copy-paste.

            /// <summary>L18.</summary>
            public const string ImagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>";
            /// <summary>L29.</summary>
            public const string VideoPlaceholder = "<|vision_start|><|video_pad|><|vision_end|>";
            /// <summary>L16, with <c>add_vision_id</c>: <c>'Picture ' ~ n ~ ': '</c>.</summary>
            public const string PictureIdPrefix = "Picture ";
            /// <summary>L27, with <c>add_vision_id</c>: <c>'Video ' ~ n ~ ': '</c>.</summary>
            public const string VideoIdPrefix = "Video ";
            /// <summary>L16/L27: what follows the 1-based index in a vision id.</summary>
            public const string VisionIdSuffix = ": ";

            // ---------------------------------------------------------------- tokenized render
            // The engine never builds the prompt as a string: it appends the special-token IDS from
            // Qwen3_5Config and text-tokenizes only what sits BETWEEN them (one id instead of the
            // several BPE pieces a literal "<|im_start|>" would split into, and the ids are what the
            // model was trained on). The template writes the same tags as text, so every fragment
            // below is a SLICE of a literal above it — sliced, never re-typed.

            /// <summary>Follows the <c>&lt;|im_start|&gt;</c> id for a system turn: <c>"system\n"</c>.</summary>
            public static readonly string SystemRoleLine = TailAfter(SystemTurnOpen, ImStart);

            /// <summary>Follows the <c>&lt;|im_start|&gt;</c> id for a user turn: <c>"user\n"</c>.
            /// L88 builds it as <c>'&lt;|im_start|&gt;' + message.role + '\n'</c> — the role is data,
            /// not a literal, so the name is taken from L133 (the one place it is spelled out) and
            /// the newline from the same L88 shape (in the tool-result branch the template gets that
            /// newline from <see cref="ToolResponseOpen"/>'s head instead — same bytes, other seam).</summary>
            public static readonly string UserRoleLine = TailAfter(ToolResponseTurnOpen, ImStart) + "\n";

            /// <summary>Follows the <c>&lt;|im_start|&gt;</c> id for an assistant turn: <c>"assistant\n"</c>.</summary>
            public static readonly string AssistantRoleLine = TailAfter(AssistantTurnOpen, ImStart);

            /// <summary>Follows the <c>&lt;|im_end|&gt;</c> id that closes a turn: <c>"\n"</c>.</summary>
            public static readonly string TurnEndTail = TailAfter(TurnEnd, ImEnd);

            /// <summary>Follows the <c>&lt;tool_response&gt;</c> id: <c>"\n"</c>.</summary>
            public static readonly string ToolResponseOpenTail = TailAfter(ToolResponseOpen, ToolResponseTag);

            /// <summary>Precedes the <c>&lt;/tool_response&gt;</c> id: <c>"\n"</c>.</summary>
            public static readonly string ToolResponseCloseHead = HeadBefore(ToolResponseClose, ToolResponseEndTag);

            /// <summary>Follows the <c>&lt;think&gt;</c> id when thinking is ENABLED (the model is
            /// left inside the block): <c>"\n"</c>.</summary>
            public static readonly string ThinkPrefillTail = TailAfter(ThinkPrefill, ThinkTag);

            /// <summary>Sits between the <c>&lt;think&gt;</c> and <c>&lt;/think&gt;</c> ids of the
            /// EMPTY block emitted when thinking is off: <c>"\n\n"</c>.</summary>
            public static readonly string EmptyThinkMid =
                HeadBefore(TailAfter(EmptyThinkBlock, ThinkTag), ThinkEndTag);

            /// <summary>Follows the <c>&lt;/think&gt;</c> id of the empty block: <c>"\n\n"</c>.</summary>
            public static readonly string EmptyThinkTail = TailAfter(EmptyThinkBlock, ThinkEndTag);

            static string TailAfter(string literal, string tag)
                => literal.Substring(literal.IndexOf(tag, StringComparison.Ordinal) + tag.Length);

            static string HeadBefore(string literal, string tag)
                => literal.Substring(0, literal.IndexOf(tag, StringComparison.Ordinal));

            // ---------------------------------------------------------------- render helpers

            /// <summary>The CANONICAL # Tools block for these schemas — byte-identical to what the
            /// template emits for the same <c>tools</c> list. Each schema must already be the
            /// compact one-line JSON <c>tool | tojson</c> produces.</summary>
            public static string RenderToolsBlock(IEnumerable<string> toolSchemasJson)
                => RenderToolsBlock(toolSchemasJson, null);

            /// <summary>The canonical block with <paramref name="extraReminderBullets"/> spliced in
            /// at the <see cref="ToolsSpecOpen"/> seam — i.e. as extra <c>&lt;IMPORTANT&gt;</c>
            /// bullets after Qwen's and before the terminator, which is the only place a host may
            /// add to this block without moving a byte of Qwen's own text. Each bullet must start
            /// with <c>"- "</c> and end with <c>"\n"</c>, like the four above it.</summary>
            public static string RenderToolsBlock(IEnumerable<string> toolSchemasJson, string extraReminderBullets)
            {
                var sb = new StringBuilder(ToolsHeader);
                if (toolSchemasJson != null)
                    foreach (string s in toolSchemasJson)
                        sb.Append(ToolSchemaSeparator).Append(s);
                sb.Append(ToolsClose).Append(ToolsSpecOpen);
                if (!string.IsNullOrEmpty(extraReminderBullets)) sb.Append(extraReminderBullets);
                return sb.Append(ReminderTerminator).ToString();
            }

            /// <summary>The whole system message (L46-60 with tools, L64 without): the block, then
            /// the persona after a blank line, then the turn end. Pass an already-composed
            /// <paramref name="toolsBlock"/> or null/empty for the no-tools branch. Content is
            /// <c>Trim()</c>ed, which is the template's <c>| trim</c> (L55).</summary>
            public static string RenderSystemTurn(string content, string toolsBlock = null)
            {
                content = Trim(content);
                var sb = new StringBuilder(SystemTurnOpen);
                if (!string.IsNullOrEmpty(toolsBlock))
                {
                    sb.Append(toolsBlock);
                    if (content.Length > 0) sb.Append(SystemContentSeparator);
                }
                return sb.Append(content).Append(TurnEnd).ToString();
            }

            /// <summary>A plain user turn (L88), content <c>| trim</c>ed per L63.</summary>
            public static string RenderUserTurn(string content)
                => ImStart + UserRoleLine + Trim(content) + TurnEnd;

            /// <summary>The user turn ONE tool result rides in (L133-141):
            /// <c>&lt;|im_start|&gt;user\n&lt;tool_response&gt;\n{json}\n&lt;/tool_response&gt;&lt;|im_end|&gt;\n</c>.
            /// Consecutive results share one turn in the template; see the note above — DeepUnity
            /// cannot produce that case.</summary>
            public static string RenderToolResponseTurn(string resultJson)
                => ToolResponseTurnOpen + ToolResponseOpen + Trim(resultJson) + ToolResponseClose + TurnEnd;

            /// <summary>Jinja's <c>| trim</c>, which is Python <c>str.strip()</c>: whitespace off both
            /// ends, null treated as empty. Applied by every content-carrying renderer above and by
            /// the tokenized path (<c>Qwen3_5.cs</c> system and user turns) so the two agree.</summary>
            static string Trim(string s) => (s ?? "").Trim();

            /// <summary>The generation prompt (L147-153). <paramref name="enableThinking"/> false —
            /// the template's default whenever the flag is undefined — still emits an empty think
            /// block, so the model always answers past a closed <c>&lt;/think&gt;</c>.</summary>
            public static string RenderGenerationPrompt(bool enableThinking)
                => AssistantTurnOpen + (enableThinking ? ThinkPrefill : EmptyThinkBlock);

            /// <summary>One assistant <c>&lt;tool_call&gt;</c> in the template's XML shape
            /// (L110-127). <paramref name="precededByContent"/> picks L112's blank-line spacing over
            /// L114's bare open. Argument values must already be stringified the way L122 does it
            /// (JSON for objects/arrays, plain string otherwise).</summary>
            public static string RenderToolCall(string functionName,
                                               IEnumerable<KeyValuePair<string, string>> arguments,
                                               bool precededByContent = true)
            {
                var sb = new StringBuilder(precededByContent ? ToolCallOpenAfterContent : ToolCallOpenFirst);
                sb.Append(functionName).Append(TagNameClose);
                if (arguments != null)
                    foreach (var a in arguments)
                        sb.Append(ToolCallParamOpen).Append(a.Key).Append(TagNameClose)
                          .Append(a.Value).Append(ToolCallParamClose);
                return sb.Append(ToolCallClose).ToString();
            }

            // ---------------------------------------------------------------- template BEHAVIOUR
            // Rules the template enforces that emit no text of their own, and where this engine
            // knowingly parts company with them. Kept here, next to the strings, because a byte
            // match on the strings is not parity on its own.
            //
            // * |trim on EVERY content (L55, L63, L71, L82). MATCHED, on both paths: the renderers
            //   above go through Trim(), and the tokenized path trims before Encode (Qwen3_5.cs
            //   system + user turns). This is not cosmetic — the inspector's descriptionAndRules is
            //   a TextArea and AskNPC only rejects all-whitespace input, so untrimmed content was
            //   reachable from normal use, and one stray trailing newline shifts every byte after it:
            //   a different KV-cache key for a prompt that is the same prompt, and a parity dump that
            //   disagrees with apply_chat_template over invisible characters.
            // * Assistant history loses its reasoning (L100-104): only the assistant turn AFTER the
            //   last user query keeps a <think> block; earlier ones are re-rendered with the
            //   reasoning split off. This engine never re-renders history at all — it decodes into a
            //   growing KV — so whatever the model emitted, empty think block or full reasoning,
            //   stays in context for the rest of the conversation. With thinking off the difference
            //   is the 8 tokens of an empty block per past turn; with thinking on it is every
            //   thought the NPC ever had.
            // * Truncation: the engine encodes the system prompt and each user turn with
            //   max_length 2048. Nothing in the template truncates.
            // * A resume/compaction prefix (NPCChatBase.BuildResumePrompt, LLM.Compact) folds the
            //   transcript into the SYSTEM message as prose under "## MEMORY" instead of replaying
            //   it as turns. Deliberately off-template: it is a re-seed prefix, and one flat layout
            //   is what the dataset teaches.
            // * The parser accepts a <tool_call> body in Hermes JSON shape as well as the XML shape
            //   above (see NPCChatBase.ParseToolCall) — deliberately MORE permissive than the
            //   template, which only ever emits XML.
            // * raise_exception guards that carry no output: no messages at all (L43); an image or
            //   video in a system message (L10/L21); an unexpected content item or type (L33/L39);
            //   no non-tool-result user message anywhere in the list (L79); a system message that is
            //   not first (L85); an unknown role (L144).
        }
    }
}
