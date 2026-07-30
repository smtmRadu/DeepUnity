using System.Diagnostics;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// EDITOR-ONLY look at what an NPC's model actually receives as its system message: the ## NAME
    /// heading, the authored Description And Rules (tools and all) and the ## MEMORY compact, assembled
    /// exactly as <see cref="NPCChatBase.EffectivePromptPreview"/> builds them.
    /// <para>Opened by the "See Effective System Prompt" button in the inspector. It exists because the
    /// gap between the field and the real prompt was genuinely confusing — the author measured the field
    /// in a browser tokenizer, got ~360 tokens, and the engine reported 943 (user 2026-07-25). So the
    /// count here is the REAL one: the prompt goes through the model's own
    /// <see cref="Qwen3_5TokenizerFast"/>, not a chars/3.6 guess. That costs a 13 MB
    /// vocab load, which is exactly why this is a button and not something drawn every repaint; the
    /// tokenizer is then kept alive for the rest of the editor session.</para>
    /// </summary>
    public class EffectivePromptWindow : EditorWindow
    {
        static Qwen3_5TokenizerFast _tok;   // built once per editor session
        static string _tokError;

        string _npcName, _model, _text, _countLine;
        Vector2 _scroll;

        public static void Open(NPCChatBase npc)
        {
            if (npc == null) return;
            var w = GetWindow<EffectivePromptWindow>(utility: false, title: "Effective System Prompt",
                                                     focus: true);
            w.minSize = new Vector2(520f, 320f);
            w.Load(npc);
            w.Show();
        }

        void Load(NPCChatBase npc)
        {
            // read through SerializedObject rather than widening the engine's fields to public just so
            // an editor window can label itself
            var so = new SerializedObject(npc);
            _npcName = so.FindProperty("NpcName")?.stringValue;
            if (string.IsNullOrEmpty(_npcName)) _npcName = npc.name;
            _model = so.FindProperty("model")?.stringValue ?? "";
            _text = npc.EffectivePromptPreview ?? "";
            _countLine = Measure(_text, _model);
        }

        // The exact count when the NPC runs a Qwen3.5 (every NPC in the demos does); an explicit
        // estimate otherwise, labelled as one, rather than a number that pretends to be exact for a
        // tokenizer we did not run.
        static string Measure(string text, string model)
        {
            int chars = text.Length;
            if (string.IsNullOrEmpty(text)) return "empty";
            bool qwen = model != null && model.StartsWith("Qwen3.5");
            if (!qwen)
                return $"{chars:N0} chars · ~{Mathf.RoundToInt(chars / 3.6f):N0} tokens (ESTIMATE — no "
                       + $"tokenizer wired for '{model}')";

            if (_tok == null && _tokError == null)
            {
                try
                {
                    var sw = Stopwatch.StartNew();
                    EditorUtility.DisplayProgressBar("Effective System Prompt",
                                                     "Loading the Qwen3.5 tokenizer (13 MB, once)…", 0.5f);
                    _tok = new Qwen3_5TokenizerFast();
                    sw.Stop();
                }
                catch (System.Exception e) { _tokError = e.Message; }
                finally { EditorUtility.ClearProgressBar(); }
            }
            if (_tok == null || !_tok.IsReady)
                return $"{chars:N0} chars · ~{Mathf.RoundToInt(chars / 3.6f):N0} tokens (ESTIMATE — "
                       + $"tokenizer unavailable: {_tokError ?? "not ready"})";

            var (ids, _) = _tok.Encode(text, add_special_tokens: false);
            int n = ids.Size(-1);
            return $"{chars:N0} chars · {n:N0} tokens (exact, Qwen3.5 tokenizer)";
        }

        void OnGUI()
        {
            if (_text == null)
            {
                EditorGUILayout.HelpBox("Select an NPC and press \"See Effective System Prompt\".",
                                        MessageType.Info);
                return;
            }

            EditorGUILayout.Space(4f);
            EditorGUILayout.LabelField(_npcName, EditorStyles.boldLabel);
            EditorGUILayout.LabelField(_countLine, EditorStyles.miniLabel);
            EditorGUILayout.HelpBox("## NAME (from NPC Name) → Description And Rules verbatim, tools and "
                                    + "all → the Compact Summary under ## MEMORY when there is one. This is "
                                    + "the whole system message and nothing else is added at runtime.",
                                    MessageType.None);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Copy", GUILayout.Width(70f)))
                    EditorGUIUtility.systemCopyBuffer = _text;
                GUILayout.FlexibleSpace();
            }

            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            var style = new GUIStyle(EditorStyles.textArea) { wordWrap = true, richText = false };
            EditorGUILayout.SelectableLabel(_text, style,
                GUILayout.ExpandHeight(true),
                GUILayout.MinHeight(style.CalcHeight(new GUIContent(_text), position.width - 26f)));
            EditorGUILayout.EndScrollView();
        }
    }
}
