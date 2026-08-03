using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // Inspector for every NPCChatBase subclass (2D + 3D): in LlmOnly mode the whole
    // Voice (TTS) section disappears — those fields (and their header) are simply skipped.
    [CustomEditor(typeof(NPCChatBase), true)]
    [CanEditMultipleObjects]
    public class NPCChatBaseEditor : Editor
    {
        static readonly string[] VOICE_FIELDS = { "ttsModel", "voicePitch", "voiceVolume", "ttsVoice", "ttsQuantization", "clonedVoiceClip",
                                                  "clausesPerChunk", "clausePauseSeconds", "replyTailSeconds" };

        // first field of each inspector category — a thin separator line is drawn above it
        static readonly string[] SECTION_STARTS = { "model", "temperature", "ttsModel", "interactPrompt", "usePrefetchZone", "decisions" };

        // Braced groups: (first field, last field, the label that reads down the brace). The three
        // system-prompt fields are the third, drawn by DrawSystemPromptGroup because they are pulled
        // out of iteration order. A group replaces its section header — the brace says it better and
        // takes no row (user 2026-07-25).
        static readonly (string first, string last, string label)[] BRACE_GROUPS =
        {
            ("chatWindow", "cacheKVCache", "CONVERSATION"),
            // backendTradeoff used to end this group; it moved into CONVERSATION 2026-07-27 (it is
            // not an LLM setting — it paces the TTS too), so the last sampling field ends it now.
            ("model", "repetitionPenalty", "LLM"),
            // worldAudioWhileInteracting moved to the END of the TTS block 2026-08-03 (user) —
            // field declaration order IS inspector order, so the group's last name moved with it.
            ("ttsModel", "worldAudioWhileInteracting", "TTS"),
            ("usePrefetchZone", "slowPrefetchSeconds", "PREFETCH"),
        };

        // Several small numeric fields on ONE control row (user 2026-08-03), each with a short
        // inline label carrying the field's real tooltip. Horizontal layout splits the row evenly;
        // labelWidth is per-row because "Repetition" needs more room than "Top K".
        void DrawInlineRow(float labelWidth, params (string prop, string label)[] items)
        {
            EditorGUILayout.BeginHorizontal();
            float oldLw = EditorGUIUtility.labelWidth;
            EditorGUIUtility.labelWidth = labelWidth;
            foreach (var (name, label) in items)
            {
                var p = serializedObject.FindProperty(name);
                if (p != null) EditorGUILayout.PropertyField(p, new GUIContent(label, p.tooltip));
            }
            EditorGUIUtility.labelWidth = oldLw;
            EditorGUILayout.EndHorizontal();
        }

        static void SectionDivider()
        {
            EditorGUILayout.Space(8f);
            var r = EditorGUILayout.GetControlRect(false, 1f);
            EditorGUI.DrawRect(r, new Color(0.5f, 0.5f, 0.5f, 0.35f));
            EditorGUILayout.Space(2f);
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            var mode = serializedObject.FindProperty("conversationMode");
            bool llmOnly = mode != null && mode.enumValueIndex == (int)NPCChatBase.ConversationMode.LlmOnly;

            SerializedProperty it = serializedObject.GetIterator();
            bool enterChildren = true;
            while (it.NextVisible(enterChildren))
            {
                enterChildren = false;
                if (it.propertyPath == "m_Script")
                    continue;   // hidden (user 2026-07-25): it is locked anyway, and it only takes up a row
                if (llmOnly && System.Array.IndexOf(VOICE_FIELDS, it.name) >= 0) continue;
                if (System.Array.IndexOf(SECTION_STARTS, it.name) >= 0) SectionDivider();
                // maxContextLength is shown in EVERY history mode (user 2026-07-22): even in
                // ResetEveryTime it sizes the KV cache, so it's a real (VRAM) knob worth seeing.
                // The three fields that ARE the system prompt are drawn as one braced group
                if (it.propertyPath == "NpcName") { DrawSystemPromptGroup(); continue; }
                if (it.propertyPath == "descriptionAndRules" || it.propertyPath == "compactSummary") continue;

                // …and the LLM / TTS runs get one brace each, so the component reads as what it is:
                // what the model is TOLD, what the model IS, and how it speaks.
                foreach (var g in BRACE_GROUPS)
                    if (it.propertyPath == g.first) { OpenBraceGroup(g.label); break; }

                // Sampling knobs share rows (user 2026-08-03 — compress the inspector's height):
                // Top K | Top P | Min P on one line, Presence | Repetition on another. Each row is
                // DRAWN at its LAST member so the brace-group close below (keyed on
                // repetitionPenalty, the LLM group's last field) still fires on a drawn iteration.
                if (it.propertyPath == "topK" || it.propertyPath == "topP"
                    || it.propertyPath == "presencePenalty") continue;

                if (it.propertyPath == "model") DrawModelPopup(it);
                else if (it.propertyPath == "backendTradeoff") DrawBackendTradeoff(it);
                else if (it.propertyPath == "ttsVoice") DrawVoicePopup(it);
                else if (it.propertyPath == "clonedVoiceClip") DrawCloneClip(it);
                else if (it.propertyPath == "minP")
                    DrawInlineRow(42f, ("topK", "Top K"), ("topP", "Top P"), ("minP", "Min P"));
                else if (it.propertyPath == "repetitionPenalty")
                    DrawInlineRow(70f, ("presencePenalty", "Presence"), ("repetitionPenalty", "Repetition"));
                else EditorGUILayout.PropertyField(it, true);

                if (_groupOpen && it.propertyPath == _groupLast)
                {
                    if (_groupLabel == "CONVERSATION") DrawResetConversationButton();
                    CloseBraceGroup();
                }
            }
            if (_groupOpen) CloseBraceGroup();   // last field missing from a subclass: never leak the layout
            serializedObject.ApplyModifiedProperties();

            // AFTER Apply, never during the draw: ResetConversation clears compactSummary on the
            // instance, and applying the (still stale) serialized snapshot on top would write the old
            // summary straight back in.
            if (_resetRequested)
            {
                _resetRequested = false;
                if (target is NPCChatBase npc) npc.ResetConversation();
            }
        }

        bool _resetRequested;

        // Last thing in the CONVERSATION group (user 2026-07-25) — it acts on the history the two
        // fields above define. Also on the component's right-click menu. Hidden in ResetEveryTime,
        // where there is never anything to reset.
        //
        // DISABLED (greyed, not hidden) unless there is an actual conversation to send back to state
        // 0 (user 2026-07-28): play mode, AND the player has already said something. Stopped, there is
        // no model and no live chat; before the player's first line the NPC is holding nothing but its
        // system prompt, which is exactly what a reset produces. Disabled rather than hidden so the
        // button stays where the author expects it and simply reads as unavailable, with the reason on
        // the line under it.
        //
        // NOTE, so this is not filed as a regression later: gating the BUTTON to play mode means the
        // edit-mode reset path (no llm — it clears the fields, the compact sidecar and the on-disk
        // conversation snapshots) is no longer reachable from HERE. It stays reachable, and must keep
        // working, through the component's right-click [ContextMenu("Reset Conversation")].
        //
        // The ResetEveryTime hide and the HasPlayerMessage gate AGREE, and neither is redundant —
        // change one and you have to think about the other. ResetEveryTime NPCs never record turns at
        // all (Talk only builds a Turn when historyMode != ResetEveryTime), so HasPlayerMessage is
        // permanently false in that mode: un-hiding the button there would only ever show a disabled
        // one. The reset ITSELF works perfectly well in ResetEveryTime — mid-dialogue it is exactly as
        // necessary, since the live KV is the thing being cleared — and it is reached there through the
        // right-click menu.
        void DrawResetConversationButton()
        {
            var hm = serializedObject.FindProperty("historyMode");
            if (hm == null || hm.enumValueIndex == (int)NPCChatBase.HistoryMode.ResetEveryTime) return;
            // Gated on `target`, the PRIMARY selection — which is also the only NPC the click resets
            // (see the _resetRequested handler); that scoping predates this gate and is unchanged.
            bool enabled = Application.isPlaying
                        && target is NPCChatBase npc && npc.HasPlayerMessage;
            EditorGUILayout.Space(2f);
            using (new EditorGUI.DisabledScope(!enabled))
            {
                if (GUILayout.Button("Reset Conversation")) _resetRequested = true;
            }
            if (!enabled)
                EditorGUILayout.LabelField(Application.isPlaying
                    ? "nothing to reset yet — the player has not spoken in this conversation"
                    : "play mode only — there is no live conversation while stopped",
                    EditorStyles.miniLabel);
        }

        // ---- the three fields that ARE the system prompt ----------------------------------
        // NPC Name, Description And Rules and Compact Summary are not three unrelated settings: they
        // are the model's system message, in that order (## NAME / the authored text / ## MEMORY).
        // Nothing about three stacked fields said so, and the confusion was real (user 2026-07-25), so
        // a brace is drawn around them with the label reading down its side.
        const float GROUP_GUTTER = 30f;

        bool _groupOpen;
        string _groupLabel, _groupLast;

        void OpenBraceGroup(string label, string lastField = null)
        {
            if (_groupOpen) CloseBraceGroup();     // a group that never saw its last field
            _groupLabel = label;
            _groupLast = lastField ?? System.Array.Find(BRACE_GROUPS, g => g.label == label).last;
            _groupOpen = true;
            EditorGUILayout.BeginHorizontal();
            GUILayout.Space(GROUP_GUTTER);         // the brace lives on the LEFT (user 2026-07-25)
            EditorGUILayout.BeginVertical();
        }

        void CloseBraceGroup()
        {
            EditorGUILayout.EndVertical();
            Rect fields = GUILayoutUtility.GetLastRect();
            EditorGUILayout.EndHorizontal();
            _groupOpen = false;
            if (Event.current.type == EventType.Repaint && fields.height > 4f)
                DrawBraceAndLabel(new Rect(fields.x - GROUP_GUTTER + 2f, fields.y + 1f,
                                           GROUP_GUTTER - 4f, fields.height - 2f), _groupLabel);
        }

        void DrawSystemPromptGroup()
        {
            OpenBraceGroup("SYSTEM PROMPT", "compactSummary");
            foreach (string path in new[] { "NpcName", "descriptionAndRules", "compactSummary" })
            {
                var p = serializedObject.FindProperty(path);
                if (p != null) EditorGUILayout.PropertyField(p, true);
            }
            // Inside the brace, under the Compact Summary: the three fields above are the prompt, and
            // this is the prompt as the model receives it — an editor-only look, in its own window
            // (user 2026-07-25). A button rather than a foldout: it tokenizes for a real count, which
            // is a 13 MB vocab load, not something to do on every inspector repaint.
            if (!serializedObject.isEditingMultipleObjects && target is NPCChatBase npcForPrompt)
                if (GUILayout.Button("See Effective System Prompt"))
                    EffectivePromptWindow.Open(npcForPrompt);
            CloseBraceGroup();
        }

        // A '{' built from rects: the spine sits against the fields with a tick at each end reaching
        // toward them, and a nub in the middle pointing back at the label. Glyph-free on purpose — a
        // real curly character scales with the editor font and never lines up with the group it is
        // supposed to embrace.
        static void DrawBraceAndLabel(Rect r, string label)
        {
            var col = new Color(0.55f, 0.55f, 0.55f, 0.85f);
            float x = r.xMax - 5f, mid = r.y + r.height * 0.5f;
            EditorGUI.DrawRect(new Rect(x, r.y, 1f, r.height), col);                 // spine
            EditorGUI.DrawRect(new Rect(x, r.y, 4f, 1f), col);                       // top tick →
            EditorGUI.DrawRect(new Rect(x, r.yMax - 1f, 4f, 1f), col);               // bottom tick →
            EditorGUI.DrawRect(new Rect(x - 4f, mid, 4f, 1f), col);                  // middle nub ←

            var style = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                fontStyle = FontStyle.Bold,
                normal = { textColor = col }
            };
            Matrix4x4 m = GUI.matrix;
            var pivot = new Vector2(r.x + 8f, mid);
            GUIUtility.RotateAroundPivot(90f, pivot);
            GUI.Label(new Rect(pivot.x - r.height * 0.5f, pivot.y - 8f, r.height, 16f), label, style);
            GUI.matrix = m;
        }

        // The write/refresh-# Tools button lived here until 2026-08-03 (user): it existed to keep
        // descriptionAndRules in sync with the per-NPC tool TOGGLES, and it died with them. The
        // runtime handles both interactive tools unconditionally now; whether an NPC's prompt
        // advertises them is authored by hand (or by the scene builders, which state it at their
        // WithToolsBlock calls). Tools still live IN the field (user 2026-07-25) — there is just
        // no inspector machinery deciding what belongs there any more.


        // ---- TTS voice dropdown -----------------------------------------------------------
        // Voices are REAL assets on disk, shipped inside each engine's weights export:
        //   <weights_dir>/voices/<name>/…    pocket-tts / CosyVoice3 — baked audio-prompt tensors
        //   <weights_dir>/voices/<name>.bin  Kokoro — style-vector voicepacks
        // ttsVoice is the string KEY into that folder (the name IS the asset id, no mapping).
        // The dropdown lists what actually exists on disk; pocket-tts appends "Clone (reference
        // clip)" — picking it reveals the clonedVoiceClip field (a non-null clip always means
        // clone mode, and overrides the baked name at runtime). Engines with no on-disk catalog
        // fall back to the plain string field.
        static readonly System.Collections.Generic.Dictionary<int, string[]> _voiceCache =
            new System.Collections.Generic.Dictionary<int, string[]>();
        const string CLONE_OPTION = "Clone (reference clip)";
        bool _clonePicked;   // transient: "Clone" chosen in the dropdown, clip not assigned yet

        static string[] DiscoverVoices(int ttsModel)
        {
            if (_voiceCache.TryGetValue(ttsModel, out var cached)) return cached;
            string prefix = ttsModel == (int)NPCChatBase.TtsModel.Chatterbox ? "weights_chatterbox"
                          : ttsModel == (int)NPCChatBase.TtsModel.CosyVoice3 ? "weights_cosyvoice3"
                          : ttsModel == (int)NPCChatBase.TtsModel.Kokoro ? "weights_kokoro"
                          : "weights_pockettts";
            var names = new System.Collections.Generic.SortedSet<string>();
            const string ROOT = "Assets/Resources/Weights";
            if (System.IO.Directory.Exists(ROOT))
                foreach (string dir in System.IO.Directory.GetDirectories(ROOT))
                {
                    if (!System.IO.Path.GetFileName(dir).StartsWith(prefix)) continue;
                    string vdir = System.IO.Path.Combine(dir, "voices");
                    if (!System.IO.Directory.Exists(vdir)) continue;
                    foreach (string d in System.IO.Directory.GetDirectories(vdir))
                        names.Add(System.IO.Path.GetFileName(d));
                    foreach (string f in System.IO.Directory.GetFiles(vdir, "*.bin"))
                        names.Add(System.IO.Path.GetFileNameWithoutExtension(f));
                }
            var arr = new string[names.Count];
            names.CopyTo(arr);
            _voiceCache[ttsModel] = arr;
            return arr;
        }

        void DrawVoicePopup(SerializedProperty prop)
        {
            var modelProp = serializedObject.FindProperty("ttsModel");
            if (modelProp == null || serializedObject.isEditingMultipleObjects)
            { EditorGUILayout.PropertyField(prop, true); return; }
            int m = modelProp.enumValueIndex;
            bool canClone = m == (int)NPCChatBase.TtsModel.PocketTTS;
            var cloneProp = serializedObject.FindProperty("clonedVoiceClip");
            bool cloneActive = canClone && (_clonePicked || (cloneProp != null && cloneProp.objectReferenceValue != null));

            string[] voices = DiscoverVoices(m);
            if (voices.Length == 0 && !canClone)   // no on-disk catalog for this engine — plain field
            { EditorGUILayout.PropertyField(prop, true); return; }

            int cloneIdx = canClone ? voices.Length : -1;
            int missingIdx = -1;
            int current;
            if (cloneActive) current = cloneIdx;
            else
            {
                current = System.Array.IndexOf(voices, prop.stringValue);
                if (current < 0) { missingIdx = voices.Length + (canClone ? 1 : 0); current = missingIdx; }
            }
            var options = new string[voices.Length + (canClone ? 1 : 0) + (missingIdx >= 0 ? 1 : 0)];
            voices.CopyTo(options, 0);
            if (canClone) options[cloneIdx] = CLONE_OPTION;
            if (missingIdx >= 0) options[missingIdx] = $"{prop.stringValue} <missing>";

            int pick = EditorGUILayout.Popup(prop.displayName, current, options);
            if (pick == current) return;
            if (pick == cloneIdx) _clonePicked = true;   // clip field appears below (DrawCloneClip)
            else if (pick < voices.Length)
            {
                prop.stringValue = voices[pick];
                _clonePicked = false;
                if (cloneProp != null) cloneProp.objectReferenceValue = null;   // clip would override the pick
            }
        }

        // ---- PocketTTS voice-clone precompute (clip field + bake status + button) --------------
        // Shown only in clone mode (pocket-tts + "Clone" picked / a clip assigned). The key is
        // content-hashed from the clip, so the status line always reflects THIS clip's bake state;
        // the button runs the Mimi encoder in edit mode and writes Resources/Cache/<key>.bytes
        // (ships in builds) so runtime never re-encodes.
        AudioClip _keyClip; string _cloneKey; PocketTTSModeling.PocketTTS.CropInfo _crop;
        double _nextKeyRetry;   // failed-read rehash cooldown — see the loadState gate in DrawCloneClip

        void DrawCloneClip(SerializedProperty prop)
        {
            var modelProp = serializedObject.FindProperty("ttsModel");
            if (modelProp == null || modelProp.enumValueIndex != (int)NPCChatBase.TtsModel.PocketTTS)
                return;   // clone-from-clip is a pocket-tts feature — hidden for other engines
            if (!_clonePicked && prop.objectReferenceValue == null)
                return;   // a baked voice is selected in the dropdown — clone UI hidden

            EditorGUILayout.PropertyField(prop, true);
            var clip = prop.objectReferenceValue as AudioClip;
            if (clip == null || serializedObject.isEditingMultipleObjects) { _keyClip = null; _cloneKey = null; return; }

            // Content hash once per clip change, not per repaint — and NEVER inside a blocking
            // wait (2026-08-03, the "Hold on … OnInspectorGUI" dialog on every play-stop): leaving
            // play mode unloads the clip's audio data, and CloneKey's ClipToMono used to sit in
            // its up-to-3 s sleep-wait right here in IMGUI while it reloaded. Gate on loadState
            // instead — kick the async load, show a quiet placeholder, poll via Repaint — and only
            // hash when the data is actually there. A failed read (bad import) retries on a 2 s
            // cooldown rather than per repaint (the hash is ~200 ms of resample+SHA every time):
            // still self-healing after a reimport, never a per-frame bill.
            if (_keyClip != clip || _cloneKey == null || _crop.totalSeconds <= 0f)
            {
                if (clip.loadState != AudioDataLoadState.Loaded)
                {
                    clip.LoadAudioData();
                    EditorGUILayout.HelpBox("Reading clip sample data…", MessageType.None);
                    Repaint();
                    return;
                }
                if (_keyClip != clip || EditorApplication.timeSinceStartup >= _nextKeyRetry)
                {
                    _keyClip = clip;
                    _cloneKey = PocketTTSModeling.PocketTTS.CloneKey(clip, out _crop);
                    _nextKeyRetry = EditorApplication.timeSinceStartup + 2.0;
                }
            }
            if (_cloneKey == null || _crop.totalSeconds <= 0f)
            {
                EditorGUILayout.HelpBox("Clip sample data isn't readable (yet) — if this persists, set its Load Type to 'Decompress On Load'.", MessageType.Warning);
                return;
            }

            string assetPath = $"{PocketTTSModeling.PocketTTSVoiceBaker.ASSET_DIR}/{_cloneKey}.bytes";
            bool baked = System.IO.File.Exists(assetPath);
            float cap = PocketTTSModeling.PocketTTS.MAX_REF_SECONDS;
            string capNote = _crop.cropped
                ? $"  Clip {_crop.totalSeconds:F1}s → cut to {_crop.croppedSeconds:F2}s ({cap:F1}s cap)."
                : $"  Clip {_crop.totalSeconds:F1}s (under the {cap:F1}s cap, used in full).";
            EditorGUILayout.HelpBox((baked
                ? "Baked ✓ — runtime is a pure load (editor + builds)."
                : "Not precomputed — first runtime use encodes once (~1-2 s). Bake for instant load.")
                + capNote,
                baked ? MessageType.Info : MessageType.None);
            using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
            {
                if (GUILayout.Button(baked ? "Recompute voice-clone cache" : "Precompute voice-clone cache"))
                {
                    var q = serializedObject.FindProperty("ttsQuantization");
                    bool int8 = q == null || (LLMQuant)q.intValue != LLMQuant.FP16;
                    PocketTTSModeling.PocketTTSVoiceBaker.Bake(clip, int8);
                }
            }
        }

        // ---- Backend Tradeoff (how capable is this machine) --------------------------------------
        // Five fixed rows, so what a level MEANS is a lookup the inspector can just show. The old
        // continuous slider could only promise that an auto-tuner would work the numbers out, and
        // printed its measured verdict on this line after the fact (2026-07-26 — BackendTradeoff.cs
        // documents why that turned out to be the wrong instrument). The label is passed explicitly
        // for the same reason it always was: the drawer, not Unity's nicifier, owns what this row
        // says. TWO mini-lines since 2026-07-27, one per pipeline — the second is where the
        // counter-intuition is visible at a glance (ticks/frame FALL as the tier rises).
        static void DrawBackendTradeoff(SerializedProperty prop)
        {
            EditorGUILayout.Space(4f);
            // The two mini-lines of per-tier numbers that used to print under the dropdown moved
            // into the TOOLTIP (user 2026-08-03: the exact values are reference material, not
            // something to spend two inspector rows on). Selection-specific, so hovering the row
            // answers exactly what the choice means; BackendTradeoffTable stays the one source.
            string tip = prop.tooltip;
            if (!prop.hasMultipleDifferentValues)
            {
                var row = BackendTradeoffTable.At((BackendTradeoffLevel)prop.enumValueIndex);
                tip += $"\n\nThis tier: fetch {row.fetchBytesPerFrame / 1e6:0.0} MB/frame, " +
                       $"prefill {row.prefillStepsPerFrame} steps/frame, decode {row.decodeTokensPerFrame} tok/frame; " +
                       $"tts {row.ttsSpeakingTicksPerFrame} ticks/frame speaking, {row.ttsSilentTicksPerFrame} refilling, " +
                       $"prebuffer {row.ttsPrebufferSeconds:0.##}s, chunk {row.ttsStreamChunkFrames}f " +
                       $"({row.ttsStreamChunkFrames * 0.08f:0.00}s), cede above {row.ttsCedeHeadroomSeconds:0.#}s.";
            }
            EditorGUILayout.PropertyField(prop, new GUIContent("Backend Tradeoff", tip));
        }

        // The LLM is a string id backed by LLMRegistry (auto-discovered [LLMEntry] methods) —
        // drawn as a dropdown so new model ports appear with zero inspector code. An id that
        // fell out of the registry shows as an extra "<missing>" row until re-picked.
        static void DrawModelPopup(SerializedProperty prop)
        {
            string[] ids = LLMRegistry.Ids;
            string current = prop.stringValue;
            int idx = System.Array.IndexOf(ids, current);
            string[] options = ids;
            if (idx < 0)
            {
                options = new string[ids.Length + 1];
                ids.CopyTo(options, 0);
                options[ids.Length] = $"{current} <missing>";
                idx = ids.Length;
            }
            int pick = EditorGUILayout.Popup(prop.displayName, idx, options);
            if (pick >= 0 && pick < ids.Length)
                prop.stringValue = ids[pick];
        }
    }
}
