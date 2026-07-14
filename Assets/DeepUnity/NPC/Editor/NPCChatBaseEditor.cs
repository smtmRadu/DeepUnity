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
        static readonly string[] VOICE_FIELDS = { "ttsModel", "voicePitch", "ttsVoice", "ttsQuantization", "clonedVoiceClip" };

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
                {
                    using (new EditorGUI.DisabledScope(true)) EditorGUILayout.PropertyField(it);
                    continue;
                }
                if (llmOnly && System.Array.IndexOf(VOICE_FIELDS, it.name) >= 0) continue;
                if (it.propertyPath == "maxContextLength")   // only used by the continue modes
                {
                    var hm = serializedObject.FindProperty("historyMode");
                    if (hm == null || hm.enumValueIndex == (int)NPCChatBase.HistoryMode.ResetEveryTime)
                        continue;
                }
                if (it.propertyPath == "model") { DrawModelPopup(it); continue; }
                if (it.propertyPath == "smoothVsSpeed") { DrawSmoothSpeed(it); continue; }
                if (it.propertyPath == "ttsVoice") { DrawVoicePopup(it); continue; }
                if (it.propertyPath == "clonedVoiceClip") { DrawCloneClip(it); continue; }
                EditorGUILayout.PropertyField(it, true);
            }
            serializedObject.ApplyModifiedProperties();

            // Manual conversation reset (also on the component's right-click context menu). Useful
            // for ContinueWhereLeftOff once it halts at the context limit, and for wiping a
            // ResumeFromCompact/continue history during testing. Works in play mode.
            var hmProp = serializedObject.FindProperty("historyMode");
            if (target is NPCChatBase npc && hmProp != null
                && hmProp.enumValueIndex != (int)NPCChatBase.HistoryMode.ResetEveryTime)
            {
                EditorGUILayout.Space();
                if (GUILayout.Button("Reset Conversation")) npc.ResetConversation();
            }
        }

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

            // content hash once per clip change, not per repaint — but NEVER cache a failure or a
            // zero-length read (clip data still loading when first drawn): retry until it heals,
            // otherwise a just-imported clip shows "0.0s" forever.
            if (_keyClip != clip || _cloneKey == null || _crop.totalSeconds <= 0f)
            {
                _keyClip = clip;
                _cloneKey = PocketTTSModeling.PocketTTS.CloneKey(clip, out _crop);
            }
            if (_cloneKey == null || _crop.totalSeconds <= 0f)
            {
                EditorGUILayout.HelpBox("Clip sample data isn't readable (yet) — if this persists, set its Load Type to 'Decompress On Load'.", MessageType.Warning);
                return;
            }

            string assetPath = $"{PocketTTSModeling.PocketTTSVoiceBaker.ASSET_DIR}/{_cloneKey}.bytes";
            bool baked = System.IO.File.Exists(assetPath);
            float cap = PocketTTSModeling.PocketTTS.MAX_REF_SECONDS;
            float min = PocketTTSModeling.PocketTTS.MIN_CROP_SECONDS;
            string capNote = _crop.cropped
                ? (_crop.atPause
                    ? $"\nClip is {_crop.totalSeconds:F1}s — cropped at a natural pause to {_crop.croppedSeconds:F2}s, never mid-word ({cap:F0}s is the model's native reference length). The cached latents cover exactly this cropped audio."
                    : $"\nClip is {_crop.totalSeconds:F1}s — no natural pause detected in the {min:F0}-{cap:F0}s window, hard cut at {_crop.croppedSeconds:F2}s (the model's native reference length). The cached latents cover exactly this cropped audio.")
                : $"\nClip is {_crop.totalSeconds:F1}s — fits the model's native {cap:F0}s reference window, used in full (no cropping).";
            EditorGUILayout.HelpBox((baked
                ? $"Voice-clone cache baked ✓ — runtime is a pure load (editor + builds).\n{assetPath}"
                : "Not precomputed — the first runtime use encodes this clip once (~1-2 s on approach). Bake it to make runtime loading instant.")
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

        // ---- Smooth ⇄ Speed (reply pacing preference) ---------------------------------------
        // The auto-detection always computes for a stable 60+ fps; this dial only biases around
        // that result while the player talks to THIS NPC. Drawn with named ends + the current
        // mode (and the live AutoTune decision in play mode).
        static void DrawSmoothSpeed(SerializedProperty prop)
        {
            EditorGUILayout.LabelField(new GUIContent("Reply Pacing",
                "Hardware adaptation is fully automatic (AutoTune measures the GPU each session, 60 fps anchor). This slider is pure preference for this NPC's dialogues."), EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label("Smooth", GUILayout.Width(50));
                prop.floatValue = GUILayout.HorizontalSlider(prop.floatValue, 0f, 1f);
                GUILayout.Label("Speed", GUILayout.Width(42));
            }
            float v = prop.floatValue;
            string mode = v <= 0.02f ? "forced gentlest: async decode, 1 layer/frame prefill"
                        : v >= 0.98f ? "forced fastest: sync decode, bulk prefill"
                        : Mathf.Approximately(v, 0.5f) ? "pure auto — computed for a stable 60+ fps"
                        : $"auto (60 fps anchor) with bias ×{Mathf.Pow(4f, (v - 0.5f) * 2f):F2} on the measured budgets";
            EditorGUILayout.LabelField(Application.isPlaying
                ? $"{mode}   |   AutoTune: {InferencePerf.AutoTuneStatus}"
                : mode, EditorStyles.miniLabel);
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
