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
                if (it.propertyPath == "model") { DrawModelPopup(it); continue; }
                if (it.propertyPath == "clonedVoiceClip") { DrawCloneClip(it); continue; }
                EditorGUILayout.PropertyField(it, true);
            }
            serializedObject.ApplyModifiedProperties();
        }

        // ---- PocketTTS voice-clone precompute (clip field + bake status + button) --------------
        // Shown only when ttsModel == PocketTTS. The key is content-hashed from the clip, so the
        // status line always reflects THIS clip's bake state; the button runs the Mimi encoder in
        // edit mode and writes Resources/PocketTTSVoices/<key>.bytes (ships in builds) so runtime
        // never re-encodes.
        AudioClip _keyClip; string _cloneKey;

        void DrawCloneClip(SerializedProperty prop)
        {
            var modelProp = serializedObject.FindProperty("ttsModel");
            if (modelProp == null || modelProp.enumValueIndex != (int)NPCChatBase.TtsModel.PocketTTS)
                return;   // clone-from-clip is a pocket-tts feature — hidden for other engines

            EditorGUILayout.PropertyField(prop, true);
            var clip = prop.objectReferenceValue as AudioClip;
            if (clip == null || serializedObject.isEditingMultipleObjects) { _keyClip = null; _cloneKey = null; return; }

            if (_keyClip != clip)   // content hash once per clip change, not per repaint
            {
                _keyClip = clip;
                _cloneKey = PocketTTSModeling.PocketTTS.CloneKey(clip);
            }
            if (_cloneKey == null)
            {
                EditorGUILayout.HelpBox("Clip sample data isn't readable — set its Load Type to 'Decompress On Load'.", MessageType.Warning);
                return;
            }

            string assetPath = $"{PocketTTSModeling.PocketTTSVoiceBaker.ASSET_DIR}/{_cloneKey}.bytes";
            bool baked = System.IO.File.Exists(assetPath);
            float cap = PocketTTSModeling.PocketTTS.MAX_REF_SECONDS;
            string capNote = clip.length > cap + 0.05f
                ? $"\nClip is {clip.length:F1}s — auto-capped: only the first {cap:F0}s are used (the model's native reference length)."
                : $"\nReference audio is auto-capped at {cap:F0}s (the model's native prompt length).";
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
