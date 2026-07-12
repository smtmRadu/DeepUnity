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
        static readonly string[] VOICE_FIELDS = { "ttsModel", "voicePitch", "ttsVoice", "ttsQuantization" };

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
                EditorGUILayout.PropertyField(it, true);
            }
            serializedObject.ApplyModifiedProperties();
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
