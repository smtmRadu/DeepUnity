#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    // Model-aware inspector for LMUnitTest: the selectors adapt to the chosen LM so you never
    // see options that don't apply to it —
    //   * size           : shown for Qwen only (Gemma3 ships a single 270m size)
    //   * quantization   : FP16/INT8/INT4 for Qwen, FP16/INT8 only for Gemma (no INT4 runtime)
    //   * enable_thinking: shown for Qwen only (Gemma3 ignores it)
    [CustomEditor(typeof(LMUnitTest))]
    public class LMUnitTestEditor : Editor
    {
        // FP16/INT8 only — what Gemma3 can actually run (index maps 1:1 to LLMQuant 0/1).
        static readonly string[] GemmaQuantNames = { "FP16", "INT8" };

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            using (new EditorGUI.DisabledScope(true))
                EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Script"));

            var lm = serializedObject.FindProperty("lm");
            var size = serializedObject.FindProperty("size");
            var quant = serializedObject.FindProperty("quantization");

            EditorGUILayout.PropertyField(lm);
            bool isQwen = lm.enumValueIndex == (int)LMUnitTest.LM.Qwen3_5;

            if (isQwen)
            {
                EditorGUILayout.PropertyField(size);
                EditorGUILayout.PropertyField(quant);
            }
            else
            {
                // Gemma: clamp a leftover INT4 selection (from a previous Qwen run) to FP16, then
                // present only the runnable formats.
                int cur = Mathf.Min(quant.enumValueIndex, GemmaQuantNames.Length - 1);
                int sel = EditorGUILayout.Popup(
                    new GUIContent("Quantization", "FP16 = reference; INT8 = per-row, ~lossless, half VRAM (recommended)."),
                    cur, GemmaQuantNames);
                quant.enumValueIndex = sel;
            }

            // Everything else in declaration order; enable_thinking is drawn manually (Qwen only).
            DrawPropertiesExcluding(serializedObject, "m_Script", "lm", "size", "quantization", "enable_thinking");
            if (isQwen)
                EditorGUILayout.PropertyField(serializedObject.FindProperty("enable_thinking"));

            serializedObject.ApplyModifiedProperties();
        }
    }
}
#endif
