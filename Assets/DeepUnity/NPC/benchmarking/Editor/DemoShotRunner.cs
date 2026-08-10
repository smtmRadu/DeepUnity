using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Batch entrypoints for the dissertation screenshot rig: open a demo scene, flip every NPC
    // to LlmOnly (batch mode has no audio device — the audio-synced reveal would never type),
    // inject a configured DemoShotProbe and enter play mode. The probe captures its frames and
    // exits the editor process itself, so one Unity launch = one scene's shots.
    public static class DemoShotRunner
    {
        [MenuItem("DeepUnity/NPC/Demo Shots/2D farm (Hobb)")]
        public static void Shot2D() => Prep(
            "Assets/DeepUnity/Tutorials/ChatDemo2D/ChatDemo2D.unity", "Hobb",
            "Good morning! How is the harvest coming along this year?", "2d_hobb", false);

        [MenuItem("DeepUnity/NPC/Demo Shots/3D castle (Velmire)")]
        public static void Shot3D() => Prep(
            "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity", "Velmire",
            "Who waits beyond the wall of golden mist?", "3d_velmire", false,
            midDelay: 7f);   // the 2B decodes ~19 tok/s — give the reply time to fill the frame

        [MenuItem("DeepUnity/NPC/Demo Shots/Trading village (Bram + banter)")]
        public static void ShotVillage() => Prep(
            "Assets/DeepUnity/Tutorials/ChatDemo3D/TradingVillage.unity", "Bram",
            "What is fresh today, friend? And how much for the trout?", "village_bram", true);

        static void Prep(string scenePath, string npcContains, string question, string prefix, bool banter,
                         float midDelay = 2.5f)
        {
            EditorSceneManager.OpenScene(scenePath);

            // text-only replies for the whole scene (the token-stream reveal works headless, and
            // no TTS weights stream alongside the LLM), and a FRESH conversation: Velmire's
            // ResumeFromCompact restored a months-old haggling transcript into the first shot,
            // which is a great persistence demo and a terrible figure
            foreach (var npc in Object.FindObjectsOfType<NPCChatBase>(true))
            {
                var so = new SerializedObject(npc);
                so.FindProperty("conversationMode").enumValueIndex = 0;   // LlmOnly
                so.FindProperty("historyMode").enumValueIndex = 0;        // ResetEveryTime
                so.FindProperty("compactSummary").stringValue = "";
                so.ApplyModifiedPropertiesWithoutUndo();
            }

            var go = new GameObject(nameof(DemoShotProbe));
            var p = go.AddComponent<DemoShotProbe>();
            p.npcNameContains = npcContains;
            p.question = question;
            p.shotPrefix = prefix;
            p.villageBanterShot = banter;
            p.midShotDelay = midDelay;

            // scene edits (mode flips + the probe GO) are never saved — play mode discards them
            EditorApplication.isPlaying = true;
        }
    }
}
