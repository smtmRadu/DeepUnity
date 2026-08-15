#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    // Batch wrapper for the pocket-tts clone bake: the TradingVillage strollers clone from
    // Moore.mp3 (Odo and Bram) and Ansbach_4-15s.mp3 (Fenn), and the bake turns the
    // first ambient line's one-time Mimi encode into a pure cache load. BakeAllVoicesClips
    // covers every clip under Tutorials/*/Voices/, so the souls-scene clones just refresh.
    public static class VillageVoiceBakeBatch
    {
        public static void BakeBatch()
        {
            try
            {
                DeepUnity.PocketTTSModeling.PocketTTSVoiceBaker.BakeAllVoicesClips();
                Debug.Log("[VillageVoiceBakeBatch] BAKE OK");
                EditorApplication.Exit(0);
            }
            catch (System.Exception e)
            {
                Debug.LogError("[VillageVoiceBakeBatch] BAKE FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }
    }
}
#endif
