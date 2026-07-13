#if UNITY_EDITOR
using System;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // Editor-side voice-clone precompute ("bake"). Runs a reference AudioClip through the Mimi
        // encoder ONCE in edit mode and stores the resulting audio_prompt as a .bytes TextAsset at
        // Assets/Resources/PocketTTSVoices/<key>.bytes — the key is the same content hash
        // CloneVoice(clip) computes at runtime, so runtime (editor play mode AND player builds)
        // resolves the clone as a pure load, never re-encoding. Driven by the NPCChatBase inspector
        // button; callable from any editor tooling.
        public static class PocketTTSVoiceBaker
        {
            public const string ASSET_DIR = "Assets/Resources/" + PocketTTS.RES_VOICE_DIR;

            /// <summary>The baked asset path for a clip (null if the clip isn't readable).</summary>
            public static string BakedAssetPath(AudioClip clip)
            {
                string key = PocketTTS.CloneKey(clip);
                return key == null ? null : $"{ASSET_DIR}/{key}.bytes";
            }

            /// <summary>Bake every AudioClip under Tutorials/*/Voices/ into the shared
            /// Resources/Cache (one-click refresh after the cache key/location convention
            /// changes, or after adding new reference clips).</summary>
            [MenuItem("DeepUnity/PocketTTS/Bake Voice-Clone Cache (all Voices clips)")]
            public static void BakeAllVoicesClips()
            {
                string[] guids = AssetDatabase.FindAssets("t:AudioClip", new[] { "Assets/DeepUnity/Tutorials" });
                int baked = 0, seen = 0;
                foreach (string g in guids)
                {
                    string path = AssetDatabase.GUIDToAssetPath(g);
                    if (!path.Replace('\\', '/').Contains("/Voices/")) continue;
                    seen++;
                    var clip = AssetDatabase.LoadAssetAtPath<AudioClip>(path);
                    if (clip != null && Bake(clip) != null) baked++;
                }
                Debug.Log($"[PocketTTS] baked {baked}/{seen} voice-clone cache entries from Tutorials/*/Voices/ into {ASSET_DIR}.");
            }

            /// <summary>Encode + write the Resources cache entry for a clip. int8 picks the int8
            /// weights dir (matches the NPC's ttsQuantization; the resulting prompts are ~equal —
            /// encoder int8 parity 0.99998). Returns the asset path or null on failure.</summary>
            public static string Bake(AudioClip clip, bool int8 = false)
            {
                if (clip == null) return null;
                string dir = int8 ? PocketTTSConfig.WEIGHTS_DIR_INT8 : PocketTTSConfig.WEIGHTS_DIR_FP16;
                PocketTTS tts = null;
                try
                {
                    EditorUtility.DisplayProgressBar("pocket-tts voice clone", "Loading weights (incl. Mimi encoder)…", 0.15f);
                    tts = new PocketTTS(dir);
                    tts.LoadBlocking();
                    EditorUtility.DisplayProgressBar("pocket-tts voice clone", $"Encoding '{clip.name}'…", 0.55f);
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    byte[] bytes = tts.PrecomputePromptBytes(clip, out string key, out PocketTTS.CropInfo crop);
                    if (bytes == null)
                    {
                        EditorUtility.DisplayDialog("pocket-tts voice clone",
                            "Could not encode the clip. Either the Mimi encoder weights are missing from the " +
                            "weights dir (re-export with import_params.py pocket-tts --include-encoder) or the clip's " +
                            "sample data isn't readable (set its Load Type to 'Decompress On Load').", "OK");
                        return null;
                    }
                    System.IO.Directory.CreateDirectory(ASSET_DIR);
                    string assetPath = $"{ASSET_DIR}/{key}.bytes";
                    System.IO.File.WriteAllBytes(assetPath, bytes);
                    AssetDatabase.ImportAsset(assetPath);
                    string cropNote = crop.cropped
                        ? (crop.atPause
                            ? $"reference {crop.totalSeconds:F1}s cropped at a natural pause to {crop.croppedSeconds:F2}s"
                            : $"reference {crop.totalSeconds:F1}s hard-cut to {crop.croppedSeconds:F2}s (no pause found)")
                        : $"reference {crop.totalSeconds:F1}s used in full";
                    Debug.Log($"[PocketTTS] voice-clone cache baked → {assetPath} ({bytes.Length / 1024} KB, " +
                              $"{sw.ElapsedMilliseconds} ms; {cropNote}). CloneVoice('{clip.name}') now loads instantly at runtime (editor + builds).");
                    return assetPath;
                }
                finally
                {
                    tts?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }
        }
    }
}
#endif
