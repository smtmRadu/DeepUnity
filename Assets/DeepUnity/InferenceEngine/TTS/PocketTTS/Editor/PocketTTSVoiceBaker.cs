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
                    byte[] bytes = tts.PrecomputePromptBytes(clip, out string key);
                    if (bytes == null)
                    {
                        EditorUtility.DisplayDialog("pocket-tts voice clone",
                            "Could not encode the clip. Either the Mimi encoder weights are missing from the " +
                            "weights dir (re-export with import_pocket_tts.py --include-encoder) or the clip's " +
                            "sample data isn't readable (set its Load Type to 'Decompress On Load').", "OK");
                        return null;
                    }
                    System.IO.Directory.CreateDirectory(ASSET_DIR);
                    string assetPath = $"{ASSET_DIR}/{key}.bytes";
                    System.IO.File.WriteAllBytes(assetPath, bytes);
                    AssetDatabase.ImportAsset(assetPath);
                    Debug.Log($"[PocketTTS] voice-clone cache baked → {assetPath} ({bytes.Length / 1024} KB, " +
                              $"{sw.ElapsedMilliseconds} ms). CloneVoice('{clip.name}') now loads instantly at runtime (editor + builds).");
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
