#if UNITY_EDITOR
using System.IO;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace DeepUnity
{
    // Ships the InferenceEngine's file-streamed data with player builds. Weights (.bin/.tsv) and
    // tokenizer/lexicon files are read with plain System.IO from Assets/-relative paths — Resources
    // only packs assets loaded via Resources.Load, so none of it would reach a build on its own
    // (see DeepUnityMeta.ResolvePath, the runtime half of this mechanism). After every player
    // build this hook copies them VERBATIM into <Game>_Data/StreamingAssets/<same Assets/-tail>,
    // where ResolvePath finds them. Copying into the OUTPUT (not Assets/StreamingAssets) keeps
    // the multi-GB weight folders out of the project's import pipeline.
    public class DeepUnityBuildStep : IPostprocessBuildWithReport
    {
        public int callbackOrder => 100;

        const string WEIGHTS_ROOT = "Assets/Resources/Weights";

        // Runtime data files read via File IO outside the weights folders (tokenizers, G2P
        // lexicons, presets) — copied with their Assets/-tail preserved.
        static readonly (string dir, string pattern)[] DATA_FILES =
        {
            ("Assets/DeepUnity/InferenceEngine/LLM/Qwen3_5",   "Qwen3_5TokenizerFast.json"),
            ("Assets/DeepUnity/InferenceEngine/LLM/Gemma3",    "Gemma3TokenizerFast.json"),
            ("Assets/DeepUnity/InferenceEngine/LLM/MiniCPM5",  "MiniCPM5TokenizerFast.json"),
            ("Assets/DeepUnity/InferenceEngine/TTS/Chatterbox","ChatterboxTokenizer*"),
            ("Assets/DeepUnity/InferenceEngine/TTS/CosyVoice", "CosyVoiceTokenizer*"),
            ("Assets/DeepUnity/InferenceEngine/TTS/Kokoro",    "KokoroG2P.*.tsv"),
            ("Assets/DeepUnity/InferenceEngine/TTS/VoiceLab",  "voice_presets.json"),
        };

        public void OnPostprocessBuild(BuildReport report)
        {
            string exe = report.summary.outputPath;
            string dataDir = Path.Combine(Path.GetDirectoryName(exe) ?? ".",
                                          Path.GetFileNameWithoutExtension(exe) + "_Data");
            if (!Directory.Exists(dataDir))
            {
                Debug.LogWarning($"[DeepUnityBuildStep] no _Data folder next to '{exe}' — " +
                                 "unsupported build target for the file-streamed weights; copy " +
                                 $"{WEIGHTS_ROOT} into the build's StreamingAssets manually.");
                return;
            }
            string sa = Path.Combine(dataDir, "StreamingAssets");

            long bytes = 0; int files = 0;
            if (Directory.Exists(WEIGHTS_ROOT))
                CopyTree(WEIGHTS_ROOT, Path.Combine(sa, "Resources", "Weights"), ref bytes, ref files);
            else
                Debug.LogWarning($"[DeepUnityBuildStep] {WEIGHTS_ROOT} not found — the build will have " +
                                 "no model weights (export them with import_params.py first).");

            foreach (var (dir, pattern) in DATA_FILES)
            {
                if (!Directory.Exists(dir)) continue;
                foreach (string f in Directory.GetFiles(dir, pattern))
                {
                    if (f.EndsWith(".meta")) continue;
                    // Assets/<tail> -> StreamingAssets/<tail> (same mapping ResolvePath applies)
                    string tail = f.Replace('\\', '/').Substring("Assets/".Length);
                    CopyFile(f, Path.Combine(sa, tail), ref bytes, ref files);
                }
            }

            Debug.Log($"[DeepUnityBuildStep] shipped {files} files, {bytes / (1024f * 1024f):F0} MB " +
                      $"of weights/tokenizer data → {sa} (delete unused weights_* folders there to slim the build).");
        }

        static void CopyTree(string src, string dst, ref long bytes, ref int files)
        {
            Directory.CreateDirectory(dst);
            foreach (string f in Directory.GetFiles(src))
            {
                if (f.EndsWith(".meta")) continue;
                CopyFile(f, Path.Combine(dst, Path.GetFileName(f)), ref bytes, ref files);
            }
            foreach (string d in Directory.GetDirectories(src))
                CopyTree(d, Path.Combine(dst, Path.GetFileName(d)), ref bytes, ref files);
        }

        static void CopyFile(string src, string dst, ref long bytes, ref int files)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(dst));
            File.Copy(src, dst, overwrite: true);
            bytes += new FileInfo(src).Length;
            files++;
        }
    }
}
#endif
