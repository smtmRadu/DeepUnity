using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // P8b — voice-clone cache tier test (the "precompute" feature). Verifies the full chain on
        // the reference wav: (1) PrecomputePromptBytes -> Resources/PocketTTSVoices/<key>.bytes,
        // (2) CloneVoice with no persistent cache loads FROM RESOURCES (no encode), (3) with the
        // bake deleted it ENCODES and the persistent bin it writes is byte-equivalent to the baked
        // prompt (corr ~1), (4) a second call hits the PERSISTENT tier. Editor-mode synchronous;
        // batch-runnable: -executeMethod DeepUnity.PocketTTSModeling.PocketTTSCloneCacheTest.Run
        public static class PocketTTSCloneCacheTest
        {
            const string DUMP = "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/dump";
            const string WEIGHTS = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string LABEL = "p8b_selftest";
            const string REPORT = "ProbeLogs/pockettts_clone_cache_test.md";

            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketCloneCache] " + s); }

            [MenuItem("DeepUnity/PocketTTS/P8b Clone Cache Tiers")]
            public static void Run()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                PocketTTS tts = null;
                string assetPath = null, persistentPath = null;
                try
                {
                    Log($"# pocket-tts P8b — clone-cache tiers (resources/encoded/persistent) — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    float[] wav = Floats(Path.Combine(DUMP, "voice_ref_audio.npy"));   // 24 kHz mono ref
                    Log($"ref wav: {wav.Length} samples ({wav.Length / 24000f:F2}s)");

                    tts = new PocketTTS(WEIGHTS);
                    tts.LoadBlocking();

                    // ---- bake (what the inspector button does) ----
                    byte[] baked = tts.PrecomputePromptBytes(wav, 24000, LABEL, out string key);
                    if (baked == null) throw new Exception("PrecomputePromptBytes failed (encoder missing?)");
                    Directory.CreateDirectory(PocketTTSVoiceBaker.ASSET_DIR);
                    assetPath = $"{PocketTTSVoiceBaker.ASSET_DIR}/{key}.bytes";
                    File.WriteAllBytes(assetPath, baked);
                    AssetDatabase.ImportAsset(assetPath);
                    Log($"baked {baked.Length / 1024} KB -> {assetPath}");

                    persistentPath = Path.Combine(Application.persistentDataPath, "pockettts_voices", key + ".bin");
                    if (File.Exists(persistentPath)) File.Delete(persistentPath);

                    // ---- tier 1: resources (persistent absent) ----
                    bool ok1 = tts.CloneVoice(wav, 24000, LABEL);
                    Log($"## clone #1: ok={ok1} source={tts.LastCloneSource} (expect resources)");
                    failed |= !ok1 || tts.LastCloneSource != "resources";

                    // ---- tier 2: encoded (bake deleted) + content equality ----
                    AssetDatabase.DeleteAsset(assetPath);
                    bool ok2 = tts.CloneVoice(wav, 24000, LABEL);
                    Log($"## clone #2: ok={ok2} source={tts.LastCloneSource} (expect encoded)");
                    failed |= !ok2 || tts.LastCloneSource != "encoded";

                    var persisted = File.Exists(persistentPath) ? File.ReadAllBytes(persistentPath) : null;
                    float corr = persisted != null ? Corr(ToFloats(baked), ToFloats(persisted)) : 0f;
                    Log($"## baked vs freshly-encoded prompt: corr {corr:F6} (expect ~1)");
                    failed |= corr < 0.99999f;

                    // ---- tier 3: persistent ----
                    bool ok3 = tts.CloneVoice(wav, 24000, LABEL);
                    Log($"## clone #3: ok={ok3} source={tts.LastCloneSource} (expect persistent)");
                    failed |= !ok3 || tts.LastCloneSource != "persistent";
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    tts?.Dispose();
                    // cleanup: test artifacts only (leave real bakes + the folder if it holds any)
                    try
                    {
                        if (assetPath != null) AssetDatabase.DeleteAsset(assetPath);
                        if (persistentPath != null && File.Exists(persistentPath)) File.Delete(persistentPath);
                    }
                    catch { }
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText("ClaudeBridge/p8b_done.txt", failed ? "FAIL" : "PASS");
                }
            }

            static float[] ToFloats(byte[] b) { var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, f.Length * 4); return f; }

            static float Corr(float[] a, float[] b)
            {
                int n = Math.Min(a.Length, b.Length);
                double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                { sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i]; }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return (float)(cov / Math.Sqrt(Math.Max(va * vb, 1e-20)));
            }

            // minimal npy reader (f4 only — dump_reference.save() casts everything to float32)
            static float[] Floats(string path)
            {
                byte[] all = File.ReadAllBytes(path);
                if (all[0] != 0x93) throw new Exception($"not npy: {path}");
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);
                if (!header.Contains("f4")) throw new Exception($"unsupported npy dtype: {header}");
                var r = new float[(all.Length - dataStart) / 4];
                Buffer.BlockCopy(all, dataStart, r, 0, r.Length * 4);
                return r;
            }
        }
    }
}
