using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // Regression probe for the 2026-07-22 "female-voice-2 gibberish" bug: a GPU device reset
        // (DXGI_ERROR_DEVICE_REMOVED) mid-encode produced an ALL-ZERO audio_prompt that CloneVoice
        // cached to persistentDataPath — every later run bound the poisoned prompt (checked before
        // the healthy editor bake) and spoke gibberish. Verifies the fix end-to-end, objectively
        // (no listening): clone-source precedence (resources first), prompt validity stats, a fresh
        // Mimi encode vs the baked prompt (corr ~1), and speech-like waveform stats of a fixed
        // phrase synthesized with BOTH reference clips vs the baked 'jean' control.
        // Bridge-invokable: DeepUnity.PocketTTSModeling.PocketTTSCloneFixProbe.Run
        public static class PocketTTSCloneFixProbe
        {
            const string REPORT = "ProbeLogs/pockettts_clone_fix_probe.md";
            const string DONE = "ClaudeBridge/clonefix_done.txt";
            const string WEIGHTS = PocketTTSConfig.WEIGHTS_DIR_INT8;   // matches the Anya demo (ttsQuantization INT8)
            const string CLIP_FEMALE = "Assets/DeepUnity/Tutorials/AnyaChatDemo/Art/VoiceRefs/female-voice-2.wav";
            const string CLIP_ANYA = "Assets/DeepUnity/Tutorials/AnyaChatDemo/Art/VoiceRefs/anya_voice_ref.mp3";
            const string PHRASE = "Testing the cloned voice pipeline.";

            static readonly StringBuilder report = new StringBuilder();
            static bool failed;
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketCloneFix] " + s); }
            static void Check(bool ok, string what) { if (!ok) { failed = true; Log($"**FAIL**: {what}"); } }

            [MenuItem("DeepUnity/PocketTTS/Clone Gibberish Fix Probe")]
            public static void Run()
            {
                report.Clear(); failed = false;
                Directory.CreateDirectory("ProbeLogs");
                PocketTTS tts = null;
                try
                {
                    Log($"# pocket-tts clone-fix probe — {DateTime.Now:yyyy-MM-dd HH:mm} ({WEIGHTS})");
                    tts = new PocketTTS(WEIGHTS);
                    tts.LoadBlocking();

                    var jeanStats = SynthStats(tts, "jean-baked");   // control FIRST (current voice is jean)

                    var female = AssetDatabase.LoadAssetAtPath<AudioClip>(CLIP_FEMALE);
                    var anya = AssetDatabase.LoadAssetAtPath<AudioClip>(CLIP_ANYA);
                    Check(female != null, $"clip missing: {CLIP_FEMALE}");
                    Check(anya != null, $"clip missing: {CLIP_ANYA}");

                    var femaleStats = CloneAndSynth(tts, female, "female-voice-2");
                    var anyaStats = CloneAndSynth(tts, anya, "anya_voice_ref");

                    // the clones must synthesize in the same speech-like regime as the known-good control
                    CompareToControl(femaleStats, jeanStats, "female-voice-2");
                    CompareToControl(anyaStats, jeanStats, "anya_voice_ref");

                    // fresh GPU encode (never cached) must reproduce the healthy editor bake
                    if (female != null) FreshEncodeParity(tts, female);
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    tts?.Dispose();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    Directory.CreateDirectory("ClaudeBridge");
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            struct Stats { public int n; public float rms, peak, zcr, silentFrac; public int nan; }

            static Stats CloneAndSynth(PocketTTS tts, AudioClip clip, string name)
            {
                if (clip == null) return default;
                bool ok = tts.CloneVoice(clip);
                Log($"## clone '{name}': ok={ok} source={tts.LastCloneSource}");
                Check(ok, $"CloneVoice('{name}') returned false");

                float[] p = tts.CurrentVoicePrompt;
                double acc = 0; int nan = 0; float mn = float.MaxValue, mx = float.MinValue;
                for (int i = 0; i < p.Length; i++)
                {
                    if (float.IsNaN(p[i]) || float.IsInfinity(p[i])) nan++;
                    acc += (double)p[i] * p[i];
                    if (p[i] < mn) mn = p[i];
                    if (p[i] > mx) mx = p[i];
                }
                float rms = (float)Math.Sqrt(acc / Math.Max(p.Length, 1));
                Log($"prompt: frames={p.Length / 1024} min={mn:F4} max={mx:F4} rms={rms:F4} nan/inf={nan}");
                Check(nan == 0, $"'{name}' prompt has NaN/Inf");
                Check(rms > 0.01f && rms < 0.5f, $"'{name}' prompt RMS {rms:F5} outside the healthy 0.01-0.5 band (all-zero = the poisoned-cache bug)");
                return SynthStats(tts, name);
            }

            static Stats SynthStats(PocketTTS tts, string name)
            {
                float[] wav = tts.GenerateOffline(tts.Tokenize(PHRASE), maxFrames: 96);
                var s = new Stats { n = wav.Length };
                double acc = 0; int zc = 0;
                for (int i = 0; i < wav.Length; i++)
                {
                    float v = wav[i];
                    if (float.IsNaN(v) || float.IsInfinity(v)) s.nan++;
                    acc += (double)v * v;
                    if (v > s.peak) s.peak = v; else if (-v > s.peak) s.peak = -v;
                    if (i > 0 && (wav[i - 1] < 0f) != (v < 0f)) zc++;
                }
                s.rms = (float)Math.Sqrt(acc / Math.Max(wav.Length, 1));
                s.zcr = zc / (float)Math.Max(wav.Length - 1, 1);
                int hop = 720;   // 30 ms @ 24 kHz
                int silent = 0, hops = 0;
                for (int off = 0; off + hop <= wav.Length; off += hop, hops++)
                {
                    double h = 0;
                    for (int i = 0; i < hop; i++) { float v = wav[off + i]; h += v * v; }
                    if (Math.Sqrt(h / hop) < 0.1f * s.rms) silent++;
                }
                s.silentFrac = hops > 0 ? silent / (float)hops : 0f;
                Log($"synth '{name}': {s.n} samples ({s.n / 24000f:F2}s) rms={s.rms:F4} peak={s.peak:F3} " +
                    $"zcr={s.zcr:F4} silentHops={s.silentFrac:P0} nan={s.nan}");
                Check(s.n > 24000, $"'{name}' synth shorter than 1 s ({s.n} samples)");
                Check(s.nan == 0, $"'{name}' synth contains NaN/Inf");
                Check(s.rms > 0.005f && s.rms < 0.4f, $"'{name}' synth RMS {s.rms:F5} not speech-like");
                Check(s.peak > 0.02f && s.peak <= 1.5f, $"'{name}' synth peak {s.peak:F4} not speech-like");
                return s;
            }

            // Gibberish shows up as a wildly different signal regime vs a known-good baked voice
            // speaking the SAME phrase: near-noise ZCR, flat energy (no silent hops), or off-scale
            // RMS. Loose factors — voices legitimately differ, broken audio differs by far more.
            static void CompareToControl(Stats v, Stats c, string name)
            {
                if (v.n == 0 || c.n == 0) return;
                Check(v.rms < c.rms * 6f && v.rms > c.rms / 6f, $"'{name}' RMS {v.rms:F4} vs control {c.rms:F4} — off-regime");
                Check(v.zcr < c.zcr * 3.5f, $"'{name}' ZCR {v.zcr:F4} vs control {c.zcr:F4} — noise-like output");
            }

            static void FreshEncodeParity(PocketTTS tts, AudioClip clip)
            {
                string bakedPath = $"{PocketTTSVoiceBaker.ASSET_DIR}/{PocketTTS.CloneKey(clip)}.bytes";
                if (!File.Exists(bakedPath)) { Log($"(no bake at {bakedPath} — parity check skipped)"); return; }
                byte[] fresh = tts.PrecomputePromptBytes(clip, out _);
                Check(fresh != null, "fresh PrecomputePromptBytes returned null");
                if (fresh == null) return;
                float[] a = ToFloats(fresh), b = ToFloats(File.ReadAllBytes(bakedPath));
                float corr = Corr(a, b);
                Log($"## fresh encode vs baked .bytes: corr {corr:F6} (expect ~1)");
                Check(corr > 0.999f, $"fresh encode does not match the bake (corr {corr:F6}) — runtime encode path unhealthy");
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
        }
    }
}
