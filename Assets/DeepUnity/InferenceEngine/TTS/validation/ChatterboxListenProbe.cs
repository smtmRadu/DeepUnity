using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    namespace ChatterboxModeling
    {
        // Listen probe: runs the REAL sampled Speak path (turbo defaults: temp 0.8, top-k 1000,
        // top-p 0.95, rep 1.2) on a few lines and writes each to ProbeLogs/chatterbox_listen_N.wav.
        // This is the audio-quality check — parity used greedy decoding, which degenerates by design.
        public class ChatterboxListenProbe : MonoBehaviour
        {
            public static readonly string[] Lines =
            {
                "Hello world! This is a test of the DeepUnity port.",
                "Ah, another undead approaches the fog gate. Turn back, while your soul is still yours.",
                "The boss beyond this door has felled a thousand warriors. [laugh] What makes you think you are different?",
            };
            public string doneMarker = "ProbeLogs/chatterbox_listen.done";

            void Start() => StartCoroutine(Run());

            IEnumerator Run()
            {
                Directory.CreateDirectory("ProbeLogs");
                var tts = new ChatterboxTTS(voice: "conds_elder");   // falls back to default if not baked
                while (!tts.IsReady) yield return null;
                yield return tts.Warmup();

                var report = new StringBuilder();
                for (int i = 0; i < Lines.Length; i++)
                {
                    float[] wav = null;
                    float t0 = Time.realtimeSinceStartup;
                    int tokens = 0;
                    yield return tts.Synthesize(Lines[i], w => wav = w, onSpeechToken: _ => tokens++);
                    float dt = Time.realtimeSinceStartup - t0;
                    if (wav != null)
                    {
                        string path = $"ProbeLogs/chatterbox_listen_{i}.wav";
                        SaveWav(path, wav, ChatterboxTTS.SampleRate);
                        float sec = wav.Length / (float)ChatterboxTTS.SampleRate;
                        report.AppendLine($"[{i}] \"{Lines[i]}\" -> {tokens} tokens, {sec:F1}s audio in {dt:F1}s (RTF {dt / sec:F2}) -> {path}");
                    }
                    else report.AppendLine($"[{i}] FAILED");
                }
                Debug.Log("[ChatterboxListen]\n" + report);
                File.WriteAllText("ProbeLogs/chatterbox_listen_report.txt", report.ToString());
                File.WriteAllText(doneMarker, "DONE");
                tts.Release();
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#endif
            }

            static void SaveWav(string path, float[] samples, int sr)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var w = new BinaryWriter(fs);
                int byteLen = samples.Length * 2;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + byteLen);
                w.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16);
                w.Write((short)1); w.Write((short)1); w.Write(sr); w.Write(sr * 2);
                w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(byteLen);
                foreach (float s in samples)
                    w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
            }
        }
    }
}
