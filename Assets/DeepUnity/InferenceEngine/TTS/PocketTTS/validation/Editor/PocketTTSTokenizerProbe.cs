using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // P7 probe — C# SentencePiece encoder parity. Encodes the SAME two texts the dump tokenized
        // with the REAL SentencePiece (text_ids / names_ids) and asserts the id sequences match
        // EXACTLY. If these match, PocketTTSVoice.Say(string) tokenizes identically to Python.
        public static class PocketTTSTokenizerProbe
        {
            const string DUMP = "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/dump";
            const string VOCAB = "Assets/Resources/Weights/weights_pockettts_english_fp16/tokenizer.vocab.json";
            const string REPORT = "ProbeLogs/pockettts_tokenizer.md";
            const string DONE = "ProbeLogs/pockettts_tokenizer.done";

            // the exact strings dump_reference.py tokenized (English defaults: pad=false, removeSemis=false)
            const string TEXT = "Hello world. This is a test of the pocket TTS port.";
            const string NAMES = "Hi, my name is Sebastien Aigner, and I work with Radu Ciobanu and Nguyen.";

            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketTok] " + s); }

            [MenuItem("DeepUnity/PocketTTS/P7 Tokenizer Parity")]
            public static void Run()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                try
                {
                    Log($"# pocket-tts P7 — SentencePiece encoder parity — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    var tok = new PocketTTSTokenizer(VOCAB);
                    Log($"vocab loaded: {tok.VocabSize} pieces.");

                    failed |= GateIds("text_ids", TEXT, tok);
                    failed |= GateIds("names_ids", NAMES, tok);
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            static bool GateIds(string dumpName, string text, PocketTTSTokenizer tok)
            {
                int[] refIds = Ints(dumpName);
                int[] ours = tok.Encode(text, padWithSpacesForShortInputs: false, removeSemicolons: false);
                bool match = ours.Length == refIds.Length;
                if (match) for (int i = 0; i < ours.Length; i++) if (ours[i] != refIds[i]) { match = false; break; }
                Log($"## {dumpName}: \"{text.Substring(0, Math.Min(40, text.Length))}...\"");
                Log($"   ref  ({refIds.Length}): [{string.Join(",", refIds)}]");
                Log($"   ours ({ours.Length}): [{string.Join(",", ours)}]");
                Log(match ? "   MATCH" : "   <-- MISMATCH");
                return !match;
            }

            static int[] Ints(string name)
            {
                byte[] all = File.ReadAllBytes(Path.Combine(DUMP, name + ".npy"));
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);
                string shapeStr = header.Substring(header.IndexOf('(') + 1);
                shapeStr = shapeStr.Substring(0, shapeStr.IndexOf(')'));
                long count = 1;
                foreach (string s in shapeStr.Split(',')) if (!string.IsNullOrWhiteSpace(s)) count *= int.Parse(s.Trim());
                var r = new int[count];
                // dump saves int64 ids cast to float32 (save() casts to f4); handle both.
                if (header.Contains("f4")) { for (int i = 0; i < count; i++) r[i] = (int)Math.Round(BitConverter.ToSingle(all, dataStart + i * 4)); }
                else if (header.Contains("i8")) { for (int i = 0; i < count; i++) r[i] = (int)BitConverter.ToInt64(all, dataStart + i * 8); }
                else { for (int i = 0; i < count; i++) r[i] = BitConverter.ToInt32(all, dataStart + i * 4); }
                return r;
            }
        }
    }
}
