using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace DeepUnity
{
    namespace QwenASRModeling
    {
        // Byte-level BPE tokenizer (Qwen2 family, vocab 151936) for Qwen3-ASR.
        // Pure C# (no UnityEngine) — shared by the Unity runtime and the net8.0 parity harness.
        //
        // Loads the plain-text export written by validation/import_qwen3asr.py next to the weights:
        //   tokenizer/vocab.txt     line i = token string for id i (byte-level BPE space)
        //   tokenizer/merges.txt    "a b" per line, rank = line index
        //   tokenizer/specials.tsv  id\tcontent for added tokens (skipped on decode, matched on encode)
        //
        // Encode is needed only for the tiny chat-scaffold text pieces ("system\n", "user\n", ...)
        // plus optional context-injection / forced-language strings; decode runs on every generated
        // token. Same pipeline as Qwen3_5TokenizerFast (regex pretokenize → GPT-2 byte map → ranked
        // merges → vocab lookup), reimplemented standalone so it stays Unity-independent.
        public class QwenASRTokenizer
        {
            // Qwen2 pretokenizer regex, verbatim from tokenizer.json (NOT the Qwen3.5 variant — no \p{M}).
            const string PRETOK_PATTERN =
                @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

            readonly Dictionary<string, int> vocab = new Dictionary<string, int>();
            readonly string[] idToToken;
            readonly Dictionary<(string, string), int> mergeRank = new Dictionary<(string, string), int>();
            readonly Dictionary<int, string> specials = new Dictionary<int, string>();
            readonly Regex preTok = new Regex(PRETOK_PATTERN, RegexOptions.Compiled);

            readonly char[] byteToChar = new char[256];
            readonly Dictionary<char, byte> charToByte = new Dictionary<char, byte>();

            public QwenASRTokenizer(string tokenizerDir)
            {
                BuildByteMap();

                string[] lines = File.ReadAllLines(Path.Combine(tokenizerDir, "vocab.txt"));
                idToToken = lines;
                for (int i = 0; i < lines.Length; i++)
                    if (lines[i].Length > 0 && !vocab.ContainsKey(lines[i]))
                        vocab[lines[i]] = i;

                string[] merges = File.ReadAllLines(Path.Combine(tokenizerDir, "merges.txt"));
                for (int i = 0; i < merges.Length; i++)
                {
                    int sp = merges[i].IndexOf(' ');
                    if (sp <= 0) continue;
                    var key = (merges[i].Substring(0, sp), merges[i].Substring(sp + 1));
                    if (!mergeRank.ContainsKey(key)) mergeRank[key] = i;
                }

                foreach (string line in File.ReadAllLines(Path.Combine(tokenizerDir, "specials.tsv")))
                {
                    string[] p = line.Split('\t');
                    if (p.Length == 2) specials[int.Parse(p[0])] = p[1];
                }
            }

            // GPT-2 byte ↔ printable-unicode map (identical construction to Qwen3_5TokenizerFast).
            void BuildByteMap()
            {
                var bs = new List<int>();
                for (int b = '!'; b <= '~'; b++) bs.Add(b);
                for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
                for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
                var cs = new List<int>(bs);
                int n = 0;
                for (int b = 0; b < 256; b++)
                    if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n); n++; }
                for (int i = 0; i < bs.Count; i++)
                {
                    byteToChar[(byte)bs[i]] = (char)cs[i];
                    charToByte[(char)cs[i]] = (byte)bs[i];
                }
            }

            /// <summary>BPE-encode a PLAIN text string (no special-token matching — the ASR prompt
            /// scaffold inserts special ids explicitly by constant).</summary>
            public List<int> Encode(string text)
            {
                var ids = new List<int>();
                if (string.IsNullOrEmpty(text)) return ids;

                foreach (Match m in preTok.Matches(text))
                {
                    byte[] utf8 = Encoding.UTF8.GetBytes(m.Value);
                    var parts = new List<string>(utf8.Length);
                    for (int i = 0; i < utf8.Length; i++) parts.Add(byteToChar[utf8[i]].ToString());

                    // ranked merges: repeatedly merge the lowest-rank adjacent pair
                    while (parts.Count > 1)
                    {
                        int bestRank = int.MaxValue, bestPos = -1;
                        for (int i = 0; i < parts.Count - 1; i++)
                            if (mergeRank.TryGetValue((parts[i], parts[i + 1]), out int r) && r < bestRank)
                            { bestRank = r; bestPos = i; }
                        if (bestPos < 0) break;
                        parts[bestPos] = parts[bestPos] + parts[bestPos + 1];
                        parts.RemoveAt(bestPos + 1);
                    }

                    foreach (string p in parts)
                    {
                        if (vocab.TryGetValue(p, out int id)) ids.Add(id);
                        else foreach (char c in p)                      // unreachable for valid BPE; byte fallback
                            if (vocab.TryGetValue(c.ToString(), out int cid)) ids.Add(cid);
                    }
                }
                return ids;
            }

            /// <summary>Decode ids to text. Special/added tokens are skipped (transcript parsing cuts
            /// at the <asr_text> id before decode, so specials never reach the output anyway).</summary>
            public string Decode(IList<int> ids)
            {
                var bytes = new List<byte>(ids.Count * 4);
                for (int i = 0; i < ids.Count; i++)
                {
                    int id = ids[i];
                    if (id < 0 || id >= idToToken.Length || specials.ContainsKey(id)) continue;
                    string tok = idToToken[id];
                    for (int c = 0; c < tok.Length; c++)
                        if (charToByte.TryGetValue(tok[c], out byte b)) bytes.Add(b);
                }
                return Encoding.UTF8.GetString(bytes.ToArray());
            }
        }
    }
}
