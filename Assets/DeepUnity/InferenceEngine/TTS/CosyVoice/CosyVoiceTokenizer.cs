using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // Qwen2.5 byte-level BPE tokenizer for CosyVoice3 (CosyVoice-BlankEN vocab, 151936).
        // Engine adapted from Qwen3_5TokenizerFast (same Qwen pre-tokenizer regex + GPT-2 byte
        // map + ranked merges), but loads the classic two-file form exported by import_params:
        //   CosyVoiceTokenizer.vocab.json   {"token": id, ...}
        //   CosyVoiceTokenizer.merges.txt   one "a b" pair per line (# header skipped)
        // Encode-only (speech tokens are never decoded back to text). The only special the TTS
        // ever needs is <|endofprompt|> (151646) — baked prompt transcripts already contain it;
        // it is still matched here in case user text carries specials verbatim.
        public class CosyVoiceTokenizer : TokenizerBase
        {
            public override bool IsReady { get; protected set; }

            readonly Dictionary<string, int> vocab = new Dictionary<string, int>();
            readonly Dictionary<(string, string), int> mergeRank = new Dictionary<(string, string), int>();
            readonly Dictionary<string, int> specials = new Dictionary<string, int>
            {
                { "<|endoftext|>", 151643 },
                { "<|im_start|>", 151644 },
                { "<|im_end|>", 151645 },
                { "<|endofprompt|>", CosyVoiceConfig.ENDOFPROMPT_TEXT_ID },
            };

            const string PRETOK_PATTERN =
                @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
            readonly Regex preTokRegex;
            readonly Regex specialRegex;

            readonly Dictionary<byte, char> byteEncoder = new Dictionary<byte, char>();

            public CosyVoiceTokenizer(string basePath = "Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/CosyVoiceTokenizer")
            {
                BuildByteMap();
                preTokRegex = new Regex(PRETOK_PATTERN, RegexOptions.Compiled);
                var ordered = new List<string>(specials.Keys);
                ordered.Sort((a, b) => b.Length.CompareTo(a.Length));
                for (int i = 0; i < ordered.Count; i++) ordered[i] = Regex.Escape(ordered[i]);
                specialRegex = new Regex(string.Join("|", ordered), RegexOptions.Compiled);

                string vocabPath = basePath + ".vocab.json";
                string mergesPath = basePath + ".merges.txt";
                if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
                    throw new FileNotFoundException($"CosyVoiceTokenizer files missing at {basePath}.*" +
                        " (exported by import_params.py cosyvoice3-0.5b).");

                ParseVocabJson(File.ReadAllText(vocabPath, Encoding.UTF8));
                int rank = 0;
                foreach (string line in File.ReadLines(mergesPath, Encoding.UTF8))
                {
                    if (line.Length == 0 || line[0] == '#') continue;
                    int sp = line.IndexOf(' ');
                    if (sp <= 0 || sp >= line.Length - 1) continue;
                    mergeRank[(line.Substring(0, sp), line.Substring(sp + 1))] = rank++;
                }
                IsReady = true;
                ConsoleMessage.Info($"CosyVoice tokenizer loaded (vocab={vocab.Count}, merges={mergeRank.Count})");
            }

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
                    byteEncoder[(byte)bs[i]] = (char)cs[i];
            }

            // The whole file IS the vocab object: {"token": id, ...} with JSON-escaped keys.
            void ParseVocabJson(string content)
            {
                int i = content.IndexOf('{') + 1;
                int end = content.LastIndexOf('}');
                while (i < end)
                {
                    if (content[i] != '"') { i++; continue; }
                    string key = ReadJsonString(content, i, out int qClose);
                    int j = qClose + 1;
                    while (j < end && (content[j] == ':' || char.IsWhiteSpace(content[j]))) j++;
                    int numStart = j;
                    while (j < end && (char.IsDigit(content[j]) || content[j] == '-')) j++;
                    if (j > numStart && int.TryParse(content.AsSpan(numStart, j - numStart), out int id))
                        vocab[key] = id;
                    i = j;
                }
            }

            static string ReadJsonString(string content, int start, out int closeIdx)
            {
                var sb = new StringBuilder();
                int i = start + 1;
                while (i < content.Length)
                {
                    char c = content[i];
                    if (c == '\\' && i + 1 < content.Length)
                    {
                        char n = content[i + 1];
                        switch (n)
                        {
                            case '"': sb.Append('"'); i += 2; break;
                            case '\\': sb.Append('\\'); i += 2; break;
                            case '/': sb.Append('/'); i += 2; break;
                            case 'b': sb.Append('\b'); i += 2; break;
                            case 'f': sb.Append('\f'); i += 2; break;
                            case 'n': sb.Append('\n'); i += 2; break;
                            case 'r': sb.Append('\r'); i += 2; break;
                            case 't': sb.Append('\t'); i += 2; break;
                            case 'u':
                                if (i + 5 < content.Length &&
                                    int.TryParse(content.Substring(i + 2, 4),
                                        System.Globalization.NumberStyles.HexNumber, null, out int code))
                                { sb.Append((char)code); i += 6; }
                                else { sb.Append(c); i++; }
                                break;
                            default: sb.Append(n); i += 2; break;
                        }
                    }
                    else if (c == '"') { closeIdx = i; return sb.ToString(); }
                    else { sb.Append(c); i++; }
                }
                closeIdx = content.Length;
                return sb.ToString();
            }

            List<string> BPE(string word)
            {
                if (word.Length <= 1) return new List<string> { word };
                var parts = new List<string>(word.Length);
                for (int i = 0; i < word.Length; i++) parts.Add(word[i].ToString());
                while (parts.Count > 1)
                {
                    int bestRank = int.MaxValue, bestIdx = -1;
                    for (int i = 0; i < parts.Count - 1; i++)
                        if (mergeRank.TryGetValue((parts[i], parts[i + 1]), out int r) && r < bestRank)
                        { bestRank = r; bestIdx = i; }
                    if (bestIdx == -1) break;
                    parts[bestIdx] += parts[bestIdx + 1];
                    parts.RemoveAt(bestIdx + 1);
                }
                return parts;
            }

            void EncodePlain(string text, List<int> ids)
            {
                if (string.IsNullOrEmpty(text)) return;
                foreach (Match m in preTokRegex.Matches(text))
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(m.Value);
                    var sb = new StringBuilder(bytes.Length);
                    for (int i = 0; i < bytes.Length; i++) sb.Append(byteEncoder[bytes[i]]);
                    foreach (string tok in BPE(sb.ToString()))
                        if (vocab.TryGetValue(tok, out int id)) ids.Add(id);
                }
            }

            public int[] EncodeIds(string input)
            {
                if (!IsReady) throw new InvalidOperationException("CosyVoice tokenizer not loaded.");
                var ids = new List<int>();
                int pos = 0;
                foreach (Match m in specialRegex.Matches(input))
                {
                    if (m.Index > pos) EncodePlain(input.Substring(pos, m.Index - pos), ids);
                    ids.Add(specials[m.Value]);
                    pos = m.Index + m.Length;
                }
                if (pos < input.Length) EncodePlain(input.Substring(pos), ids);
                return ids.ToArray();
            }
        }
    }
}
