using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace DeepUnity
{
    namespace ChatterboxModeling
    {
        // GPT2 byte-level BPE tokenizer for Chatterbox-Turbo (vocab 50276 = GPT2 50257 + 19
        // paralinguistic tags like [laugh]/[cough]). Loads the plain-text twins written by
        // import_params.py: ChatterboxTokenizer.vocab.txt (line i = token for id i) +
        // ChatterboxTokenizer.merges.txt. Encode-only (speech tokens are never decoded to text).
        // No BOS/EOS are added (turbo: add_bos_token=false). Includes the reference punc_norm.
        public class ChatterboxTokenizer
        {
            readonly Dictionary<string, int> vocab = new Dictionary<string, int>();
            readonly Dictionary<(string, string), int> mergeRanks = new Dictionary<(string, string), int>();
            readonly Dictionary<string, int[]> bpeCache = new Dictionary<string, int[]>();
            readonly List<string> addedTokens = new List<string>();      // matched literally, pre-BPE
            readonly Dictionary<string, int> addedIds = new Dictionary<string, int>();
            readonly Dictionary<byte, char> byteToUnicode = new Dictionary<byte, char>();

            // GPT2 pre-tokenizer pattern
            static readonly Regex PreTok = new Regex(
                @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                RegexOptions.Compiled);

            public bool IsReady { get; private set; }

            public ChatterboxTokenizer(string basePath)
            {
                // basePath e.g. "Assets/DeepUnity/TTS/Chatterbox/ChatterboxTokenizer" (no extension)
                string[] vlines = System.IO.File.ReadAllLines(basePath + ".vocab.txt", Encoding.UTF8);
                for (int i = 0; i < vlines.Length; i++)
                {
                    vocab[vlines[i]] = i;
                    if (i >= 50257) { addedTokens.Add(vlines[i]); addedIds[vlines[i]] = i; }
                }
                string[] mlines = System.IO.File.ReadAllLines(basePath + ".merges.txt", Encoding.UTF8);
                for (int i = 0; i < mlines.Length; i++)
                {
                    int sp = mlines[i].IndexOf(' ');
                    if (sp <= 0) continue;
                    mergeRanks[(mlines[i].Substring(0, sp), mlines[i].Substring(sp + 1))] = i;
                }
                BuildByteMap();
                IsReady = true;
            }

            void BuildByteMap()
            {
                // GPT2 bytes_to_unicode: printable ranges map to themselves, the rest shift to 256+n
                var bs = new List<int>();
                for (int b = '!'; b <= '~'; b++) bs.Add(b);
                for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
                for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
                var cs = new List<int>(bs);
                int n = 0;
                for (int b = 0; b < 256; b++)
                {
                    if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n); n++; }
                }
                for (int i = 0; i < bs.Count; i++) byteToUnicode[(byte)bs[i]] = (char)cs[i];
            }

            /// <summary>Reference punc_norm (tts_turbo.py): capitalize, collapse spaces, replace
            /// uncommon punctuation, ensure ending punctuation.</summary>
            public static string PuncNorm(string text)
            {
                if (string.IsNullOrEmpty(text)) return "You need to add some text for me to talk.";
                if (char.IsLower(text[0])) text = char.ToUpper(text[0]) + text.Substring(1);
                text = string.Join(" ", text.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries));
                (string, string)[] rep = {
                    ("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","),
                    ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
                };
                foreach (var (o, r) in rep) text = text.Replace(o, r);
                text = text.TrimEnd(' ');
                if (text.Length == 0 || ".!?-,".IndexOf(text[text.Length - 1]) < 0) text += ".";
                return text;
            }

            public int[] Encode(string text)
            {
                var ids = new List<int>(text.Length / 3);
                // split around added tokens ([laugh], [cough], ...) — literal longest-first match
                EncodeSegment(text, ids);
                return ids.ToArray();
            }

            void EncodeSegment(string text, List<int> ids)
            {
                if (text.Length == 0) return;
                int best = -1, bestPos = int.MaxValue; string bestTok = null;
                foreach (string t in addedTokens)
                {
                    int p = text.IndexOf(t, System.StringComparison.Ordinal);
                    if (p >= 0 && (p < bestPos || (p == bestPos && t.Length > (bestTok?.Length ?? 0))))
                    { bestPos = p; bestTok = t; best = addedIds[t]; }
                }
                if (bestTok != null)
                {
                    EncodeSegment(text.Substring(0, bestPos), ids);
                    ids.Add(best);
                    EncodeSegment(text.Substring(bestPos + bestTok.Length), ids);
                    return;
                }
                foreach (Match m in PreTok.Matches(text))
                    ids.AddRange(BpeWord(m.Value));
            }

            int[] BpeWord(string word)
            {
                if (bpeCache.TryGetValue(word, out int[] cached)) return cached;

                // bytes -> unicode surrogate string
                byte[] raw = Encoding.UTF8.GetBytes(word);
                var parts = new List<string>(raw.Length);
                for (int i = 0; i < raw.Length; i++) parts.Add(byteToUnicode[raw[i]].ToString());

                while (parts.Count > 1)
                {
                    int bestRank = int.MaxValue, bestIdx = -1;
                    for (int i = 0; i < parts.Count - 1; i++)
                        if (mergeRanks.TryGetValue((parts[i], parts[i + 1]), out int rank) && rank < bestRank)
                        { bestRank = rank; bestIdx = i; }
                    if (bestIdx < 0) break;
                    parts[bestIdx] = parts[bestIdx] + parts[bestIdx + 1];
                    parts.RemoveAt(bestIdx + 1);
                }

                var ids = new List<int>(parts.Count);
                foreach (string p in parts)
                    if (vocab.TryGetValue(p, out int id)) ids.Add(id);
                    // unknown pieces are silently dropped (GPT2 byte-level BPE can't actually miss)

                int[] result = ids.ToArray();
                bpeCache[word] = result;
                return result;
            }
        }
    }
}
