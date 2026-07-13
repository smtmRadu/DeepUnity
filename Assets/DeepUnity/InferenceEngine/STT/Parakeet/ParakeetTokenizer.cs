using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DeepUnity
{
    namespace ParakeetModeling
    {
        // DECODE-ONLY SentencePiece-BPE detokenizer (SPEC.md §7). ASR never encodes text, so
        // this is a pure id -> string table: skip specials/blank, concat, Metaspace '▁' -> ' ',
        // trim the single leading space. Verified: neither variant's vocab has byte-fallback
        // tokens (<0xNN>), so no byte accumulation is needed.
        // Loads tokenizer/vocab.txt (line i = token for id i) + tokenizer/specials.tsv
        // (id \t content \t special \t byte) exported next to the weight manifests.
        // Pure C# (no UnityEngine) — shared by the dotnet parity harness and Unity.
        public class ParakeetTokenizer : TokenizerBase
        {
            readonly string[] vocab;
            readonly HashSet<int> specials = new HashSet<int>();

            public int Count => vocab.Length;

            public ParakeetTokenizer(string weightsDir)
            {
                string vocabPath = Path.Combine(weightsDir, "tokenizer", "vocab.txt");
                if (!File.Exists(vocabPath))
                    throw new FileNotFoundException(
                        $"tokenizer/vocab.txt missing in '{weightsDir}' (re-run import_parakeet.py).");
                vocab = File.ReadAllLines(vocabPath);

                string specialsPath = Path.Combine(weightsDir, "tokenizer", "specials.tsv");
                foreach (string line in File.ReadAllLines(specialsPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] p = line.Split('\t');       // id content special byte
                    if (p.Length >= 3 && p[2] == "1") specials.Add(int.Parse(p[0]));
                }
            }

            /// <summary>Emitted token ids (blank never appears — the TDT loop drops it, and it is
            /// flagged special anyway) -> final transcript.</summary>
            public string Decode(IReadOnlyList<int> ids)
            {
                var sb = new StringBuilder(ids.Count * 4);
                for (int i = 0; i < ids.Count; i++)
                {
                    int id = ids[i];
                    if (id < 0 || id >= vocab.Length || specials.Contains(id)) continue;
                    sb.Append(vocab[id]);
                }
                sb.Replace('▁', ' ');               // Metaspace word-boundary mark
                if (sb.Length > 0 && sb[0] == ' ') sb.Remove(0, 1);
                return sb.ToString();
            }
        }
    }
}
