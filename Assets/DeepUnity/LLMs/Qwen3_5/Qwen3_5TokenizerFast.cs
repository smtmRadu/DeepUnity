using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    // Byte-level BPE tokenizer for Qwen3.5. Parses HuggingFace tokenizers `tokenizer.json`
    // directly (no preprocessing).
    //
    // Pipeline (Encode):
    //   1. Locate special tokens in the input via regex; tokenize the surrounding text via BPE.
    //   2. Pre-tokenize each plain-text segment with the GPT-2/Qwen regex.
    //   3. UTF-8 encode each chunk, map bytes to printable unicode chars (GPT-2 byte map).
    //   4. Run BPE merges by priority (rank = position in merges list).
    //   5. Look up resulting strings in vocab → token IDs.
    //
    // Decode reverses the byte map and UTF-8 decodes the concatenated bytes.
    public class Qwen3_5TokenizerFast
    {
        public static readonly int ENDOFTEXT_TOKEN_ID = Qwen3_5Modeling.Qwen3_5Config.ENDOFTEXT_TOKEN_ID;
        public static readonly int IM_START_TOKEN_ID  = Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID;
        public static readonly int IM_END_TOKEN_ID    = Qwen3_5Modeling.Qwen3_5Config.IM_END_TOKEN_ID;
        public static readonly int EOS_TOKEN_ID       = Qwen3_5Modeling.Qwen3_5Config.EOS_TOKEN_ID;

        public bool IsReady { get; private set; }

        readonly Dictionary<string, int> vocab = new();
        readonly Dictionary<int, string> idToToken = new();
        readonly Dictionary<(string, string), int> mergeRank = new();
        readonly Dictionary<string, int> addedTokens = new();
        readonly Dictionary<int, string> addedTokensInv = new();

        // Pre-tokenizer regex (Qwen / GPT-2 family). NB: .NET supports \p{L}, \p{N}, \p{M}.
        const string PRETOK_PATTERN =
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        Regex preTokRegex;
        Regex specialRegex;

        // GPT-2 byte ↔ unicode map
        readonly Dictionary<byte, char> byteEncoder = new();
        readonly Dictionary<char, byte> byteDecoder = new();

        public Qwen3_5TokenizerFast(string path = "Assets/DeepUnity/LLMs/Qwen3_5/Qwen3_5TokenizerFast.json", bool load_async = false)
        {
            BuildByteMaps();
            preTokRegex = new Regex(PRETOK_PATTERN, RegexOptions.Compiled);

            if (!File.Exists(path))
                throw new ArgumentException($"Qwen3_5TokenizerFast: vocab file not found at {path}");

            if (load_async) _ = LoadAsync(path);
            else LoadSync(path);
        }

        // -------- byte map --------
        void BuildByteMaps()
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
                byteEncoder[(byte)bs[i]] = (char)cs[i];
                byteDecoder[(char)cs[i]] = (byte)bs[i];
            }
        }

        // -------- loading --------
        async Task LoadAsync(string path)
        {
            await Task.Run(() => LoadSync(path));
        }

        void LoadSync(string path)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            string content = File.ReadAllText(path, Encoding.UTF8);
            ParseAddedTokens(content);
            ParseVocab(content);
            ParseMerges(content);
            BuildSpecialRegex();
            IsReady = true;
            ConsoleMessage.Info($"Qwen3.5 tokenizer loaded ({sw.Elapsed.TotalSeconds:0.00} s, vocab={vocab.Count}, merges={mergeRank.Count}, added={addedTokens.Count})");
        }

        // Find `"section": {` or `"section": [` and return the index just past the bracket.
        static int FindSectionStart(string content, string section, char openBracket)
        {
            int needle = content.IndexOf("\"" + section + "\"", StringComparison.Ordinal);
            if (needle < 0) return -1;
            int br = content.IndexOf(openBracket, needle);
            return br < 0 ? -1 : br + 1;
        }

        // Match the closing bracket given an opening one, respecting strings.
        static int FindMatchingClose(string content, int from, char openBracket, char closeBracket)
        {
            int depth = 1;
            bool inStr = false;
            for (int i = from; i < content.Length; i++)
            {
                char c = content[i];
                if (inStr)
                {
                    if (c == '\\') { i++; continue; }
                    if (c == '"') inStr = false;
                }
                else
                {
                    if (c == '"') inStr = true;
                    else if (c == openBracket) depth++;
                    else if (c == closeBracket) { depth--; if (depth == 0) return i; }
                }
            }
            return -1;
        }

        // Read a JSON-escaped string starting at content[i]=='"'. Returns the unescaped
        // string and writes the index of the closing '"' to outClose.
        static string ReadJsonString(string content, int start, out int closeIdx)
        {
            // Assumes content[start] == '"'
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
                        case '"':  sb.Append('"');  i += 2; break;
                        case '\\': sb.Append('\\'); i += 2; break;
                        case '/':  sb.Append('/');  i += 2; break;
                        case 'b':  sb.Append('\b'); i += 2; break;
                        case 'f':  sb.Append('\f'); i += 2; break;
                        case 'n':  sb.Append('\n'); i += 2; break;
                        case 'r':  sb.Append('\r'); i += 2; break;
                        case 't':  sb.Append('\t'); i += 2; break;
                        case 'u':
                            if (i + 5 < content.Length &&
                                int.TryParse(content.Substring(i + 2, 4),
                                    System.Globalization.NumberStyles.HexNumber, null, out int code))
                            {
                                sb.Append((char)code); i += 6;
                            }
                            else { sb.Append(c); i++; }
                            break;
                        default: sb.Append(n); i += 2; break;
                    }
                }
                else if (c == '"')
                {
                    closeIdx = i;
                    return sb.ToString();
                }
                else
                {
                    sb.Append(c); i++;
                }
            }
            closeIdx = content.Length;
            return sb.ToString();
        }

        void ParseVocab(string content)
        {
            int start = FindSectionStart(content, "vocab", '{');
            if (start < 0) throw new InvalidDataException("vocab section not found");
            int end = FindMatchingClose(content, start, '{', '}');
            // Walk: "<key>": <int>, ...
            int i = start;
            while (i < end)
            {
                if (content[i] != '"') { i++; continue; }
                string key = ReadJsonString(content, i, out int qClose);
                int j = qClose + 1;
                while (j < end && (content[j] == ':' || char.IsWhiteSpace(content[j]))) j++;
                int numStart = j;
                while (j < end && (char.IsDigit(content[j]) || content[j] == '-')) j++;
                if (j > numStart && int.TryParse(content.AsSpan(numStart, j - numStart), out int id))
                {
                    vocab[key] = id;
                    idToToken[id] = key;
                }
                i = j;
            }
        }

        void ParseMerges(string content)
        {
            int start = FindSectionStart(content, "merges", '[');
            if (start < 0) throw new InvalidDataException("merges section not found");
            int end = FindMatchingClose(content, start, '[', ']');
            int i = start;
            int rank = 0;
            while (i < end)
            {
                if (content[i] != '"') { i++; continue; }
                string s = ReadJsonString(content, i, out int qClose);
                // "a b" — first space splits the two halves
                int sp = s.IndexOf(' ');
                if (sp > 0 && sp < s.Length - 1)
                {
                    string a = s.Substring(0, sp);
                    string b = s.Substring(sp + 1);
                    mergeRank[(a, b)] = rank++;
                }
                i = qClose + 1;
            }
        }

        void ParseAddedTokens(string content)
        {
            int start = FindSectionStart(content, "added_tokens", '[');
            if (start < 0) return;
            int end = FindMatchingClose(content, start, '[', ']');

            // Each element: { "id": N, "content": "...", ... }. We extract id+content per object.
            int i = start;
            while (i < end)
            {
                if (content[i] == '{')
                {
                    int objEnd = FindMatchingClose(content, i + 1, '{', '}');
                    string body = content.Substring(i + 1, objEnd - i - 1);
                    int id = -1; string ctt = null;
                    // id
                    int idIdx = body.IndexOf("\"id\"");
                    if (idIdx >= 0)
                    {
                        int colon = body.IndexOf(':', idIdx);
                        int k = colon + 1;
                        while (k < body.Length && char.IsWhiteSpace(body[k])) k++;
                        int ks = k;
                        while (k < body.Length && (char.IsDigit(body[k]) || body[k] == '-')) k++;
                        int.TryParse(body.AsSpan(ks, k - ks), out id);
                    }
                    // content
                    int cIdx = body.IndexOf("\"content\"");
                    if (cIdx >= 0)
                    {
                        int q = body.IndexOf('"', cIdx + "\"content\"".Length);
                        if (q >= 0) ctt = ReadJsonString(body, q, out _);
                    }
                    if (id >= 0 && ctt != null)
                    {
                        addedTokens[ctt] = id;
                        addedTokensInv[id] = ctt;
                        idToToken[id] = ctt;
                    }
                    i = objEnd + 1;
                }
                else i++;
            }
        }

        void BuildSpecialRegex()
        {
            if (addedTokens.Count == 0) { specialRegex = null; return; }
            // Longest first to avoid prefix matches
            var ordered = addedTokens.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape);
            specialRegex = new Regex(string.Join("|", ordered), RegexOptions.Compiled);
        }

        // ---------------------------------------------------------------- BPE
        List<string> BPE(string word)
        {
            if (word.Length <= 1) return new List<string> { word };

            // Each unicode char (after byte-mapping) is a starting "part"
            var parts = new List<string>(word.Length);
            for (int i = 0; i < word.Length; i++) parts.Add(word[i].ToString());

            while (parts.Count > 1)
            {
                int bestRank = int.MaxValue;
                int bestIdx = -1;
                for (int i = 0; i < parts.Count - 1; i++)
                {
                    if (mergeRank.TryGetValue((parts[i], parts[i + 1]), out int r) && r < bestRank)
                    {
                        bestRank = r;
                        bestIdx = i;
                    }
                }
                if (bestIdx == -1) break;
                parts[bestIdx] = parts[bestIdx] + parts[bestIdx + 1];
                parts.RemoveAt(bestIdx + 1);
            }
            return parts;
        }

        string ByteEncode(string chunk)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(chunk);
            var sb = new StringBuilder(bytes.Length);
            for (int i = 0; i < bytes.Length; i++) sb.Append(byteEncoder[bytes[i]]);
            return sb.ToString();
        }

        void EncodePlain(string text, List<int> ids)
        {
            if (string.IsNullOrEmpty(text)) return;
            foreach (Match m in preTokRegex.Matches(text))
            {
                string mapped = ByteEncode(m.Value);
                foreach (string tok in BPE(mapped))
                    if (vocab.TryGetValue(tok, out int id)) ids.Add(id);
            }
        }

        public (Tensor, Tensor) Encode(string input, bool add_special_tokens = true, bool truncation = false, int max_length = 512)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (!IsReady) throw new InvalidOperationException("Qwen3.5 tokenizer not yet loaded. Check IsReady first.");

            var ids = new List<int>();

            if (specialRegex != null)
            {
                int pos = 0;
                foreach (Match m in specialRegex.Matches(input))
                {
                    if (m.Index > pos) EncodePlain(input.Substring(pos, m.Index - pos), ids);
                    if (addedTokens.TryGetValue(m.Value, out int sid)) ids.Add(sid);
                    pos = m.Index + m.Length;
                }
                if (pos < input.Length) EncodePlain(input.Substring(pos), ids);
            }
            else
            {
                EncodePlain(input, ids);
            }

            if (truncation && ids.Count > max_length)
                ids.RemoveRange(max_length, ids.Count - max_length);

            float[] arr = new float[ids.Count];
            for (int i = 0; i < ids.Count; i++) arr[i] = ids[i];
            Tensor inputTensor = Tensor.Constant(arr);
            Tensor maskTensor = Tensor.Ones(arr.Length);
            return (inputTensor, maskTensor);
        }

        // Decode: tokens → mapped string → bytes → UTF-8 text. Handles 1D and 2D inputs.
        public List<string> Decode(Tensor input_ids)
        {
            if (input_ids.Rank <= 1)
                return new List<string> { DecodeSingle(input_ids) };
            if (input_ids.Rank == 2)
            {
                int B = input_ids.Size(-2), L = input_ids.Size(-1);
                var outs = new string[B];
                Parallel.For(0, B, b =>
                {
                    var sb = new StringBuilder();
                    for (int l = 0; l < L; l++) AppendDecoded(sb, (int)input_ids[b, l]);
                    outs[b] = MappedToUtf8(sb.ToString());
                });
                return outs.ToList();
            }
            throw new ArgumentException($"Decode only supports 1D/2D tensors (got rank {input_ids.Rank})");
        }

        string DecodeSingle(Tensor input_ids)
        {
            int n = input_ids.Rank == 0 ? 1 : input_ids.Size(-1);
            var sb = new StringBuilder();
            for (int i = 0; i < n; i++)
            {
                int id = input_ids.Rank == 0 ? (int)input_ids[0] : (int)input_ids[i];
                AppendDecoded(sb, id);
            }
            return MappedToUtf8(sb.ToString());
        }

        void AppendDecoded(StringBuilder sb, int id)
        {
            if (idToToken.TryGetValue(id, out string s)) sb.Append(s);
        }

        string MappedToUtf8(string mapped)
        {
            // Special-token strings (e.g. "<|im_end|>") aren't in the byte map; pass through.
            // We do a single-pass conversion: known mapped char → byte; unknown → keep as-is in a parallel raw buffer.
            var bytes = new List<byte>(mapped.Length);
            var raw = new StringBuilder();
            void FlushBytes()
            {
                if (bytes.Count == 0) return;
                raw.Append(Encoding.UTF8.GetString(bytes.ToArray()));
                bytes.Clear();
            }
            foreach (char c in mapped)
            {
                if (byteDecoder.TryGetValue(c, out byte b)) bytes.Add(b);
                else { FlushBytes(); raw.Append(c); }
            }
            FlushBytes();
            return raw.ToString();
        }

        // ----------------------- Chat template (Qwen3.5) -----------------------
        // Mirrors `tok.apply_chat_template(..., add_generation_prompt=True, enable_thinking=...)`.
        // HF template: thinking ON emits only `<think>\n` (model fills the block and closes itself);
        // thinking OFF emits the full empty wrap `<think>\n\n</think>\n\n` so the model skips it.
        public Tensor ApplyChatTemplate(List<Dictionary<string, string>> messages, bool add_generation_prompt = true, bool enable_thinking = false)
        {
            var ids = new List<float>();
            foreach (var m in messages)
            {
                ids.Add(IM_START_TOKEN_ID);
                AppendEncoded(m["role"], ids);
                AppendEncoded("\n", ids);
                AppendEncoded(m["content"], ids);
                ids.Add(IM_END_TOKEN_ID);
                AppendEncoded("\n", ids);
            }
            if (add_generation_prompt)
            {
                ids.Add(IM_START_TOKEN_ID);
                AppendEncoded("assistant\n", ids);
                ids.Add(Qwen3_5Modeling.Qwen3_5Config.THINK_OPEN_TOKEN_ID);
                if (enable_thinking)
                {
                    AppendEncoded("\n", ids);
                }
                else
                {
                    AppendEncoded("\n\n", ids);
                    ids.Add(Qwen3_5Modeling.Qwen3_5Config.THINK_CLOSE_TOKEN_ID);
                    AppendEncoded("\n\n", ids);
                }
            }
            return Tensor.Constant(ids.ToArray());
        }

        void AppendEncoded(string text, List<float> dst)
        {
            var (t, _) = Encode(text, add_special_tokens: false);
            for (int i = 0; i < t.Size(-1); i++) dst.Add(t[i]);
        }
    }
}
