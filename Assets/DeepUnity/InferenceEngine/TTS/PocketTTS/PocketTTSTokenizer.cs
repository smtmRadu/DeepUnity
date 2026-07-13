using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // C# port of the pocket-tts SentencePiece encoder (Unigram model, byte-fallback, NO Unicode
        // normalization). Loads tokenizer.vocab.json (pieces + scores + types, emitted by
        // import_params.py pocket-tts) and reproduces sp.encode(text, out_type=int) exactly:
        //   1) prepare_text_prompt: strip, \n\r -> space, "  " -> " ", (optional ; -> ,), uppercase
        //      first letter, ensure trailing punctuation, (optional 8-space pad for short inputs).
        //   2) SentencePiece normalize (identity here — this model applies no NFKC; verified that
        //      ligatures/fullwidth survive as raw bytes and double spaces are preserved): prepend a
        //      dummy prefix U+2581 and replace every ' ' with U+2581.
        //   3) Unigram Viterbi best-segmentation over the piece lattice (max sum of log-scores),
        //      with byte-fallback: at each position the single source char may also be emitted as
        //      its UTF-8 byte pieces (<0x00>..<0xFF> = ids byte_base_id..+255), so unknown chars
        //      decompose to bytes exactly like SentencePiece.
        // Validated against the dump's text_ids/names_ids (real SentencePiece) — ids must match.
        public class PocketTTSTokenizer
        {
            const char SPACE = '▁';   // ▁ SentencePiece whitespace marker

            struct Piece { public string s; public float score; public int type; }   // type 6 = BYTE
            readonly Dictionary<string, int> _pieceToId = new Dictionary<string, int>();
            Piece[] _pieces;
            int _byteBase, _unkId, _maxPieceLen = 1;
            float _byteEdgeScore;   // SentencePiece byte-fallback edge score = min_piece_score - 10 (per byte)
            const float UNK_PENALTY = 10f;   // SentencePiece kUnkPenalty
            public int VocabSize { get; private set; }
            public bool IsReady { get; private set; }

            public PocketTTSTokenizer(string vocabJsonPath)
            {
                if (!File.Exists(vocabJsonPath))
                    throw new FileNotFoundException(
                        $"pocket-tts tokenizer.vocab.json not found at '{vocabJsonPath}'. Re-run " +
                        "import_params.py pocket-tts (it emits tokenizer.vocab.json next to tokenizer.model).");
                Load(File.ReadAllText(vocabJsonPath, Encoding.UTF8));
            }

            // ---- minimal JSON parse (the file is a flat, machine-written object; no general parser
            // needed). Shape: {"vocab_size":N,"unk_id":..,"byte_base_id":..,"pieces":[{"piece":"..",
            // "score":-1.23,"type":1},...]}. Uses JsonUtility for the pieces array via a wrapper.
            [Serializable] class PieceJson { public string piece; public float score; public int type; }
            [Serializable] class VocabJson { public int vocab_size; public int unk_id; public int bos_id;
                                             public int eos_id; public int pad_id; public int byte_base_id;
                                             public PieceJson[] pieces; }

            void Load(string json)
            {
                var v = JsonUtility.FromJson<VocabJson>(json);
                if (v == null || v.pieces == null)
                    throw new FormatException("pocket-tts tokenizer.vocab.json: could not parse pieces.");
                VocabSize = v.vocab_size;
                _unkId = v.unk_id;
                _byteBase = v.byte_base_id;
                _pieces = new Piece[v.pieces.Length];
                float minScore = float.PositiveInfinity;
                for (int i = 0; i < v.pieces.Length; i++)
                {
                    var pj = v.pieces[i];
                    _pieces[i] = new Piece { s = pj.piece, score = pj.score, type = pj.type };
                    if (pj.score < minScore) minScore = pj.score;
                    // NORMAL / USER_DEFINED pieces are match candidates; CONTROL/UNKNOWN/BYTE are not
                    // matched by string (byte pieces are addressed positionally via _byteBase).
                    if (pj.type == 1 || pj.type == 4)
                    {
                        _pieceToId[pj.piece] = i;
                        if (pj.piece.Length > _maxPieceLen) _maxPieceLen = pj.piece.Length;
                    }
                }
                // SentencePiece scores byte-fallback (unk) lattice edges at (min_score - kUnkPenalty)
                // per byte, so a byte path only wins when NO vocab piece covers the span. Byte
                // pieces' own proto score is 0; this penalty is what makes them lose to real pieces.
                _byteEdgeScore = minScore - UNK_PENALTY;
                IsReady = true;
            }

            /// <summary>Full text -> token ids, reproducing model.flow_lm.conditioner.prepare(
            /// prepare_text_prompt(text, pad, removeSemis)). English defaults: pad=false, removeSemis=false.</summary>
            public int[] Encode(string text, bool padWithSpacesForShortInputs = false, bool removeSemicolons = false)
            {
                string prepped = PrepareTextPrompt(text, padWithSpacesForShortInputs, removeSemicolons);
                return EncodeRaw(prepped);
            }

            /// <summary>prepare_text_prompt (tts_model.py) — text preprocessing before tokenization.</summary>
            public static string PrepareTextPrompt(string text, bool pad, bool removeSemicolons)
            {
                text = (text ?? "").Trim();
                if (text.Length == 0) throw new ArgumentException("Text prompt cannot be empty");
                text = text.Replace("\n", " ").Replace("\r", " ").Replace("  ", " ");
                if (removeSemicolons) text = text.Replace(";", ",");
                // uppercase first letter (char.ToUpper mirrors Python str.isupper()/upper() for the
                // common ASCII/Latin case; SentencePiece has no case normalization so this only
                // changes which piece matches).
                if (!char.IsUpper(text[0])) text = char.ToUpper(text[0]) + text.Substring(1);
                // ensure trailing punctuation
                if (char.IsLetterOrDigit(text[text.Length - 1])) text += ".";
                if (pad && CountWords(text) < 5) text = new string(' ', 8) + text;
                return text;
            }

            static int CountWords(string s)
            {
                int n = 0; bool inW = false;
                foreach (char c in s)
                {
                    bool ws = char.IsWhiteSpace(c);
                    if (!ws && !inW) n++;
                    inW = !ws;
                }
                return n;
            }

            /// <summary>SentencePiece encode of an ALREADY-prepared string (no prepare_text_prompt):
            /// normalize (dummy prefix + space->U+2581) then Unigram Viterbi + byte-fallback.</summary>
            public int[] EncodeRaw(string text)
            {
                // normalize: add_dummy_prefix + replace ' ' with the SentencePiece space marker.
                var sb = new StringBuilder(text.Length + 1);
                sb.Append(SPACE);
                foreach (char c in text) sb.Append(c == ' ' ? SPACE : c);
                string norm = sb.ToString();

                int n = norm.Length;
                // Viterbi over UTF-16 char positions. best[i] = (max total score to reach i,
                // start-of-last-piece, id-of-last-piece OR -1 for a byte-fallback char at [j..i)).
                var bestScore = new float[n + 1];
                var backPos = new int[n + 1];
                var backId = new int[n + 1];      // >=0 vocab piece; -1 = byte-fallback char span
                for (int i = 1; i <= n; i++) bestScore[i] = float.NegativeInfinity;
                bestScore[0] = 0f;

                for (int i = 0; i < n; i++)
                {
                    if (float.IsNegativeInfinity(bestScore[i])) continue;
                    // (a) all vocab pieces matching starting at i (surrogate-safe substring lengths)
                    int maxLen = Math.Min(_maxPieceLen, n - i);
                    for (int len = 1; len <= maxLen; len++)
                    {
                        // avoid splitting a surrogate pair boundary
                        if (i + len < n && char.IsLowSurrogate(norm[i + len]) && char.IsHighSurrogate(norm[i + len - 1]))
                            continue;
                        string sub = norm.Substring(i, len);
                        if (_pieceToId.TryGetValue(sub, out int pid))
                        {
                            float ns = bestScore[i] + _pieces[pid].score;
                            if (ns > bestScore[i + len]) { bestScore[i + len] = ns; backPos[i + len] = i; backId[i + len] = pid; }
                        }
                    }
                    // (b) byte-fallback for the single source codepoint at i: one lattice edge that
                    // the backtrace decodes into that codepoint's UTF-8 byte pieces. Scored below so
                    // it only wins when no vocab piece covers the span (matches SentencePiece).
                    int clen = (char.IsHighSurrogate(norm[i]) && i + 1 < n && char.IsLowSurrogate(norm[i + 1])) ? 2 : 1;
                    // byte-fallback edge for this whole codepoint: (min_score - kUnkPenalty) per UTF-8
                    // byte, so it only wins when no vocab piece covers the span (matches SP exactly).
                    int nBytes = Encoding.UTF8.GetByteCount(norm.Substring(i, clen));
                    float bs = bestScore[i] + _byteEdgeScore * nBytes;
                    if (bs > bestScore[i + clen]) { bestScore[i + clen] = bs; backPos[i + clen] = i; backId[i + clen] = -1; }
                }

                // backtrace -> ids (reverse, then flip)
                var rev = new List<int>();
                int p = n;
                while (p > 0)
                {
                    int j = backPos[p];
                    int id = backId[p];
                    if (id >= 0) rev.Add(id);
                    else
                    {
                        // byte-fallback span [j..p): emit UTF-8 bytes of that codepoint, REVERSED
                        // (we reverse the whole list at the end, so push bytes in reverse here).
                        byte[] bytes = Encoding.UTF8.GetBytes(norm.Substring(j, p - j));
                        for (int b = bytes.Length - 1; b >= 0; b--) rev.Add(_byteBase + bytes[b]);
                    }
                    p = j;
                }
                rev.Reverse();
                return rev.ToArray();
            }
        }
    }
}
