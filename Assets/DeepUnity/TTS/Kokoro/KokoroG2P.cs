using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // English G2P for the Kokoro-82M port — a C# port of misaki.en.G2P (Apache-2.0,
        // https://github.com/hexgrad/misaki, en.py @ v0.9.4), the exact G2P Kokoro was trained
        // against. See ../Kokoro/G2P_PROPOSAL.md (approved option (b) v1) and SPEC.md §8.
        //
        // Data: KokoroG2P.gold.tsv / KokoroG2P.silver.tsv next to this file, produced by
        // validation/export_g2p_data.py from misaki us_gold/us_silver (90,201 + 93,361 entries,
        // 790 tag-keyed heteronyms). grow_dictionary (capitalized/lowercase twins) is applied at
        // load, mirroring en.py Lexicon.__init__.
        //
        // Faithful ports: Lexicon (special cases, lookup, get_NNP, stem_s/_ed/_ing, numbers),
        // apply_stress/restress, subtokenize regex, retokenize, resolve_tokens, merge_tokens,
        // the right-to-left context pass (future_vowel/future_to), final ɾ→T / ʔ→t post-map.
        //
        // Deviations from the reference (marked // DEV: at the sites):
        //   1. Tokenizer+tagger replace spaCy (rule-based; heteronyms only need coarse tags —
        //      see G2P_PROPOSAL.md analysis: 790 heteronyms, keys DEFAULT/NOUN/VERB/ADJ + rare
        //      VBD/VBN/VBP). Mistags yield a *valid* variant, not garbage.
        //   2. OOD fallback = get_NNP spell-out for capitalized unknowns, else token skipped
        //      (misaki's unk='' behavior in the Kokoro pipeline). No espeak (GPL), no BART (v2).
        //   3. Markdown link features ([word](/ph/)) not supported (game text never has them).
        //   4. num2words subset: cardinal/ordinal/year/decimal/currency for |n| < 10^15.
        //
        // Validation gate (CHECKLIST.md B1): Phonemize(text) must match validation/dump/
        // t{0,1,2}_phonemes.txt byte-for-byte.
        public class KokoroG2P
        {
            // ---------------------------------------------------------------- constants (en.py)
            const string DIPHTHONGS = "AIOQWYʤʧ";
            const string VOWELS = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
            const string CONSONANTS = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
            const string PUNCTS = ";:,.!?—…\"“”";
            const string NON_QUOTE_PUNCTS = ";:,.!?—…";
            const string SUBTOKEN_JUNKS = "',-._‘’/";
            const char PRIMARY = 'ˈ', SECONDARY = 'ˌ';
            const string US_TAUS = "AIOWYiuæɑəɛɪɹʊʌ";
            static readonly HashSet<string> ORDINALS = new HashSet<string> { "st", "nd", "rd", "th" };
            static readonly Dictionary<string, string> SYMBOLS = new Dictionary<string, string>
                { { "%", "percent" }, { "&", "and" }, { "+", "plus" }, { "@", "at" } };
            static readonly Dictionary<string, (string unit, string sub)> CURRENCIES =
                new Dictionary<string, (string, string)>
                { { "$", ("dollar", "cent") }, { "£", ("pound", "pence") }, { "€", ("euro", "cent") } };
            static readonly HashSet<string> PUNCT_TAGS = new HashSet<string>
                { ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP" };
            static readonly Dictionary<string, string> PUNCT_TAG_PHONEMES = new Dictionary<string, string>
                { { "-LRB-", "(" }, { "-RRB-", ")" }, { "``", "“" }, { "\"\"", "”" }, { "''", "”" } };

            // misaki subtokenize regex (en.py), .NET syntax-compatible as-is
            static readonly Regex SubtokenRegex = new Regex(
                @"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|" +
                @"\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$",
                RegexOptions.Compiled);

            // ---------------------------------------------------------------- token
            class Tok
            {
                public string text = "", tag = "", whitespace = "", currency = null, numFlags = "";
                public string phonemes = null;      // null = unresolved
                public bool isHead = true, prespace = false;
                public float? stress = null;        // feature stress (unused without link syntax)
                public int rating = 0;

                public Tok Clone() => (Tok)MemberwiseClone();
            }

            struct Ctx { public bool? futureVowel; public bool futureTo; }

            // ---------------------------------------------------------------- lexicon data
            Dictionary<string, object> golds;   // string | Dictionary<string,string>
            Dictionary<string, object> silvers;
            volatile bool ready;
            public bool IsReady => ready;

            public KokoroG2P(string dataPathBase = "Assets/DeepUnity/TTS/Kokoro/KokoroG2P")
            {
                // 4.6 MB of TSV (~200k lexicon entries): parsing is heavy string work — doing it
                // here synchronously stalled the game for seconds the moment a prefetch zone
                // built the TTS. Parse on the pool; every synthesis path gates on IsReady.
                System.Threading.Tasks.Task.Run(() =>
                {
                    try
                    {
                        golds = Grow(LoadTsv(dataPathBase + ".gold.tsv"));
                        silvers = Grow(LoadTsv(dataPathBase + ".silver.tsv"));
                        ready = true;
                    }
                    catch (Exception e)
                    {
                        ConsoleMessage.Error($"KokoroG2P lexicon load failed: {e.Message}");
                    }
                });
            }

            static Dictionary<string, object> LoadTsv(string path)
            {
                if (!File.Exists(path))
                    throw new FileNotFoundException(
                        $"KokoroG2P data missing: '{path}'. Generate with validation/export_g2p_data.py.");
                var d = new Dictionary<string, object>(200_000, StringComparer.Ordinal);
                foreach (string line in File.ReadLines(path))
                {
                    if (line.Length == 0) continue;
                    string[] p = line.Split('\t');
                    if (p.Length == 2 && p[1].IndexOf('=') < 0) { d[p[0]] = p[1]; continue; }
                    var het = new Dictionary<string, string>(p.Length - 1, StringComparer.Ordinal);
                    for (int i = 1; i < p.Length; i++)
                    {
                        int eq = p[i].IndexOf('=');
                        string ps = p[i].Substring(eq + 1);
                        het[p[i].Substring(0, eq)] = ps == "\\N" ? null : ps;
                    }
                    d[p[0]] = het;
                }
                return d;
            }

            static string PyCapitalize(string s) =>
                s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s.Substring(1).ToLowerInvariant();

            static Dictionary<string, object> Grow(Dictionary<string, object> d)
            {
                var e = new Dictionary<string, object>(d.Count, StringComparer.Ordinal);
                foreach (var kv in d)
                {
                    string k = kv.Key;
                    if (k.Length < 2) continue;
                    if (k == k.ToLowerInvariant())
                    { if (k != PyCapitalize(k)) e[PyCapitalize(k)] = kv.Value; }
                    else if (k == PyCapitalize(k.ToLowerInvariant()))
                        e[k.ToLowerInvariant()] = kv.Value;
                }
                foreach (var kv in d) e[kv.Key] = kv.Value;   // originals win
                return e;
            }

            // ---------------------------------------------------------------- public API
            /// <summary>text -> Kokoro phoneme string (misaki-en compatible, American).</summary>
            public string Phonemize(string text)
            {
                List<Tok> tokens = Tokenize(text);
                Tag(tokens);
                List<object> words = Retokenize(tokens);   // Tok | List<Tok>

                var ctx = new Ctx();
                for (int i = words.Count - 1; i >= 0; i--)
                {
                    if (words[i] is Tok w)
                    {
                        if (w.phonemes == null)
                            (w.phonemes, w.rating) = LexLookupToken(w, ctx);
                        if (w.phonemes == null && IsCapAlpha(w.text))                 // DEV: v1 fallback
                            (w.phonemes, w.rating) = GetNNP(w.text);
                        ctx = TokenContext(ctx, w.phonemes, w);
                        continue;
                    }
                    var g = (List<Tok>)words[i];
                    ResolveGroup(g, ref ctx);
                }

                var final = new List<Tok>(words.Count);
                foreach (object o in words)
                    final.Add(o is Tok t ? t : MergeTokens((List<Tok>)o, ""));

                var sb = new StringBuilder();
                foreach (Tok tk in final)
                {
                    string ps = tk.phonemes ?? "";
                    ps = ps.Replace('ɾ', 'T').Replace('ʔ', 't');   // misaki v1 American post-map
                    sb.Append(ps).Append(tk.whitespace);
                }
                return sb.ToString().Trim();
            }

            static bool IsCapAlpha(string s) =>
                s.Length > 0 && char.IsUpper(s[0]) && s.All(char.IsLetter);

            // ---------------------------------------------------------------- tokenizer  // DEV
            // Replaces spaCy tokenization: whitespace split + punct peeling + contraction split.
            static readonly string[] Contractions = { "n't", "'s", "'re", "'ve", "'ll", "'d", "'m" };
            const string OpenPunct = "(\"“‘[";
            const string ClosePunct = ")\"”’]!?.,;:…%";

            static List<Tok> Tokenize(string text)
            {
                var toks = new List<Tok>();
                var chunks = Regex.Matches(text.TrimStart(), @"\S+(\s*)");
                foreach (Match m in chunks)
                {
                    string chunk = m.Value.TrimEnd();
                    string ws = m.Groups[1].Value.Length > 0 ? " " : "";
                    var parts = new List<string>();
                    // peel opening punct / currency
                    while (chunk.Length > 1 && (OpenPunct.IndexOf(chunk[0]) >= 0 ||
                                                CURRENCIES.ContainsKey(chunk[0].ToString()) ||
                                                chunk[0] == '#'))
                    { parts.Add(chunk[0].ToString()); chunk = chunk.Substring(1); }
                    // peel closing punct (keep number-internal . ,)
                    var tail = new List<string>();
                    while (chunk.Length > 1)
                    {
                        char last = chunk[chunk.Length - 1];
                        if (chunk.EndsWith("...")) { tail.Insert(0, "..."); chunk = chunk.Substring(0, chunk.Length - 3); continue; }
                        if (ClosePunct.IndexOf(last) < 0 && last != '—' && last != '…') break;
                        if (last == '.' && Regex.IsMatch(chunk, @"^[A-Za-z](\.[A-Za-z])+\.$"))
                            break;                                     // dotted abbreviation: a.m., U.S.
                        // NOTE: number-internal . , ("3.14", "1,000") never reach here — the loop
                        // only ever peels the FINAL char, and those chunks end in a digit.
                        tail.Insert(0, last.ToString()); chunk = chunk.Substring(0, chunk.Length - 1);
                    }
                    // em-dash splits mid-chunk ("night—cold")
                    foreach (string piece in Regex.Split(chunk, "(—)").Where(s => s.Length > 0))
                    {
                        string w = piece;
                        // contraction split (don't -> do + n't; it's -> it + 's)
                        string suf = null;
                        if (w.Length > 2 && w != "n't")
                            foreach (string c in Contractions)
                                if (w.EndsWith(c, StringComparison.OrdinalIgnoreCase) && w.Length > c.Length)
                                { suf = w.Substring(w.Length - c.Length); w = w.Substring(0, w.Length - c.Length); break; }
                        parts.Add(w);
                        if (suf != null) parts.Add(suf);
                    }
                    parts.AddRange(tail);
                    for (int i = 0; i < parts.Count; i++)
                        toks.Add(new Tok { text = parts[i], whitespace = i == parts.Count - 1 ? ws : "" });
                }
                return toks;
            }

            // ---------------------------------------------------------------- tagger  // DEV
            static readonly HashSet<string> DTs = new HashSet<string> { "the", "a", "an", "this", "that", "these", "those", "each", "every", "no", "some", "any" };
            static readonly HashSet<string> INs = new HashSet<string> { "in", "of", "on", "at", "by", "with", "from", "as", "for", "into", "over", "under", "after", "before", "between", "through", "during", "against", "about", "than", "if", "because", "while", "since", "until", "unless", "vs", "vs." };
            static readonly HashSet<string> PRPs = new HashSet<string> { "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them" };
            static readonly HashSet<string> PRPDs = new HashSet<string> { "my", "your", "his", "its", "our", "their" };
            static readonly HashSet<string> CCs = new HashSet<string> { "and", "or", "but", "nor", "yet", "so" };
            static readonly HashSet<string> MDs = new HashSet<string> { "will", "would", "can", "could", "shall", "should", "may", "might", "must" };
            static readonly HashSet<string> BeHave = new HashSet<string> { "is", "are", "was", "were", "been", "be", "am", "has", "have", "had", "being" };
            static readonly HashSet<string> IrregularPast = new HashSet<string> { "said", "told", "saw", "went", "came", "took", "made", "found", "gave", "knew", "thought", "felt", "got", "heard", "kept", "began", "brought", "stood", "held", "ran", "met", "paid", "sat", "spoke", "led", "wrote", "drew", "drove", "ate", "fell", "grew", "threw", "wore", "won", "sent", "built", "spent", "lost", "meant", "caught", "taught", "bought", "fought", "sought", "sold" };

            static void Tag(List<Tok> toks)
            {
                bool quoteOpen = false;
                for (int i = 0; i < toks.Count; i++)
                {
                    Tok t = toks[i];
                    string w = t.text, lo = w.ToLowerInvariant();
                    string prevTag = i > 0 ? toks[i - 1].tag : "";
                    string prevLo = i > 0 ? toks[i - 1].text.ToLowerInvariant() : "";
                    if (w == "\"")                                          // straight quote: parity
                    { t.tag = quoteOpen ? "''" : "``"; quoteOpen = !quoteOpen; continue; }
                    if (SYMBOLS.ContainsKey(w))
                        t.tag = "NN";                                       // %, &, +, @ -> lexicon symbols
                    else if (w.Length > 0 && !w.Any(char.IsLetterOrDigit))
                        t.tag = w == "$" || w == "£" || w == "€" ? "$"
                              : w == "#" ? "#"
                              : w == "(" || w == "[" ? "-LRB-"
                              : w == ")" || w == "]" ? "-RRB-"
                              : w == "“" ? "``"
                              : w == "”" ? "''"
                              : w == "," ? ","
                              : w == "." || w == "!" || w == "?" || w == "…" || w == "..." ? "."
                              : ":";                                       // ; : — - –
                    else if (w.Any(char.IsDigit) && Regex.IsMatch(w, @"^-?[\d,.]+$"))
                        t.tag = "CD";
                    else if (lo == "to") t.tag = "TO";
                    else if (lo == "that" && (prevTag.StartsWith("VB") || prevTag == "MD"))
                        t.tag = "IN";                                       // complementizer "said that ..."
                    else if (IrregularPast.Contains(lo)) t.tag = "VBD";
                    else if (w == "I") t.tag = "PRP";
                    else if (DTs.Contains(lo) && w != "May") t.tag = "DT";
                    else if (INs.Contains(lo)) t.tag = "IN";
                    else if (PRPs.Contains(lo)) t.tag = "PRP";
                    else if (PRPDs.Contains(lo)) t.tag = "PRP$";
                    else if (CCs.Contains(lo)) t.tag = "CC";
                    else if (MDs.Contains(lo)) t.tag = "MD";
                    else if (BeHave.Contains(lo)) t.tag = "VBP";
                    else if (lo == "n't" || lo == "not") t.tag = "RB";
                    else if (lo.EndsWith("ly") && lo.Length > 4) t.tag = "RB";
                    else if (prevTag == "MD" || prevTag == "TO") t.tag = "VB";
                    else if (prevTag == "PRP") t.tag = "VBP";
                    else if (prevTag == "VBP" && BeHave.Contains(prevLo)) t.tag = "VBN";
                    else if (prevTag == "NNS") t.tag = "VBD";              // "researchers read"
                    else if (lo.EndsWith("ed") && lo.Length > 3 && (prevTag == "NN" || prevTag == "NNS" || prevTag == "NNP"))
                        t.tag = "VBD";                                     // "merchant leaned"
                    else if ((lo.EndsWith("er") || lo.EndsWith("est")) && lo.Length > 4 &&
                             (prevTag == "VBD" || prevTag == "VBN" || prevTag == "VB" || prevTag == "VBP"))
                        t.tag = "RBR";                                     // "leaned closer" -> ADV parent
                    else if (i > 0 && char.IsUpper(w.FirstOrDefault()) && !(prevTag == "." || prevTag == ""))
                        t.tag = "NNP";
                    else if (lo.EndsWith("ing") && lo.Length > 4 && (prevTag == "VBP" || prevTag == "IN"))
                        t.tag = "VBG";
                    else if (lo.EndsWith("s") && !lo.EndsWith("ss") && lo.Length > 3 &&
                             (prevTag == "DT" || prevTag == "JJ" || prevTag == "PRP$" || prevTag == "" || prevTag == "IN" || prevTag == "CD"))
                        t.tag = "NNS";
                    else t.tag = "NN";
                }
            }

            // ---------------------------------------------------------------- retokenize (en.py)
            List<object> Retokenize(List<Tok> tokens)
            {
                var words = new List<object>();
                string currency = null;
                for (int i = 0; i < tokens.Count; i++)
                {
                    Tok token = tokens[i];
                    List<Tok> tks;
                    if (token.phonemes == null)
                    {
                        tks = SubtokenRegex.Matches(token.text).Cast<Match>().Select(m =>
                        {
                            Tok c = token.Clone();
                            c.text = m.Value; c.whitespace = ""; c.isHead = true; c.prespace = false;
                            return c;
                        }).ToList();
                        if (tks.Count == 0) tks.Add(token.Clone());
                    }
                    else tks = new List<Tok> { token };
                    tks[tks.Count - 1].whitespace = token.whitespace;

                    for (int j = 0; j < tks.Count; j++)
                    {
                        Tok tk = tks[j];
                        if (tk.phonemes != null) { }
                        else if (tk.tag == "$" && CURRENCIES.ContainsKey(tk.text))
                        { currency = tk.text; tk.phonemes = ""; tk.rating = 4; }
                        else if (tk.tag == ":" && (tk.text == "-" || tk.text == "–"))
                        { tk.phonemes = "—"; tk.rating = 3; }
                        else if (PUNCT_TAGS.Contains(tk.tag) && !tk.text.All(c => char.IsLetter(c) && c < 128))
                        {
                            tk.phonemes = PUNCT_TAG_PHONEMES.TryGetValue(tk.tag, out string pp)
                                ? pp : new string(tk.text.Where(c => PUNCTS.IndexOf(c) >= 0).ToArray());
                            tk.rating = 4;
                        }
                        else if (currency != null)
                        {
                            if (tk.tag != "CD") currency = null;
                            else if (j + 1 == tks.Count && (i + 1 == tokens.Count || tokens[i + 1].tag != "CD"))
                                tk.currency = currency;
                        }
                        else if (j > 0 && j < tks.Count - 1 && tk.text == "2" &&
                                 char.IsLetter(tks[j - 1].text.LastOrDefault()) && char.IsLetter(tks[j + 1].text.FirstOrDefault()))
                            tk.text = "to";   // DEV: alias applied in place ("B2B")

                        if (tk.phonemes != null)
                            words.Add(tk);
                        else if (words.Count > 0 && words[words.Count - 1] is List<Tok> prev && prev[prev.Count - 1].whitespace == "")
                        { tk.isHead = false; prev.Add(tk); }
                        else
                            words.Add(tk.whitespace == "" ? (object)new List<Tok> { tk } : tk);
                    }
                }
                for (int i = 0; i < words.Count; i++)
                    if (words[i] is List<Tok> l && l.Count == 1) words[i] = l[0];
                return words;
            }

            // ---------------------------------------------------------------- group resolution (en.py __call__)
            void ResolveGroup(List<Tok> w, ref Ctx ctx)
            {
                int left = 0, right = w.Count;
                bool shouldFallback = false;
                while (left < right)
                {
                    Tok merged = w.Skip(left).Take(right - left).Any(t => t.phonemes != null)
                        ? null : MergeTokens(w.GetRange(left, right - left), null);
                    (string ps, int rating) = merged == null ? (null, 0) : LexLookupToken(merged, ctx);
                    if (ps != null)
                    {
                        w[left].phonemes = ps; w[left].rating = rating;
                        for (int x = left + 1; x < right; x++) { w[x].phonemes = ""; w[x].rating = rating; }
                        ctx = TokenContext(ctx, ps, merged);
                        right = left; left = 0;
                    }
                    else if (left + 1 < right) left++;
                    else
                    {
                        right--;
                        Tok tk = w[right];
                        if (tk.phonemes == null)
                        {
                            if (tk.text.All(c => SUBTOKEN_JUNKS.IndexOf(c) >= 0)) { tk.phonemes = ""; tk.rating = 3; }
                            else { shouldFallback = true; break; }
                        }
                        left = 0;
                    }
                }
                if (shouldFallback)
                {
                    Tok tk = MergeTokens(w, null);
                    (w[0].phonemes, w[0].rating) = IsCapAlpha(tk.text) ? GetNNP(tk.text) : ("", 1); // DEV
                    for (int x = 1; x < w.Count; x++) { w[x].phonemes = ""; w[x].rating = w[0].rating; }
                }
                else ResolveTokens(w);
            }

            static Ctx TokenContext(Ctx ctx, string ps, Tok token)
            {
                bool? vowel = ctx.futureVowel;
                if (!string.IsNullOrEmpty(ps))
                    foreach (char c in ps)
                    {
                        bool inV = VOWELS.IndexOf(c) >= 0, inC = CONSONANTS.IndexOf(c) >= 0,
                             inP = NON_QUOTE_PUNCTS.IndexOf(c) >= 0;
                        if (inV || inC || inP) { vowel = inP ? (bool?)null : inV; break; }
                    }
                bool futureTo = token.text == "to" || token.text == "To" ||
                                (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"));
                return new Ctx { futureVowel = vowel, futureTo = futureTo };
            }

            // ---------------------------------------------------------------- merge/resolve (en.py)
            static Tok MergeTokens(List<Tok> tokens, string unk)
            {
                var stresses = tokens.Where(t => t.stress.HasValue).Select(t => t.stress.Value).Distinct().ToList();
                var currencies = tokens.Where(t => t.currency != null).Select(t => t.currency).Distinct().ToList();
                string phonemes = null;
                if (unk != null)
                {
                    var sb = new StringBuilder();
                    foreach (Tok tk in tokens)
                    {
                        if (tk.prespace && sb.Length > 0 && !char.IsWhiteSpace(sb[sb.Length - 1]) &&
                            !string.IsNullOrEmpty(tk.phonemes))
                            sb.Append(' ');
                        sb.Append(tk.phonemes == null ? unk : tk.phonemes);
                    }
                    phonemes = sb.ToString();
                }
                var text = new StringBuilder();
                for (int i = 0; i < tokens.Count - 1; i++) text.Append(tokens[i].text).Append(tokens[i].whitespace);
                text.Append(tokens[tokens.Count - 1].text);
                Tok tagTok = tokens.OrderByDescending(t => t.text.Sum(c => c == char.ToLowerInvariant(c) ? 1 : 2)).First();
                return new Tok
                {
                    text = text.ToString(),
                    tag = tagTok.tag,
                    whitespace = tokens[tokens.Count - 1].whitespace,
                    phonemes = phonemes,
                    stress = stresses.Count == 1 ? stresses[0] : (float?)null,
                    currency = currencies.Count > 0 ? currencies.OrderByDescending(c => c).First() : null,
                    numFlags = new string(tokens.SelectMany(t => t.numFlags).Distinct().OrderBy(c => c).ToArray()),
                    isHead = tokens[0].isHead,
                    prespace = tokens[0].prespace,
                };
            }

            static int StressWeight(string ps) =>
                string.IsNullOrEmpty(ps) ? 0 : ps.Sum(c => DIPHTHONGS.IndexOf(c) >= 0 ? 2 : 1);

            static void ResolveTokens(List<Tok> tokens)
            {
                var text = new StringBuilder();
                for (int i = 0; i < tokens.Count - 1; i++) text.Append(tokens[i].text).Append(tokens[i].whitespace);
                text.Append(tokens[tokens.Count - 1].text);
                string full = text.ToString();
                var classes = new HashSet<int>(full.Where(c => SUBTOKEN_JUNKS.IndexOf(c) < 0)
                    .Select(c => char.IsLetter(c) ? 0 : char.IsDigit(c) ? 1 : 2));
                bool prespace = full.Contains(' ') || full.Contains('/') || classes.Count > 1;
                for (int i = 0; i < tokens.Count; i++)
                {
                    Tok tk = tokens[i];
                    if (tk.phonemes == null)
                    {
                        if (i == tokens.Count - 1 && tk.text.Length == 1 && NON_QUOTE_PUNCTS.IndexOf(tk.text[0]) >= 0)
                        { tk.phonemes = tk.text; tk.rating = 3; }
                        else if (tk.text.All(c => SUBTOKEN_JUNKS.IndexOf(c) >= 0))
                        { tk.phonemes = ""; tk.rating = 3; }
                    }
                    else if (i > 0) tk.prespace = prespace;
                }
                if (prespace) return;
                var idx = tokens.Select((tk, i) => (tk, i)).Where(p => !string.IsNullOrEmpty(p.tk.phonemes))
                    .Select(p => (hasP: p.tk.phonemes.Contains(PRIMARY) ? 1 : 0,
                                  w: StressWeight(p.tk.phonemes), i: p.i)).ToList();
                if (idx.Count == 2 && tokens[idx[0].i].text.Length == 1)
                {
                    int i2 = idx[1].i;
                    tokens[i2].phonemes = ApplyStress(tokens[i2].phonemes, -0.5f);
                    return;
                }
                if (idx.Count < 2 || idx.Sum(t => t.hasP) <= (idx.Count + 1) / 2) return;
                foreach (var t in idx.OrderBy(t => t.hasP).ThenBy(t => t.w).ThenBy(t => t.i).Take(idx.Count / 2))
                    tokens[t.i].phonemes = ApplyStress(tokens[t.i].phonemes, -0.5f);
            }

            // ---------------------------------------------------------------- apply_stress (en.py)
            static string Restress(string ps)
            {
                var pos = ps.Select((c, i) => ((float)i, c)).ToList();
                for (int i = 0; i < pos.Count; i++)
                    if (pos[i].c == PRIMARY || pos[i].c == SECONDARY)
                        for (int j = i; j < pos.Count; j++)
                            if (VOWELS.IndexOf(ps[j]) >= 0) { pos[i] = (j - 0.5f, pos[i].c); break; }
                return new string(pos.OrderBy(p => p.Item1).Select(p => p.c).ToArray());
            }

            static string ApplyStress(string ps, float? stress)
            {
                if (stress == null || ps == null) return ps;
                float s = stress.Value;
                if (s < -1) return ps.Replace(PRIMARY.ToString(), "").Replace(SECONDARY.ToString(), "");
                if (s == -1 || ((s == 0 || s == -0.5f) && ps.Contains(PRIMARY)))
                    return ps.Replace(SECONDARY.ToString(), "").Replace(PRIMARY, SECONDARY);
                if ((s == 0 || s == 0.5f || s == 1) && !ps.Contains(PRIMARY) && !ps.Contains(SECONDARY))
                    return ps.Any(c => VOWELS.IndexOf(c) >= 0) ? Restress(SECONDARY + ps) : ps;
                if (s >= 1 && !ps.Contains(PRIMARY) && ps.Contains(SECONDARY))
                    return ps.Replace(SECONDARY, PRIMARY);
                if (s > 1 && !ps.Contains(PRIMARY) && !ps.Contains(SECONDARY))
                    return ps.Any(c => VOWELS.IndexOf(c) >= 0) ? Restress(PRIMARY + ps) : ps;
                return ps;
            }

            // ---------------------------------------------------------------- Lexicon (en.py)
            (string, int) LexLookupToken(Tok tk, Ctx ctx)
            {
                string word = tk.text.Replace('‘', '\'').Replace('’', '\'');
                word = word.Normalize(NormalizationForm.FormKC);
                float? capStress = word == word.ToLowerInvariant() ? (float?)null
                                 : (word == word.ToUpperInvariant() ? 2f : 0.5f);
                (string ps, int rating) = GetWord(word, tk.tag, capStress, ctx);
                if (ps != null)
                    return (ApplyStress(AppendCurrency(ps, tk.currency), tk.stress), rating);
                var tm = Regex.Match(word, @"^(\d{1,2}):(\d{2})$");         // DEV: clock time "9:30"
                if (tm.Success)                                             // (misaki resolves via espeak)
                {
                    var (hh, r1) = GetNumber(tm.Groups[1].Value, null, true, "");
                    var (mm, r2) = GetNumber(tm.Groups[2].Value, null, true, "");
                    if (hh != null && mm != null) return (hh + ":" + mm, Math.Min(r1, r2));
                }
                if (IsNumber(word, tk.isHead))
                {
                    (ps, rating) = GetNumber(word, tk.currency, tk.isHead, tk.numFlags);
                    return (ApplyStress(ps, tk.stress), rating);
                }
                if (!word.All(c => c == '\'' || c == '-' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
                    return (null, 0);
                return (null, 0);
            }

            (string, int) GetWord(string word, string tag, float? stress, Ctx ctx)
            {
                (string ps, int rating) = GetSpecialCase(word, tag, stress, ctx);
                if (ps != null) return (ps, rating);
                string wl = word.ToLowerInvariant();
                if (word.Length > 1 && word.Replace("'", "").All(char.IsLetter) && word != wl &&
                    (tag != "NNP" || word.Length > 7) && !golds.ContainsKey(word) && !silvers.ContainsKey(word) &&
                    (word == word.ToUpperInvariant() || word.Substring(1) == word.Substring(1).ToLowerInvariant()) &&
                    (golds.ContainsKey(wl) || silvers.ContainsKey(wl) ||
                     StemS(wl, tag, stress, ctx).Item1 != null || StemEd(wl, tag, stress, ctx).Item1 != null ||
                     StemIng(wl, tag, stress, ctx).Item1 != null))
                    word = wl;
                if (IsKnown(word, tag)) return Lookup(word, tag, stress, ctx);
                if (word.EndsWith("s'") && IsKnown(word.Substring(0, word.Length - 2) + "'s", tag))
                    return Lookup(word.Substring(0, word.Length - 2) + "'s", tag, stress, ctx);
                if (word.EndsWith("'") && IsKnown(word.Substring(0, word.Length - 1), tag))
                    return Lookup(word.Substring(0, word.Length - 1), tag, stress, ctx);
                var s = StemS(word, tag, stress, ctx); if (s.Item1 != null) return s;
                var e = StemEd(word, tag, stress, ctx); if (e.Item1 != null) return e;
                var g = StemIng(word, tag, stress ?? 0.5f, ctx); if (g.Item1 != null) return g;
                return (null, 0);
            }

            (string, int) GetSpecialCase(string word, string tag, float? stress, Ctx ctx)
            {
                if (SYMBOLS.TryGetValue(word, out string sym)) return Lookup(sym, null, null, ctx);
                if (word.Trim('.').Contains('.') && word.Replace(".", "").All(char.IsLetter) &&
                    word.Split('.').Max(p => p.Length) < 3)
                    return GetNNP(word);
                if (word == "a" || word == "A")
                    return (tag == "DT" ? "ɐ" : "ˈA", 4);
                if (word == "am" || word == "Am" || word == "AM")
                {
                    if (tag.StartsWith("NN")) return GetNNP(word);
                    if (ctx.futureVowel == null || word != "am" || (stress.HasValue && stress > 0))
                        return ((string)golds["am"], 4);
                    return ("ɐm", 4);
                }
                if (word == "an" || word == "An" || word == "AN")
                {
                    if (word == "AN" && tag.StartsWith("NN")) return GetNNP(word);
                    return ("ɐn", 4);
                }
                if (word == "I" && tag == "PRP") return (SECONDARY + "I", 4);
                if ((word == "by" || word == "By" || word == "BY") && ParentTag(tag) == "ADV")
                    return ("bˈI", 4);
                if (word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")))
                    return (ctx.futureVowel == null ? (string)golds["to"]
                          : ctx.futureVowel == false ? "tə" : "tʊ", 4);
                if (word == "in" || word == "In" || (word == "IN" && tag != "NNP"))
                    return ((ctx.futureVowel == null || tag != "IN" ? PRIMARY.ToString() : "") + "ɪn", 4);
                if (word == "the" || word == "The" || (word == "THE" && tag == "DT"))
                    return (ctx.futureVowel == true ? "ði" : "ðə", 4);
                if (tag == "IN" && Regex.IsMatch(word, @"(?i)^vs\.?$"))
                    return Lookup("versus", null, null, ctx);
                if (word == "used" || word == "Used" || word == "USED")
                {
                    var used = (Dictionary<string, string>)golds["used"];
                    if ((tag == "VBD" || tag == "JJ") && ctx.futureTo) return (used["VBD"], 4);
                    return (used["DEFAULT"], 4);
                }
                return (null, 0);
            }

            static string ParentTag(string tag) =>
                tag == null ? null
                : tag.StartsWith("VB") ? "VERB"
                : tag.StartsWith("NN") ? "NOUN"
                : tag.StartsWith("ADV") || tag.StartsWith("RB") ? "ADV"
                : tag.StartsWith("ADJ") || tag.StartsWith("JJ") ? "ADJ" : tag;

            bool IsKnown(string word, string tag)
            {
                if (golds.ContainsKey(word) || SYMBOLS.ContainsKey(word) || silvers.ContainsKey(word))
                    return true;
                if (!word.All(char.IsLetter) ||
                    !word.All(c => c == '\'' || c == '-' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
                    return false;
                if (word.Length == 1) return true;
                if (word == word.ToUpperInvariant() && golds.ContainsKey(word.ToLowerInvariant())) return true;
                return word.Substring(1) == word.Substring(1).ToUpperInvariant();
            }

            (string, int) Lookup(string word, string tag, float? stress, Ctx? ctx)
            {
                bool? isNNP = null;
                if (word == word.ToUpperInvariant() && !golds.ContainsKey(word))
                {
                    word = word.ToLowerInvariant();
                    isNNP = tag == "NNP";
                }
                object entry = golds.TryGetValue(word, out object gv) ? gv : null;
                int rating = 4;
                if (entry == null && isNNP != true)
                { entry = silvers.TryGetValue(word, out object sv) ? sv : null; rating = 3; }
                string ps = null;
                if (entry is Dictionary<string, string> het)
                {
                    if (ctx.HasValue && ctx.Value.futureVowel == null && het.ContainsKey("None"))
                        tag = "None";
                    else if (tag == null || !het.ContainsKey(tag))
                        tag = ParentTag(tag);
                    ps = het.TryGetValue(tag ?? "", out string v) ? v : het["DEFAULT"];
                }
                else ps = (string)entry;
                if (ps == null || (isNNP == true && !ps.Contains(PRIMARY)))
                {
                    var nnp = GetNNP(word);
                    if (nnp.Item1 != null) return nnp;
                }
                return (ApplyStress(ps, stress), rating);
            }

            (string, int) GetNNP(string word)
            {
                var parts = new List<string>();
                foreach (char c in word)
                {
                    if (!char.IsLetter(c)) continue;
                    if (!golds.TryGetValue(char.ToUpperInvariant(c).ToString(), out object v) || v is not string s)
                        return (null, 0);
                    parts.Add(s);
                }
                string ps = ApplyStress(string.Concat(parts), 0);
                int cut = ps.LastIndexOf(SECONDARY);
                if (cut >= 0) ps = ps.Substring(0, cut) + PRIMARY + ps.Substring(cut + 1);
                return (ps, 3);
            }

            // ---------------------------------------------------------------- affixes (en.py)
            string SuffixS(string stem)
            {
                if (string.IsNullOrEmpty(stem)) return null;
                char last = stem[stem.Length - 1];
                if ("ptkfθ".IndexOf(last) >= 0) return stem + "s";
                if ("szʃʒʧʤ".IndexOf(last) >= 0) return stem + "ᵻz";
                return stem + "z";
            }

            (string, int) StemS(string word, string tag, float? stress, Ctx? ctx)
            {
                string stem = null;
                if (word.Length < 3 || !word.EndsWith("s")) return (null, 0);
                if (!word.EndsWith("ss") && IsKnown(word.Substring(0, word.Length - 1), tag))
                    stem = word.Substring(0, word.Length - 1);
                else if ((word.EndsWith("'s") || (word.Length > 4 && word.EndsWith("es") && !word.EndsWith("ies"))) &&
                         IsKnown(word.Substring(0, word.Length - 2), tag))
                    stem = word.Substring(0, word.Length - 2);
                else if (word.Length > 4 && word.EndsWith("ies") && IsKnown(word.Substring(0, word.Length - 3) + "y", tag))
                    stem = word.Substring(0, word.Length - 3) + "y";
                else return (null, 0);
                var (ps, rating) = Lookup(stem, tag, stress, ctx);
                return (SuffixS(ps), rating);
            }

            string SuffixEd(string stem)
            {
                if (string.IsNullOrEmpty(stem)) return null;
                char last = stem[stem.Length - 1];
                if ("pkfθʃsʧ".IndexOf(last) >= 0) return stem + "t";
                if (last == 'd') return stem + "ᵻd";
                if (last != 't') return stem + "d";
                if (stem.Length < 2) return stem + "ɪd";
                if (US_TAUS.IndexOf(stem[stem.Length - 2]) >= 0)
                    return stem.Substring(0, stem.Length - 1) + "ɾᵻd";
                return stem + "ᵻd";
            }

            (string, int) StemEd(string word, string tag, float? stress, Ctx? ctx)
            {
                string stem = null;
                if (word.Length < 4 || !word.EndsWith("d")) return (null, 0);
                if (!word.EndsWith("dd") && IsKnown(word.Substring(0, word.Length - 1), tag))
                    stem = word.Substring(0, word.Length - 1);
                else if (word.Length > 4 && word.EndsWith("ed") && !word.EndsWith("eed") &&
                         IsKnown(word.Substring(0, word.Length - 2), tag))
                    stem = word.Substring(0, word.Length - 2);
                else return (null, 0);
                var (ps, rating) = Lookup(stem, tag, stress, ctx);
                return (SuffixEd(ps), rating);
            }

            string SuffixIng(string stem)
            {
                if (string.IsNullOrEmpty(stem)) return null;
                if (stem.Length > 1 && stem[stem.Length - 1] == 't' && US_TAUS.IndexOf(stem[stem.Length - 2]) >= 0)
                    return stem.Substring(0, stem.Length - 1) + "ɾɪŋ";
                return stem + "ɪŋ";
            }

            (string, int) StemIng(string word, string tag, float? stress, Ctx? ctx)
            {
                string stem = null;
                if (word.Length < 5 || !word.EndsWith("ing")) return (null, 0);
                if (word.Length > 5 && IsKnown(word.Substring(0, word.Length - 3), tag))
                    stem = word.Substring(0, word.Length - 3);
                else if (IsKnown(word.Substring(0, word.Length - 3) + "e", tag))
                    stem = word.Substring(0, word.Length - 3) + "e";
                else if (word.Length > 5 && Regex.IsMatch(word, @"([bcdgklmnprstvxz])\1ing$|cking$") &&
                         IsKnown(word.Substring(0, word.Length - 4), tag))
                    stem = word.Substring(0, word.Length - 4);
                else return (null, 0);
                var (ps, rating) = Lookup(stem, tag, stress, ctx);
                return (SuffixIng(ps), rating);
            }

            // ---------------------------------------------------------------- numbers (en.py + num2words subset)
            static bool IsDigits(string s) => s.Length > 0 && s.All(char.IsDigit);

            static bool IsNumber(string word, bool isHead)
            {
                if (word.All(c => !char.IsDigit(c))) return false;
                foreach (string suf in new[] { "ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s" })
                    if (word.EndsWith(suf)) { word = word.Substring(0, word.Length - suf.Length); break; }
                return word.Select((c, i) => char.IsDigit(c) || c == ',' || c == '.' || (isHead && i == 0 && c == '-'))
                           .All(b => b);
            }

            static bool IsCurrencyAmount(string word)
            {
                if (!word.Contains('.')) return true;
                if (word.Count(c => c == '.') > 1) return false;
                string cents = word.Split('.')[1];
                return cents.Length < 3 || cents.All(c => c == '0');
            }

            (string, int) GetNumber(string word, string currency, bool isHead, string numFlags)
            {
                var m = Regex.Match(word, "[a-z']+$");
                string suffix = m.Success ? m.Value : null;
                if (suffix != null) word = word.Substring(0, word.Length - suffix.Length);
                var result = new List<(string ps, int rating)>();
                if (word.StartsWith("-")) { result.Add(Lookup("minus", null, null, null)); word = word.Substring(1); }

                void ExtendNum(string num, bool first = true, bool escape = false)
                {
                    string verbal = num;
                    if (!escape)
                    {
                        if (!long.TryParse(num, out long v))   // >15 digits: spell digit-by-digit
                        { foreach (char c in num.Where(char.IsDigit)) ExtendNum(c.ToString(), first: false); return; }
                        verbal = NumToWords(v);
                    }
                    string[] splits = Regex.Split(verbal, "[^a-z]+").Where(s => s.Length > 0).ToArray();
                    for (int i = 0; i < splits.Length; i++)
                    {
                        string w = splits[i];
                        if (w != "and" || numFlags.Contains('&'))
                        {
                            if (first && i == 0 && splits.Length > 1 && w == "one" && numFlags.Contains('a'))
                                result.Add(("ə", 4));
                            else
                                result.Add(Lookup(w, null, w == "point" ? -2f : (float?)null, null));
                        }
                        else if (w == "and" && numFlags.Contains('n') && result.Count > 0)
                            result[result.Count - 1] = (result[result.Count - 1].ps + "ən", result[result.Count - 1].rating);
                    }
                }

                if (IsDigits(word) && suffix != null && ORDINALS.Contains(suffix))
                    ExtendNum(NumToOrdinal(long.Parse(word)), escape: true);
                else if (result.Count == 0 && word.Length == 4 && currency == null && IsDigits(word))
                    ExtendNum(NumToYear(int.Parse(word)), escape: true);
                else if (!isHead && !word.Contains('.'))
                {
                    string num = word.Replace(",", "");
                    if (num[0] == '0' || num.Length > 3)
                        foreach (char n in num) ExtendNum(n.ToString(), first: false);
                    else if (num.Length == 3 && !num.EndsWith("00"))
                    {
                        ExtendNum(num[0].ToString());
                        if (num[1] == '0') { result.Add(Lookup("O", null, -2, null)); ExtendNum(num[2].ToString(), first: false); }
                        else ExtendNum(num.Substring(1), first: false);
                    }
                    else ExtendNum(num);
                }
                else if (word.Count(c => c == '.') > 1 || !isHead)
                {
                    bool first = true;
                    foreach (string num in word.Replace(",", "").Split('.'))
                    {
                        if (num.Length == 0) { }
                        else if (num[0] == '0' || (num.Length != 2 && num.Skip(1).Any(n => n != '0')))
                            foreach (char n in num) ExtendNum(n.ToString(), first: false);
                        else ExtendNum(num, first: first);
                        first = false;
                    }
                }
                else if (currency != null && CURRENCIES.ContainsKey(currency) && IsCurrencyAmount(word))
                {
                    var units = CURRENCIES[currency];
                    string[] halves = word.Replace(",", "").Split('.');
                    var pairs = new List<(long num, string unit)>();
                    for (int i = 0; i < halves.Length && i < 2; i++)
                        pairs.Add((halves[i].Length > 0 ? long.Parse(halves[i]) : 0, i == 0 ? units.unit : units.sub));
                    if (pairs.Count > 1)
                    {
                        if (pairs[1].num == 0) pairs.RemoveAt(1);
                        else if (pairs[0].num == 0) pairs.RemoveAt(0);
                    }
                    for (int i = 0; i < pairs.Count; i++)
                    {
                        if (i > 0) result.Add(Lookup("and", null, null, null));
                        ExtendNum(pairs[i].num.ToString(), first: i == 0);
                        result.Add(Math.Abs(pairs[i].num) != 1 && pairs[i].unit != "pence"
                            ? StemS(pairs[i].unit + "s", null, null, null)
                            : Lookup(pairs[i].unit, null, null, null));
                    }
                }
                else
                {
                    string words;
                    if (IsDigits(word)) words = NumToWords(long.Parse(word));
                    else if (!word.Contains('.'))
                        words = suffix != null && ORDINALS.Contains(suffix)
                            ? NumToOrdinal(long.Parse(word.Replace(",", "")))
                            : NumToWords(long.Parse(word.Replace(",", "")));
                    else
                    {
                        word = word.Replace(",", "");
                        if (word[0] == '.')
                            words = "point " + string.Join(" ", word.Substring(1).Select(n => NumToWords(n - '0')));
                        else
                        {
                            string[] halves = word.Split('.');
                            words = NumToWords(long.Parse(halves[0])) + " point " +
                                    string.Join(" ", halves[1].Select(n => NumToWords(n - '0')));
                        }
                    }
                    ExtendNum(words, escape: true);
                }

                if (result.Count == 0) return (null, 0);
                string ps = string.Join(" ", result.Select(r => r.ps));
                int rating = result.Min(r => r.rating);
                if (suffix == "s" || suffix == "'s") return (SuffixS(ps), rating);
                if (suffix == "ed" || suffix == "'d") return (SuffixEd(ps), rating);
                if (suffix == "ing") return (SuffixIng(ps), rating);
                return (ps, rating);
            }

            string AppendCurrency(string ps, string currency)
            {
                if (string.IsNullOrEmpty(currency) || !CURRENCIES.TryGetValue(currency, out var units)) return ps;
                var (cs, _) = StemS(units.unit + "s", null, null, null);
                return cs != null ? ps + " " + cs : ps;
            }

            // -- num2words subset (matches num2words en output incl. "and" after hundred) -------
            static readonly string[] Units = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
            static readonly string[] Tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
            static readonly (long v, string n)[] Scales = { (1_000_000_000_000L, "trillion"),
                (1_000_000_000L, "billion"), (1_000_000L, "million"), (1000L, "thousand") };

            static string NumToWords(long n)
            {
                if (n < 0) return "minus " + NumToWords(-n);
                if (n < 20) return Units[n];
                if (n < 100)
                    return Tens[n / 10] + (n % 10 != 0 ? "-" + Units[n % 10] : "");
                if (n < 1000)
                    return Units[n / 100] + " hundred" + (n % 100 != 0 ? " and " + NumToWords(n % 100) : "");
                foreach (var (v, name) in Scales)
                    if (n >= v)
                        return NumToWords(n / v) + " " + name +
                               (n % v == 0 ? "" : (n % v < 100 ? " and " : " ") + NumToWords(n % v));
                return Units[0];
            }

            static string NumToOrdinal(long n)
            {
                string c = NumToWords(n);
                int cut = Math.Max(c.LastIndexOf(' '), c.LastIndexOf('-'));
                string head = cut < 0 ? "" : c.Substring(0, cut + 1), last = c.Substring(cut + 1);
                var irregular = new Dictionary<string, string> { { "one", "first" }, { "two", "second" },
                    { "three", "third" }, { "five", "fifth" }, { "eight", "eighth" }, { "nine", "ninth" }, { "twelve", "twelfth" } };
                if (irregular.TryGetValue(last, out string ir)) return head + ir;
                if (last.EndsWith("y")) return head + last.Substring(0, last.Length - 1) + "ieth";
                return head + last + "th";
            }

            static string NumToYear(int y)   // num2words to='year' behavior for 1000-9999
            {
                if (y < 1000 || y >= 10000) return NumToWords(y);
                int high = y / 100, low = y % 100;
                if (low == 0) return y % 1000 == 0 ? NumToWords(y) : NumToWords(high) + " hundred";
                if (low < 10) return high % 10 == 0 ? NumToWords(y)        // 2005 -> two thousand and five
                                                    : NumToWords(high) + " oh-" + NumToWords(low);  // 1905
                return NumToWords(high) + " " + NumToWords(low);           // 2024 -> twenty twenty-four
            }
        }
    }
}
