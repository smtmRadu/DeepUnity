# Kokoro G2P in Unity — Decision Document

Kokoro-82M consumes a **phoneme string** (misaki's custom IPA-superset, 178-symbol vocab from
`config.json`), NOT raw text. Phoneme fidelity is the single biggest quality lever: the model was
trained on misaki-produced transcriptions, so the closer our runtime G2P matches
`misaki.en.G2P(british=False, fallback=EspeakFallback)`, the closer we get to reference audio.
This doc compares the options for producing those phonemes inside Unity at runtime and recommends one.

## What the reference actually does (misaki EN, mined from source)

`misaki/en.py` (738 lines) + `misaki/espeak.py` (107 lines), pipeline for `lang_code='a'`:

1. **Tokenize + POS-tag** with spaCy `en_core_web_sm` (tok2vec+tagger only).
2. **Retokenize**: split merged tokens (regex subtokenizer), currency/symbol/punct handling.
3. **Lexicon lookup** per token, right-to-left (so `future_vowel`/`future_to` context flows backwards):
   - `us_gold.json` — **90,201 entries** (curated, rating 4). Only **790 are heteronyms** stored as
     tag-keyed dicts; the tag keys used are just `DEFAULT`(790), `NOUN`(425), `VERB`(272), `ADJ`(63),
     `None`(32, = future_vowel-conditioned like "the/to/in"), `VBD/VBN/VBP`(10), `ADV`(3), `DT`(1).
   - `us_silver.json` — **93,361 entries** (rating 3), no heteronyms.
   - `grow_dictionary`: adds Capitalized/lowercase twins at load.
   - ~15 special-case rules (`a/an/am/the/to/in/by/I/used/vs`...), mostly needing only the coarse tag
     + `future_vowel` bit.
   - Affix rules `stem_s/_ed/_ing` (regular morphology on known stems, exact phonology rules ~30 lines).
   - Number verbalization via `num2words` (cardinal/ordinal/year/currency/decimal paths).
   - Acronym spell-out `get_NNP` via per-letter gold entries.
   - Final post-map: `ɾ→T`, `ʔ→t` (American), stress juggling via `apply_stress`.
4. **OOD fallback** (words not resolvable above): espeak-ng via phonemizer
   (`--ipa --tie`) + a ~30-rule espeak→misaki phoneme remap (`EspeakFallback.E2M`), rating 2.
   (misaki standalone defaults to a BART seq2seq fallback `PeterReid/graphemes_to_phonemes_en_us`;
   the Kokoro pipeline overrides that with espeak.)

**Empirical coverage** (140-word game-dialogue sample incl. deliberate heteronyms + rare words):
138/140 direct dictionary hits, 1 affix hit, **1 miss — an invented fantasy proper noun ("Aldric")**.
The fallback path is nearly irrelevant for normal English text; it exists for names/neologisms.

## Options

### (a) espeak-ng as native Windows plugin (P/Invoke) as the PRIMARY G2P
Ship `libespeak-ng.dll` + espeak-ng-data (~12 MB), P/Invoke `espeak_TextToPhonemes` with IPA+tie
mode, port the `E2M` remap (~40 string replaces) to C#.
- Effort: **~2 days** (the pip wheel `espeakng-loader` already ships prebuilt win_amd64/linux/mac
  binaries we could lift; no compiling needed).
- Quality: **measurably worse than reference.** espeak is only the *rating-2 fallback* in the real
  pipeline; using it for everything loses the 90k curated gold entries, misaki's stress logic,
  heteronym handling and number verbalization. Audible on common words (espeak's stress placement
  and vowel choices deviate; Kokoro is sensitive to stress marks `ˈ ˌ`).
- **Blocker: espeak-ng is GPL-3.0.** P/Invoking it from a Unity game makes the combined work
  GPL-encumbered — unacceptable as a hard dependency for a redistributable engine module.
  phonemizer (its usual wrapper) is GPL-3.0 too. This kills (a) as the mandatory path regardless
  of quality.
- Platforms: desktop only (DLL/dylib/so); no consoles, WebGL, iOS static-linking pain.

### (b) Full misaki-EN port to C#
Port `en.py` 1:1 including a POS tagger equivalent.
- The scary part was always the spaCy tagger — but the data above shows we do NOT need spaCy-grade
  tagging: only 790 heteronyms exist and they key on `NOUN/VERB/ADJ` parent tags; the special cases
  need `DT/IN/TO/PRP/NNP`-level distinctions. A ~200-line rule/suffix tagger (Brill-style: closed-class
  word lists + suffix heuristics + 2-3 context rules) disambiguates the overwhelming majority; ties
  fall back to the `DEFAULT` pronunciation, which is what misaki itself does for unknown tags.
- Effort: **4–6 days** (Lexicon + apply_stress + retokenizer + affix rules + number verbalizer +
  mini-tagger + tests against reference dumps).
- Quality: near-reference. On normal text, phoneme output is expected to match misaki on ≳99% of
  tokens (deviations only where the mini-tagger mislabels a heteronym → still a *valid* English
  pronunciation, just occasionally the wrong variant of read/present/record...).
- License: misaki is **Apache-2.0** — dictionaries are redistributable. No native code. All platforms.

### (c) CMUdict + letter-to-sound fallback
- CMUdict is ARPABET; mapping ARPABET→misaki's vocab is lossy (no `ᵊ ᵻ`, no flap `T`, different
  diphthong inventory, coarser unstressed vowels). We'd be converting a worse dictionary into the
  right alphabet when misaki's own gold/silver dicts — already IN the exact target alphabet, already
  curated against this exact model — are Apache-2.0. Strictly dominated by (b). **Rejected.**

### (d) Neural G2P as OOD fallback (later, optional)
misaki's own `FallbackNetwork` is a small BART (`PeterReid/graphemes_to_phonemes_en_us`) that emits
misaki phonemes directly. DeepUnity already has transformer inference; exporting it and running it
for the ~1-in-100 OOD word is a natural **v2** upgrade that would beat espeak quality-wise for
invented names, with zero licensing issues. Not needed for v1.

## Recommendation: (b) — C# port of misaki's EN Lexicon path, staged

**v1 (port this now, ~4–6 days):**
1. Exporter converts `us_gold.json`/`us_silver.json` (+grown variants) into one binary/text lookup
   file shipped next to the weights (~6 MB raw JSON → ~4 MB flattened; load into a
   `Dictionary<string,string>` + a small tag-keyed sidecar for the 790 heteronyms).
2. C# `KokoroG2P`: retokenizer (the `subtokenize` regex is portable), special cases, affix stemming
   (`stem_s/_ed/_ing`), `apply_stress`, number verbalization (port the num2words cardinal/ordinal/
   year/currency subset — bounded, ~300 lines), acronym spell-out, `ɾ→T`/`ʔ→t` post-map,
   punctuation passthrough (Kokoro's vocab includes `; : , . ! ? — … " ( ) “ ”` and they carry
   prosodic meaning — keep them).
3. Mini POS tagger (rule-based, ~200 lines): closed-class lists (DT/IN/TO/PRP/MD/CC), suffix rules,
   `to/will/have`+word → VERB, `the/a/adj`+word → NOUN, default NOUN. Only consulted for the 790
   heteronyms + special cases.
4. OOD fallback v1 = misaki's own last resorts, no espeak: `get_NNP` letter spell-out for
   capitalized unknowns, else skip token (misaki's `unk=''` behavior in the Kokoro pipeline).
   Measured miss rate ≈ 0.7% of words on realistic text, and misses are mostly names where
   spell-out or a rough LTS guess is acceptable.

**v2 (optional quality top-up):** export the BART fallback (d) for OOD words; and/or an *optional*,
desktop-only espeak-ng plugin for users who accept GPL — never a core dependency.

**Validation:** `validation/dump_reference.py` stores the reference phoneme string for each test
text; CHECKLIST.md gates the C# G2P on exact string match for the fixed texts, then on a larger
corpus we track token-level agreement vs misaki (target ≥99%).

## Why not just require espeak?
It's the only option that is BOTH lower quality on the 99% case AND legally radioactive (GPL-3.0)
AND platform-restricted. The dictionary path is more work up front but it is the same data the
model was trained against, pure C#, Apache-licensed, and runs everywhere Unity runs.
