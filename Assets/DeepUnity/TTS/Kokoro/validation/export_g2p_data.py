#!/usr/bin/env python3
"""
Misaki EN dictionaries -> DeepUnity KokoroG2P data files (B1, G2P_PROPOSAL.md option (b) v1).

Converts misaki's us_gold.json / us_silver.json (Apache-2.0, the exact lexicon Kokoro was
trained against) into two TSV files next to the C# G2P (ChatterboxTokenizer.* convention):

    Assets/DeepUnity/TTS/Kokoro/KokoroG2P.gold.tsv
    Assets/DeepUnity/TTS/Kokoro/KokoroG2P.silver.tsv

Line formats (UTF-8, \n, no escaping needed - words/phonemes never contain tabs):
    simple entry:     word<TAB>phonemes
    heteronym entry:  word<TAB>TAG=phonemes<TAB>TAG=phonemes...   (always includes DEFAULT=)
                      a None pronunciation for a tag is written as TAG=\\N
RAW dict only - the C# loader applies misaki's grow_dictionary (capitalized/lowercase twins)
at load time, mirroring en.py Lexicon.__init__.

Run anywhere with python3 (no torch needed):
    python export_g2p_data.py [--misaki /mnt/c/dev/_model_staging/kokoro/misaki-repo/misaki/data]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MISAKI = "/mnt/c/dev/_model_staging/kokoro/misaki-repo/misaki/data" if os.name != "nt" \
    else "C:/dev/_model_staging/kokoro/misaki-repo/misaki/data"


def write_tsv(src_json, dst_tsv):
    with open(src_json, encoding="utf-8") as f:
        d = json.load(f)
    n_het = 0
    with open(dst_tsv, "w", encoding="utf-8", newline="\n") as f:
        for word, v in d.items():
            assert "\t" not in word and "\n" not in word, repr(word)
            if isinstance(v, str):
                assert "\t" not in v and "\n" not in v, repr(v)
                f.write(f"{word}\t{v}\n")
            else:
                assert "DEFAULT" in v, (word, v)
                n_het += 1
                fields = []
                for tag in sorted(v):  # deterministic order
                    ps = v[tag]
                    assert "=" not in tag
                    fields.append(f"{tag}={'\\N' if ps is None else ps}")
                f.write(word + "\t" + "\t".join(fields) + "\n")
    print(f"{os.path.basename(dst_tsv)}: {len(d)} entries ({n_het} heteronym) "
          f"-> {os.path.getsize(dst_tsv) / 1024 / 1024:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misaki", default=DEFAULT_MISAKI, help="misaki-repo/misaki/data dir")
    args = ap.parse_args()
    out_dir = os.path.normpath(os.path.join(HERE, ".."))   # Assets/DeepUnity/TTS/Kokoro
    write_tsv(os.path.join(args.misaki, "us_gold.json"), os.path.join(out_dir, "KokoroG2P.gold.tsv"))
    write_tsv(os.path.join(args.misaki, "us_silver.json"), os.path.join(out_dir, "KokoroG2P.silver.tsv"))
    print("Done. C# KokoroG2P loads these relative to its own folder (grow_dictionary at load).")


if __name__ == "__main__":
    main()
