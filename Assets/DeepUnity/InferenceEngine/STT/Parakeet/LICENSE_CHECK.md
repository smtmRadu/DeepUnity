# Parakeet-TDT — License Check (E0)

**Verdict: PASS — CC-BY-4.0 on both repos. Redistribution-friendly; requires ATTRIBUTION.
No blocker for a public GitHub repo.**
Checked 2026-07-11 against the live Hugging Face repos (front-matter + API tags).

## Repos and their licenses

| Repo | License | Evidence |
|---|---|---|
| `nvidia/parakeet-tdt-0.6b-v2` | CC-BY-4.0 | raw README.md front-matter `license: cc-by-4.0`; HF API tag `license:cc-by-4.0`; not gated |
| `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | HF API tag `license:cc-by-4.0`; model card "governed by the CC-BY-4.0 license"; not gated |
| transformers `parakeet` integration (modeling/generation/feature-extraction/convert code mined for the spec) | Apache-2.0 | File headers: `Copyright 2025/2026 The HuggingFace Inc. team … Apache License, Version 2.0` |
| NVIDIA NeMo sources (rnnt_greedy_decoding.py etc., mined for the TDT loop) | Apache-2.0 | Repo LICENSE (NVIDIA/NeMo, Apache-2.0) |

Links:
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- CC-BY-4.0 text: https://creativecommons.org/licenses/by/4.0/
- Papers: https://arxiv.org/abs/2305.05084 (FastConformer) · https://arxiv.org/abs/2304.06795 (TDT)

## What CC-BY-4.0 means for DeepUnity

- **Allowed**: redistribution, modification (fp16/int8 repack into manifest.tsv + .bin), commercial
  use — for both the original weights and our converted exports (they are "Adapted Material").
- **Required**: (a) credit NVIDIA as the creator, (b) link the license, (c) indicate that changes
  were made (format conversion / quantization), (d) don't imply NVIDIA endorses DeepUnity.
- **Weights are gitignored** in the DeepUnity repo; users regenerate the `.bin` export locally from
  the HF checkpoints via `validation/import_parakeet.py`. That means the public repo itself ships
  NO CC-BY material — attribution is then only *strictly* required wherever weights are actually
  distributed (e.g. if a release bundle ever includes the exported .bin files). We still ship the
  attribution file unconditionally: it costs nothing, covers the docs' use of model names/figures,
  and makes any future weight-bundling automatically compliant.
- The **code** we write (C#, shaders, exporter) is ours — CC-BY-4.0 applies to the model weights,
  not to independently written inference code. The transformers/NeMo sources we *read* are
  Apache-2.0; our C# is a re-implementation from spec, not a translation of GPL-encumbered code.

## Attribution text draft (goes in the weight folders' NOTICE / repo README credits section)

> **Parakeet-TDT 0.6B** speech-to-text models © NVIDIA Corporation, licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
> Sources: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (English) and
> [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (25 languages).
> Changes: weights converted to DeepUnity's fp16/int8 binary format (`manifest.tsv` + `.bin`,
> BatchNorm folded, tokenizer flattened to `vocab.txt`) by the DeepUnity exporter; no retraining.
> Architecture: FastConformer encoder (arXiv 2305.05084) with a Token-and-Duration Transducer
> decoder (arXiv 2304.06795), built with [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).
> This use does not imply endorsement by NVIDIA.

## Notes

- Both repos are public and ungated — anonymous `hf download` works (verified 2026-07-11, both
  checkpoints staged without a token).
- No usage restrictions beyond attribution (contrast: no RAIL clauses, no NVIDIA Open Model
  License terms on these two — plain CC-BY-4.0).
- v3's `tokenizer.json`/config files ship in-repo under the same CC-BY-4.0; the flattened
  `tokenizer/vocab.txt` we export is Adapted Material → covered by the same attribution.
