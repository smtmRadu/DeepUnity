# Qwen3-ASR — License Check (D0)

**Verdict: PASS — Apache-2.0, redistribution-friendly. No blocker for a public GitHub repo.**
Checked 2026-07-11 against the live Hugging Face repos and the QwenLM GitHub sources.

## Repos and their licenses

| Repo | License | Evidence |
|---|---|---|
| `Qwen/Qwen3-ASR-0.6B` | Apache-2.0 | Model card front-matter: `license: apache-2.0`; HF API tag `license:apache-2.0` |
| `Qwen/Qwen3-ASR-1.7B` | Apache-2.0 | Model card front-matter (raw README.md): `--- license: apache-2.0 pipeline_tag: automatic-speech-recognition ---`; HF API tag `license:apache-2.0` |
| `Qwen/Qwen3-ASR-0.6B-hf` | Apache-2.0 | HF API tag `license:apache-2.0` |
| `Qwen/Qwen3-ASR-1.7B-hf` | Apache-2.0 | HF API tag `license:apache-2.0` |
| GitHub `QwenLM/Qwen3-ASR` (qwen_asr library) | Apache-2.0 | Source headers: `# Copyright 2026 The Alibaba Qwen team.` / `# SPDX-License-Identifier: Apache-2.0` / `Licensed under the Apache License, Version 2.0` |
| transformers `qwen3_asr` integration (modeling/processing/feature-extraction code we mined) | Apache-2.0 | File headers: `Copyright 2026 The HuggingFace Inc. team … Licensed under the Apache License, Version 2.0` |

Links:
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf
- https://github.com/QwenLM/Qwen3-ASR
- Tech report: https://arxiv.org/abs/2601.21337 · Blog: https://qwen.ai/blog?id=qwen3asr

## What this means for DeepUnity

- Apache-2.0 permits redistribution of the weights (including converted/quantized `manifest.tsv` + `.bin`
  exports) in a public repo, commercial use, and modification. Same license family as the rest of the
  Qwen3 releases DeepUnity already ships (Qwen3.5).
- Obligations: keep the Apache-2.0 license text + a NOTICE attribution for the Qwen3-ASR weights in the
  distribution, and state changes (quantization/repacking). Mirror what was done for the Qwen3.5 weight
  folders.
- No acceptance gate: the HF repos are public, not gated — `hf download` works anonymously
  (verified: both `-hf` checkpoints downloaded 2026-07-11 without a token).
- The tokenizer (Qwen2 BPE, `tokenizer.json`) and chat template ship under the same repo license.

## Notes

- Weights are genuinely open — this is NOT the API-only "Qwen3-ASR-Flash" service; the 0.6B/1.7B
  checkpoints (released 2026-01-29) are full safetensors downloads (1.56 GB / 4.08 GB bf16).
- Companion model `Qwen/Qwen3-ForcedAligner-0.6B` (timestamps) is also Apache-2.0 — out of scope for D0
  but unblocked if we ever want word timestamps.
