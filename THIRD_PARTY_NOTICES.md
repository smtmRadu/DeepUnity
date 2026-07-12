# Third-Party Notices

DeepUnity's inference engine ports the following third-party models and assets. All weights are
re-exported into the engine's own format (`Assets/Resources/Weights/weights_*`) by the scripts in
this repo; no third-party runtime code is redistributed.

## TTS models

- **Fun-CosyVoice3-0.5B-2512** — FunAudioLLM (Alibaba), Apache License 2.0.
  Source: https://github.com/FunAudioLLM/CosyVoice · https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
  Used for: CosyVoice3 port (`Assets/DeepUnity/TTS/CosyVoice/`) — LM, DiT flow, CausalHiFT
  vocoder weights; voices baked offline with the upstream speech tokenizer + campplus x-vector
  models (Python-side only, `validation/make_voice.py`).
- **Kokoro-82M** — hexgrad, Apache License 2.0.
  Source: https://huggingface.co/hexgrad/Kokoro-82M
  Used for: Kokoro port (`Assets/DeepUnity/TTS/Kokoro/`) incl. baked voicepacks and blends.
- **misaki** (G2P lexicons) — hexgrad, Apache License 2.0.
  Source: https://github.com/hexgrad/misaki
  Used for: the English gold/silver lexicon TSVs consumed by `KokoroG2P`.
- **Chatterbox (Turbo)** — Resemble AI, MIT License.
  Source: https://github.com/resemble-ai/chatterbox
  Used for: Chatterbox port (`Assets/DeepUnity/TTS/Chatterbox/`).

## LLMs

- **Qwen3.5 (0.8B / 2B)** — Alibaba Cloud, Apache License 2.0.
  Source: https://huggingface.co/Qwen
- **Gemma 3 (270M) / EmbeddingGemma** — Google. Distributed under the **Gemma Terms of Use**
  (not an OSI license — see https://ai.google.dev/gemma/terms before redistribution).
- **MiniCPM5 (1B)** — OpenBMB. Code Apache License 2.0; model under the MiniCPM Model License.
  Source: https://huggingface.co/openbmb

## STT models (ports in progress)

- **Qwen3-ASR (0.6B / 1.7B)** — Alibaba Cloud, Apache License 2.0.
- **Parakeet-TDT 0.6B (v2/v3)** — NVIDIA, CC-BY-4.0.
  Source: https://huggingface.co/nvidia

## Demo art & audio

- **Quaternius packs** (RPG Characters Nov-2020, Ultimate Modular Women Apr-2022, Ultimate
  Modular Ruins, Animated Knight, Medieval Weapons, Universal Animation Library 1/2) — CC0 /
  public domain, https://quaternius.com. Per-file mapping:
  `Assets/DeepUnity/Tutorials/ChatDemo3D/Art/LICENSES.txt`.
- 2D demo sprites/audio: generated in-repo or CC0 (see the ChatDemo2D builder comments).

License texts: Apache-2.0 https://www.apache.org/licenses/LICENSE-2.0 · MIT
https://opensource.org/license/mit · CC-BY-4.0 https://creativecommons.org/licenses/by/4.0 ·
CC0 https://creativecommons.org/publicdomain/zero/1.0
