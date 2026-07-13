# Pocket-TTS (Kyutai) — DeepUnity port spec (frozen 2026-07-12 from weights + code)

Source: github.com/kyutai-labs/pocket-tts · HF `kyutai/pocket-tts` (voice-cloning) · MIT · paper arxiv 2509.06926.
Model file: `languages/english/model.safetensors` — **208.9 MB, 214 tensors** (bf16). Future DeepUnity DEFAULT TTS.

## Two components: FlowLM (127 tensors) + Mimi codec (87 tensors)

Frame rate **12.5 Hz**, sample rate **24000**. Per audio frame the FlowLM predicts ONE continuous
Mimi latent (dim 32); Mimi's decoder turns the latent stream into 24 kHz waveform. AR loop is tiny
(125 frames = 10 s audio).

### FlowLM (`flow_lm.*`) — autoregressive continuous-latent transformer
- **Text conditioner**: `conditioner.embed.weight [4001, 1024]` — SentencePiece vocab 4001, model dim **1024**. (tokenizer.model from `kyutai/pocket-tts-without-voice-cloning`, 58 KB, open repo.)
- **Transformer backbone** (yaml `depformer`/main): **6 layers, dim 1024, 16 heads, FFN 4096**, LayerNorm w/ bias (norm1/norm2), linear1 [4096,1024] + linear2 [1024,4096]. Standard pre-LN transformer (confirm attention: causal + RoPE vs learned — check modules/transformer.py + rope.py at P0).
- **Latent space**: ldim/`inner_dim` = **32**; `emb_mean[32]`/`emb_std[32]` normalize latents; `bos_emb[32]`; `bos_before_voice [1,1,1024]`.
- **Flow-matching head** `flow_net` = **SimpleMLPAdaLN** (modules/mlp.py), flow-net dim **512**, **6 res_blocks**:
  - `input_proj [512,32]` (noisy latent 32→512), `cond_embed [512,1024]` (transformer output 1024→512, the AdaLN condition)
  - res_block: `in_ln[512]` + `mlp.0[512,512]` SiLU `mlp.2[512,512]` + `adaLN_modulation.1[1536,512]` (1536 = 3×512 → shift/scale/gate)
  - `final_layer`: `adaLN_modulation.1[1024,512]` (2×512 shift/scale) + `linear[32,512]` (→ back to ldim 32)
  - Runs **N Euler steps** per frame from clamped noise (DEFAULT_LSD_DECODE_STEPS / DEFAULT_NOISE_CLAMP in default_parameters.py — read at P0). = exactly CosyVoice DiT's AdaLN + Euler pattern (reuse AdaLNModulate/EulerCfgStep kernels).
- **EOS**: `out_norm` LayerNorm + `out_eos` Linear(1024→1) — stop when EOS prob > DEFAULT_EOS_THRESHOLD.

### Mimi codec (`mimi.*`) — streaming neural audio codec (24 kHz)
- `mimi.decoder` (22): SEANet transposed-conv stack — model.0 conv[512,512,7] → model.2 convtr[512,256,12] → model.3 resblock → model.5 convtr[256,128,10] → ... → model.11 conv[1,64,3] = 1-channel waveform. = HiFT/Chatterbox Conv1d/ConvTranspose1d/resblock kernels (validated family).
- `mimi.decoder_transformer` (20): small transformer, **dim 512, 8 heads, 2 layers**, layer_scale 0.01, FFN 2048 — runs on the latent stream before the conv decoder.
- `mimi.upsample` / `downsample` (resample between encoder_frame_rate and 12.5 Hz), `mimi.quantizer` (1 — dummy/passthrough: latents are CONTINUOUS, not RVQ-discrete here).
- `mimi.encoder` (22) + `mimi.encoder_transformer` (20): ONLY needed to encode a reference clip for voice cloning → bake offline in Python, NOT needed at runtime if voices are pre-baked.

### Voice cloning — `embeddings/<name>.safetensors` → `audio_prompt [1, 125, 1024]`
125 frames × 1024-dim transformer-space prefix (= 10 s reference @ 12.5 Hz), **prepended** to the
sequence before generation. Voices are just [125,1024] tensors — bake any voice offline (run Mimi
encoder on a reference clip, Python-side), drop the tensor in. 20+ prebuilt voices in the repo
(embeddings/ + embeddings_v2/ + embeddings_v3/).

## Why this is the cleanest DeepUnity fit yet
- FlowLM transformer (6L/1024d/16h, vanilla LayerNorm) → Qwen/CosyVoice-LM attn+FFN kernels (simpler — no GQA).
- SimpleMLPAdaLN flow head → CosyVoice DiT AdaLN+Euler kernels ALREADY EXIST.
- Mimi SEANet decoder + small transformer → HiFT/Chatterbox conv + LM-transformer kernels.
- Frame streaming → Kokoro/CosyVoice ring-buffer voice component.
- 100M params, 12.5 Hz (short AR loop), tiny flow head ⇒ RTF well under 1 on 4060, int8 → smaller. Clear Kokoro upgrade WITH voice cloning.

## Plan (mirror CosyVoice A0-A7 + Kokoro optims)
- **P0**: freeze this SPEC ✅ · read modules/{transformer,rope,mlp,seanet,mimi_transformer}.py + default_parameters.py for exact attention type / Euler step count / noise clamp / act fns · `import_pocket_tts.py` exporter → Assets/Resources/Weights/weights_pockettts_english_{fp16,int8} · `dump_reference.py` per-stage .npy (fixed noise) · bake a voice.
- **P1** Mimi decoder (conv + small transformer) parity · **P2** FlowLM transformer parity · **P3** SimpleMLPAdaLN flow head + Euler parity · **P4** offline e2e (text→audio) · **P5** streaming (frame-level ring buffer) + RTF/TTFA · **P6** int8 + Kokoro-style throughput autopsy · **P7** TtsModel enum entry + registry + NPC integration, make it DEFAULT.
Module: Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/ + Resources/ComputeShaders/PocketTTSCS.compute.
