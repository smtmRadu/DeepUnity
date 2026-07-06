# Chatterbox-Turbo → DeepUnity Port Spec (exact, from source)

Source of truth: `C:\dev\chatterbox-ref` (resemble-ai/chatterbox @ master) + HF `ResembleAI/chatterbox-turbo`.
This file records the EXACT inference graph as implemented in Python; the C#/HLSL port must match it tensor-for-tensor.

## 0. Checkpoint files (HF repo ResembleAI/chatterbox-turbo)
- `t3_turbo_v1.safetensors` — T3 (GPT2-medium backbone + heads). `tfmr.wte.weight` present but UNUSED (deleted after load).
- `s3gen_meanflow.safetensors` — full S3Gen (tokenizer(unused at runtime) + flow + estimator + vocoder + CAMPPlus).
- `ve.safetensors` — VoiceEncoder (cloning only; NOT needed for v1 baked voice).
- `conds.pt` — baked default voice: `{t3: {speaker_emb, cond_prompt_speech_tokens, emotion_adv(unused)}, gen: {prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding}}`.
- `vocab.json` + `merges.txt` + `added_tokens.json` + `special_tokens_map.json` + `tokenizer_config.json` — GPT2 byte-level BPE, vocab 50276.

## 1. Top-level pipeline (tts_turbo.py, offline reference)
1. `punc_norm(text)`: capitalize first letter; collapse spaces; replace `… : — – “ ” ‘ ’` and `" ,"`; append `.` if no ending punc in `.!?-,`.
2. `text_tokens = AutoTokenizer(text).input_ids` (GPT2 BPE; NO BOS/EOS added; `_ensure_BOT_EOT` NOT called in turbo).
3. `speech_tokens = t3.inference_turbo(...)` — temp 0.8, top_k 1000, top_p 0.95, rep_penalty 1.2, max_gen_len 1000.
4. Filter: keep `tok < 6561`; append silence `[4299, 4299, 4299]` (S3GEN_SIL).
5. `wav = s3gen.inference(speech_tokens, ref_dict=conds.gen, n_cfm_timesteps=2)`.
6. Fade-in: `wav[:960] *= trim_fade` where trim_fade = zeros(480) ++ (cos(linspace(π,0,480))+1)/2.
7. Watermark (Perth) — SKIPPED in our port.
- Output 24 kHz mono float.

## 2. T3-turbo — GPT2-medium backbone (NOT Llama!)
Config (`GPT2_MEDIUM_CONFIG` + turbo hp): hidden 1024, 24 layers, 16 heads × head_dim 64 (MHA, no GQA),
n_inner 4096, LayerNorm eps 1e-5 (affine + bias), act **gelu_new** (tanh approx), n_positions 8196 (wpe),
text vocab 50276, speech vocab **6563** (start 6561, stop 6562), speaker_embed 256,
`speech_cond_prompt_len=375` (turbo hp; conds.pt prompt token count should match), NO perceiver, NO emotion_adv,
NO learned T3 pos emb (`input_pos_emb=None`) — GPT2 backbone adds `wpe[past_len + i]` internally.
All GPT2 projections have BIAS. HF GPT2 `Conv1D.weight` is stored **(in, out) — TRANSPOSE at export** to (out, in).

### Modules + safetensors names (t3.*):
- `cond_enc.spkr_enc` Linear(256→1024) w/ bias
- `text_emb` Embedding(50276, 1024); `speech_emb` Embedding(6563, 1024)
- `speech_head` Linear(1024→6563) **with bias** (is_gpt); (`text_head` unused at inference)
- `tfmr.wpe` Embedding(8196, 1024) — position embedding, index = absolute position in the KV stream
- per layer i in 0..23: `tfmr.h.{i}.ln_1.{weight,bias}`, `tfmr.h.{i}.attn.c_attn.{weight,bias}` (1024→3072 fused QKV),
  `tfmr.h.{i}.attn.c_proj.{weight,bias}` (1024→1024), `tfmr.h.{i}.ln_2.{weight,bias}`,
  `tfmr.h.{i}.mlp.c_fc.{weight,bias}` (1024→4096), `tfmr.h.{i}.mlp.c_proj.{weight,bias}` (4096→1024)
- `tfmr.ln_f.{weight,bias}` final LayerNorm
- SKIP at export: `tfmr.wte.weight`, `text_head.weight`, all `ve.*`.

### GPT2 block (pre-LN):
x += attn(ln_1(x)); x += mlp(ln_2(x)) where attn = c_proj(softmax(QK^T/√64 + causal_mask)V),
mlp = c_proj(gelu_new(c_fc(x))). Final hidden = ln_f(x). Dropouts all no-op at eval.
gelu_new(x) = 0.5x(1+tanh(√(2/π)(x+0.044715x³))) — same as DeepUnity's inline `gelu()`.

### inference_turbo decode loop (exact):
- cond_emb = concat[spkr_enc(speaker_emb) (1 tok), speech_emb(cond_prompt_speech_tokens) (375 toks)] → (1,376,1024)
- embeds = concat[cond_emb, text_emb(text_tokens), speech_emb([6561])] → prefill through backbone (causal, KV cache)
- speech_logits = speech_head(hidden[-1]); sample; then loop: embed sampled token → forward 1 pos → sample …
- Logits processor ORDER (HF LogitsProcessorList as appended): temperature(0.8) → top_k(1000) → top_p(0.95) →
  repetition_penalty(1.2, over previously GENERATED speech tokens only, BOS not included; HF rule:
  score>0 ? score/p : score*p). Then softmax → multinomial sample.
- Stop on token 6562 (EOS, dropped from output) or 1000 tokens. Positions: wpe index runs over the whole
  stream (cond+text+speech contiguous).
- Context budget: 376 + text(≤~402) + speech(≤1000) < 1800 → allocate KV 2048.

### New GPU kernels needed for T3 (vs Gemma3CS):
LayerNorm(+bias) kernel; matmul-with-bias variants (or bias-add kernel); MHA is FlashAttention with heads_kv=16;
add positional lookup (wpe) fused into embed step. No RoPE, no RMSNorm, no q/k-norm. Sampler kernels reusable
(check processor order above; rep-penalty domain = generated speech tokens only).

## 3. S3Gen flow (token → mel), meanflow turbo mode
Constants: S3GEN_SR=24000, S3_SR=16000, token rate 25Hz, token_mel_ratio=2 (mel 50Hz), S3GEN_SIL=4299,
speech vocab (flow) 6561, pre_lookahead_len=3 (finalize=True at offline → no trim).

### CausalMaskedDiffWithXvec.inference graph (batch=1):
1. `embedding = spk_embed_affine_layer(normalize(x_vector_192))` — Linear(192→80); x_vector from conds.gen["embedding"].
2. `token = concat[prompt_token (conds.gen), gen_tokens+3×SIL]`
3. `input_embedding` Embedding(6561, 512) (masked by token pad mask — B=1 → all ones)
4. `h = UpsampleConformerEncoder(token_emb)` → (1, 2·T, 512)
5. `mu = encoder_proj(h)` Linear(512→80)
6. `conds_feat` = zeros(1, mel_total, 80); `conds_feat[:, :len(prompt_feat)] = prompt_feat` (from conds.gen); transpose → (1,80,T)
7. meanflow euler (`basic_euler`, NO CFG — distilled in): z = randn(1,80,mel_total) (prompt region) with
   z[..., prompt_len:] = separately generated randn (noised_mels; both are just N(0,1) — dump for parity),
   t_span = linspace(0,1,3) = [0,.5,1] (NO cosine warp in meanflow), 2 estimator calls:
   x ← x + (r−t)·estimator(x, mask, mu, t, spks=embedding, cond=conds_feat, r) for (t,r) ∈ {(0,.5),(.5,1)}
8. output mel = x[:, :, prompt_len:].

### Estimator = ConditionalDecoder (matcha-style causal "U-Net" — actually FLAT, no resampling):
in 320, out 80, channels=[256], n_blocks=4 (tfmr per level), num_mid_blocks=12, heads 8×64 (inner 512), act gelu, causal.
- time: t → SinusoidalPosEmb(320) (scale=1000: emb = 1000·t·exp(-log(10000)·i/159), cat[sin,cos]) →
  TimestepEmbedding: Linear(320→1024) → silu → Linear(1024→1024).
  meanflow: same pipeline for r; t_final = time_embed_mixer(cat[t_emb, r_emb]) — Linear(2048→1024, NO bias).
- input: x = cat[x(80), mu(80), spks·repeat_T(80), cond(80)] → (1, 320, T)
- down (1 level): CausalResnetBlock1D(320→256) → 4×BasicTransformerBlock(256) → CausalConv1d(256,256,3) [stride 1!]
  (skip saved BEFORE the k3 conv; masks all-ones at B=1)
- mid: 12 × [CausalResnetBlock1D(256→256) → 4×BasicTransformerBlock(256)]
- up (1 level): cat[x, skip] (512) → CausalResnetBlock1D(512→256) → 4×BTB(256) → CausalConv1d(256,256,3)
- final: CausalBlock1D(256→256) → final_proj Conv1d(256→80, k1) → ·mask
- CausalConv1d(k=3): left-pad (2,0) zeros, stride 1, bias.
- CausalBlock1D: CausalConv1d(k3) → **LayerNorm over channels** (transpose→LN(256)→transpose) → Mish. (base Block1D
  uses GroupNorm(8) — causal variant REPLACES it with LayerNorm.)
- CausalResnetBlock1D: h = block1(x·m); h += Linear(1024→C_out)(Mish(t_emb)) broadcast T; h = block2(h·m);
  out = h + res_conv(x·m) (Conv1d k1).
- BasicTransformerBlock (diffusers, num_embeds_ada_norm=None → **timestep IGNORED**, plain LN):
  x += to_out(attn(LN(x))) with to_q/k/v Linear(256→512, NO bias), to_out.0 Linear(512→256, bias), scale 1/√64,
  additive attn bias mask (pad-only; full bidirectional attention — B=1 offline ⇒ no mask needed);
  x += FF(LN(x)): GELU-proj Linear(256→1024) + exact gelu → Linear(1024→256, bias).
- Mish(x) = x·tanh(softplus(x)).
- 14 resnet + 56 transformer blocks, ALL at full mel resolution T (mask [:, ::2] append/pop is a no-op net effect).

## 4. Values to bake from conds.pt (exporter)
- t3.speaker_emb (1,256); t3.cond_prompt_speech_tokens (1,375?) — verify length; (emotion_adv ignored)
- gen.prompt_token (1,P); gen.prompt_feat (1,2P,80); gen.embedding (1,192)
- verify: prompt_feat frames = 2 × prompt_token count.

## 5. Sampling/parity notes
- Estimator/flow noise: N(0,1) — dump exact noise tensors in the validation harness and inject the same in Unity for parity.
- Everything runs fp32 activations / fp16 weights in the port (v1); Python reference runs fp32.

## 6. UpsampleConformerEncoder (25Hz → 50Hz), d=512, 8 heads × d_k 64
All LayerNorms in encoder LAYERS use **eps=1e-12**; embed/after_norm LN eps=1e-5.
1. `embed` (LinearNoSubsampling): Linear(512→512)+bias → LayerNorm(512, eps 1e-5) → EspnetRelPositionalEncoding:
   x·√512; pos table pe = sinusoids over relative positions [T−1 … −(T−1)] (len 2T−1):
   pos_emb[k] for rel r: standard interleaved sin/cos with div_term = exp(−ln(10000)·2i/512); positive part flipped.
   (CPU-precompute per T; upload.)
2. `pre_lookahead_layer`: (B,T,512)→transpose→ right-pad 3 → Conv1d(512,512,k4) → leaky_relu(0.01 default!) →
   left-pad 2 → Conv1d(512,512,k3) → transpose → + residual.
3. 6 × ConformerEncoderLayer (transformer-only): x += attn(LN_mha(x)); x += FF(LN_ff(x)); pre-norm, eps 1e-12.
   - RelPositionMultiHeadedAttention: linear_q/k/v (512→512, bias), linear_out (bias), linear_pos (NO bias),
     pos_bias_u/v (8,64). scores = [(q+u)·kᵀ + shift((q+v)·pᵀ)]/√64.
     GPU shortcut: bd[i,j] = (q[i]+v)·p[(T−1)−(i−j)] — direct index, NO rel_shift needed.
   - FF: Linear(512→2048)+bias → SiLU → Linear(2048→512)+bias.
   - full bidirectional attention (B=1 offline: no mask).
4. `up_layer`: nearest-neighbor ×2 on time → left-pad 4 → Conv1d(512,512,k5,s1)+bias.
5. `up_embed`: fresh Linear+LN+pos-enc (same structure as embed, own weights; pos table for 2T).
6. 4 × ConformerEncoderLayer (same structure, `up_encoders`).
7. `after_norm` LayerNorm(512, eps 1e-5).
Output → `encoder_proj` Linear(512→80).

## 7. HiFTGenerator (turbo cfg) — mel(80, 50Hz) → wav(24kHz)
cfg: base_channels 512, nb_harmonics 8, upsample_rates [8,5,3], upsample_kernels [16,11,7],
istft n_fft 16 / hop 4 (Hann periodic 16), resblock_kernels [3,7,11], dilations [1,3,5] each,
source_resblock_kernels [7,7,11] dil [1,3,5]×3, lrelu_slope 0.1, audio_limit 0.99, voiced_threshold 10,
nsf_alpha (sine_amp) 0.1, nsf_sigma (noise_std) 0.003.
1. `f0_predictor` (ConvRNNF0Predictor — despite the name: conv-only): 5 × [Conv1d(·,512,k3,p1)+bias → ELU(α=1)]
   (80→512 then 512→512×4) → Linear(512→1) → **abs** → f0 (B,T).
2. `f0_upsamp`: nearest ×480 → (B,1,480T). SourceModuleHnNSF/SineGen:
   F_mat[i] = f0·(i+1)/24000, i=0..8; theta = 2π·(cumsum(F_mat, time) mod 1);
   phase_vec[i] ~ U(−π,π) but phase[0]=0; sine = 0.1·sin(theta+phase);
   uv = f0 > 10; noise_amp = uv·0.003 + (1−uv)·0.1/3; noise = noise_amp·randn;
   s_harm = sine·uv + noise → tanh(Linear(9→1)) → s (B,1,480T). (cumsum over FULL length — sequential scan;
   dump noise/phases in harness for parity.)
3. s_stft: STFT(s, n_fft 16, hop 4, win hann16, center=True reflect-pad 8) → concat[real(9), imag(9)] → (B,18,120T+1)
4. Main: conv_pre Conv1d(80→512,k7,p3) → for i in 0..2: leaky(0.1) → ups[i] ConvTranspose1d:
   (512→256,k16,s8,p4) / (256→128,k11,s5,p3) / (128→64,k7,s3,p2); at i==2: reflection_pad left 1 (repeat x[1]);
   fusion: x += source_resblocks[i](source_downs[i](s_stft)); then x = mean of 3 ResBlocks(x) [k3,7,11].
   source_downs: i0 Conv1d(18→256,k30,s15,p7); i1 Conv1d(18→128,k6,s3,p1); i2 Conv1d(18→64,k1,s1).
   source_resblocks: ResBlock(256,k7)/(128,k7)/(64,k11), dil [1,3,5].
5. ResBlock(C,k,d=[1,3,5]): 3× { x += conv2(snake2(conv1(snake1(x)))) }, conv1 dilated (pad (k·d−d)/2), conv2 dil 1
   (pad (k−1)/2). Snake per-channel α (alpha_logscale=False): x + sin²(αx)/(α+1e-9).
6. leaky_relu(**0.01**) → conv_post Conv1d(64→18,k7,p3) → mag = exp(clip(x[:9], max=1e2)); phase = sin(x[9:])
   → real = mag·cos(phase), imag = mag·sin(phase) → iSTFT(n_fft 16, hop 4, hann16, center) → clamp(±0.99).
   iSTFT = per-frame irFFT(9 onesided bins→16 samples) · window → overlap-add (hop 4) → divide by window² OLA sum
   → trim center pad 8 each side. Output length 480·T.
7. Post: wav[:960] *= trim_fade (480 zeros + half-cosine rise 480).

## 8. Text tokenizer (turbo)
GPT2 byte-level BPE, vocab.json (50257) + merges.txt + 19 added tokens 50257..50275 ([angry],[fear],[surprised],
[whispering],[advertisement],[dramatic],[narration],[crying],[happy],[sarcastic],[clear throat],[sigh],[shush],
[cough],[groan],[sniff],[gasp],[chuckle],[laugh]). `add_bos_token=false` — NO BOS/EOS around input ids.
<|endoftext|>=50256. Port: reuse Qwen3_5TokenizerFast pattern (GPT2 byte-level BPE); exporter converts
vocab.json+merges.txt+added_tokens → a single tokenizer JSON for C#. Added tokens must be matched BEFORE BPE
(longest-match on the literal strings).

## 9. Checkpoint key map (verified via safetensors headers)
- **t3_turbo_v1.safetensors** (299 tensors, ALL F32): as §2. c_attn.weight [1024,3072], c_fc [1024,4096],
  c_proj(attn) [1024,1024], c_proj(mlp) [4096,1024] — HF Conv1D (in,out): **TRANSPOSE to (out,in) at export**.
  speech_head.weight [6563,1024] already (out,in). Skip: tfmr.wte.weight, text_head.weight.
- **s3gen_meanflow.safetensors** (2491 tensors, F32): prefixes:
  - `flow.input_embedding.weight` [6561,512]; `flow.spk_embed_affine_layer.{w,b}` [80,192];
    `flow.encoder_proj.{w,b}` [80,512]
  - `flow.encoder.embed.out.{0,1}` (Linear 512×512, LN 512); `flow.encoder.pre_lookahead_layer.conv{1,2}`;
    `flow.encoder.encoders.{0..5}.{self_attn.{linear_q,k,v,out(+bias)},linear_pos(no bias),pos_bias_u,pos_bias_v},
    feed_forward.w_1[2048,512],w_2[512,2048], norm_mha, norm_ff}`; `flow.encoder.up_layer.conv` [512,512,5];
    `flow.encoder.up_embed.out.{0,1}`; `flow.encoder.up_encoders.{0..3}.*` (same); `flow.encoder.after_norm`
  - estimator: `flow.decoder.estimator.time_mlp.linear_{1,2}`, `time_embed_mixer.weight` [1024,2048];
    `down_blocks.0.{0(resnet: block1.block.{0,2},block2.block.{0,2},mlp.1,res_conv), 1.{0..3}(BTB:
    norm1,attn1.to_{q,k,v}[512,256 no bias],attn1.to_out.0[256,512+bias],norm3,ff.net.0.proj[1024,256],ff.net.2[256,1024]),
    2(CausalConv1d k3)}`; `mid_blocks.{0..11}.{0,1.{0..3}}`; `up_blocks.0.{0,1.{0..3},2}`; `final_block.block.{0,2}`;
    `final_proj` [80,256,1].
    NOTE resnet block.N indices: block.0=CausalConv1d, block.2=LayerNorm (block = Seq(conv,transpose,LN,transpose,Mish));
    mlp.1 = time Linear (mlp = Seq(Mish, Linear)).
  - vocoder `mel2wav.*`: conv_pre/conv_post/ups.{0,1,2}/resblocks.{0..8}.convs{1,2}.{0,1,2}/source_downs.{0,1,2}/
    source_resblocks.{0..2}/f0_predictor.condnet.{0,2,4,6,8}+classifier/m_source.l_linear.
    **weight_norm = NEW parametrization style**: `<conv>.parametrizations.weight.original0` (g, [out,1,1]) +
    `original1` (v, full) → FOLD at export: w = g · v / ‖v‖₂(dims≠0). Snake alphas: resblocks.N.activations{1,2}.N.alpha.
  - SKIP at export (v1): `speaker_encoder.*` (CAMPPlus, 911 tensors), `tokenizer.*` (S3 audio tokenizer, 22).
- **conds.pt**: torch pickle; export t3.speaker_emb, t3.cond_prompt_speech_tokens, gen.prompt_token,
  gen.prompt_feat, gen.embedding as .bin (verify shapes at export; expect prompt 375 tokens / 750 mel frames).

## 10. Mel spectrogram (cloning path only — NOT needed for v1 runtime)
n_fft 1920, hop 480, win 1920 hann, 80 mels (librosa fmin 0 fmax 8000, sr 24000), center=False with reflect pad
720 both sides, log(clamp(mag, min=1e-5)).

## 11. DeepUnity port conventions (from Gemma3 stack — FOLLOW THESE)
- **Shader style** (Gemma3CS.compute): uniforms as loose globals at top; fp16 weights in
  `StructuredBuffer<uint>` read via `readH(buf,i)` (2 halves/uint, low=even); FP32 activations in
  `(RW)StructuredBuffer<float>`; per-kernel `#pragma kernel` + `[numthreads(...)]` + early-out guard;
  helpers `gelu` (tanh approx = gelu_new ✓), `silu`, `safe_tanh`, `runif()` (uses int uniform rng_seed).
  **GOTCHA:** Gemma RmsNorm applies `(1+γ)` — chatterbox norms are PLAIN γ (LayerNorm: (x−μ)·rstd·γ+β).
- **Dispatch idiom** (Gemma3Model.cs): cache kernel ids once (`FindKernel`); per op: `SetInt/SetFloat` +
  `SetBuffer` + `Dispatch(ceil)`; persistent scratch `ComputeBuffer`s with `EnsureScratch`/`Realloc`;
  `Div256(n)=(n+255)/256`. ForwardYielding yields ~1 layer/frame. Sampling on GPU → `argmaxBuf` →
  `AsyncGPUReadback` in SampleYielding(result[]). PrewarmKernels: dispatch every kernel once with zeroed
  size-uniforms + dummy buffers (one per frame), idempotent static.
- **Weights streaming** (Gemma3Weights.cs): ctor only builds manifest; background `Task` file readers
  (MAX_IO_JOBS=4, pooled byte[]) → `ConcurrentQueue<UploadJob>` → main-thread `UploadPump()` coroutine via
  `DeepUnityDispatcher.Run`, lazy buffer creation + sliced `SetData` under `LLM.UploadBudgetBytes`/frame;
  `IsReady` flips at end. **Chatterbox loader: parse `manifest.json`** (name→{file,shape,dtype}) instead of
  hardcoding — one generic pump. fp16 → packed-uint buffers (HalfBuf(count/2)); i32 → int buffer.
- **API surface** (mirror Gemma3ForCausalLM): ctor(quant, params_path=null→ResolveParamsDir, tokenizer_path,
  ...), static `Prewarm()`, `Warmup()`, `IsReady`, coroutine generate loop yielding per token, `Release()` +
  `OnReleased()`, `WarnIfNotInResources`. TTS resolves `Assets/Resources/DeepUnity/TTS/Chatterbox/<dir>`.
- **FlashAttention kernel** (Gemma3CS): supports heads_kv≠heads_q, `bidirectional` uniform, sliding window;
  K/V read through KVCache.hlsl macros (KV_FP16 keyword). Reusable for T3 (16/16, causal, KV cache incl.
  `WriteCacheFull`, declare head_dim/num_heads_kv before `#include "KVCache.hlsl"`). S3Gen attentions get
  own kernels: (a) encoder rel-pos attention (adds (q+u)kᵀ + (q+v)p[T−1−i+j]ᵀ), (b) estimator plain
  bidirectional MHA 8×64 over 512 (or 4-step scores/softmax/AV path), no cache.
- **T3 extras vs Gemma3 kernels**: LayerNorm(γ,β); matmul+bias variants (QKV fused [3072,1024]+bias,
  attn_out, fc gelu_new, mlp_out, speech_head [6563]+bias); embed = text/speech emb lookup + wpe[pos] add
  (no ×√H scale, embed_scale=1... write own kernel EmbedT3: emb[tok] + wpe[abs_pos]); NO RoPE/q-k-norm.
  Prefill embeds are CONCAT of [cond_spkr(1), prompt_speech_emb(375), text_emb(N), speech_emb(1)] — build
  hiddenBuf by 3 lookups + 1 tiny matmul (spkr_enc·speaker_emb), all GPU.
- **Sampler**: reuse SampleToken/ArgMax pattern; add HF-style repetition_penalty over generated-speech-token
  set (Qwen3_5CS has ApplyRepetitionPresencePenalty to crib). Turbo defaults: temp 0.8, top_k 1000,
  top_p 0.95, rep 1.2.
- Shaders live FLAT in `Assets/Resources/ComputeShaders/` (loaded via `DeepUnityMeta` / `Resources.Load`):
  new files `T3CS.compute` + `ChatterboxS3GenCS.compute` (+ shared snippets copied in, not .hlsl includes
  except KVCache.hlsl for T3). C# under `Assets/DeepUnity/TTS/{Chatterbox,T3,S3Gen}`.
