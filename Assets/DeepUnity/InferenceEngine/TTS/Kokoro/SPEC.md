# Kokoro-82M → DeepUnity Port Spec (exact, from source)

Source of truth: `C:\dev\_model_staging\kokoro\kokoro-repo` (hexgrad/kokoro @ master, pkg v0.9.4) +
HF `hexgrad/Kokoro-82M` (`kokoro-v1_0.pth`, `config.json`, `voices/*.pt`) +
`C:\dev\_model_staging\kokoro\misaki-repo` (G2P — see G2P_PROPOSAL.md).
This file records the EXACT inference graph as implemented in Python (`kokoro/model.py`
`KModel.forward_with_tokens`, `kokoro/modules.py`, `kokoro/istftnet.py`); the C#/HLSL port must
match it tensor-for-tensor. Style follows `Assets/DeepUnity/TTS/SPEC.md` (Chatterbox).

## 0. Big picture
StyleTTS2-lineage, NON-autoregressive (single feed-forward pass per text chunk — no KV cache,
no sampling loop; the only stochastic part is the vocoder's NSF noise):

```
text ──G2P──> phoneme string ps (≤510 chars) ──vocab──> input_ids [1,T]  (T = len(ps)+2)
voicepack[len(ps)-1] -> ref_s [1,256] = [ acoustic style s_d (0:128) | prosody style s_p (128:256) ]

input_ids ─> PLBERT (ALBERT, 12 shared layers) ─> bert_dur [1,T,768]
          ─> bert_encoder Linear ─> d_en [1,512,T]
(d_en, s_p) ─> DurationEncoder(3×{biLSTM,AdaLN}) ─> d [1,T,640]
d ─> biLSTM ─> duration_proj ─> sigmoid.sum/speed ─> round ─> pred_dur [T]  (frames @ 40 Hz)
pred_dur ─> alignment matrix aln [T,F], F = Σ pred_dur
dᵀ@aln ─> en [1,640,F] ─> shared biLSTM ─> F0/N conv stacks (×2 time upsample) ─> F0,N [1,2F] (80 Hz)
input_ids ─> TextEncoder(emb+3×CNN+biLSTM) ─> t_en [1,512,T];  asr = t_en@aln [1,512,F]
(asr, F0, N, s_d) ─> Decoder(AdainResBlk1d stack) ─> [1,512,2F]
                 ─> Generator (NSF source + 2×upsample + Snake resblocks + iSTFT) ─> wav [600F] @ 24 kHz
```
1 duration frame = 600 samples = 25 ms. Output: 24 kHz mono fp32.
Whole model 82 M params, fp32 checkpoint 327 MB → fp16 export ≈ 164 MB.

## 1. Checkpoint + assets (HF hexgrad/Kokoro-82M)
- `kokoro-v1_0.pth` — torch pickle: dict of 5 sub-state-dicts keyed `bert`, `bert_encoder`,
  `predictor`, `text_encoder`, `decoder` (VERIFIED: 25+2+122+375+24 = 548 tensors, ~82M params
  incl. skipped pooler, all fp32, every inner key `module.`-prefixed). Weight norm is stored
  OLD-style (`<conv>.weight_g` [out,1,1] + `<conv>.weight_v` full) → fold w = g·v/‖v‖₂(dims≠0)
  at export. Loading REQUIRES torch (pickle) — exporter runs in the WSL `kokoro` conda env.
- `config.json` — dims + the phoneme vocab (`vocab`: symbol → id, 114 symbols used of n_token=178).
  Reproduced in §8.
- `voices/<name>.pt` — voicepack: fp32 tensor **[510, 1, 256]**. Runtime picks ONE row:
  `ref_s = pack[len(phonemes)-1]` → [1,256]. `ref_s[:, :128]` styles the decoder/generator
  (AdaIN/AdaInResBlk), `ref_s[:, 128:]` styles the prosody predictor (AdaLayerNorm + AdainResBlk).
  Multi-voice mixing = plain mean of packs. Staged voices: af_heart, af_bella, am_michael.
- Config values: n_token 178, hidden_dim 512, style_dim 128, n_layer 3, max_dur 50,
  text_encoder_kernel_size 5, plbert{hidden 768, heads 12, inter 2048, max_pos 512, layers 12},
  istftnet{rates [10,6], kernels [20,12], initial_ch 512, resblock_k [3,7,11], dil [1,3,5]×3,
  n_fft 20, hop 5}.

## 2. Input handling (KModel.forward)
- `input_ids = [0, *[vocab[p] for p in ps if p in vocab], 0]` — id 0 = `$` boundary token at BOTH
  ends (unknown symbols silently dropped). Assert len ≤ 512 (ALBERT max_position_embeddings).
- B=1 everywhere → text_mask all-false; every `masked_fill_`/`pack_padded_sequence` is a no-op.
  All dropouts eval no-op (incl. the explicit `dropout(x, 0.5, training=False)` before duration_proj).
- `speed` (float, default 1) divides the summed duration BEFORE rounding.

## 3. PLBERT = HF AlbertModel (CustomAlbert returns last_hidden_state; pooler UNUSED — skip export)
ALBERT ⇒ **ONE shared transformer layer applied 12 times** (huge export saving; loop the same
weights). embedding_size=128 (AlbertConfig default) ≠ hidden 768.
- Embeddings: word_emb [178,128] + pos_emb [512,128] (absolute, index 0..T-1) + token_type_emb
  [2,128] (all index 0) → LayerNorm(128, eps 1e-12) → `embedding_hidden_mapping_in` Linear(128→768).
- Layer (×12, post-LN): MHA 12 heads × d64: q/k/v/dense Linear(768→768)+bias,
  softmax(QKᵀ/√64) **bidirectional** (no causal mask); x = LN(x + attn_dense_out);
  ffn Linear(768→2048) → **gelu_new** (tanh approx — DeepUnity `gelu()`) → ffn_output
  Linear(2048→768); x = LN(x + ffn_out). Both LN eps 1e-12, affine.
- Output bert_dur [1,T,768] → `bert_encoder` Linear(768→512), transpose → d_en [1,512,T].

## 4. ProsodyPredictor (style = s_p = ref_s[:,128:], 128-d)
### 4.1 DurationEncoder (predictor.text_encoder)
x = cat[d_en_t, s_p broadcast] → [1,T,640]; then 3 × { biLSTM(640→2×256) → AdaLayerNorm(128,512)
→ re-cat style (→640) }. Output d [1,T,640] (last cat included).
- AdaLayerNorm: `[γ,β] = fc(s)` Linear(128→1024) split; out = (1+γ)·LN_noaffine(x, 512, eps 1e-5)+β.
  NOTE the (1+γ) — same convention trap as Gemma RmsNorm.
### 4.2 Duration head
biLSTM `predictor.lstm` (640→2×256) → [1,T,512] → `duration_proj` LinearNorm(512→50, bias) →
sigmoid → sum over 50 dims → /speed → round → clamp(min=1) → long → pred_dur [T].
(Boundary $ tokens get durations too — they pad the audio ends.)
### 4.3 Alignment
`aln` [T,F]: row t has 1s over the pred_dur[t] consecutive frame columns (CPU integer build +
upload, or index-map kernel). en = dᵀ@aln [1,640,F].
### 4.4 F0/N predictors (F0Ntrain)
shared biLSTM (640→2×256) over en → x [1,F,512] → transpose [1,512,F].
- F0 path: AdainResBlk1d(512→512, s128) → AdainResBlk1d(512→256, **upsample ×2**) →
  AdainResBlk1d(256→256) → F0_proj Conv1d(256→1,k1,p0) → F0 [1,2F] (80 Hz).
- N path: identical twin (N.0/N.1/N.2, N_proj).
- AdainResBlk1d (istftnet.py; also used in decoder §6): pre-act residual, ALL convs weight-normed:
  `res = conv2(lrelu0.2(norm2(conv1(lrelu0.2(norm1(x,s)) via pool), s)))` where
  norm* = AdaIN1d, conv* = Conv1d(k3,p1,bias), pool = Identity or (upsample: depthwise
  ConvTranspose1d(C_in,C_in,k3,s2,groups=C_in,p1,output_padding=1));
  `short = nearest×2?(x) then conv1x1 (no bias) if C_in≠C_out`; out = (res+short)·rsqrt(2).
- AdaIN1d: InstanceNorm1d(C) with PER-SAMPLE stats (no running stats): over time dim per channel:
  μ_c,σ_c → (x-μ)/√(σ²+1e-5), THEN style: (1+γ_s)·IN(x)+β_s with [γ_s,β_s] = fc(s) Linear(128→2C).
  NOTE: current code sets InstanceNorm affine=True (ONNX-bug workaround) but the CHECKPOINT has no
  `norm.weight/bias` keys — the reference's strict=False load leaves them at init (w=1,b=0) =
  identity. Port as PLAIN InstanceNorm + style affine.

## 5. TextEncoder (t_en branch)
Embedding(178→512) → transpose [1,512,T] → 3 × { weight-norm Conv1d(512,512,k5,p2,bias) →
LayerNorm over channels (transpose→LN(512, eps 1e-5, affine γ,β)→transpose) → LeakyReLU(0.2) } →
biLSTM(512→2×256) → t_en [1,512,T]. asr = t_en@aln [1,512,F].

## 6. Decoder (istftnet.Decoder; style = s_d = ref_s[:,:128])
- F0 = F0_conv(F0_pred[1,1,2F]) — weight-norm Conv1d(1,1,k3,**s2**,p1) → [1,1,F]; N likewise.
- x = cat[asr(512), F0(1), N(1)] = [1,514,F] → `encode` AdainResBlk1d(514→1024).
- asr_res = weight-norm Conv1d(512→64,k1) of asr.
- decode.0..2: x = AdainResBlk1d(cat[x,asr_res,F0,N] = 1090 → 1024).
- decode.3: AdainResBlk1d(1090→512, upsample=True) → [1,512,2F] (nearest×2 shortcut + depthwise
  ConvT pool in residual). After this block NO re-cat (res flag drops).
- → Generator(x, s_d, F0_pred).

## 7. Generator (istftnet.Generator — HiFiGAN-NSF + iSTFT head, ADAIN-conditioned)
Constants: 2 upsample stages [10,6] k[20,12], resblocks 3 kernels [3,7,11] dil [1,3,5],
n_fft 20 → 11 onesided bins, hop 5, win = hann(20, periodic). All convs (ups, resblocks,
conv_post) weight-normed; noise_convs are NOT weight-normed.
### 7.1 NSF source (torch.no_grad branch)
- f0_upsamp = nearest ×300 of F0_pred [1,2F] → f0 [1, S=600F, 1] @24 kHz.
- SineGen(24000, upsample_scale=300, harmonics 8+1, sine_amp 0.1, noise_std 0.003, thresh 10):
  fn = f0·[1..9]; rad = (fn/24000) mod 1; **rand_ini** ~U(0,1) per harmonic (index 0 forced 0)
  ADDED to rad[:,0,:] (first sample only);
  rad ↓×(1/300) linear-interp → [1,9,2F] → transpose → cumsum over time ×2π
  → (phase·300) ↑×300 linear-interp → sin → sine_waves ×0.1.
  uv = (f0>10); noise_amp = uv·0.003+(1-uv)·0.1/3; **noise** = noise_amp·randn[1,S,9];
  sine = sine·uv + noise.  (Second SourceModule output `noise` and `uv` are UNUSED downstream.)
- har = tanh(l_linear(9→1)(sine)) [1,S,1] → squeeze → STFT(n_fft 20, hop 5, win hann20,
  center=True reflect-pad 10) → mag [1,11,120F+1], phase(angle) → har_cat = [mag;phase] [1,22,·].
  NOTE: cat is [magnitude, ANGLE] (not real/imag like Chatterbox HiFT).
### 7.2 Main trunk
- i=0: x = lrelu(x, **0.1**); x_src = noise_res0(noise_convs0(har_cat), s_d)
  (Conv1d(22→256,k12,s6,p3) → AdaINResBlock1(256,k7,dil[1,3,5])); x = ups0(x)
  ConvTranspose1d(512→256,k20,s10,p5) → [1,256,20F]; x += x_src;
  x = mean(resblocks0..2(x, s_d)) (AdaINResBlock1(256, k3/7/11)).
- i=1: x = lrelu(x, 0.1); x_src = noise_res1(noise_convs1(har_cat), s_d)
  (Conv1d(22→128,k1) → AdaINResBlock1(128,k11)); x = ups1(x)
  ConvTranspose1d(256→128,k12,s6,p3) → [1,128,120F]; **reflection_pad (1,0)** (prepend x[:,:,1])
  → [1,128,120F+1]; x += x_src; x = mean(resblocks3..5(x, s_d)) (128, k3/7/11).
- x = lrelu(x, **0.01 — DEFAULT slope, differs from the 0.1 in-loop!**) → conv_post
  Conv1d(128→22,k7,p3) → spec = exp(x[:,:11]); phase = sin(x[:,11:])
  → iSTFT(spec·e^{i·phase}, n_fft 20, hop 5, hann20, center=True) → wav [1,600F], no final clamp.
### 7.3 AdaINResBlock1 (kernel k, dil [1,3,5])
3 × { xt = AdaIN1d(x,s); xt = xt + (1/α1)·sin²(α1·xt)  (Snake, α [1,C,1], NO eps);
      xt = conv1_d(xt) (dilated, pad (k·d−d)/2); xt = AdaIN1d(xt,s); Snake α2;
      xt = conv2_d1(xt) (pad (k−1)/2); x = x + xt }.
Same Snake as Chatterbox but α indexes per-channel and 1/α exact (no +1e-9 guard; α init 1, never 0).

## 8. Phoneme vocab (config.json `vocab`, id: symbol)
0=`$`(boundary, implicit — not in the json map), 1=`;` 2=`:` 3=`,` 4=`.` 5=`!` 6=`?` 9=`—` 10=`…`
11=`"` 12=`(` 13=`)` 14=`“` 15=`”` 16=` `(space) 17=`◌̃` 18-23=`ʣʥʦʨᵝꭧ` 24=A 25=I 31=O 33=Q 35=S
36=T(flap ɾ) 39=W 41=Y 42=`ᵊ` 43-68=a..z(some gaps) 69+=IPA (ɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽ
ʂʃʈʧʊʋʌɣɤχʎʒʔ) 156=`ˈ` 157=`ˌ` 158=`ː` 162=`ʰ` 164=`ʲ` 169=`↓` 171=`→` 172=`↗` 173=`↘` 177=`ᵻ`.
Full map exported by import_kokoro.py as `vocab.txt` (line i = symbol for id i) next to the weights.
American English G2P additionally post-maps `ɾ→T`, `ʔ→t` (misaki v1 behavior).

## 9. GPU/CPU split for the port
**CPU (C#, fp32):**
- G2P (dictionary lookup — G2P_PROPOSAL.md) + vocab mapping.
- All 6 biLSTMs (DurationEncoder ×3, duration lstm, shared, TextEncoder's).
  Sequential dependency chains, H=256/dir, T≤512 / F≤~1500 steps → a per-step GPU dispatch would
  be dispatch-bound (cf. Chatterbox decode); flat C# loops with fp32 math are ~10 MFLOP/LSTM-pass,
  microseconds-to-ms territory. Gate order torch: i,f,g,o; out = cat[fwd, bwd reversed].
- Duration sigmoid-sum/round/clamp + alignment build (integer); SineGen phase pipeline at frame
  rate (downsample-interp, cumsum over 2F ≤ ~3000 steps, upsample-interp — cheap, and keeps the
  RNG on CPU for parity injection).
**GPU (HLSL kernels, fp16 weights / fp32 activations, mirroring Gemma3CS/T3CS style):**
- ALBERT: embed+LN, Linear(+bias) matmuls, bidirectional FlashAttention 12×64 (Chatterbox
  estimator/conformer attention kernels reusable — no cache, no rel-pos), gelu_new, post-LN.
- TextEncoder conv stack; all AdainResBlk1d stacks (F0/N, decoder); Generator convs/ConvT/Snake;
  InstanceNorm (two-pass mean/var over time per channel, like LayerNorm kernel but reduce over T);
  AdaIN style FCs (tiny matmuls); t_en@aln + dᵀ@aln matmuls; STFT/iSTFT.
- STFT/iSTFT: n_fft **20** is NOT a power of two — Chatterbox's radix-2 FFT16 shader does not apply.
  11 bins × 20 samples → direct DFT matmul kernel (20×11 sin/cos tables), trivially cheap.
  center=True reflect-pad both directions; iSTFT = per-frame inverse DFT ×window → overlap-add
  (hop 5) → divide by OLA window² sum → trim pad 10 each side (mirror Chatterbox iSTFT kernel).
**Precision watch-list:** cumsum phase fp32 (S up to ~360k samples → phase magnitude ~10⁴ rad;
fp32 sin fine, fp16 NOT); duration sigmoid-sum near .5 rounding boundaries (fp16 weights shift
durations by ±1 frame occasionally — acceptable; parity probe compares durations with tolerance).

## 10. Weight export map (import_kokoro.py — fp16 manifest.tsv, ChatterboxWeights format)
Folder `Assets/Resources/DeepUnity/TTS/Kokoro/weights_kokoro_fp16/`; name → file/dtype/numel/shape
manifest lines; weight-norm FOLDED at export (old-style `weight_g`/`weight_v` → g·v/‖v‖₂, dims≠0);
LSTMs exported as 8 tensors each (wih/whh/bih/bhh × fwd + `_r` reverse; torch gate order i,f,g,o).
Naming: `bert/*` (ONE shared layer under `bert/layer/`), `benc.{w,b}` (bert_encoder),
`pred/*` (DurationEncoder `durenc/lstm{0-2}`+`adaln{0-2}`, `lstm`, `dur_proj`, `shared`,
`F0_{0-2}`/`N_{0-2}`, `F0_proj`/`N_proj`), `tenc/*` (TextEncoder), `dec/*` (encode, decode{0-3},
F0_conv/N_conv, asr_res, `gen/*`), `voices/<name>` [510,256] (flattened from [510,1,256]).
Skipped: bert pooler only. Exporter tracks consumed keys and FAILS on leftovers — the manifest is
the schema and is guaranteed complete.

## 11. API surface (future C#, mirrors ChatterboxTTS/ChatterboxVoice)
`KokoroTTS(params_path=null→Resources resolve, voice="af_heart", LLMQuant.FP16)` → `Warmup()` →
`Speak(text, onClip, speed=1f)` / `Synthesize(text, onWav)` coroutines → `Release()`.
Text chunking at sentence punctuation (KPipeline waterfall: split so each chunk's phoneme string
≤510), one forward per chunk, chunks streamed into `KokoroVoice` ring buffer (ChatterboxVoice
pattern) — Kokoro has no token-level streaming (non-AR), so time-to-first-audio = first-chunk
synth time; chunker should cut the first clause short (firstChunkChars pattern).
No sampler, no temperature: the ONLY nondeterminism is SineGen rand_ini + noise (inject for parity).

## 12. Known gotchas (write these into the C# port)
1. AdaIN/AdaLayerNorm apply **(1+γ_style)**. AdaIN1d's InstanceNorm affine is identity at runtime
   (params absent from the checkpoint; strict=False load leaves init 1/0) — port plain IN.
2. InstanceNorm uses per-utterance statistics — output depends on the whole chunk (no streaming
   through the decoder mid-chunk).
3. `duration_proj` output passes sigmoid THEN sums 50 channels (NOT softmax-argmax).
4. F0/N are at 80 Hz (2F) but decoder immediately strided-conv's them back to 40 Hz; generator's
   NSF consumes the 80 Hz curve. Don't "optimize" the ×2/÷2 pair away — F0_conv/N_conv have
   learned weights.
5. Final pre-conv_post leaky_relu slope is 0.01, in-loop ones are 0.1 (same trap as Chatterbox).
6. har_cat = [magnitude, PHASE-ANGLE], not [real, imag].
7. torch nearest-neighbor ×2 (shortcut) vs depthwise-ConvT ×2 (residual pool) differ — both exist
   in the same block.
8. `reflection_pad (1,0)` prepends index-1 (not index-0) sample.
9. Phoneme ids not in vocab are DROPPED silently before the [0,...,0] wrap.
10. Voicepack row selection uses len(phoneme string) BEFORE vocab filtering (`pack[len(ps)-1]`).
