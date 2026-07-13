# Kokoro INT8 weight variant — operational notes

Export: `python import_kokoro.py --quant int8` (WSL, torch env) →
`Assets/Resources/DeepUnity/TTS/Kokoro/weights_kokoro_int8/` (self-contained: vocab.txt,
all voices/*, CPU-side families unchanged f16). Scheme = `import_params.py quantize_int8`:
symmetric int8, ONE fp16 scale per OUTPUT ROW (`scale_r = max|w_r|/127`); manifest dtype
`q8` (`<name>.int8.bin`, GPU-packed 4-per-uint, low byte = element 0) + sibling f16 row
`<name>.scales` (`<name>.scales.bin`, shape `[rows]`). Worst recon |err| 0.01066
(`bert/layer/ffn_out.w`).

## C# rule (one line)

Per tensor, in `KokoroModel.Linear`: **if `weights.Has(name + ".w.scales")` → dispatch
`LinearBiasQ8` and additionally bind `W_scales = weights.Get(name + ".w.scales")`;
else keep `LinearBias`.** Everything else (X/W/W_bias/Y binds, uniforms, dispatch dims)
is identical. `KokoroWeights` already uploads q8 correctly (BytesPerElem 1, packed
`(numel+3)/4`-word buffer) — no loader changes needed.

## Kernel: `LinearBiasQ8` (KokoroCS.compute)

Exact twin of `LinearBias`: `[numthreads(1, 8, 32)]`, dispatch `(1, (T+7)/8, (O+31)/32)`,
same uniforms (`seq_len`, `in_dim`, `out_dim`, `activation_type`, `has_bias`,
`leaky_slope`) and fused `apply_act`. Buffers: `X` (fp32 in), `Y` (fp32 out),
`W` (q8 words), `W_bias` (fp16), `W_scales` (fp16 per-row, NEW). Per-row scale is applied
ONCE per dot product, before bias/activation:
`Y = apply_act(sum * readH(W_scales, o) + bias)`. Requires `in_dim % 4 == 0` — all q8
tensors are in_dim 128 / 768 / 2048. All 18 kernels compile (dxc `-T cs_6_0`).

## q8 tensor list (78 = every KokoroModel `Linear()` weight; each has a `.scales` sibling)

- `bert/map.w` [768,128], `benc.w` [512,768]
- `bert/layer/{attn_q,attn_k,attn_v,attn_o}.w` [768,768],
  `bert/layer/ffn.w` [2048,768] (gelu fused), `bert/layer/ffn_out.w` [768,2048]
- AdainResBlk style FCs (in 128, out 2C):
  `pred/F0_{0,1,2}/{norm1_fc,norm2_fc}.w`, `pred/N_{0,1,2}/{norm1_fc,norm2_fc}.w`,
  `dec/encode/{norm1_fc,norm2_fc}.w`, `dec/decode{0,1,2,3}/{norm1_fc,norm2_fc}.w`
- Generator Snake-block AdaIN FCs (in 128, out 2C):
  `dec/gen/noise_res{0,1}/ada{1,2}_{0,1,2}_fc.w`, `dec/gen/rb{0..5}/ada{1,2}_{0,1,2}_fc.w`

NOT quantized (stay f16): all biases, convs/convT, embeddings, norms, LSTMs,
`pred/dur_proj`, `pred/{F0,N}_proj` (convs), `dec/gen/nsf_linear`, `tenc/*`, voicepacks —
includes every family KokoroCPU reads CPU-side. The CPU full-forward ORACLE reads all
tensors as f16, so parity/oracle runs must keep using `weights_kokoro_fp16`.
