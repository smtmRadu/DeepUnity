# DeepUnity LLM Benchmark

Canonical results table for the paper. Every number is for one **(model × config × GPU)** cell.
Models: **Qwen3.5-0.8B** (24 layers, 18 Gated-DeltaNet + 6 full-attn, vocab 248320) and
**Gemma3-270M** (18 layers, vocab 262144).

**Standard benchmark configs (weight quant → KV-cache quant)** — 3 per model, **auto-paired by the
runner** (`LMBenchmarkProbeRunner.StandardKV`); weight-only quant keeps norms + tied embedding/lm_head
fp16 (see `OPTIMIZATIONS.md`). DeltaNet recurrent/conv state is always fp32.
- **fp16 → fp16 KV** — lossless reference tier
- **int8 → int8 KV** — VRAM-optimized tier
- **int4 → int8 KV** — max-compression tier

Run a config with `-quant fp16|int8|int4` (KV auto-pairs); `-kvquant fp32|fp16|int8` overrides for
one-off KV-precision experiments. Fill order: fp16 config first (all models, both GPUs), then int8, int4.

## Machines (both GPUs)

| tag | machine | GPU | notes |
|---|---|---|---|
| **4060** | Victus (hostname rpc, Win11) | RTX 4060 Laptop 8 GB | primary dev box, D3D11 |
| **pavilion** | Pavilion Gaming 15-dk0xxx (Win10) | _TBD — fill from probe `machine.gpu`_ | second GPU |

Each probe stamps the exact GPU/CPU/driver into its `summary.json` `machine` block — the aggregator
keys rows off `machine.gpu`, so a row can always be traced to the box it ran on.

## How to produce the numbers (headless)

ONE model+quant per editor run, `-batchmode` **without** `-nographics` (compute shaders need a graphics
device). Reports land in `ProbeLogs/<tag>_<timestamp>/` (`report.md`, `summary.json`, `per_token.csv`).
Headless gotcha: Unity can report a misleading exit code — confirm `summary.json` exists with
`"success": true` rather than trusting the exit code.

```
# prefill tok/s (2048-token prompt)        -> summary.json {prefill_speed}
Unity.exe -projectPath <proj> -batchmode -executeMethod DeepUnity.LMBenchmarkProbeRunner.RunPrefillProbe     -model qwen  -quant fp16
# decode tok/s + decay over context         -> summary.json {decode_decay}
Unity.exe -projectPath <proj> -batchmode -executeMethod DeepUnity.LMBenchmarkProbeRunner.RunDecodeDecayProbe  -model gemma -quant fp16
# boot / load / frame pacing                 -> summary.json {boot_load}
Unity.exe -projectPath <proj> -batchmode -executeMethod DeepUnity.LMBenchmarkProbeRunner.RunBootProbe         -model qwen  -quant int8
# quality vs fp16 (int8/int4 ONLY; fp16 = 0 reference) -> summary.json {quant_quality}
#   FlashAttnProbeRunner.RunQwenInt8 / RunQwenInt4 / RunGemmaInt8 / RunGemmaInt4   (no -model/-quant args)
```

Then aggregate: `python Assets/DeepUnity/LLMs/benchmarking/aggregate_benchmarks.py` — scans `ProbeLogs/*/summary.json`
and rewrites Tables 2–4 below (between the AUTO markers), one block per distinct GPU.

### Run the full campaign on a new GPU (e.g. Pavilion) — copy/paste

Reproduces every cell for a new machine. **Prereqs:** Unity **closed**; editor **2022.3.43f1**; all 6 weight
sets present under `Assets/Resources/DeepUnity/LLMs/{Qwen3_5,Gemma3}/weights_*_{fp16,int8,int4}/` (regenerate
with `import_params.py` if missing — they're gitignored); run from the project root in Git-Bash. 22 runs,
~20 min, sequential (one Unity batch per probe — never two on the same project at once). KV auto-pairs to the
weight quant (`StandardKV`: fp16→fp16, int8→int8, int4→int8); quality probes take **no** `-model/-quant`.

```bash
UNITY="/c/Program Files/Unity/Hub/Editor/2022.3.43f1/Editor/Unity.exe"   # adjust per machine
PROJ="C:\\dev\\DeepUnity"                                                # Windows-style path for -projectPath
R=DeepUnity.LMBenchmarkProbeRunner; F=DeepUnity.FlashAttnProbeRunner
run(){ echo "== $1 =="; "$UNITY" -batchmode -projectPath "$PROJ" "${@:2}" -logFile "ProbeLogs/_run_$1.log"; echo "rc=$?"; }
for q in fp16 int8 int4; do
  run prefill_qwen_$q  -executeMethod $R.RunPrefillProbe     -model qwen  -quant $q
  run prefill_gemma_$q -executeMethod $R.RunPrefillProbe     -model gemma -quant $q
  run decode_qwen_$q   -executeMethod $R.RunDecodeDecayProbe -model qwen  -quant $q
  run decode_gemma_$q  -executeMethod $R.RunDecodeDecayProbe -model gemma -quant $q
  run boot_qwen_$q     -executeMethod $R.RunBootProbe        -model qwen  -quant $q
  run boot_gemma_$q    -executeMethod $R.RunBootProbe        -model gemma -quant $q
done
# quality A/B vs fp16 (int8/int4 only; fp16 is the 0 reference)
run quality_qwen_int8  -executeMethod $F.RunQwenInt8
run quality_qwen_int4  -executeMethod $F.RunQwenInt4
run quality_gemma_int8 -executeMethod $F.RunGemmaInt8
run quality_gemma_int4 -executeMethod $F.RunGemmaInt4
python Assets/DeepUnity/LLMs/benchmarking/aggregate_benchmarks.py   # appends a new GPU block automatically
```

Verify each produced `ProbeLogs/<tag>_<ts>/summary.json` with `"success": true` (gemma int4 quality is
*expected* `false` — documented collapse; its speed/boot rows are still valid). The aggregator keys on
`machine.gpu`, so the new machine becomes its own block under the AUTO markers — then fill that GPU's row in
the **Machines** table above from any `summary.json` `machine.gpu` string.

---

## Table 1 — Weight memory (offline, from `import_params.py` export) ✅ COMPLETE

Same bytes on disk and in VRAM (fp16-packed). GPU-independent.

| model | quant | weight (MB) | vs fp16 |
|---|---|---:|---:|
| Qwen3.5-0.8B | fp16 | 1436 | — |
| Qwen3.5-0.8B | int8 | 963 | −33% |
| Qwen3.5-0.8B | int4 | 755 | −47% |
| Gemma3-270M | fp16 | 512 | — |
| Gemma3-270M | int8 | 417 | −19% |
| Gemma3-270M | int4 | 375 | −27% |

KV cache (FP32, separate from weights) — _formula TBD/fill_: Qwen full-attn ≈ 2·kv_heads·head_dim·6·4 B/token;
Gemma sliding-window layers cap at window=512, only the full layers grow with context.

| model | KV KB/token | KV @ max ctx (MB) | max ctx |
|---|---:|---:|---:|
| Qwen3.5-0.8B | _TBD_ | _TBD_ | 8192 |
| Gemma3-270M | _TBD_ | _TBD_ | 2048 |

---

## Tables 2–4 — Speed / Quality / Boot (auto-generated)

Generated by `aggregate_benchmarks.py` from `ProbeLogs/*/summary.json`. **Do not hand-edit between the
markers** — re-run the aggregator to refresh. One block per distinct GPU (`machine.gpu`).

<!-- BEGIN:AUTO -->
### GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

#### Table 2 — Speed

| model | weight | kv | prefill tok/s (2048) | decode tok/s (ctx≈0) | decode tok/s (max ctx) | decay % |
|---|---|---|---:|---:|---:|---:|
| qwen3.5-0.8B | fp16 | fp16 | 134.4 | 31.3 | 27.6 | 11.8 |
| qwen3.5-0.8B | int8 | int8 | 133.8 | 31.3 | 26.8 | 14.2 |
| qwen3.5-0.8B | int4 | int8 | 101.3 | 25.1 | 22.1 | 11.9 |
| gemma3-270M | fp16 | fp16 | 416.9 | 59.0 | 52.2 | 11.4 |
| gemma3-270M | int8 | int8 | 409.5 | 58.7 | 50.3 | 14.3 |
| gemma3-270M | int4 | int8 | 352.4 | 46.0 | 40.9 | 11.1 |

#### Table 3 — Quality vs fp16 (fp16 = 0 reference)

| model | weight | kv | max logit Δ | mean logit Δ | argmax match | greedy div (char) | decode speedup |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3.5-0.8B | int8 | int8 | 0.4596 | 0.075648 | 7/8 | -1 | 1.01x |
| qwen3.5-0.8B | int4 | int8 | 3.3893 | 0.503522 | 7/8 | 1 | 0.8x |
| gemma3-270M | int8 | int8 | 3.8474 | 0.757592 | 8/8 | 7 | 0.98x |
| gemma3-270M | int4 | int8 | 24.9383 | 3.775131 | 1/8 | 1 | 0.77x |

#### Table 4 — Boot / load & frame pacing

| model | weight | kv | total boot s | prewarm ms | tokenizer ready ms | ctor ms | stream s | stream worst ms | stream >33ms | GC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5-0.8B | fp16 | fp16 | 1.86 | 511.0 | 577.8 | 66.7 | 0.58 | 73.73 | 1 | 3 |
| qwen3.5-0.8B | int8 | int8 | 2.06 | 496.9 | 651.9 | 155.0 | 0.74 | 159.03 | 2 | 3 |
| qwen3.5-0.8B | int4 | int8 | 2.0 | 477.7 | 598.7 | 121.0 | 0.65 | 124.68 | 1 | 3 |
| gemma3-270M | fp16 | fp16 | 1.21 | 215.1 | 1011.6 | 54.3 | 0.8 | 58.15 | 2 | 3 |
| gemma3-270M | int8 | int8 | 1.36 | 220.4 | 1139.6 | 89.7 | 0.92 | 96.87 | 2 | 3 |
| gemma3-270M | int4 | int8 | 1.36 | 225.3 | 1129.5 | 87.2 | 0.9 | 91.11 | 2 | 3 |

<!-- END:AUTO -->

---

## Status

**4060 (Victus) matrix — ✅ COMPLETE.** All 3 standard tiers × 2 models populated above (fp16→fp16 KV,
int8→int8 KV, int4→int8 KV): speed (Table 2), quality (Table 3, int8/int4 only), boot (Table 4). 22
`summary.json` records. Quality probes now A/B the full shipped config (quant weights + int8 KV vs fp16+fp16 KV)
and tag `kv` accordingly.

Headline reads:
- **Speed** — decode is dispatch-bound: int8 is speed-neutral vs fp16; **int4 is *slower*** (Q4_0 group
  dequant overhead with no bandwidth win at this size — qwen 0.80×, gemma 0.77× decode). int4 is a
  memory-footprint play, not a speed play.
- **Quality** — int8 safe on both (qwen 7/8 argmax & identical greedy text; gemma 8/8 argmax). int4: qwen
  usable (coherent text, 7/8), **gemma int4 collapses** (1/8 argmax, maxΔ 24.9, garbled output) — confirms
  the documented "int4 = benchmark-only for gemma".

**Remaining:**
- ⬜ **Pavilion GPU** — re-run the same 3-tier campaign there; the aggregator auto-adds a second GPU block.
- ⬜ KV cache size formula (Table 1, second sub-table).
- ⬜ (optional) KL divergence + fixed-text perplexity in the quant probes for a stronger quality axis.

### Tooling (all ✅)
- Weight memory (Table 1) — measured.
- `LMPrefillProbe` / `LMDecodeDecayProbe` — emit `summary.json` (prefill / decode+decay), kv-tagged.
- `QuantProbe` / `GemmaQuantProbe` — emit `summary.json` (logit Δ, argmax match, decode tok/s, speedup),
  kv-tagged, batch-safe (timeout + self-Exit).
- `LMBootProbe` — parameterized by model+quant+kv, emits `summary.json`.
- `aggregate_benchmarks.py` — fills Tables 2–4 from `ProbeLogs`, one block per GPU.
