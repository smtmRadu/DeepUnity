# DeepUnity LLM Benchmark

Canonical results table for the paper. Every number is for one **(model × config × GPU)** cell.
Models: **Qwen3.5-0.8B / 2B** (24 layers, 18 Gated-DeltaNet + 6 full-attn, vocab 248320; the 2B only
widens hidden/MLP), **MiniCPM5-1B** (vanilla llama, 24 full-attn layers, UNTIED lm_head, vocab 130560)
and **Gemma3-270M** (18 layers, 15 SWA + 3 global, vocab 262144).

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
| **pavilion** | Pavilion Gaming 15-dk0xxx (Win10) | GTX 1650 Laptop 4 GB (i5-9300H) | second GPU, D3D11 |

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

Then aggregate: `python Assets/DeepUnity/LLM/benchmarking/aggregate_benchmarks.py` — scans `ProbeLogs/*/summary.json`
and rewrites Tables 2–4 below (between the AUTO markers), one block per distinct GPU.

### Run the full campaign on a new GPU (e.g. Pavilion) — copy/paste

Reproduces every cell for a new machine. **Prereqs:** Unity **closed**; editor **2022.3.43f1**; all 6 weight
sets present under `Assets/Resources/DeepUnity/LLM/{Qwen3_5,Gemma3}/weights_*_{fp16,int8,int4}/` (regenerate
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
python Assets/DeepUnity/LLM/benchmarking/aggregate_benchmarks.py   # appends a new GPU block automatically
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
| Qwen3.5-2B | fp16 | 3590 | — |
| Qwen3.5-2B | int8 | 2284 | −36% |
| Qwen3.5-2B | int4 | 1710 | −52% |
| MiniCPM5-1B | fp16 | 2062 | — |
| MiniCPM5-1B | int8 | 1415 | −31% |
| MiniCPM5-1B | int4 | 1131 | −45% |
| Qwen3.5-0.8B | fp16 | 1436 | — |
| Qwen3.5-0.8B | int8 | 963 | −33% |
| Qwen3.5-0.8B | int4 | 755 | −47% |
| Gemma3-270M | fp16 | 512 | — |
| Gemma3-270M | int8 | 417 | −19% |
| Gemma3-270M | int4 | 375 | −27% |

MiniCPM5-1B's int savings are proportionally smaller than Qwen's: its **untied** lm_head doubles the
always-fp16 embedding cost (2 × 130560·1536 fp16 ≈ 0.77 GB floor that no weight quant touches).

KV cache (FP32, separate from weights). Per-token cost = `Σ_layers 2(K,V)·kv_heads·head_dim·4 B`, summed only
over layers that actually cache K/V at the given context. DeltaNet recurrent/conv state is fixed-size (does not
grow with context) and is excluded here.
- **Qwen3.5-0.8B / 2B** — only the **6 full-attn** layers (idx 3,7,11,15,19,23; the 18 DeltaNet layers cache no
  KV). kv_heads=2, head_dim=256 → `6·2·2·256·4 B` = **24 KB/token**, growing linearly to 8192 ctx. The 2B has
  the SAME attention config (only hidden/intermediate grow), so its KV cost is identical to the 0.8B's.
- **MiniCPM5-1B** — vanilla llama: **all 24 layers** are full attention and cache KV. kv_heads=2, head_dim=128
  → `24·2·2·128·4 B` = **48 KB/token** — the highest KV cost of the four models despite the smallest head_dim,
  because nothing (DeltaNet or SWA) caps it.
- **Gemma3-270M** — **3 global** layers (idx 5,11,17) grow with context; the **15 sliding-window** layers cap at
  window=512. kv_heads=1, head_dim=256 → global = `3·2·1·256·4 B` = 6 KB/token; SWA contributes a fixed
  `15·2·1·256·4·512 B` = 15 MB once ctx ≥ 512.

| model | KV KB/token | KV @ max ctx (MB) | max ctx |
|---|---:|---:|---:|
| Qwen3.5-2B | 24.0 | 192 | 8192 |
| MiniCPM5-1B | 48.0 | 384 | 8192 |
| Qwen3.5-0.8B | 24.0 | 192 | 8192 |
| Gemma3-270M | 6.0 (global) + 15 MB fixed SWA | 27 | 2048 |

(max ctx here = the KV capacity the engine pre-allocates by default, not the checkpoint's positional limit —
MiniCPM5 supports 131k positions; Qwen3.5 262k.)

---

## Tables 2–4 — Speed / Quality / Boot (auto-generated)

Generated by `aggregate_benchmarks.py` from `ProbeLogs/*/summary.json`. **Do not hand-edit between the
markers** — re-run the aggregator to refresh. One block per distinct GPU (`machine.gpu`).

<!-- BEGIN:AUTO -->
### GPU: `NVIDIA GeForce GTX 1650`

#### Table 2 — Speed

| model | weight | kv | prefill tok/s (2048) | decode tok/s (ctx≈0) | decode tok/s (max ctx) | decay % |
|---|---|---|---:|---:|---:|---:|
| qwen3.5-2B | fp16 | fp16 | — | 0.0 | 0.0 | 0.0 |
| qwen3.5-2B | int8 | int8 | 12.3 | 4.1 | 3.9 | 3.1 |
| qwen3.5-2B | int4 | int8 | — | 1.5 | 1.5 | 0.9 |
| minicpm5-1B | fp16 | fp16 | 21.9 | 7.0 | 6.0 | 13.3 |
| minicpm5-1B | int8 | int8 | 21.6 | 7.4 | 6.2 | 15.8 |
| minicpm5-1B | int4 | int8 | 17.2 | 2.2 | 2.1 | 5.3 |
| qwen3.5-0.8B | fp16 | fp16 | 34.1 | 8.2 | 7.9 | 4.1 |
| qwen3.5-0.8B | int8 | int8 | 34.0 | 8.3 | 7.9 | 5.1 |
| qwen3.5-0.8B | int4 | int8 | 25.9 | 0.0 | 0.0 | 0.0 |
| gemma3-270M | fp16 | fp16 | 124.5 | 22.3 | 20.9 | 6.5 |
| gemma3-270M | int8 | int8 | 122.8 | 22.8 | 20.8 | 8.5 |
| gemma3-270M | int4 | int8 | 110.7 | 19.5 | 18.1 | 7.1 |

#### Table 3 — Quality vs fp16 (fp16 = 0 reference)

| model | weight | kv | max logit Δ | mean logit Δ | argmax match | greedy div (char) | decode speedup |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3.5-2B | int8 | int8 | 0.5007 | 0.063713 | 8/8 | -1 | 1.08x |
| qwen3.5-2B | int4 | int8 | 3.0684 | 0.477537 | 4/8 | 1 | 0.39x |
| minicpm5-1B | int8 | int8 | 0.9556 | 0.103492 | 8/8 | 101 | 1.06x |
| minicpm5-1B | int4 | int8 | 6.2644 | 1.007999 | 7/8 | 97 | 0.31x |
| qwen3.5-0.8B | int8 | int8 | 0.4596 | 0.075648 | 7/8 | -1 | 1.01x |
| qwen3.5-0.8B | int4 | int8 | 3.3893 | 0.503522 | 7/8 | 1 | 0.56x |
| gemma3-270M | int8 | int8 | 3.8474 | 0.757592 | 8/8 | 7 | 1.02x |
| gemma3-270M | int4 | int8 | 24.9383 | 3.775131 | 1/8 | 1 | 0.87x |

#### Table 4 — Boot / load & frame pacing

| model | weight | kv | total boot s | prewarm ms | tokenizer ready ms | ctor ms | stream s | stream worst ms | stream >33ms | GC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5-2B | fp16 | fp16 | 9.83 | 772.7 | 1600.7 | 827.9 | 7.38 | 829.94 | 2 | 5 |
| qwen3.5-2B | int8 | int8 | 7.07 | 727.0 | 1510.1 | 783.1 | 4.61 | 785.23 | 2 | 4 |
| qwen3.5-2B | int4 | int8 | 4.31 | 736.4 | 1120.5 | 384.0 | 0.98 | 386.18 | 1 | 3 |
| minicpm5-1B | fp16 | fp16 | 1.71 | 341.9 | 416.8 | 74.8 | 0.83 | 87.7 | 1 | 2 |
| minicpm5-1B | int8 | int8 | 2.02 | 344.3 | 603.9 | 259.5 | 1.13 | 280.17 | 1 | 2 |
| minicpm5-1B | int4 | int8 | 1.74 | 329.8 | 397.4 | 67.5 | 0.49 | 78.03 | 1 | 2 |
| qwen3.5-0.8B | fp16 | fp16 | 3.02 | 852.5 | 941.6 | 89.0 | 0.78 | 94.27 | 2 | 3 |
| qwen3.5-0.8B | int8 | int8 | 2.83 | 850.6 | 946.6 | 95.9 | 0.66 | 101.2 | 2 | 3 |
| qwen3.5-0.8B | int4 | int8 | 2.99 | 905.3 | 1003.1 | 97.6 | 0.64 | 102.85 | 2 | 3 |
| gemma3-270M | fp16 | fp16 | 1.89 | 1025.3 | 1188.7 | 163.3 | 0.53 | 165.75 | 2 | 2 |
| gemma3-270M | int8 | int8 | 1.83 | 316.1 | 1508.3 | 89.6 | 1.19 | 91.75 | 2 | 3 |
| gemma3-270M | int4 | int8 | 1.88 | 340.0 | 1544.9 | 104.8 | 1.21 | 110.08 | 2 | 3 |

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

**1650 (Pavilion) matrix — ✅ COMPLETE.** Same 3 tiers × 2 models run on the GTX 1650 (4 GB, i5-9300H, Win10,
D3D11). 21/22 cells valid; aggregator added the second GPU block above. Two non-success cells: `gemma int4
quality` collapses (same documented failure as the 4060), and **`qwen int4 decode` timed out** — int4 decode is
the slowest tier (Q4_0 dequant overhead) and on this card it can't complete a measurement bucket within the
probe's wall-clock limit, so that Table-2 cell reads 0.0 (effectively unusable, ~4–5 tok/s extrapolated).

Headline (1650 vs 4060):
- **~3–4× slower across the board.** qwen fp16: prefill 34 vs 134 tok/s, decode 8.2 vs 31. gemma fp16: prefill
  124 vs 417, decode 22 vs 59. gemma (~22 tok/s decode) is borderline usable; qwen (~8 tok/s) is not.
- Same quant story as the 4060: int8 speed-neutral & safe; int4 a memory play that *slows* decode; gemma int4
  collapses. Quant trends are GPU-independent — the 1650 just shifts the whole speed axis down.

**Scaling campaign (1650 only) — ✅ COMPLETE.** Two larger models added to test the entry-GPU boundary:
**Qwen3.5-2B** (same hybrid arch, hidden 2048) and **MiniCPM5-1B** (vanilla llama on the SAME kernel set —
zero shader changes; only `import_params.py` + a config record). Probe timeouts got a `-timeout` CLI arg
(the 600/900 s defaults were tuned for sub-1B models). Headline:
- **The VRAM wall is binary.** qwen2b fp16 = 3.59 GB weights on a 3.9 GB card → D3D11 pages to shared
  memory and prefill/decode never complete ("—"/0.0 cells above are the *result*, not an error). The int8
  export of the SAME model (2.28 GB) runs: 12.3 prefill / 4.1 decode tok/s. Quantization = runnability.
- **minicpm5-1B ≈ the upper end of usable on a 1650**: 22 prefill / 7 decode tok/s, <2 s boot, int8 free
  (8/8, maxΔ 0.96). qwen2b int8 (4 tok/s decode) is patience-only.
- **int4 = memory play, never speed**, on a third and fourth architecture: decode 0.31–0.37× vs fp16;
  quality 4/8 (qwen2b) / 7/8 (minicpm). And the untied-lm_head floor (Table 1) shrinks int savings.
- Two implementation gotchas worth recording for posterity (both found by the fp16 quality baseline
  producing garbage, both fixed): (1) llama norms are `x̂·γ` but the shared RmsNorm kernel computes
  `x̂·(1+γ)` (Gemma convention) → llama exports now write `γ−1` (`import_params.py` shim); (2) newer HF
  `tokenizer.json` files store BPE merges as `[["a","b"],…]` pair-arrays, which the legacy `"a b"` parser
  silently ignored → single-character tokenization. `MiniCPM5TokenizerFast.ParseMerges` handles both.

**Remaining:**
- ✅ **Pavilion GPU** — done (GTX 1650 block above).
- ✅ KV cache size formula (Table 1, second sub-table) — filled from configs.
- ✅ **Qwen3.5-2B + MiniCPM5-1B scaling campaign** (1650) — done, see above.
- ⬜ **4060 (Victus) cells for the two new models** — run `run_bench_2b_minicpm.sh` there (2B fp16 should
  FIT in 8 GB — the interesting contrast cell).
- ⬜ (optional) KL divergence + fixed-text perplexity in the quant probes for a stronger quality axis.
- ⬜ (cleanup) `qwen int4 decode` 1650 cell shows 0.0 (timed out) — re-run with `-timeout 3600` for a real number, or annotate.

### Tooling (all ✅)
- Weight memory (Table 1) — measured.
- `LMPrefillProbe` / `LMDecodeDecayProbe` — emit `summary.json` (prefill / decode+decay), kv-tagged.
- `QuantProbe` / `GemmaQuantProbe` — emit `summary.json` (logit Δ, argmax match, decode tok/s, speedup),
  kv-tagged, batch-safe (timeout + self-Exit).
- `LMBootProbe` — parameterized by model+quant+kv, emits `summary.json`.
- `aggregate_benchmarks.py` — fills Tables 2–4 from `ProbeLogs`, one block per GPU.
