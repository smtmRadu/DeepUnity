#!/usr/bin/env bash
# Re-run of the cells that timed out (or were invalidated) in the first 2B+MiniCPM campaign:
#   - prefill/decode timeouts: the 600/900 s defaults were tuned for sub-1B models; the 1-2B
#     models on the 1650 legitimately need more wall-clock -> -timeout 2400/3600.
#   - minicpm quality: first run used weights exported BEFORE the llama norm-convention shim
#     (gamma-1 for the shared (1+g) RmsNorm kernel) — fp16 baseline was garbage. Weights are
#     re-exported; quality A/Bs re-run. (Speed/boot cells that already succeeded are kept:
#     norm VALUES don't change the arithmetic, so those timings remain valid.)
#   - qwen2b fp16 prefill/decode are NOT re-run: 3.6 GB fp16 weights on a 3.9 GB card spill to
#     shared memory — the recorded "(timed out)" cells ARE the result.
set -u
cd "$(dirname "$0")" || exit 1

UNITY="/e/Programs/Unity/2022.3.43f1/Editor/Unity.exe"
PROJ="E:\\Development\\DeepUnity"
R=DeepUnity.LMBenchmarkProbeRunner
F=DeepUnity.FlashAttnProbeRunner

run(){ echo "== $1 == $(date '+%H:%M:%S')"; "$UNITY" -batchmode -projectPath "$PROJ" "${@:2}" -logFile "ProbeLogs/_run_$1.log"; echo "   rc=$?"; }

# prefill (2048 tokens x5 reps — slowest cells first would waste nothing; keep model order)
run rr_prefill_qwen2b_int8   -executeMethod $R.RunPrefillProbe     -model qwen2b  -quant int8 -timeout 2400
run rr_prefill_qwen2b_int4   -executeMethod $R.RunPrefillProbe     -model qwen2b  -quant int4 -timeout 2400
run rr_prefill_minicpm_fp16  -executeMethod $R.RunPrefillProbe     -model minicpm -quant fp16 -timeout 2400
run rr_prefill_minicpm_int8  -executeMethod $R.RunPrefillProbe     -model minicpm -quant int8 -timeout 2400
run rr_prefill_minicpm_int4  -executeMethod $R.RunPrefillProbe     -model minicpm -quant int4 -timeout 2400

# decode decay (4096 steps; ~4 tok/s worst case ≈ 17 min + warmup)
run rr_decode_qwen2b_int8    -executeMethod $R.RunDecodeDecayProbe -model qwen2b  -quant int8 -timeout 3600
run rr_decode_qwen2b_int4    -executeMethod $R.RunDecodeDecayProbe -model qwen2b  -quant int4 -timeout 3600
run rr_decode_minicpm_int4   -executeMethod $R.RunDecodeDecayProbe -model minicpm -quant int4 -timeout 3600
# minicpm decode fp16/int8 succeeded in the first pass — not re-run.

# minicpm speed cells with corrected weights are unnecessary, but quality NEEDS the fixed norms:
run rr_quality_minicpm_int8  -executeMethod $F.RunMiniCPMInt8
run rr_quality_minicpm_int4  -executeMethod $F.RunMiniCPMInt4
# also re-run minicpm decode fp16/int8? -> not needed (same arithmetic), see header note.
# qwen2b int4 quality FAIL verdict from the first pass is genuine (int4 lossy) — kept.

echo "== rerun done == $(date '+%H:%M:%S')"
