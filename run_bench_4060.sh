#!/usr/bin/env bash
# RTX 4060 (Victus) benchmark campaign — reproduces the full BENCHMARK.md matrix on the
# CURRENT build (coalesced GEMV/GEMM #31 + tiled Mimi #30 + v0.15.x). Batch mode, Unity CLOSED.
# The 4060 block in BENCHMARK.md dates from 2026-07-08 (legacy kernels) — this refreshes it.
#
# 4 models x 3 quants x {prefill, decode, boot} + 8 quality runs. Sequential (one Unity batch per
# probe; NEVER two on the same project at once). ~40-55 min on the 4060.
set -u
cd "$(dirname "$0")" || exit 1

# ── SET THESE FOR THE VICTUS ───────────────────────────────────────────────────────────────────
UNITY="/c/Program Files/Unity/Hub/Editor/2022.3.43f1/Editor/Unity.exe"   # <-- confirm Victus Unity path
PROJ="C:\\dev\\DeepUnity"                                                 # <-- confirm Victus project path (Windows-style)
# ─────────────────────────────────────────────────────────────────────────────────────────────────
R=DeepUnity.LMBenchmarkProbeRunner
F=DeepUnity.FlashAttnProbeRunner

run(){ echo "== $1 == $(date '+%H:%M:%S')"; "$UNITY" -batchmode -projectPath "$PROJ" "${@:2}" -logFile "ProbeLogs/_run_$1.log"; echo "   rc=$?"; }

MODE="${1:-full}"

# smoke = one fast run to confirm paths/weights/entry points before the full ~45 min campaign
if [ "$MODE" = "smoke" ]; then
  run prefill_gemma_fp16 -executeMethod "$R".RunPrefillProbe -model gemma -quant fp16
  echo "smoke done — check ProbeLogs/prefill_gemma3-270M_*/summary.json for \"success\": true"
  exit 0
fi

# ── sub-1B models: Qwen3.5-0.8B + Gemma3-270M (default timeouts) ──
for q in fp16 int8 int4; do
  run prefill_qwen_$q   -executeMethod "$R".RunPrefillProbe     -model qwen  -quant $q
  run prefill_gemma_$q  -executeMethod "$R".RunPrefillProbe     -model gemma -quant $q
  run decode_qwen_$q    -executeMethod "$R".RunDecodeDecayProbe -model qwen  -quant $q
  run decode_gemma_$q   -executeMethod "$R".RunDecodeDecayProbe -model gemma -quant $q
  run boot_qwen_$q      -executeMethod "$R".RunBootProbe        -model qwen  -quant $q
  run boot_gemma_$q     -executeMethod "$R".RunBootProbe        -model gemma -quant $q
done

# ── 1-2B scaling models: Qwen3.5-2B + MiniCPM5-1B (longer -timeout; 2B fp16 fits in the 4060's 8 GB) ──
for q in fp16 int8 int4; do
  run prefill_qwen2b_$q   -executeMethod "$R".RunPrefillProbe     -model qwen2b   -quant $q -timeout 3600
  run prefill_minicpm_$q  -executeMethod "$R".RunPrefillProbe     -model minicpm5 -quant $q -timeout 3600
  run decode_qwen2b_$q    -executeMethod "$R".RunDecodeDecayProbe -model qwen2b   -quant $q -timeout 3600
  run decode_minicpm_$q   -executeMethod "$R".RunDecodeDecayProbe -model minicpm5 -quant $q -timeout 3600
  run boot_qwen2b_$q      -executeMethod "$R".RunBootProbe        -model qwen2b   -quant $q -timeout 3600
  run boot_minicpm_$q     -executeMethod "$R".RunBootProbe        -model minicpm5 -quant $q -timeout 3600
done

# ── quality A/B vs fp16 (int8/int4 only; fp16 is the 0 reference; no -model/-quant args) ──
run quality_qwen_int8     -executeMethod "$F".RunQwenInt8
run quality_qwen_int4     -executeMethod "$F".RunQwenInt4
run quality_gemma_int8    -executeMethod "$F".RunGemmaInt8
run quality_gemma_int4    -executeMethod "$F".RunGemmaInt4
run quality_qwen2b_int8   -executeMethod "$F".RunQwen2BInt8
run quality_qwen2b_int4   -executeMethod "$F".RunQwen2BInt4
run quality_minicpm_int8  -executeMethod "$F".RunMiniCPMInt8
run quality_minicpm_int4  -executeMethod "$F".RunMiniCPMInt4

# ── kernel A/B: legacy vs new, Qwen decode (#31) + pocket-tts Mimi (#30). These probes don't write
#    summary.json and don't self-Exit, so use -quit and read the numbers from the logs (they are
#    transcribed into BENCHMARK.md's A/B section by hand, not auto-aggregated). ──
ab(){ echo "== $1 == $(date '+%H:%M:%S')"; "$UNITY" -batchmode -quit -projectPath "$PROJ" "${@:2}" -logFile "ProbeLogs/_run_$1.log"; echo "   rc=$?"; }
ab qwenprof_legacy -executeMethod DeepUnity.LegacyKernelABRunner.QwenLegacy
ab qwenprof_new    -executeMethod DeepUnity.Qwen3_5Modeling.QwenDecodeProfileProbe.Run
ab pocket_legacy   -executeMethod DeepUnity.LegacyKernelABRunner.PocketLegacy
ab pocket_new      -executeMethod DeepUnity.PocketTTSModeling.PocketTTSRtfProbe.RunInt8
echo "A/B numbers: grep '\[QwenProfile\]' ProbeLogs/_run_qwenprof_*.log ; grep '\[PocketRTF\]' ProbeLogs/_run_pocket_*.log"

echo "== campaign done == $(date '+%H:%M:%S')"
echo "Now aggregate: python Assets/DeepUnity/InferenceEngine/LLM/benchmarking/aggregate_benchmarks.py"
echo "(rewrites the 4060 block in BENCHMARK.md from ProbeLogs/*/summary.json)"
