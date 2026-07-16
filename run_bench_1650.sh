#!/usr/bin/env bash
# GTX 1650 benchmark campaign — adapted from BENCHMARK.md for the Pavilion box.
set -u
cd "$(dirname "$0")" || exit 1

UNITY="/e/Programs/Unity/2022.3.43f1/Editor/Unity.exe"
PROJ="E:\\Development\\DeepUnity"
R=DeepUnity.LMBenchmarkProbeRunner
F=DeepUnity.FlashAttnProbeRunner

run(){ echo "== $1 == $(date '+%H:%M:%S')"; "$UNITY" -batchmode -projectPath "$PROJ" "${@:2}" -logFile "ProbeLogs/_run_$1.log"; echo "   rc=$?"; }

MODE="${1:-full}"

if [ "$MODE" = "smoke" ]; then
  run prefill_gemma_fp16 -executeMethod $R.RunPrefillProbe -model gemma -quant fp16
  exit 0
fi

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
echo "== campaign done == $(date '+%H:%M:%S')"
