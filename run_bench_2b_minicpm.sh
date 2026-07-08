#!/usr/bin/env bash
# GTX 1650 benchmark campaign for the NEW models: Qwen3.5-2B + MiniCPM5-1B.
# Same probe set as run_bench_1650.sh (speed/boot per quant + quality A/B vs fp16).
# NOTE: qwen2b fp16 (~4 GB weights) is expected to OOM/spill on the 4 GB 1650 — the run is
# kept anyway; a failed/timed-out cell is itself a datapoint (record what happens).
set -u
cd "$(dirname "$0")" || exit 1

UNITY="/e/Programs/Unity/2022.3.43f1/Editor/Unity.exe"
PROJ="E:\\Development\\DeepUnity"
R=DeepUnity.LMBenchmarkProbeRunner
F=DeepUnity.FlashAttnProbeRunner

run(){ echo "== $1 == $(date '+%H:%M:%S')"; "$UNITY" -batchmode -projectPath "$PROJ" "${@:2}" -logFile "ProbeLogs/_run_$1.log"; echo "   rc=$?"; }

MODE="${1:-full}"

if [ "$MODE" = "smoke" ]; then
  # cheapest end-to-end sanity: minicpm int8 prefill (fits easily in 4 GB)
  run prefill_minicpm_int8 -executeMethod $R.RunPrefillProbe -model minicpm -quant int8
  exit 0
fi

for q in fp16 int8 int4; do
  run prefill_qwen2b_$q  -executeMethod $R.RunPrefillProbe     -model qwen2b  -quant $q
  run prefill_minicpm_$q -executeMethod $R.RunPrefillProbe     -model minicpm -quant $q
  run decode_qwen2b_$q   -executeMethod $R.RunDecodeDecayProbe -model qwen2b  -quant $q
  run decode_minicpm_$q  -executeMethod $R.RunDecodeDecayProbe -model minicpm -quant $q
  run boot_qwen2b_$q     -executeMethod $R.RunBootProbe        -model qwen2b  -quant $q
  run boot_minicpm_$q    -executeMethod $R.RunBootProbe        -model minicpm -quant $q
done
# quality A/B vs fp16 (int8/int4 only; fp16 is the 0 reference).
# qwen2b quality boots the fp16 reference FIRST — on the 1650 this cell may time out.
run quality_qwen2b_int8  -executeMethod $F.RunQwen2BInt8
run quality_qwen2b_int4  -executeMethod $F.RunQwen2BInt4
run quality_minicpm_int8 -executeMethod $F.RunMiniCPMInt8
run quality_minicpm_int4 -executeMethod $F.RunMiniCPMInt4
echo "== campaign done == $(date '+%H:%M:%S')"
