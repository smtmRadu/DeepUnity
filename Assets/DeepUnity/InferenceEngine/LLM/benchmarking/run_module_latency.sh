#!/usr/bin/env bash
# ============================================================================
# Per-module latency for fig:eval-pertoken-1650 — all four models, ONE headless
# Unity launch. Fills the values the figure currently carries as dummies.
#
# GTX 1650, int8 (the tier the engine actually ships on a 4 GB card).
#
# PREREQS
#   * Unity must be CLOSED. One batch process per project, never two.
#   * Weights present under DeepUnity/Assets/Resources/Weights/:
#         weights_qwen3.5_0.8B_int8   present
#         weights_qwen3.5_2B_int8     present
#         weights_pockettts_english_int8  present
#         weights_gemma3_270M_int8    *** MISSING as of 2026-08-08 *** — gitignored;
#             regenerate with import_params.py or the Gemma arm throws immediately
#             (the probe checks first and names the directory it wants).
#   * NO -nographics: every model here runs on compute shaders, which need a device.
#
# Runtime ~5-8 min: 3 passes per arm, and only the LAST pass is reported (the first
# still carries shader compilation even after the warmup steps).
#
# Then:  python tools/fill_latency_figure.py     (writes the numbers into the thesis)
# ============================================================================
set -u
UNITY="/e/Programs/Unity/2022.3.43f1/Editor/Unity.exe"
PROJ_WIN='E:\Development\DeepUnity'
PROJ_NIX=/e/Development/DeepUnity

# ONE UNITY PROCESS PER MODEL. Not RunAll: the engine is one-model-per-session by design
# (Qwen3_5Config.ApplySize mutates statics, and the compute-shader quant keyword is shared), so
# loading a second model into a live editor crashed Unity natively after the first arm on
# 2026-08-09. BENCHMARK.md says the same -- "one Unity batch per probe".
METHODS="RunQwen08 RunQwen2B RunGemma RunPocketTts"

[ -x "$UNITY" ] || { echo "Unity not found at $UNITY"; exit 1; }

for M in $METHODS; do
  LOG="$PROJ_NIX/ProbeLogs/_module_latency_$M.log"
  echo "======== $M ========"
  "$UNITY" -batchmode -projectPath "$PROJ_WIN" -executeMethod "DeepUnity.ModuleLatencyProbe.$M" -logFile "$LOG"
  echo "rc=$?"
  grep -a "ModuleLatency\]" "$LOG" | grep -av "wrote" | head -8
  grep -aq "A crash has been intercepted" "$LOG" && echo "  !! UNITY CRASHED during $M"
  grep -a "DirectoryNotFoundException\|error CS" "$LOG" | head -3
done

echo
echo "---- json produced ----"
ls -1dt "$PROJ_NIX"/ProbeLogs/module_latency_* 2>/dev/null | head -8

echo
echo "Next: python /mnt/e/Development/Dissertation/tools/fill_latency_figure.py"
