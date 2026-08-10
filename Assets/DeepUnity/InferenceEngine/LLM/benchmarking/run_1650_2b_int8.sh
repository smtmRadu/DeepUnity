#!/usr/bin/env bash
# ============================================================================
# 2B arm of the cross-engine comparison (fig:eval-engine-comparison, middle panel).
# GTX 1650, headless (WSL, no display). Fills the four values still marked DUMMY in
# chapters/evaluation/benchmark_data_2026.tex.
#
# WHY int8 FOR THE PYTORCH ARMS: Qwen3.5-2B in fp16 is 3.59 GB of weights on a 3.9 GB
# card. An fp16 arm there measures the driver paging, not the library, so both PyTorch
# arms run bitsandbytes LLM.int8() -- the same reason the engine arm is int8. It is NOT
# the same quantizer as the engine's (see bench_frameworks.py docstring); the panel
# compares what each stack DELIVERS at 8 bits on this card, which is the question a
# developer targeting 4 GB actually has.
#
# EXPECT: LLM.int8() is frequently SLOWER than fp16 at batch 1 (its per-matmul outlier
# split costs more than the narrower weight loads save). A decode number below the fp16
# one is a real result, not a broken run -- do not "fix" it.
#
# PROTOCOL CHECK FIRST: the 0.8B fp16 run must reproduce HF 358 / unsloth 343 tok/s
# prefill. If it does not, the restored --logits_to_keep path no longer matches what
# produced results_1650/framework_bench_1650_lastlogit.jsonl, and nothing measured next
# to it is comparable. Read that line before trusting the 2B rows.
#
# Launch detached -- a plain shell gets reaped mid-install:
#   setsid nohup bash run_1650_2b_int8.sh > ~/bench2b_int8.log 2>&1 < /dev/null &
# Then, in the thesis repo:
#   python tools/fill_engine_2b.py        (reads the jsonl below, writes the figure macros)
#
# Runtime: ~25-35 min including two venv builds and the HF downloads.
# ============================================================================
set -u
# This script lives next to bench_frameworks.py in the engine repo; results land here too,
# and the thesis-side filler reads them from this path.
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$BENCH/results_1650/framework_bench_1650_2b_int8.jsonl"

# DISK: the WSL vhdx lives at E:\Development\wsl\Ubuntu, so ANYTHING written inside WSL grows a
# file on E: -- which on this box runs at ~5 GB free. Two torch venvs are ~5 GB and would refill it.
# So the venvs go on C: and the HF cache points at the models already downloaded on E:
# (Qwen3.5-{0.8B,2B} and their unsloth mirrors are all present, ~12 GB), which means this script
# downloads NOTHING. Override either path if you run it on another machine.
VENV_ROOT="${VENV_ROOT:-/mnt/c/Users/radup/bench_venvs}"
export HF_HOME="${HF_HOME:-/mnt/e/Development/hf_cache}"
mkdir -p "$VENV_ROOT"
HF_VENV="$VENV_ROOT/bench_hf"
UNSLOTH_VENV="$VENV_ROOT/bench_unsloth"
echo "venvs:    $VENV_ROOT   ($(df -h "$VENV_ROOT" | tail -1 | awk '{print $4}') free)"
echo "HF_HOME:  $HF_HOME     (models pre-cached; no downloads)"
WITH_FP16=${WITH_FP16:-0}      # 1 = also attempt the 2B in fp16 (expected to OOM/page)
mkdir -p "$BENCH/results_1650"

run() {   # run <python> <framework> <model> <quant>
  echo "=== $2 $3 $4 ==="
  "$1" "$BENCH/bench_frameworks.py" "$3" --framework "$2" --quant "$4" --out "$OUT" \
    && echo "OK $2 $3 $4" \
    || echo "!! FAILED $2 $3 $4 (rc=$?)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

# torch is PINNED to 2.10.0: the published lastlogit numbers were taken on 2.10.0+cu128, and an
# unpinned reinstall pulls 2.11.0+cu128, which on this WSL/WDDM driver cannot allocate at all --
# a bare 1.5 GB tensor fails with 3.45 GB free and the allocator reports a nonsense
# "17179869184 GiB in use". Do not relax this pin without re-checking that allocation works.
TORCH_PIN='torch==2.10.0'
TORCH_IDX=https://download.pytorch.org/whl/cu128

echo "############ HF arm ############"
if [ ! -x "$HF_VENV"/bin/python ]; then python3 -m venv "$HF_VENV"; fi
"$HF_VENV"/bin/pip install -q --upgrade pip
"$HF_VENV"/bin/pip install -q "$TORCH_PIN" --index-url "$TORCH_IDX"
"$HF_VENV"/bin/pip install -q 'transformers==5.13.1' accelerate sentencepiece huggingface_hub bitsandbytes
"$HF_VENV"/bin/python - <<'PY'
import torch, bitsandbytes
print("HF env: torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0), "| bnb", bitsandbytes.__version__)
PY
run "$HF_VENV"/bin/python hf Qwen/Qwen3.5-0.8B fp16     # <-- protocol check: expect prefill 358
run "$HF_VENV"/bin/python hf Qwen/Qwen3.5-0.8B int8     # so the 0.8B panel can match the 2B's format
run "$HF_VENV"/bin/python hf Qwen/Qwen3.5-2B   int8
[ "$WITH_FP16" = 1 ] && run "$HF_VENV"/bin/python hf Qwen/Qwen3.5-2B fp16

echo "############ unsloth arm ############"
if [ ! -x "$UNSLOTH_VENV"/bin/python ]; then python3 -m venv "$UNSLOTH_VENV"; fi
"$UNSLOTH_VENV"/bin/pip install -q --upgrade pip
"$UNSLOTH_VENV"/bin/pip install -q "$TORCH_PIN" --index-url "$TORCH_IDX"
# unsloth is PINNED to 2026.6.9, the build the published 343 tok/s was taken with. Unpinned, the
# resolver backs off to 2025.9.5 to satisfy torch==2.10.0, and that build dies at import against
# transformers 5.x with "NameError: auto_docstring" (it execs patched modeling source that expects a
# symbol which moved). Install unsloth LAST so it cannot drag transformers off the pin.
"$UNSLOTH_VENV"/bin/pip install -q 'transformers==5.13.1' bitsandbytes
"$UNSLOTH_VENV"/bin/pip install -q --no-deps 'unsloth==2026.6.9' unsloth_zoo
"$UNSLOTH_VENV"/bin/python -c 'import torch;print("unsloth env: torch",torch.__version__,"cuda",torch.cuda.is_available())'
run "$UNSLOTH_VENV"/bin/python unsloth Qwen/Qwen3.5-0.8B fp16   # <-- protocol check: expect 343
run "$UNSLOTH_VENV"/bin/python unsloth Qwen/Qwen3.5-0.8B int8
run "$UNSLOTH_VENV"/bin/python unsloth Qwen/Qwen3.5-2B   int8
[ "$WITH_FP16" = 1 ] && run "$UNSLOTH_VENV"/bin/python unsloth Qwen/Qwen3.5-2B fp16

echo "############ DONE -> $OUT ############"
cat "$OUT" 2>/dev/null
echo
echo "Next: python /mnt/e/Development/Dissertation/tools/fill_engine_2b.py"
