#!/usr/bin/env python3
"""
Cross-framework LLM speed benchmark for the BENCHMARK.md comparison table (dissertation).
Measures the SAME metrics as the DeepUnity probes — load s, prefill tok/s + TTFT ms
(2048-token prompt), steady decode tok/s (greedy, batch 1) and peak VRAM — on
HuggingFace transformers (SDPA) or unsloth, so the DeepUnity engine numbers have
PyTorch reference points on the same GPU.

Runs on WSL (CUDA), NOT inside Unity:
    # HF arm  (env: inference-qwen35 — tf>=5.13 for qwen3.5)
    python bench_frameworks.py Qwen/Qwen3.5-0.8B --framework hf
    # unsloth arm  (env: train-gemma3-fa — unsloth 2026.6.9)
    python bench_frameworks.py Qwen/Qwen3.5-0.8B --framework unsloth

Prints ONE json line (append with --out to a .jsonl). Prompt = random ids (speed only),
decode = manual KV-cache loop (generate() adds Python overhead that varies per version;
the manual loop is what the DeepUnity decode probes measure). fp16 weights everywhere —
the DeepUnity int8/int4 tiers have no bit-equivalent in either framework (bnb NF4 is a
different quant; bnb is also quality-banned for Qwen3.5 hybrid attention).
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--framework", choices=["hf", "unsloth"], default="hf")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default=None, help="append the json line to this file")
    a = ap.parse_args()

    # unsloth must be imported BEFORE transformers (it patches at import time)
    if a.framework == "unsloth":
        from unsloth import FastLanguageModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    if a.framework == "unsloth":
        model, tok = FastLanguageModel.from_pretrained(
            a.model_id, max_seq_length=a.prefill + a.decode + 64, dtype=dt, load_in_4bit=False)
        FastLanguageModel.for_inference(model)
    else:
        tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            a.model_id, torch_dtype=dt, attn_implementation="sdpa",
            trust_remote_code=True).cuda().eval()
    load_s = time.perf_counter() - t0

    vocab = model.get_input_embeddings().num_embeddings
    ids = torch.randint(1000, vocab - 1000, (1, a.prefill), device="cuda")

    with torch.inference_mode():
        # ---- prefill: median wall over N runs (fresh cache each) = TTFT for this prompt ----
        for _ in range(2):
            model(ids, use_cache=True)                      # warmup (kernel autotune/JIT)
        torch.cuda.synchronize()
        times = []
        for _ in range(a.runs):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = model(ids, use_cache=True)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        prefill_s = sorted(times)[len(times) // 2]

        # ---- decode: manual KV loop, greedy, steady-state over `decode` tokens ----
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(8):                                  # warm the single-token path
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(a.decode):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); decode_s = time.perf_counter() - t

    import transformers
    rec = {
        "framework": a.framework,
        "framework_version": (__import__("unsloth").__version__ if a.framework == "unsloth"
                              else transformers.__version__),
        "torch": torch.__version__,
        "model_id": a.model_id,
        "dtype": a.dtype,
        "load_s": round(load_s, 2),
        "prefill_tokens": a.prefill,
        "prefill_tok_s": round(a.prefill / prefill_s, 1),
        "ttft_ms": round(prefill_s * 1000, 1),
        "decode_tokens": a.decode,
        "decode_tok_s": round(a.decode / decode_s, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu": torch.cuda.get_device_name(0),
    }
    line = json.dumps(rec)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
