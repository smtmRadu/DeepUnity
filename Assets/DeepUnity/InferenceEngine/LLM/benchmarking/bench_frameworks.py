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
the manual loop is what the DeepUnity decode probes measure).

--quant fp16 | int8
    fp16 is the default and what the published 0.8B numbers used.
    int8 loads through bitsandbytes LLM.int8() and exists for ONE reason: Qwen3.5-2B in fp16
    is 3.59 GB of weights on a 3.9 GB card, so an fp16 arm there measures paging, not the
    library. READ THE int8 NUMBERS AS "what this library delivers at 8 bits", never as a
    format-for-format comparison with the engine: LLM.int8() is a different scheme entirely
    (row-wise int8 with an fp16 outlier path decomposed per matmul), whereas the engine ships
    symmetric per-output-row int8 dequantized inside the shader. LLM.int8() is also routinely
    SLOWER than fp16 at batch 1 -- the outlier split costs more than the narrower loads save --
    so a decode number below the fp16 one is an expected result, not a broken run.
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--framework", choices=["hf", "unsloth"], default="hf")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                    help="compute dtype (also the weight dtype when --quant fp16)")
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16",
                    help="int8 = bitsandbytes LLM.int8() weight quantization; see the module "
                         "docstring for why the 2B needs it and how not to read the number")
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    # The published 1650 numbers (results_1650/framework_bench_1650_lastlogit.jsonl) were taken
    # with this ON, but the flag was missing from this file, so they were not reproducible.
    # OFF, the model projects EVERY position onto the vocabulary: [1, 2048, 248320] fp16 is
    # 970 MiB of logits of which only the last row is read. Measured on the 0.8B that costs HF
    # 249 vs 358 tok/s and 3497 vs 1934 MB peak. The engine only ever projects the last chunk,
    # so ON is the comparison that is actually fair.
    ap.add_argument("--all-logits", action="store_true",
                    help="project every position (the unfair, memory-hungry default of the "
                         "PyTorch libraries) instead of just the last")
    ap.add_argument("--out", default=None, help="append the json line to this file")
    a = ap.parse_args()

    # unsloth must be imported BEFORE transformers (it patches at import time)
    if a.framework == "unsloth":
        from unsloth import FastLanguageModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    torch.cuda.reset_peak_memory_stats()

    int8 = a.quant == "int8"
    bnb_version = None
    if int8:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
        print(f"quant: bitsandbytes LLM.int8() {bnb_version}", flush=True)

    t0 = time.perf_counter()
    if a.framework == "unsloth":
        kw = dict(max_seq_length=a.prefill + a.decode + 64, dtype=dt, load_in_4bit=False)
        if int8:
            kw["load_in_8bit"] = True
        try:
            model, tok = FastLanguageModel.from_pretrained(a.model_id, **kw)
        except TypeError as e:
            # older unsloth builds have no load_in_8bit kwarg. Do NOT silently fall back to
            # fp16: that would file an fp16 number under an int8 label.
            raise SystemExit(f"unsloth cannot do int8 here ({e}); upgrade unsloth or drop --quant int8")
        FastLanguageModel.for_inference(model)
    else:
        tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)
        kw = dict(torch_dtype=dt, attn_implementation="sdpa", trust_remote_code=True)
        if int8:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kw["device_map"] = {"": 0}   # .cuda() is rejected on a quantized model
        model = AutoModelForCausalLM.from_pretrained(a.model_id, **kw)
        if not int8:
            model = model.cuda()
        model = model.eval()
    load_s = time.perf_counter() - t0

    # A wrong-path guard, because "int8" that silently loaded fp16 is the one failure mode that
    # would poison the figure without looking like a failure.
    if int8:
        kinds = {type(m).__name__ for m in model.modules()}
        if not any("8bit" in k or "Int8" in k for k in kinds):
            raise SystemExit("--quant int8 requested but no 8-bit Linear layers are present; "
                             "refusing to report this as an int8 run")

    # EVERY weight must be on the GPU. With bitsandbytes + accelerate, a model that does not fit is
    # silently split across CPU and GPU: it still runs, still reports a tok/s, and that number is
    # meaningless next to the engine's. Fail loudly instead.
    devs = {p.device.type for p in model.parameters()}
    if devs != {"cuda"}:
        raise SystemExit(f"model parameters are on {sorted(devs)}, not all on cuda -- "
                         "refusing to report a partly-CPU run")
    print(f"all weights on GPU ({torch.cuda.get_device_name(0)})", flush=True)

    vocab = model.get_input_embeddings().num_embeddings
    ids = torch.randint(1000, vocab - 1000, (1, a.prefill), device="cuda")

    lk = {} if a.all_logits else {"logits_to_keep": 1}
    print("prefill projects %s" % ("ALL positions" if a.all_logits else "the last position only"),
          flush=True)

    with torch.inference_mode():
        # ---- prefill: median wall over N runs (fresh cache each) = TTFT for this prompt ----
        for _ in range(2):
            model(ids, use_cache=True, **lk)                # warmup (kernel autotune/JIT)
        torch.cuda.synchronize()
        times = []
        for _ in range(a.runs):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = model(ids, use_cache=True, **lk)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        prefill_s = sorted(times)[len(times) // 2]

        # ---- decode: manual KV loop, greedy, steady-state over `decode` tokens ----
        out = model(ids, use_cache=True, **lk)
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
        "quant": a.quant,
        "bitsandbytes": bnb_version,
        "load_s": round(load_s, 2),
        "prefill_tokens": a.prefill,
        "prefill_tok_s": round(a.prefill / prefill_s, 1),
        "ttft_ms": round(prefill_s * 1000, 1),
        "decode_tokens": a.decode,
        "decode_tok_s": round(a.decode / decode_s, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu": torch.cuda.get_device_name(0),
        # recorded so a result file can never again be ambiguous about which protocol produced it
        "last_logit_only": not a.all_logits,
    }
    line = json.dumps(rec)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
