#!/usr/bin/env python3
"""
Aggregate DeepUnity LLM benchmark probe outputs into BENCHMARK.md.

Scans ProbeLogs/<run>/summary.json (written by LMPrefillProbe, LMDecodeDecayProbe,
QuantProbe/GemmaQuantProbe, LMBootProbe), groups by GPU (machine.gpu) x model x quant x probe,
keeps the LATEST run per key, and rewrites the Speed / Quality / Boot tables in BENCHMARK.md
between the <!-- BEGIN:AUTO --> / <!-- END:AUTO --> markers. One block per distinct GPU.

Usage:
    python aggregate_benchmarks.py [--probelogs DIR] [--benchmark FILE] [--print]

Defaults resolve relative to this script: ProbeLogs at the Unity project root (4 levels up),
BENCHMARK.md next to this script.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))  # Assets/DeepUnity/LLMs/benchmarking -> root

MODEL_ORDER = ["qwen3.5-0.8B", "gemma3-270M"]
QUANT_ORDER = ["FP16", "INT8", "INT4"]
KV_ORDER = ["FP32", "FP16", "INT8"]   # KV-cache precision (independent of weight quant)
BEGIN, END = "<!-- BEGIN:AUTO -->", "<!-- END:AUTO -->"


def load_summaries(probelogs):
    """Return {(gpu, model, quant, probe): record}, newest run wins."""
    out = {}
    if not os.path.isdir(probelogs):
        return out
    runs = []
    for name in os.listdir(probelogs):
        p = os.path.join(probelogs, name, "summary.json")
        if os.path.isfile(p):
            runs.append(p)
    runs.sort(key=lambda p: os.path.getmtime(p))  # oldest first; later overwrites
    for p in runs:
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  skip (bad json): {p} ({e})", file=sys.stderr)
            continue
        gpu = (rec.get("machine") or {}).get("gpu", "unknown-gpu")
        # pre-kv-feature summaries have no "kv" field and were always FP32 KV.
        key = (gpu, rec.get("model", "?"), rec.get("quant", "?"), rec.get("kv", "FP32"), rec.get("probe", "?"))
        rec["_path"] = p
        out[key] = rec
    return out


def g(rec, *keys, default="—"):
    for k in keys:
        if rec is not None and k in rec and rec[k] is not None:
            return rec[k]
    return default


def speed_table(recs, gpu):
    rows = ["| model | weight | kv | prefill tok/s (2048) | decode tok/s (ctx≈0) | decode tok/s (max ctx) | decay % |",
            "|---|---|---|---:|---:|---:|---:|"]
    for model in MODEL_ORDER:
        for quant in QUANT_ORDER:
            for kv in KV_ORDER:
                pf = recs.get((gpu, model, quant, kv, "prefill_speed"))
                dd = recs.get((gpu, model, quant, kv, "decode_decay"))
                if pf is None and dd is None:
                    continue
                rows.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                    model, quant.lower(), kv.lower(),
                    g(pf, "median_tok_s"), g(dd, "start_tok_s"), g(dd, "end_tok_s"), g(dd, "decay_pct")))
    return "\n".join(rows) if len(rows) > 2 else "_(no speed runs for this GPU yet)_"


def quality_table(recs, gpu):
    rows = ["| model | weight | kv | max logit Δ | mean logit Δ | argmax match | greedy div (char) | decode speedup |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for model in MODEL_ORDER:
        for quant in ["INT8", "INT4"]:  # fp16 weights is the reference
            for kv in KV_ORDER:
                q = recs.get((gpu, model, quant, kv, "quant_quality"))
                if q is None:
                    continue
                match = "{}/{}".format(g(q, "argmax_match"), g(q, "compare_steps"))
                spd = g(q, "decode_speedup")
                rows.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    model, quant.lower(), kv.lower(),
                    g(q, "max_logit_diff"), g(q, "mean_logit_diff"), match,
                    g(q, "divergence_char"), "{}x".format(spd) if spd != "—" else "—"))
    return "\n".join(rows) if len(rows) > 2 else "_(no quality runs for this GPU yet)_"


def boot_table(recs, gpu):
    rows = ["| model | weight | kv | total boot s | prewarm ms | tokenizer ready ms | ctor ms | stream s | stream worst ms | stream >33ms | GC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODEL_ORDER:
        for quant in QUANT_ORDER:
            for kv in KV_ORDER:
                b = recs.get((gpu, model, quant, kv, "boot_load"))
                if b is None:
                    continue
                rows.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    model, quant.lower(), kv.lower(),
                    g(b, "total_boot_s"), g(b, "prewarm_ms"), g(b, "tokenizer_ready_ms"), g(b, "ctor_ms"),
                    g(b, "weight_stream_s"), g(b, "load_worst_frame_ms"), g(b, "load_frames_over_33ms"), g(b, "gc_gen0")))
    return "\n".join(rows) if len(rows) > 2 else "_(no boot runs for this GPU yet)_"


def build_auto(recs):
    gpus = sorted({k[0] for k in recs})
    if not gpus:
        return "_(no runs aggregated yet — run the probes, then `python Assets/DeepUnity/LLMs/benchmarking/aggregate_benchmarks.py`)_"
    out = []
    for gpu in gpus:
        out.append(f"### GPU: `{gpu}`\n")
        out.append("#### Table 2 — Speed\n")
        out.append(speed_table(recs, gpu) + "\n")
        out.append("#### Table 3 — Quality vs fp16 (fp16 = 0 reference)\n")
        out.append(quality_table(recs, gpu) + "\n")
        out.append("#### Table 4 — Boot / load & frame pacing\n")
        out.append(boot_table(recs, gpu) + "\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probelogs", default=os.path.join(PROJECT_ROOT, "ProbeLogs"))
    ap.add_argument("--benchmark", default=os.path.join(HERE, "BENCHMARK.md"))
    ap.add_argument("--print", action="store_true", help="print the generated block, don't write")
    args = ap.parse_args()

    recs = load_summaries(args.probelogs)
    print(f"[aggregate] {len(recs)} summary records from {args.probelogs}")
    for k in sorted(recs):
        print(f"  - {k[4]:14s} | {k[1]:14s} w:{k[2]:5s} kv:{k[3]:5s} | {k[0]}")

    block = build_auto(recs)
    if args.print:
        print("\n" + block)
        return

    with open(args.benchmark, encoding="utf-8") as f:
        doc = f.read()
    if BEGIN not in doc or END not in doc:
        sys.exit(f"ERROR: markers {BEGIN}/{END} not found in {args.benchmark}")
    pre = doc[:doc.index(BEGIN) + len(BEGIN)]
    post = doc[doc.index(END):]
    doc2 = pre + "\n" + block + "\n" + post
    with open(args.benchmark, "w", encoding="utf-8") as f:
        f.write(doc2)
    print(f"[aggregate] wrote {args.benchmark} ({len(recs)} records, {len({k[0] for k in recs})} GPU(s))")


if __name__ == "__main__":
    main()
