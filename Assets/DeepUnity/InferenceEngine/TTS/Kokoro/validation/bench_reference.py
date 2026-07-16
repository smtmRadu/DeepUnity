#!/usr/bin/env python3
"""
hexgrad Kokoro-82M REFERENCE (PyTorch KPipeline) benchmark on CPU / CUDA — the framework
comparison points for the DeepUnity Kokoro port in BENCHMARK.md. Same 3-sentence lighthouse
passage and metrics as KokoroRtfProbe: load s, TTFA ms (first yielded chunk), generation
wall -> RTF (median of 3 after 1 warmup).

Runs on WSL in the `kokoro` env (kokoro 0.9.4 + misaki):
    python bench_reference.py --device cpu
    python bench_reference.py --device cuda   # needs a CUDA torch build in the env

Appends one json line per run via --out.
"""
import argparse, json, time

TEXT = ("The old lighthouse keeper climbed the spiral stairs every evening at dusk. "
        "He lit the great lamp and watched the beam sweep across the darkening waves; "
        "ships far at sea counted on that light to find their way home safely.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from kokoro import KPipeline

    t0 = time.perf_counter()
    try:
        pipeline = KPipeline(lang_code="a", device=a.device)
    except TypeError:                                   # older KPipeline without device kwarg
        pipeline = KPipeline(lang_code="a")
    load_s = time.perf_counter() - t0

    def gen(measure_ttfa=False):
        chunks, ttfa = [], None
        t = time.perf_counter()
        for _gs, _ps, audio in pipeline(TEXT, voice=a.voice):
            if ttfa is None:
                ttfa = (time.perf_counter() - t) * 1000
            chunks.append(audio)
        wall = time.perf_counter() - t
        n = sum(c.numel() if hasattr(c, "numel") else len(c) for c in chunks)
        return wall, n / 24000.0, (round(ttfa, 1) if measure_ttfa else None)

    gen()                                               # warmup (kernel JIT / voice load)
    walls, audio_sec, ttfa_ms = [], 0, None
    for i in range(a.runs):
        wall, audio_sec, t = gen(measure_ttfa=(i == 0))
        walls.append(wall)
        if t is not None:
            ttfa_ms = t
    wall = sorted(walls)[len(walls) // 2]

    rec = {
        "framework": "kokoro-ref",
        "torch": torch.__version__,
        "device": a.device,
        "threads": torch.get_num_threads() if a.device == "cpu" else None,
        "load_s": round(load_s, 2),
        "audio_sec": round(audio_sec, 2),
        "gen_wall_s": round(wall, 3),
        "rtf": round(wall / audio_sec, 3),
        "ttfa_ms": ttfa_ms,
        "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None,
    }
    line = json.dumps(rec)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
