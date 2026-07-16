#!/usr/bin/env python3
"""
Kyutai pocket-tts REFERENCE (PyTorch) benchmark on CPU / CPU-quantized / CUDA — the framework
comparison points for the DeepUnity GPU port's numbers in BENCHMARK.md. Same 3-sentence
lighthouse passage as PocketTTSRtfProbe, same metrics: load s, TTFA ms (first streamed chunk),
generation wall -> RTF (offline generate_audio, median of 3 after 1 warmup).

Runs on WSL in the `pocket` env (pocket-tts 2.1.0 editable install, torch 2.13):
    python bench_reference.py --device cpu        # fp32 CPU (Kyutai's design target)
    python bench_reference.py --device cpu --quantize   # their quantized CPU path
    python bench_reference.py --device cuda       # fp32 CUDA

Appends one json line per run via --out.
"""
import argparse, json, time

TEXT = ("The old lighthouse keeper climbed the spiral stairs every evening at dusk. "
        "He lit the great lamp and watched the beam sweep across the darkening waves; "
        "ships far at sea counted on that light to find their way home safely.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--voice", default="jean")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from pocket_tts.models.tts_model import TTSModel

    t0 = time.perf_counter()
    model = TTSModel.load_model(language="english", quantize=a.quantize)
    if a.device == "cuda":
        model = model.to("cuda")
    load_s = time.perf_counter() - t0

    state = model.get_state_for_audio_prompt(a.voice)

    def gen():
        return model.generate_audio(state, TEXT, frames_after_eos=2, copy_state=True)

    audio = gen()                                            # warmup (kernel JIT / autotune)
    audio_sec = audio.numel() / model.sample_rate

    walls = []
    for _ in range(a.runs):
        t = time.perf_counter()
        gen()
        walls.append(time.perf_counter() - t)
    wall = sorted(walls)[len(walls) // 2]

    # TTFA = wall to the FIRST streamed audio chunk (their streaming generator)
    ttfa_ms = None
    try:
        t = time.perf_counter()
        for _chunk in model.generate_audio_stream(state, TEXT, copy_state=True):
            ttfa_ms = round((time.perf_counter() - t) * 1000, 1)
            break
    except Exception as e:                                   # streaming API drift — RTF still valid
        ttfa_ms = f"err: {type(e).__name__}"

    rec = {
        "framework": "pocket-tts-ref",
        "torch": torch.__version__,
        "device": a.device + ("-quantized" if a.quantize else ""),
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
