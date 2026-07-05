#!/usr/bin/env python3
"""
Bake an alternative Chatterbox voice for the DeepUnity port from a reference wav.

Runs the reference cloning path (VoiceEncoder + S3Tokenizer + CAMPPlus + mel) in Python — the
heavy cloning models never need to run in Unity — and writes the resulting conditioning tensors
as fp16/i32 .bin files under conds_<name>/ in the exported weights folder, appending them to
manifest.tsv. Unity then simply selects the voice:

    var tts = new ChatterboxTTS(voice: "conds_elder");
    // or on the NPC: ChatterboxVoice.SetSharedTTS(new ChatterboxTTS(voice: "conds_elder"));

Run on WSL in the `chatterbox` conda env (same one as dump_reference.py):
    conda activate chatterbox
    export HF_HOME=/mnt/c/Users/Radu/.cache/huggingface
    python make_voice.py elder /path/to/elder_reference.wav

Reference audio: >5 s of clean single-speaker speech (10 s is ideal). For an ELDER voice use any
clip of an elderly speaker (e.g. a public-domain LibriVox narrator).
"""
import argparse
import os
import sys

import numpy as np
import torch

WEIGHTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "Resources", "DeepUnity", "TTS", "Chatterbox", "weights_chatterbox_turbo_fp16"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="voice name -> conds_<name>/ in the weights folder")
    ap.add_argument("wav", help="reference audio (>5s clean speech)")
    ap.add_argument("--out", default=WEIGHTS_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not os.path.isdir(args.out):
        sys.exit(f"weights folder not found: {args.out} (run import_params.py first)")

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    tts = ChatterboxTurboTTS.from_pretrained(args.device)
    # exaggeration is ignored by turbo (no emotion_adv), norm_loudness matches the reference flow
    tts.prepare_conditionals(args.wav, norm_loudness=True)
    t3c, gen = tts.conds.t3, tts.conds.gen

    prefix = f"conds_{args.name}"
    voice_dir = os.path.join(args.out, prefix)
    os.makedirs(voice_dir, exist_ok=True)

    manifest_lines = []

    def save(rel, arr, dtype):
        arr = arr.detach().cpu().float().numpy() if torch.is_tensor(arr) else np.asarray(arr)
        path = os.path.join(args.out, rel + ".bin")
        if dtype == "i32":
            np.ascontiguousarray(arr.astype(np.int32)).tofile(path)
        else:
            np.ascontiguousarray(arr.astype(np.float16)).tofile(path)
        shape = list(arr.shape)
        numel = int(np.prod(shape))
        manifest_lines.append(f"{rel}\t{rel}.bin\t{dtype}\t{numel}\t{','.join(map(str, shape))}")
        print(f"  {rel}: {shape} {dtype}")

    save(f"{prefix}/t3_speaker_emb", t3c.speaker_emb.reshape(-1), "f16")
    save(f"{prefix}/t3_prompt_tokens", t3c.cond_prompt_speech_tokens.reshape(-1).cpu().numpy(), "i32")
    save(f"{prefix}/prompt_token", gen["prompt_token"].reshape(-1).cpu().numpy(), "i32")
    pf = gen["prompt_feat"]
    save(f"{prefix}/prompt_feat", pf.reshape(pf.shape[-2], pf.shape[-1]), "f16")
    save(f"{prefix}/embedding", gen["embedding"].reshape(-1), "f16")

    # append to manifest.tsv (idempotent: drop existing lines for this voice first)
    man = os.path.join(args.out, "manifest.tsv")
    with open(man, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l and not l.startswith(prefix + "/")]
    lines += manifest_lines
    with open(man, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    P = gen["prompt_token"].numel()
    F = pf.shape[-2]
    print(f"\nVoice '{args.name}' baked ({P} prompt tokens / {F} mel frames"
          f"{' — OK' if F == 2 * P else ' — WARNING: F != 2P'}).")
    print(f'Unity: new ChatterboxTTS(voice: "{prefix}")')


if __name__ == "__main__":
    main()
