#!/usr/bin/env python3
"""Bake a CosyVoice3 zero-shot voice for the DeepUnity port.

Runs the official frontend (speech_tokenizer_v3 + campplus + mel) on a prompt wav +
transcript and writes voices/<name>/{prompt_text_tokens,prompt_speech_tokens,
prompt_feat,embedding} + manifest lines into the exported weights folder.

Run (WSL):
  conda activate cosyvoice
  python make_voice.py --wav velmire_prompt.wav --transcript-file velmire_prompt.txt --name velmire
"""
import argparse, os, sys
import numpy as np

MODEL_DIR = os.path.expanduser("~/cosyvoice_work/pretrained_models/Fun-CosyVoice3-0.5B")
REPO = os.path.expanduser("~/cosyvoice_work/CosyVoice")
WEIGHTS = "/mnt/c/dev/DeepUnity/Assets/Resources/Weights/weights_cosyvoice3_fp16"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--transcript-file", default=None)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    transcript = a.transcript or open(a.transcript_file, encoding="utf-8").read().strip()

    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "third_party/Matcha-TTS"))
    import torch

    from cosyvoice.cli.cosyvoice import CosyVoice3
    cv = CosyVoice3(MODEL_DIR, fp16=False)

    # capture the frontend outputs exactly like dump_reference.py (proven path)
    captured = {}
    fz = cv.frontend.frontend_zero_shot
    def fz_wrap(tts_text, prompt_text, prompt_wav, resample_rate, spk_id):
        mi = fz(tts_text, prompt_text, prompt_wav, resample_rate, spk_id)
        if not captured:
            captured.update(mi)
        return mi
    cv.frontend.frontend_zero_shot = fz_wrap

    prompt_text = transcript + "<|endofprompt|>"
    # run a tiny synthesis just to drive the frontend (LM output is discarded)
    for _ in cv.inference_zero_shot("Hello there.", prompt_text, a.wav, stream=False):
        break

    def pick(*names):
        for n in names:
            if n in captured: return captured[n].detach().cpu().numpy()
        raise KeyError(names)
    text_tok = pick("prompt_text").squeeze(0)
    speech_tok = pick("llm_prompt_speech_token", "flow_prompt_speech_token").squeeze(0)
    feat = pick("prompt_speech_feat").squeeze(0)
    emb = pick("llm_embedding", "flow_embedding").squeeze(0)

    vdir = os.path.join(WEIGHTS, "voices", a.name)
    os.makedirs(vdir, exist_ok=True)
    entries = []
    def bake(rel, arr, dtype):
        arr = np.asarray(arr)
        (arr.astype(np.int32) if dtype == "i32" else arr.astype(np.float16)).tofile(
            os.path.join(vdir, rel + ".bin"))
        shape = ",".join(map(str, arr.shape))
        entries.append(f"voices/{a.name}/{rel}\tvoices/{a.name}/{rel}.bin\t{dtype}\t{arr.size}\t{shape}\n")
        print(f"[bake] {rel:22s} {list(arr.shape)}")
    bake("prompt_text_tokens", text_tok, "i32")
    bake("prompt_speech_tokens", speech_tok, "i32")
    bake("prompt_feat", feat, "f16")
    bake("embedding", emb, "f16")

    man = os.path.join(WEIGHTS, "manifest.tsv")
    existing = open(man, encoding="utf-8").read()
    with open(man, "a", encoding="utf-8") as f:
        for e in entries:
            if e.split("\t")[0] + "\t" not in existing:
                f.write(e)
    print(f"MAKE_VOICE DONE -> voices/{a.name}")

if __name__ == "__main__":
    main()
