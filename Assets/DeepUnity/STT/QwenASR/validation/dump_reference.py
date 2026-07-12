#!/usr/bin/env python3
"""
Qwen3-ASR reference dumps for DeepUnity parity probes (D0).

Runs the HF reference model (Qwen3ASRForConditionalGeneration, transformers >= 5.13) on the wavs in
validation/clips/ and dumps every pipeline stage the Unity port must reproduce:

    reference_dumps/<size>/<clip>/
        mel.npy            [128, T_pad]  normalized log-mel, padded to a multiple of 100 frames
        mel_mask.npy       [T_pad]       1 = valid mel frame
        enc_out.npy        [N_tok, d]    encoder output AFTER ln_post, BEFORE projector
        proj_out.npy       [N_tok, hid]  projector output (what replaces <|audio_pad|> embeds)
        input_ids.npy      [S]           full prompt token ids (scaffold + N audio pads)
        logits_step0.npy   [151936]      lm_head logits at the last prompt position (first decode step)
        tokens_greedy.npy  [K]           greedy continuation token ids (no sampling)
        raw_output.txt                   raw decoded string ("language X<asr_text>...")
        transcript.txt                   parsed transcript (text after <asr_text>)
        meta.json                        shapes, token counts, argmax of step0, timing

Reference is run on CPU in fp32 by default (deterministic, full-precision parity target — same
convention as the Chatterbox dumps). --device cuda works for a quick look but is NOT the parity
reference. Greedy decode (do_sample=False) throughout, matching generation_config.json.

USAGE
    python dump_reference.py --ckpt C:/dev/_model_staging/qwen3asr/Qwen3-ASR-0.6B-hf
    python dump_reference.py --ckpt C:/dev/_model_staging/qwen3asr/Qwen3-ASR-1.7B-hf --clips clip1_hello

Requires: torch, transformers>=5.13 (qwen3_asr), numpy. (venv: _model_staging/qwen3asr/venv_ref)
"""
import argparse
import json
import os
import time
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ASR_TEXT = "<asr_text>"


def load_wav_16k_mono(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"{path}: expected 16 kHz, got {w.getframerate()}"
        assert w.getnchannels() == 1, f"{path}: expected mono"
        assert w.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="local Qwen3-ASR-*-hf checkpoint folder")
    ap.add_argument("--clips", default=None, help="comma-separated clip stems (default: all in clips/)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--language", default=None, help="optional forced language (system prompt), e.g. English")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    size = "0.6b" if "0.6" in os.path.basename(args.ckpt.rstrip("/\\")) else "1.7b"
    out_root = os.path.join(HERE, "reference_dumps", size)
    clip_dir = os.path.join(HERE, "clips")
    stems = (args.clips.split(",") if args.clips
             else sorted(f[:-4] for f in os.listdir(clip_dir) if f.endswith(".wav")))

    print(f"[load]   {args.ckpt} -> {args.device} fp32")
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(args.ckpt, dtype=torch.float32)
    model.to(args.device).eval()

    for stem in stems:
        t0 = time.time()
        wav = load_wav_16k_mono(os.path.join(clip_dir, stem + ".wav"))
        out_dir = os.path.join(out_root, stem)
        os.makedirs(out_dir, exist_ok=True)

        inputs = processor.apply_transcription_request(
            audio=[wav], language=args.language, return_tensors="pt")
        inputs = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        mel = inputs["input_features"]            # [1, 128, T_pad]
        mel_mask = inputs["input_features_mask"]  # [1, T_pad]
        ids = inputs["input_ids"]                 # [1, S]

        with torch.no_grad():
            # encoder + projector stages
            audio_out = model.model.get_audio_features(
                mel.to(model.dtype), mel_mask, return_dict=True)
            enc_out = audio_out.last_hidden_state   # [N_tok, d]  post-ln_post
            proj_out = audio_out.pooler_output      # [N_tok, hidden]

            # first-step logits over the prompt
            fwd = model(input_ids=ids, input_features=mel.to(model.dtype),
                        input_features_mask=mel_mask,
                        attention_mask=inputs.get("attention_mask"))
            logits_step0 = fwd.logits[0, -1]        # [vocab]

            # greedy decode
            gen = model.generate(**{k: v for k, v in inputs.items()},
                                 do_sample=False, max_new_tokens=args.max_new_tokens)
            new_tokens = gen[0, ids.shape[1]:]

        raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
        raw_clean = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        transcript = raw_clean.split(ASR_TEXT, 1)[-1].strip() if ASR_TEXT in raw_clean else raw_clean.strip()

        np.save(os.path.join(out_dir, "mel.npy"), mel[0].float().cpu().numpy())
        np.save(os.path.join(out_dir, "mel_mask.npy"), mel_mask[0].cpu().numpy())
        np.save(os.path.join(out_dir, "enc_out.npy"), enc_out.float().cpu().numpy())
        np.save(os.path.join(out_dir, "proj_out.npy"), proj_out.float().cpu().numpy())
        np.save(os.path.join(out_dir, "input_ids.npy"), ids[0].cpu().numpy())
        np.save(os.path.join(out_dir, "logits_step0.npy"), logits_step0.float().cpu().numpy())
        np.save(os.path.join(out_dir, "tokens_greedy.npy"), new_tokens.cpu().numpy())
        with open(os.path.join(out_dir, "raw_output.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

        n_audio_tok = int((ids[0] == model.config.audio_token_id).sum())
        meta = {
            "clip": stem, "samples": int(wav.shape[0]), "seconds": round(wav.shape[0] / 16000, 3),
            "mel_frames_padded": int(mel.shape[-1]), "mel_frames_valid": int(mel_mask.sum()),
            "audio_tokens": n_audio_tok, "enc_out_shape": list(enc_out.shape),
            "proj_out_shape": list(proj_out.shape), "prompt_len": int(ids.shape[1]),
            "step0_argmax": int(logits_step0.argmax()), "new_tokens": int(new_tokens.shape[0]),
            "raw_output": raw, "transcript": transcript,
            "expected_audio_tokens_formula": int((int(mel_mask.sum()) // 100) * 13
                + (lambda r: 0 if r == 0 else ((((r - 1) // 2 + 1) - 1) // 2 + 1 - 1) // 2 + 1)(int(mel_mask.sum()) % 100)),
            "device": args.device, "seconds_elapsed": round(time.time() - t0, 1),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        print(f"[{stem}] {meta['seconds']}s audio -> {n_audio_tok} audio tok, "
              f"prompt {meta['prompt_len']}, out {meta['new_tokens']} tok "
              f"({meta['seconds_elapsed']}s): {transcript!r}")

    print(f"\nDone -> {out_root}")


if __name__ == "__main__":
    main()
