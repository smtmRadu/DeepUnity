#!/usr/bin/env python3
"""
Parakeet-TDT reference dumps for DeepUnity parity (SPEC.md §11).

Runs the HF ParakeetForTDT chain in fp32 on 16 kHz mono wavs and dumps every stage the Unity
port must reproduce, INCLUDING a manual TDT greedy loop that mirrors SPEC.md §6 line-by-line
(then cross-checks its transcript against model.generate()).

USAGE (WSL, env with transformers>=5.6 + torch + librosa + soundfile):
    python dump_reference.py <hf_folder> <clips_dir> <out_dir> [--device cuda|cpu]

    python dump_reference.py /mnt/c/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v3-hf \
        ./clips ./reference_dumps/v3 --device cuda

Per clip -> <out_dir>/<clip_stem>/:
    mel.npy                [T_mel,128]  post-normalization features (valid frames)
    pos_emb.npy            [2*T_enc-1,1024]
    sub_out.npy            [T_enc,1024] subsampling output (post-linear, pre-scale)
    enc_layer0.npy         [T_enc,1024] first conformer block output
    enc_out.npy            [T_enc,1024] encoder output (block 23's norm_out)
    enc_proj.npy           [T_enc,640]  encoder_projector output (the CPU loop's input)
    joint_logits_first8.npy [8,V+5]     first 8 greedy-loop joint evaluations
    tokens.npy / durations.npy / frames.npy   full emission sequence (non-blank tokens)
    emissions.tsv          step-by-step log incl. blanks (step, t, k, dur, is_blank)
    transcript.txt         manual-loop transcript (line 2 = generate() transcript)
    meta.json              dims, timings, checks (blank row, transcript match)
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

from transformers import AutoTokenizer, ParakeetForTDT
from transformers.models.parakeet import ParakeetFeatureExtractor


def load_feature_extractor(folder):
    for name in ("preprocessor_config.json", "processor_config.json"):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg = cfg.get("feature_extractor", cfg)
            kw = {k: v for k, v in cfg.items() if k not in
                  ("feature_extractor_type", "processor_class", "blank_token")}
            return ParakeetFeatureExtractor(**kw)
    raise FileNotFoundError(f"no (pre)processor_config.json in {folder}")


@torch.no_grad()
def manual_tdt_greedy(model, enc_proj):
    """SPEC.md §6 loop, verbatim. enc_proj: [T,640] fp32. Returns emission log + first-8 logits."""
    cfg = model.config
    V, blank, durations, max_sym = cfg.vocab_size, cfg.blank_token_id, cfg.durations, cfg.max_symbols_per_step
    dec, joint = model.decoder, model.joint
    device = enc_proj.device

    h = torch.zeros(cfg.num_decoder_layers, 1, cfg.decoder_hidden_size, device=device)
    c = torch.zeros_like(h)
    emb = dec.embedding(torch.tensor([[blank]], device=device))          # blank row (~0)
    lstm_out, (h, c) = dec.lstm(emb, (h, c))
    pred_out = dec.decoder_projector(lstm_out)[0, 0]                     # [640]

    T = enc_proj.shape[0]
    t, step = 0, 0
    tokens, frames, durs, emissions, first_logits = [], [], [], [], []
    while t < T:
        symbols_at_frame = 0
        while True:
            logits = joint.head(joint.activation(enc_proj[t] + pred_out))  # [V+5]
            if step < 8:
                first_logits.append(logits.float().cpu().numpy().copy())
            k = int(torch.argmax(logits[:V]).item())
            d = durations[int(torch.argmax(logits[V:]).item())]
            emissions.append((step, t, k, d, int(k == blank)))
            step += 1
            if k != blank:
                tokens.append(k)
                frames.append(t)
                durs.append(d)
                emb = dec.embedding(torch.tensor([[k]], device=device))
                lstm_out, (h, c) = dec.lstm(emb, (h, c))
                pred_out = dec.decoder_projector(lstm_out)[0, 0]
            symbols_at_frame += 1
            t += d
            if d > 0:
                break
            if k == blank:            # blank + dur 0 would spin: force +1 (HF rule)
                t += 1
                break
            if symbols_at_frame >= max_sym:   # NeMo max_symbols_per_step guard
                t += 1
                break
    return tokens, frames, durs, emissions, np.stack(first_logits) if first_logits else np.zeros((0, V + 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_folder")
    ap.add_argument("clips_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    fe = load_feature_extractor(args.hf_folder)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_folder)
    model = ParakeetForTDT.from_pretrained(args.hf_folder, dtype=torch.float32).to(args.device).eval()
    cfg = model.config
    blank_row_max = float(model.decoder.embedding.weight[cfg.blank_token_id].abs().max())
    print(f"[model] vocab {cfg.vocab_size}, blank {cfg.blank_token_id}, durations {cfg.durations}, "
          f"blank-emb max|w| {blank_row_max:.2e}")

    wavs = sorted(f for f in os.listdir(args.clips_dir) if f.lower().endswith(".wav"))
    assert wavs, f"no wavs in {args.clips_dir}"

    for wav_name in wavs:
        stem = os.path.splitext(wav_name)[0]
        out = os.path.join(args.out_dir, stem)
        os.makedirs(out, exist_ok=True)
        audio, sr = sf.read(os.path.join(args.clips_dir, wav_name), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == fe.sampling_rate, f"{wav_name}: {sr} Hz, expected {fe.sampling_rate}"

        feats = fe(audio, sampling_rate=sr, return_tensors="pt")
        input_features = feats.input_features.to(args.device)          # [1,T,128] post-norm
        attention_mask = feats.attention_mask.to(args.device)
        t_mel = int(attention_mask.sum())

        # stage hooks
        grabbed = {}
        hooks = [
            model.encoder.subsampling.register_forward_hook(
                lambda m, i, o: grabbed.__setitem__("sub_out", o.detach())),
            model.encoder.encode_positions.register_forward_hook(
                lambda m, i, o: grabbed.__setitem__("pos_emb", o.detach())),
            model.encoder.layers[0].register_forward_hook(
                lambda m, i, o: grabbed.__setitem__("enc_layer0", o.detach())),
        ]
        t0 = time.perf_counter()
        enc = model.get_audio_features(input_features=input_features, attention_mask=attention_mask)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t_encoder = time.perf_counter() - t0
        for hk in hooks:
            hk.remove()

        t_enc = int(enc.attention_mask.sum())
        enc_proj = enc.pooler_output[0, :t_enc]                        # [T_enc,640]

        t0 = time.perf_counter()
        tokens, frames, durs, emissions, first_logits = manual_tdt_greedy(model, enc_proj)
        t_loop = time.perf_counter() - t0
        transcript = tokenizer.decode(tokens, skip_special_tokens=True)

        # cross-check vs the library's own generate()
        gen = model.generate(input_features=input_features, attention_mask=attention_mask)
        gen_ids = [int(i) for i in gen.sequences[0].tolist()
                   if int(i) < cfg.vocab_size and int(i) != cfg.blank_token_id]
        gen_transcript = tokenizer.decode(gen_ids, skip_special_tokens=True)
        match = transcript.strip() == gen_transcript.strip()

        np.save(os.path.join(out, "mel.npy"), input_features[0, :t_mel].cpu().numpy())
        np.save(os.path.join(out, "pos_emb.npy"), grabbed["pos_emb"][0].cpu().numpy())
        np.save(os.path.join(out, "sub_out.npy"), grabbed["sub_out"][0, :t_enc].cpu().numpy())
        np.save(os.path.join(out, "enc_layer0.npy"), grabbed["enc_layer0"][0, :t_enc].cpu().numpy())
        np.save(os.path.join(out, "enc_out.npy"), enc.last_hidden_state[0, :t_enc].cpu().numpy())
        np.save(os.path.join(out, "enc_proj.npy"), enc_proj.cpu().numpy())
        np.save(os.path.join(out, "joint_logits_first8.npy"), first_logits)
        np.save(os.path.join(out, "tokens.npy"), np.array(tokens, dtype=np.int32))
        np.save(os.path.join(out, "durations.npy"), np.array(durs, dtype=np.int32))
        np.save(os.path.join(out, "frames.npy"), np.array(frames, dtype=np.int32))
        with open(os.path.join(out, "emissions.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("step\tt\tk\tdur\tis_blank\n")
            for e in emissions:
                f.write("\t".join(map(str, e)) + "\n")
        with open(os.path.join(out, "transcript.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(transcript + "\n" + gen_transcript + "\n")
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "clip": wav_name, "samples": int(len(audio)), "seconds": round(len(audio) / sr, 3),
                "t_mel": t_mel, "t_enc": t_enc, "vocab": cfg.vocab_size, "blank": cfg.blank_token_id,
                "durations_bins": cfg.durations, "blank_embedding_row_max": blank_row_max,
                "n_tokens": len(tokens), "n_steps": len(emissions),
                "timestamps_s": [round(fr * 0.08, 2) for fr in frames],
                "encoder_seconds": round(t_encoder, 3), "loop_seconds": round(t_loop, 3),
                "manual_matches_generate": match,
                "transcript": transcript, "generate_transcript": gen_transcript,
            }, f, indent=1, ensure_ascii=False)
        print(f"[{stem}] {len(audio)/sr:.1f}s T_mel={t_mel} T_enc={t_enc} steps={len(emissions)} "
              f"tokens={len(tokens)} enc={t_encoder:.2f}s loop={t_loop:.2f}s match={match}\n  -> {transcript}")

    print("\nAll clips dumped ->", args.out_dir)


if __name__ == "__main__":
    main()
