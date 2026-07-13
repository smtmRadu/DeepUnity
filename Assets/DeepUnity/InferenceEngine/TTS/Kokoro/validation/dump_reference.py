#!/usr/bin/env python3
"""
Kokoro-82M reference dump for the DeepUnity port (parity-probe ground truth).

Runs the REAL hexgrad/Kokoro-82M pipeline (misaki G2P + KModel) on fixed texts with a fixed
voice and saves every stage boundary + every stochastic tensor as .npy, so the Unity side can
(a) gate the C# G2P on exact phoneme-string match, (b) compare per-stage tensors, and
(c) inject the exact same NSF noise for bit-comparable audio.

Run on WSL in the `kokoro` conda env (torch CPU is fine):
    conda activate kokoro
    python dump_reference.py [--staging /mnt/c/dev/_model_staging/kokoro/hf] [--out dump]

Per text i (t0 also gets generator internals + noise):
    t{i}_phonemes.txt    the reference phoneme string (C# G2P must match EXACTLY)
    t{i}_meta.json       {text, phonemes, n_ids, F(dur frames @40Hz), S(samples)}
    t{i}_input_ids.npy   [1,T] int64 ([0, ids..., 0])
    t{i}_ref_s.npy       [1,256] voicepack row (pack[len(ps)-1])
    t{i}_bert_dur.npy    [1,T,768] ALBERT output
    t{i}_d_en.npy        [1,512,T] after bert_encoder (transposed)
    t{i}_d.npy           [1,T,640] DurationEncoder output
    t{i}_duration.npy    [1,T] sigmoid-summed durations (pre-round)
    t{i}_pred_dur.npy    [T] int64 rounded/clamped
    t{i}_en.npy          [1,640,F] d^T @ alignment
    t{i}_F0_pred.npy     [1,2F] 80 Hz F0 curve
    t{i}_N_pred.npy      [1,2F]
    t{i}_t_en.npy        [1,512,T] TextEncoder output
    t{i}_asr.npy         [1,512,F]
    t{i}_wav.npy         [S] final waveform  (+ t{i}.wav 24 kHz for listening)
  t0 only:
    t0_dec_x.npy         [1,512,2F] decoder output = generator input
    t0_rand_ini.npy      [9] SineGen initial phases (U(0,1), index 0 = 0)
    t0_sine_noise.npy    [1,S,9] SineGen raw randn (multiply by noise_amp at use site)
    t0_har.npy           [1,S] NSF source after l_linear+tanh
    t0_har_cat.npy       [1,22,120F+1] STFT(mag;phase-angle) of the source

Self-check: with the same torch seed, the manual reimplementation below consumes the RNG stream
in the same prefix order as the reference KModel forward, so its wav must match model(ps, ref_s)
EXACTLY — the script asserts max|dwav| < 1e-5 and prints it.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

STAGING_DEFAULT = "/mnt/c/dev/_model_staging/kokoro/hf"
TEXTS = [
    "Hello world! This is a test of the DeepUnity port.",
    "The researchers read 42 papers in 2024, and the present record shows they will present it again.",
    "The old merchant leaned closer, lowering his voice. The northern pass is blocked by snow, "
    "and the wolves grow bolder every night.",
]
VOICE = "af_heart"
SEED = 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=STAGING_DEFAULT)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import soundfile as sf
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline

    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=os.path.join(args.staging, "config.json"),
                   model=os.path.join(args.staging, "kokoro-v1_0.pth")).eval()
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)  # G2P only
    pack = torch.load(os.path.join(args.staging, "voices", f"{VOICE}.pt"), weights_only=True)

    def save(i, name, x):
        a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        np.save(os.path.join(args.out, f"t{i}_{name}.npy"), a)
        print(f"  t{i}_{name}: {a.shape} {a.dtype}")

    for i, text in enumerate(TEXTS):
        print(f"\n=== t{i}: {text[:60]}{'...' if len(text) > 60 else ''}")
        results = list(pipe(text, voice=None))          # quiet pipeline -> G2P + chunking only
        assert len(results) == 1, f"text {i} chunked into {len(results)} pieces; keep texts short"
        ps = results[0].phonemes
        with open(os.path.join(args.out, f"t{i}_phonemes.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(ps)
        print(f"  phonemes ({len(ps)}): {ps}")

        ref_s = pack[len(ps) - 1]                       # [1,256]
        save(i, "ref_s", ref_s)

        ids = [model.vocab.get(p) for p in ps]
        ids = [x for x in ids if x is not None]
        input_ids = torch.LongTensor([[0, *ids, 0]])
        save(i, "input_ids", input_ids)

        with torch.inference_mode():
            # ---------------- reference end-to-end (RNG stream: rand_ini, sine noise, ...) ----
            torch.manual_seed(SEED)
            wav_ref = model(ps, ref_s)                  # KModel.forward -> audio tensor [S]

            # ---------------- manual stage-by-stage (same seed -> identical RNG prefix) -------
            torch.manual_seed(SEED)
            T = input_ids.shape[-1]
            input_lengths = torch.full((1,), T, dtype=torch.long)
            text_mask = torch.zeros(1, T, dtype=torch.bool)

            bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
            save(i, "bert_dur", bert_dur)
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            save(i, "d_en", d_en)

            s_p = ref_s[:, 128:]
            s_d = ref_s[:, :128]

            d = model.predictor.text_encoder(d_en, s_p, input_lengths, text_mask)
            save(i, "d", d)
            x, _ = model.predictor.lstm(d)
            duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
            save(i, "duration", duration)
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
            save(i, "pred_dur", pred_dur)

            indices = torch.repeat_interleave(torch.arange(T), pred_dur)
            aln = torch.zeros((T, indices.shape[0]))
            aln[indices, torch.arange(indices.shape[0])] = 1
            aln = aln.unsqueeze(0)
            en = d.transpose(-1, -2) @ aln
            save(i, "en", en)
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s_p)
            save(i, "F0_pred", F0_pred)
            save(i, "N_pred", N_pred)

            t_en = model.text_encoder(input_ids, input_lengths, text_mask)
            save(i, "t_en", t_en)
            asr = t_en @ aln
            save(i, "asr", asr)

            # ---------------- decoder trunk (istftnet.Decoder.forward inlined) ----------------
            dec = model.decoder
            F0c = dec.F0_conv(F0_pred.unsqueeze(1))
            Nc = dec.N_conv(N_pred.unsqueeze(1))
            xd = torch.cat([asr, F0c, Nc], axis=1)
            xd = dec.encode(xd, s_d)
            asr_res = dec.asr_res(asr)
            res = True
            for block in dec.decode:
                if res:
                    xd = torch.cat([xd, asr_res, F0c, Nc], axis=1)
                xd = block(xd, s_d)
                if block.upsample_type != "none":
                    res = False
            if i == 0:
                save(i, "dec_x", xd)

            # ---------------- generator with captured randomness (Generator.forward inlined) --
            gen = dec.generator
            sg = gen.m_source.l_sin_gen
            f0 = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)              # [1,S,1]
            S = f0.shape[1]
            fn = torch.multiply(f0, torch.FloatTensor([[range(1, 10)]]).to(f0.device))
            rad = (fn / sg.sampling_rate) % 1
            rand_ini = torch.rand(fn.shape[0], fn.shape[2])                   # CAPTURED
            rand_ini[:, 0] = 0
            rad[:, 0, :] = rad[:, 0, :] + rand_ini
            rad_ds = F.interpolate(rad.transpose(1, 2), scale_factor=1 / 300, mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_ds, dim=1) * 2 * torch.pi
            phase_us = F.interpolate(phase.transpose(1, 2) * 300, scale_factor=300, mode="linear").transpose(1, 2)
            sine_waves = torch.sin(phase_us) * sg.sine_amp
            uv = sg._f02uv(f0)
            noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
            nz = torch.randn_like(sine_waves)                                 # CAPTURED
            sine_waves = sine_waves * uv + noise_amp * nz
            har = gen.m_source.l_tanh(gen.m_source.l_linear(sine_waves))
            if i == 0:
                save(i, "rand_ini", rand_ini[0])
                save(i, "sine_noise", nz)
                save(i, "har", har.squeeze(-1))
            har_spec, har_phase = gen.stft.transform(har.transpose(1, 2).squeeze(1))  # [1,S]
            har_cat = torch.cat([har_spec, har_phase], dim=1)
            if i == 0:
                save(i, "har_cat", har_cat)

            xg = xd
            for u in range(gen.num_upsamples):
                xg = F.leaky_relu(xg, negative_slope=0.1)
                x_source = gen.noise_convs[u](har_cat)
                x_source = gen.noise_res[u](x_source, s_d)
                xg = gen.ups[u](xg)
                if u == gen.num_upsamples - 1:
                    xg = gen.reflection_pad(xg)
                xg = xg + x_source
                xs = None
                for j in range(gen.num_kernels):
                    b = gen.resblocks[u * gen.num_kernels + j](xg, s_d)
                    xs = b if xs is None else xs + b
                xg = xs / gen.num_kernels
            xg = F.leaky_relu(xg)                                             # default slope 0.01
            xg = gen.conv_post(xg)
            spec = torch.exp(xg[:, :gen.post_n_fft // 2 + 1, :])
            phase_out = torch.sin(xg[:, gen.post_n_fft // 2 + 1:, :])
            wav = gen.stft.inverse(spec, phase_out).squeeze()

            # ---------------- self-check: manual == reference, bit-for-bit RNG ----------------
            dmax = (wav - wav_ref.squeeze()).abs().max().item()
            print(f"  SELF-CHECK max|manual - reference| = {dmax:.3e}")
            assert dmax < 1e-5, "manual reimplementation drifted from KModel.forward!"

            save(i, "wav", wav)
            sf.write(os.path.join(args.out, f"t{i}.wav"), wav.cpu().numpy(), 24000)

            with open(os.path.join(args.out, f"t{i}_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"text": text, "phonemes": ps, "n_ids": int(T),
                           "F": int(pred_dur.sum().item()), "S": int(wav.numel()),
                           "voice": VOICE, "seed": SEED, "self_check_max_abs_diff": dmax},
                          f, ensure_ascii=False, indent=1)

    print(f"\nDone. Reference dump in {args.out}/ — the Unity KokoroParityProbe compares against "
          "these (G2P: exact string match on t*_phonemes.txt; tensors: corr/maxabs per stage).")


if __name__ == "__main__":
    main()
