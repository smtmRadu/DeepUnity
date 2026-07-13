#!/usr/bin/env python3
"""Windowed-S3Gen feasibility probe: can chatterbox-turbo's flow+vocoder run on overlapped
token windows (streaming) without audible seams / prosody collapse?

Controlled comparison: ONE greedy T3 token sequence, ONE global z-noise indexed by absolute
mel position (overlaps see identical noise), voice = elder_ref. Variants:
  wav_full            full-context S3Gen (baseline, = what the Unity port does per clause)
  wav_winflow_2s      flow windowed W=50tok/ctx=25/la=5, vocoder full on concat mel
  wav_winflow_1s      flow windowed W=25tok/ctx=25/la=5, vocoder full
  wav_stream_1s       W=25/ctx=25/la=5 + windowed vocoder (mel ctx 24, global phase-continuous
                      NSF source sliced per window = emulates carried phase, 10ms crossfade)

WSL env `chatterbox`, HF_HOME=/mnt/c/Users/Radu/.cache/huggingface.
"""
import os
import numpy as np
import torch
import soundfile as sf

SCRATCH = "/mnt/c/Users/Radu/AppData/Local/Temp/claude/C--Users-Radu/e8227271-9338-4185-9d69-a2ac2df97666/scratchpad"
REF = os.path.join(SCRATCH, "elder_ref.wav")
TEXT = ("Ah, another undead approaches the fog gate. Turn back, poor wanderer, while your soul "
        "is still yours. The sentinel beyond has felled every challenger, and it will fell you too.")
SIL_TOKENS = 3
SR = 24000
MEL_HOP = 480          # samples per mel frame (24000 / 50)


def main():
    torch.manual_seed(1234)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
    tts = ChatterboxTurboTTS.from_pretrained(dev)
    tts.prepare_conditionals(REF, norm_loudness=True)
    t3, s3gen, conds = tts.t3, tts.s3gen, tts.conds
    flow, voc = s3gen.flow, s3gen.mel2wav

    # ---------------- T3 tokens: capture from a seeded SAMPLED generate ----------------
    # (greedy T3 collapses into 3-4x-stretched mumble on multi-sentence texts — the probe's
    # first run produced 37s of near-silence. tts.generate's sampler is the healthy path;
    # hook the flow input to grab prompt+gen tokens and share them across all variants.)
    cap = {}
    hook = flow.input_embedding.register_forward_hook(lambda m, i, o: cap.update(full=i[0].detach()))
    _ = tts.generate(TEXT)
    hook.remove()
    P_tok = conds.gen["prompt_token"].size(1)
    tokens = cap["full"][0, P_tok:].long()          # gen tokens incl. any trailing silence
    T = tokens.numel()
    print(f"T3 sampled: {T} flow tokens ({T / 25:.1f}s of audio)")

    ref = conds.gen
    prompt_tok = ref["prompt_token"]          # [1, P]
    prompt_feat = ref["prompt_feat"]          # [1, 2P, 80]
    P_mel = prompt_feat.size(1)
    emb = torch.nn.functional.normalize(ref["embedding"], dim=1)
    spks = flow.spk_embed_affine_layer(emb)

    # ONE global noise field over prompt + full gen mel (absolute-position indexed)
    g = torch.Generator(device=dev).manual_seed(4321)
    z_global = torch.randn(1, 80, P_mel + 2 * T, device=dev, generator=g)

    def run_flow(gen_tokens, z_gen_abs_start, ctx_mel=None):
        """prompt + gen_tokens -> generated-region mel [1, 80, 2*len(gen_tokens)].
        ctx_mel [1, 80, Fc]: ALREADY-EMITTED mel for the leading context tokens, teacher-forced
        into the estimator's cond channel (the same mechanism as the voice prompt) so each window
        continues from the audio the listener actually heard, not a fresh hallucination of it."""
        full = torch.cat([prompt_tok, gen_tokens[None]], dim=1)
        with torch.inference_mode():
            h, _ = flow.encoder(flow.input_embedding(full.long()),
                                torch.LongTensor([full.size(1)]).to(dev))
            mu = flow.encoder_proj(h)
            L = h.size(1)
            cond = torch.zeros([1, L, 80], device=dev)
            cond[:, :P_mel] = prompt_feat
            if ctx_mel is not None:
                cond[:, P_mel: P_mel + ctx_mel.size(2)] = ctx_mel.transpose(1, 2)
            cond = cond.transpose(1, 2)
            n_gen = L - P_mel
            z = torch.cat([z_global[:, :, :P_mel],
                           z_global[:, :, P_mel + z_gen_abs_start: P_mel + z_gen_abs_start + n_gen]], dim=2)
            mask = torch.ones(1, 1, L, device=dev)
            x = z.clone()
            t_span = torch.linspace(0, 1, 3, device=dev)
            for t, r in zip(t_span[:-1], t_span[1:]):
                dxdt = flow.decoder.estimator(x, mask=mask, mu=mu.transpose(1, 2).contiguous(),
                                              t=t[None], spks=spks, cond=cond, r=r[None])
                x = x + (r - t) * dxdt
            return x[:, :, P_mel:]

    def make_source(mel):
        """Global phase-continuous NSF source for a mel [1,80,F] -> s [1,1,F*480], f0 [1,F]."""
        with torch.inference_mode():
            f0 = voc.f0_predictor(mel)
            s_up = voc.f0_upsamp(f0[:, None]).transpose(1, 2)
            sg = voc.m_source.l_sin_gen
            f0t = s_up.transpose(1, 2)
            F_mat = torch.cat([f0t * (i + 1) / sg.sampling_rate for i in range(9)], dim=1)
            theta = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
            gph = torch.Generator(device=dev).manual_seed(99)
            phases = torch.zeros(1, 9, 1, device=dev)
            phases[:, 1:, :] = (torch.rand(1, 8, 1, device=dev, generator=gph) * 2 - 1) * np.pi
            uv = (f0t > sg.voiced_threshold).float()
            namp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
            nz = torch.randn(F_mat.shape, device=dev, generator=gph)
            harm = (sg.sine_amp * torch.sin(theta + phases) * uv + namp * nz).transpose(1, 2)
            s = voc.m_source.l_tanh(voc.m_source.l_linear(harm))
            return s.transpose(1, 2), f0

    def vocode_full(mel):
        with torch.inference_mode():
            s, _ = make_source(mel)
            wav = voc.decode(x=mel, s=s)
            return wav[0].float().cpu().numpy()

    def fade_head(wav):
        n = SR // 50
        env = (np.cos(np.linspace(np.pi, 0, n)) + 1) / 2
        wav[:n] *= 0.0
        wav[n:2 * n] *= env
        return wav

    # ---------------- baseline: full context ----------------
    mel_full = run_flow(tokens, 0)
    wav = fade_head(vocode_full(mel_full))
    sf.write(os.path.join(SCRATCH, "wav_full.wav"), wav, SR)
    print(f"wav_full: {len(wav) / SR:.1f}s")

    # ---------------- windowed flow ----------------
    def windowed_mel(W, C, A, W0=None, A0=None):
        """W0/A0: first-window overrides. Each window's left context is TEACHER-FORCED from the
        already-emitted mel (run_flow ctx_mel), so no blend is needed — the window continues the
        real trajectory by construction."""
        out = None
        s = 0
        while s < T:
            first = s == 0
            e = min(T, s + (W0 if first and W0 else W))
            a = max(0, s - C)
            b = min(T, e + (A0 if first and A0 else A))
            ctx = out[:, :, 2 * a:] if out is not None and s > a else None
            m = run_flow(tokens[a:b], 2 * a, ctx_mel=ctx)
            keep = m[:, :, 2 * (s - a): 2 * (e - a)]
            out = keep if out is None else torch.cat([out, keep], dim=2)
            s = e
        return out


    # ---------------- windowed flow + windowed vocoder (full streaming emulation) ----------------
    # TRUE overlap-crossfade: each new piece keeps xfade samples of its own rendering of the
    # PREVIOUS region (available from the conv-context render) and blends them with the previous
    # piece's tail — two renderings of the SAME instant, not adjacent content (v1 smeared 10ms
    # of distinct audio per seam, audible as a stutter).
    def stream_wav(m):
        with torch.inference_mode():
            s_glob, _ = make_source(m)   # phase-continuous source = the carried-phase emulation
            Fm = m.size(2)
            # WM window, CM left context, RM RIGHT RENDER MARGIN: the last frames of any decode
            # are conv/istft edge-garbage (+3-5 dB over signal, measured) — render RM frames past
            # the keep region and discard them so kept audio never touches a render edge.
            WM, CM, RM = 50, 32, 16
            xfade = int(0.010 * SR)
            out = None
            st = 0
            while st < Fm:
                en = min(Fm, st + WM)
                a = max(0, st - CM)
                b = min(Fm, en + RM)
                w = voc.decode(x=m[:, :, a:b], s=s_glob[:, :, a * MEL_HOP: b * MEL_HOP])
                w = w[0].float().cpu().numpy()
                keep_from = (st - a) * MEL_HOP
                keep_to = (en - a) * MEL_HOP
                if out is None:
                    out = w[keep_from:keep_to]
                else:
                    nb = min(xfade, keep_from, len(out))
                    env = np.linspace(0, 1, nb, dtype=np.float32)
                    blended = out[-nb:] * (1 - env) + w[keep_from - nb: keep_from] * env
                    out = np.concatenate([out[:-nb], blended, w[keep_from:keep_to]])
                st = en
        return fade_head(out)

    # v7: vocoder right-margin fix. vocwin_only (full flow + windowed voc) must now be
    # numerically ~identical to full; stream_F = the real end-to-end streaming config.
    out = stream_wav(mel_full)
    sf.write(os.path.join(SCRATCH, "wav_vocwin_only.wav"), out, SR)
    n = min(len(out), len(  # numeric gate: worst 20ms artifact-to-signal bin, dB
        wf := fade_head(vocode_full(mel_full))))
    d = out[:n] - wf[:n]
    nb = n // 480
    g = np.sqrt((wf[:n] ** 2).mean())
    rd = np.sqrt((d[:nb * 480].reshape(nb, 480) ** 2).mean(1))
    rs = np.maximum(np.sqrt((wf[:nb * 480].reshape(nb, 480) ** 2).mean(1)), 0.3 * g)  # pause-floored
    rel = 20 * np.log10(rd / rs + 1e-12)
    for i in np.argsort(rel)[-5:][::-1]:
        print(f"wav_vocwin_only: bin {rel[i]:6.1f} dB @ t={i * 480 / SR:.2f}s (target < -25)")

    m = windowed_mel(25, 50, 5, W0=50)
    out = stream_wav(m)
    sf.write(os.path.join(SCRATCH, "wav_stream_G.wav"), out, SR)
    print(f"wav_stream_G: TEACHER-FORCED ctx flow (W=25 ctx=50 W0=50 la=5) + margin-fixed vocoder  "
          f"mel-MSE vs full {float(((m - mel_full) ** 2).mean()):.4f}  len={len(out)/SR:.1f}s")


if __name__ == "__main__":
    main()
