#!/usr/bin/env python3
"""
Chatterbox-Turbo reference dump for the DeepUnity port (SPEC.md task M0/M3 validation).

Runs the REAL ResembleAI/chatterbox-turbo pipeline on a fixed text with GREEDY T3 decoding
(deterministic) and saves every stage boundary + every stochastic tensor as .npy, so the Unity
side can (a) compare tensors and (b) inject the exact same noise for bit-comparable output.

Run on WSL in the inference env (vLLM box) or any env with:
    pip install chatterbox-tts   (or the cloned repo + torch, transformers, librosa)

    python dump_reference.py --out dump/ [--text "..."]

Dumped files (all float32/int64 .npy):
    text_tokens.npy            GPT2 BPE ids after punc_norm (no BOS/EOS)
    prefill_embeds.npy         T3 input embeds [1, L, 1024] AFTER wpe was added by GPT2? (NO —
                               these are the pre-backbone embeds; GPT2 adds wpe internally)
    t3_hidden_prefill.npy      backbone hidden state after the prefill forward [1, L, 1024]
    t3_logits_step0.npy        speech-head logits at the first decode position [6563]
    speech_tokens.npy          greedy speech tokens (pre-filtering)
    flow_tokens.npy            prompt+gen+SIL tokens fed to the flow
    enc_out.npy                UpsampleConformerEncoder output h [1, 2T, 512]
    mu.npy                     encoder_proj(h) [1, 2T, 80]
    spks.npy                   spk_embed_affine_layer(normalize(x-vector)) [1, 80]
    conds_feat.npy             prompt-mel conditioning [1, 80, 2T]
    z_noise.npy                the exact initial noise x0 [1, 80, 2T]
    est_in_0.npy / est_out_0.npy   estimator input x / output dxdt at (t=0,   r=0.5)
    est_out_1.npy              estimator output at (t=0.5, r=1)
    mel.npy                    final mel (prompt sliced off) [1, 80, Tg]
    f0.npy                     ConvRNNF0Predictor output [1, Tg]
    nsf_phases.npy             SineGen per-harmonic random phases [9] (phase[0]=0)
    nsf_noise.npy              SineGen noise [1, 9, S]... (captured via hook)
    source.npy                 NSF source s [1, 1, S]
    wav.npy                    final waveform (PRE-watermark) [S24k]
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello world! This is a test of the DeepUnity port.")
    ap.add_argument("--out", default="dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(1234)

    from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
    tts = ChatterboxTurboTTS.from_pretrained(args.device)
    t3, s3gen, conds = tts.t3, tts.s3gen, tts.conds
    dev = args.device

    def save(name, x):
        a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        np.save(os.path.join(args.out, name + ".npy"), a)
        print(f"  {name}: {a.shape} {a.dtype}")

    # ---------------- text tokens ----------------
    text = punc_norm(args.text)
    text_tokens = tts.tokenizer(text, return_tensors="pt").input_ids.to(dev)
    save("text_tokens", text_tokens)

    # ---------------- T3 prefill (embeds + hidden + first logits) ----------------
    with torch.inference_mode():
        speech_start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(
            t3_cond=conds.t3, text_tokens=text_tokens, speech_tokens=speech_start, cfg_weight=0.0)
        save("prefill_embeds", embeds)

        out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        hidden, past = out[0], out.past_key_values
        save("t3_hidden_prefill", hidden)
        logits0 = t3.speech_head(hidden[:, -1:])
        save("t3_logits_step0", logits0[0, -1])

        # ---------------- greedy decode (deterministic) ----------------
        gen = []
        tok = logits0[:, -1, :].argmax(dim=-1, keepdim=True)
        gen.append(tok)
        for _ in range(600):
            emb = t3.speech_emb(tok)
            out = t3.tfmr(inputs_embeds=emb, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = t3.speech_head(out[0])[:, -1, :]
            tok = logits.argmax(dim=-1, keepdim=True)
            if tok.item() == t3.hp.stop_speech_token:
                break
            gen.append(tok)
        speech_tokens = torch.cat(gen, dim=1)[0]
        save("speech_tokens", speech_tokens)

    # ---------------- S3Gen with captured noise ----------------
    speech_tokens = speech_tokens[speech_tokens < 6561].to(dev)
    from chatterbox.models.s3gen.const import S3GEN_SIL
    silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=dev)
    tokens = torch.cat([speech_tokens, silence])

    ref = conds.gen
    flow = s3gen.flow
    with torch.inference_mode():
        prompt_token = ref["prompt_token"]
        full_tokens = torch.concat([prompt_token, tokens[None]], dim=1)
        save("flow_tokens", full_tokens)

        emb = torch.nn.functional.normalize(ref["embedding"], dim=1)
        spks = flow.spk_embed_affine_layer(emb)
        save("spks", spks)

        tok_emb = flow.input_embedding(full_tokens.long())
        h, _ = flow.encoder(tok_emb, torch.LongTensor([full_tokens.size(1)]).to(dev))
        save("enc_out", h)
        mu = flow.encoder_proj(h)
        save("mu", mu)

        mel_total = h.size(1)
        prompt_feat = ref["prompt_feat"]
        conds_feat = torch.zeros([1, mel_total, 80], device=dev)
        conds_feat[:, :prompt_feat.size(1)] = prompt_feat
        conds_feat = conds_feat.transpose(1, 2)
        save("conds_feat", conds_feat)

        # exact noise + 2-step meanflow euler with estimator IO capture
        z = torch.randn(1, 80, mel_total, device=dev)
        save("z_noise", z)

        est = flow.decoder.estimator
        mask = torch.ones(1, 1, mel_total, device=dev)
        x = z.clone()
        t_span = torch.linspace(0, 1, 3, device=dev)
        for i, (t, r) in enumerate(zip(t_span[:-1], t_span[1:])):
            if i == 0:
                save("est_in_0", x)
            dxdt = est(x, mask=mask, mu=mu.transpose(1, 2).contiguous(),
                       t=t[None], spks=spks, cond=conds_feat, r=r[None])
            save(f"est_out_{i}", dxdt)
            x = x + (r - t) * dxdt

        prompt_mel = prompt_feat.size(1)
        mel = x[:, :, prompt_mel:]
        save("mel", mel)

        # ---------------- vocoder with captured NSF randomness ----------------
        voc = s3gen.mel2wav
        f0 = voc.f0_predictor(mel)
        save("f0", f0)

        s_up = voc.f0_upsamp(f0[:, None]).transpose(1, 2)
        sine_gen = voc.m_source.l_sin_gen

        # reimplement SineGen forward with captured randomness (same math as hifigan.py)
        F_mat = torch.zeros((1, 9, s_up.size(1)), device=dev)
        f0t = s_up.transpose(1, 2)
        for i in range(9):
            F_mat[:, i:i+1, :] = f0t * (i + 1) / sine_gen.sampling_rate
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        phases = torch.zeros(1, 9, 1, device=dev)
        phases[:, 1:, :] = (torch.rand(1, 8, 1, device=dev) * 2 - 1) * np.pi
        save("nsf_phases", phases[0, :, 0])
        uv = (f0t > sine_gen.voiced_threshold).float()
        noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
        nz = torch.randn_like(F_mat.transpose(1, 2)).transpose(1, 2)
        save("nsf_noise", nz)
        sine_waves = sine_gen.sine_amp * torch.sin(theta_mat + phases)
        # sine_waves/nz [1,9,S]; uv/noise_amp [1,1,S] broadcast over harmonics
        s_harm = (sine_waves * uv + noise_amp * nz).transpose(1, 2)   # -> [1,S,9] for l_linear
        s = voc.m_source.l_tanh(voc.m_source.l_linear(s_harm))
        save("source", s.transpose(1, 2))

        wav = voc.decode(x=mel, s=s.transpose(1, 2))
        n_trim = 24000 // 50
        trim_fade = torch.zeros(2 * n_trim, device=dev)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim, device=dev)) + 1) / 2
        wav[:, :len(trim_fade)] *= trim_fade
        save("wav", wav[0])

    print(f"\nDone. Reference dump in {args.out}/ — copy next to the Unity project and run the "
          "ChatterboxParityProbe (Assets/DeepUnity/InferenceEngine/TTS/validation) against it.")


if __name__ == "__main__":
    main()
