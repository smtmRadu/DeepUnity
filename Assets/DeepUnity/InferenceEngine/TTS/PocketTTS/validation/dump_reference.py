"""
Pocket-TTS reference dump for DeepUnity parity (P0). Runs the real pocket_tts model on a FIXED
text + the `jean` voice + a FIXED torch seed, and hooks every stage's tensors to dump/*.npy so the
C# port can be graded corr>0.99 stage by stage (like CosyVoice A1-A4).

Run in WSL (conda env `pocket`, HF access to kyutai/pocket-tts):
    cd /mnt/c/dev/DeepUnity/Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation
    python dump_reference.py

Dumps (dump/):
  meta.json            text, seed, dims, config (ratios/hop/frame_rate/steps)
  text_ids.npy         [1,S] int64 SentencePiece ids of the prompt (post prepare_text_prompt)
  text_embeddings.npy  [1,S,1024] conditioner output (P2 transformer input, text part)
  voice_prompt.npy     [1,125,1024] the jean audio_prompt prefix (voice conditioning)

VOICE STATE (P4 root-cause note): generation uses a state built by running the flow-LM transformer
over [bos_before_voice ; audio_prompt(125)] from kyutai/pocket-tts embeddings/jean.safetensors —
the SAME tensor baked into the C# weights as voices/jean/audio_prompt. It must NOT use
get_state_for_audio_prompt("jean"): that imports a DIFFERENT baked KV state (from
pocket-tts-without-voice-cloning) whose KV corr vs this prefix is only 0.90-0.98 → the C# prefix
reconstruction can never reproduce those latents (this was the P4 0.54-corr bug).

NOISE ALIGNMENT (P4): the TEXT-PROMPTING call also runs flow_net once (output discarded), and the
final break step computes a latent that is never emitted. flow_noise_all.npy contains ONLY the
autoregressive-phase noises (prompt call skipped): [T_emit+1, 32] = one per emitted frame + the
trailing break-step noise (unused by a bit-exact C# run — it EOS-breaks before consuming it).
  xformer_out_f0.npy   [1,1,1024] transformer backbone output at the FIRST generation frame (P2 gate)
  flow_noise_f0.npy    [1,32] the noise the flow head started from at frame 0 (P3 determinism)
  flow_cond_f0.npy     [1,1024] the AdaLN condition (transformer_out) at frame 0 (P3 input)
  flow_latent_f0.npy   [1,32] the flow head's output latent at frame 0 (P3 gate)
  latents.npy          [T,32] ALL generated latents, POST denorm (*emb_std+emb_mean) — Mimi input (P1)
  emb_mean.npy/emb_std.npy [32] latent denorm stats
  mimi_upsampled_f0.npy   [1,512,16] upsample output for latents[:1] (P1 mid-stage)
  mimi_xf_out_f0.npy      [1,512,16] decoder_transformer output for that window (P1 mid-stage)
  wav.npy              [samples] final 24kHz waveform (P1 end-to-end gate) + wav.wav for listen
"""
import json
import os

import numpy as np
import torch

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "dump")
os.makedirs(DUMP, exist_ok=True)

TEXT = "Hello world. This is a test of the pocket TTS port."
VOICE = "jean"
SEED = 0


def save(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().to(torch.float32).cpu().numpy()
    np.save(os.path.join(DUMP, name + ".npy"), np.ascontiguousarray(t))
    print(f"  dump {name} {list(np.asarray(t).shape)}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    from pocket_tts.models.tts_model import TTSModel, prepare_text_prompt

    model = TTSModel.load_model(language="english")   # temp/steps/eos from defaults
    model.eval()
    print(f"loaded pocket-tts english; ldim={model.flow_lm.ldim} dim={model.flow_lm.dim} "
          f"lsd_steps={model.lsd_decode_steps} temp={model.temp} eos_thr={model.eos_threshold}")

    save("emb_mean", model.flow_lm.emb_mean)
    save("emb_std", model.flow_lm.emb_std)

    # ---- voice state: build it from the VOICE-CLONING audio_prompt (kyutai/pocket-tts
    # embeddings/jean.safetensors), the exact [1,125,1024] tensor the C# holds as
    # voices/jean/audio_prompt. See docstring VOICE STATE note — do NOT use
    # get_state_for_audio_prompt("jean") (different baked KV state, irreproducible from C#).
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from pocket_tts.modules.stateful_module import init_states

    voice = load_file(hf_hub_download("kyutai/pocket-tts", f"embeddings/{VOICE}.safetensors"))[
        "audio_prompt"].to(model.flow_lm.dtype)                     # [1,125,1024]
    save("voice_prompt", voice)
    prompt = torch.cat([model.flow_lm.bos_before_voice.to(model.flow_lm.dtype), voice], dim=1)
    state = init_states(model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
    with torch.no_grad():
        model._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=prompt)
    print(f"voice state built from audio_prompt: current_end={model._flow_lm_current_end(state)} "
          f"(1 bbv + {voice.shape[1]} voice)")

    # capture text ids for the fixed prompt
    prep = model.flow_lm.conditioner.prepare(prepare_text_prompt(TEXT, model.pad_with_spaces_for_short_inputs, model.remove_semicolons)[0])
    save("text_ids", prep.tokens)
    save("text_embeddings", model.flow_lm.conditioner(prep))

    # ---- hooks: capture the first-frame transformer out, flow head in/out, and mimi mid-stages
    caps = {}

    def _first_tensor(x):
        # module outputs may be a tensor, or a tuple/list wrapping it (e.g. (emb,))
        while isinstance(x, (tuple, list)):
            x = x[0]
        return x

    def hook_transformer(mod, inp, out):
        # out is the backbone hidden [B,S,1024]; the text-prompt call has S=S_text, generation
        # frames have S==1. Capture the FIRST generation frame under its own key (the P2
        # no-KV reconstruction below saves xformer_out_f0) + self-check the two at the end.
        o = _first_tensor(out)
        if o.shape[1] == 1 and "xformer_out_f0_gen" not in caps:
            caps["xformer_out_f0_gen"] = o.detach().clone()

    flow_noise_all_list = []
    flow_call_idx = [0]      # 0 = the TEXT-PROMPT flow call (output discarded) — must be skipped
    in_generation = [False]  # sub-hooks only capture on the first AUTOREGRESSIVE call (frame 0)

    def hook_flow(mod, inp, out):
        # SimpleMLPAdaLN.forward(c, s, t, x): c=transformer condition [1,1024], s=start-time [1,1],
        # t=target-time [1,1], x=noisy latent (=the noise for step 0) [1,32]. lsd_decode(steps=1):
        # latent = x + flow_net(c, s=0, t=1, x).
        # Call 0 = TEXT PROMPTING (forward always samples; result discarded) -> capture NOTHING,
        # else the C# injects the discarded prompt noise as frame 0 (the P4 off-by-one).
        idx = flow_call_idx[0]; flow_call_idx[0] += 1
        if idx == 0:
            in_generation[0] = True   # sub-hooks may capture from the NEXT call (= frame 0)
            return
        flow_noise_all_list.append(inp[3].detach().clone()[:, 0, :] if inp[3].dim() == 3 else inp[3].detach().clone())  # per-frame noise x [1,32] for P4 deterministic injection
        if "flow_latent_f0" in caps:
            return
        caps["flow_c_f0"] = inp[0].detach().clone()   # transformer condition [1,1024]
        caps["flow_s_f0"] = inp[1].detach().clone()   # start time [1,1] (=0)
        caps["flow_t_f0"] = inp[2].detach().clone()   # target time [1,1] (=1)
        caps["flow_x_f0"] = inp[3].detach().clone()   # noisy latent x = the noise [1,32]
        caps["flow_latent_f0"] = _first_tensor(out).detach().clone()   # flow_net output (velocity) [1,32]; latent = x + this
        flow_captured[0] = True                                        # freeze the sub-hooks after frame 0

    flow_captured = [False]

    # P3 flow-head intermediate gates (frame 0 only — in_generation skips the text-prompt call).
    # res_blocks[0] pre-hook gives (x=input_proj out, y=cond vec).
    def flow_sub(key):
        def hk(mod, inp, out):
            if in_generation[0] and not flow_captured[0] and key not in caps:
                caps[key] = _first_tensor(out).detach().clone()
        return hk

    def flow_resblock0_pre(mod, inp):
        if in_generation[0] and not flow_captured[0]:
            caps["flow_inproj"] = inp[0].detach().clone()     # x = input_proj(noise) [1,512]
            caps["flow_cond_vec"] = inp[1].detach().clone()   # y = 0.5(t0+t1)+cond_embed(c) [1,512]

    def flow_rb0_mlp_pre(mod, inp):
        if in_generation[0] and not flow_captured[0] and "flow_rb0_modulated" not in caps:
            caps["flow_rb0_modulated"] = inp[0].detach().clone()   # modulate(in_ln(x),shift,scale) [1,512]

    def flow_final_pre(mod, inp):
        if in_generation[0] and not flow_captured[0] and "flow_final_prelinear" not in caps:
            caps["flow_final_prelinear"] = inp[0].detach().clone()  # modulate(norm_final(x)) fed to final linear [1,512]

    def _first_tensor(x):
        # module outputs may be a tensor, or a tuple/list wrapping it (e.g. (emb,))
        while isinstance(x, (tuple, list)):
            x = x[0]
        return x

    def hook_upsample(mod, inp, out):
        if "mimi_upsampled_f0" not in caps:
            caps["mimi_upsampled_f0"] = _first_tensor(out).detach().clone()

    xf_calls = [0]

    def hook_xf(mod, inp, out):
        n = xf_calls[0]; xf_calls[0] += 1
        if n == 0:
            caps["mimi_xf_out_f0"] = _first_tensor(out).detach().clone()
        if n == 40:   # deep-tail latent (abs frames 640-655, well past the 250 context window)
            caps["mimi_xf_out_f40"] = _first_tensor(out).detach().clone()

    # SEANet decoder stage gates (first-latent output): model[0]=conv0, [3]/[6]/[9]=stage resblocks.
    def make_seanet_hook(key):
        def hk(mod, inp, out):
            if key not in caps:
                caps[key] = _first_tensor(out).detach().clone()   # [B,C,T]
        return hk

    h = []
    h.append(model.flow_lm.transformer.register_forward_hook(hook_transformer))
    h.append(model.flow_lm.flow_net.register_forward_hook(hook_flow))
    fn = model.flow_lm.flow_net
    h.append(fn.time_embed[0].register_forward_hook(flow_sub("flow_temb_s")))
    h.append(fn.time_embed[1].register_forward_hook(flow_sub("flow_temb_t")))
    h.append(fn.res_blocks[0].register_forward_pre_hook(flow_resblock0_pre))
    h.append(fn.res_blocks[0].register_forward_hook(flow_sub("flow_resblock0")))
    h.append(fn.res_blocks[0].adaLN_modulation.register_forward_hook(flow_sub("flow_rb0_adaln")))
    h.append(fn.res_blocks[0].mlp.register_forward_pre_hook(flow_rb0_mlp_pre))
    h.append(fn.res_blocks[0].mlp.register_forward_hook(flow_sub("flow_rb0_mlp")))
    h.append(fn.res_blocks[1].register_forward_hook(flow_sub("flow_resblock1")))
    h.append(fn.res_blocks[3].register_forward_hook(flow_sub("flow_resblock3")))
    h.append(fn.res_blocks[5].register_forward_hook(flow_sub("flow_resblock5")))
    h.append(fn.final_layer.linear.register_forward_pre_hook(flow_final_pre))
    h.append(model.mimi.upsample.register_forward_hook(hook_upsample))
    h.append(model.mimi.decoder_transformer.register_forward_hook(hook_xf))
    dec = model.mimi.decoder.model
    h.append(dec[0].register_forward_hook(make_seanet_hook("seanet_conv0_f0")))
    h.append(dec[3].register_forward_hook(make_seanet_hook("seanet_stage0_f0")))
    h.append(dec[6].register_forward_hook(make_seanet_hook("seanet_stage1_f0")))
    h.append(dec[9].register_forward_hook(make_seanet_hook("seanet_stage2_f0")))

    # ---- also capture every latent the flow produces (post-denorm, the Mimi decode input)
    latents = []
    orig_decode = model.mimi.decode_from_latent

    def wrapped_decode(latent, mimi_state):
        latents.append(latent.detach().clone())   # [B, 512?, steps] — this is POST quantizer proj
        return orig_decode(latent, mimi_state)

    # capture the pre-quantizer denormalized latent instead (what the C# feeds Mimi): hook the
    # worker's denorm by wrapping quantizer.forward to grab its INPUT [B,32,1]
    raw_lat = []
    quant_outs = []
    orig_q = model.mimi.quantizer.forward

    def wrapped_q(x):
        raw_lat.append(x.detach().clone())     # quantizer INPUT [B,32,1] = the DENORMED latent (Mimi decode input; latents.npy)
        out = orig_q(x)
        quant_outs.append(out.detach().clone())  # quantizer OUTPUT [B,512,1] (32->512 proj) — localization gate
        return out

    model.mimi.quantizer.forward = wrapped_q

    with torch.no_grad():
        audio = model.generate_audio(state, TEXT, frames_after_eos=2, copy_state=True)

    for hh in h:
        hh.remove()
    model.mimi.quantizer.forward = orig_q

    for k, v in caps.items():
        save(k, v)
    if raw_lat:
        # each is [B,32,1]; concat over time → [T,32]. This is the DENORMED latent (quantizer
        # input) = exactly the Mimi decoder's input; the C# Decode feeds it directly (no denorm).
        allc = torch.cat([x[:, :, 0] for x in raw_lat], dim=0)  # [T,32]
        save("latents", allc)
    if quant_outs:
        save("quant_out_f0", quant_outs[0][0, :, 0])            # [512] frame-0 quantizer output (localization gate)
    if flow_noise_all_list:
        # [T_emit+1, 32]: frame-n noise at index n (text-prompt call EXCLUDED), + the trailing
        # break-step noise whose latent is never emitted (a bit-exact C# run EOS-breaks before it).
        save("flow_noise_all", torch.cat(flow_noise_all_list, dim=0))
    # P4 listen sentence (names) — tokenize for the C# probe (no C# SentencePiece encoder yet)
    from pocket_tts.models.tts_model import prepare_text_prompt as _ptp
    NAMES = "Hi, my name is Sebastien Aigner, and I work with Radu Ciobanu and Nguyen."
    names_prep = model.flow_lm.conditioner.prepare(_ptp(NAMES, model.pad_with_spaces_for_short_inputs, model.remove_semicolons)[0])
    save("names_ids", names_prep.tokens)
    save("wav", audio.reshape(-1))

    # ---- P2: reproducible transformer forward (model_state=None => plain causal, NO KV cache) over
    # the EXACT assembled sequence the C# reconstructs: [bos_before_voice ; voice(125) ; text_emb ;
    # input_linear(bos_emb)]. Removes the predefined-voice KV-cache ambiguity and isolates
    # construction (xformer_in) from math (xformer_out_f0). RoPE offset=0 => positions 0..L-1.
    with torch.no_grad():
        voice32 = voice.to(torch.float32)                                            # [1,125,1024] (same tensor the state was built from)
        bbv = model.flow_lm.bos_before_voice.to(torch.float32)                       # [1,1,1024]
        text_emb2 = model.flow_lm.conditioner(prep).to(torch.float32)                # [1,S,1024]
        bos_lat = model.flow_lm.input_linear(model.flow_lm.bos_emb.view(1, 1, -1).to(torch.float32))  # [1,1,1024]
        seq = torch.cat([bbv, voice32, text_emb2, bos_lat], dim=1)                    # [1,L,1024]
        xf = _first_tensor(model.flow_lm.transformer(seq, None))                      # [1,L,1024] pre out_norm
        save("xformer_in", seq)
        save("xformer_out_f0", xf[:, -1:])                                           # last row [1,1,1024]
        print(f"xformer_in L={seq.shape[1]} (1 bbv + {voice32.shape[1]} voice + {text_emb2.shape[1]} text + 1 bos)")
        # SELF-CHECK: the hook-captured REAL generation frame 0 (KV-cache path, correct voice
        # state) must equal this no-KV full-forward reconstruction. If corr is not ~1.0, the
        # prefix/KV equivalence is broken and the C# full-forward loop cannot match generation.
        if "xformer_out_f0_gen" in caps:
            a = caps["xformer_out_f0_gen"].reshape(-1).to(torch.float32)
            b = xf[:, -1:].reshape(-1)
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            print(f"SELF-CHECK xformer f0 gen(KV) vs reconstruction(full): corr {corr:.6f} "
                  f"maxabs {(a - b).abs().max().item():.6f}"
                  + ("" if corr > 0.9999 else "  <-- MISMATCH, KV/prefix bug in the DUMP itself"))

    # ---- P8: voice-clone parity. Encode a fixed reference wav through the FULL Mimi encoder path
    # (encoder SEANet -> encoder_transformer -> downsample -> [1,32,T]) then speaker_proj -> the
    # audio_prompt [1,T,1024]. This is exactly _encode_audio(). The C# CloneVoice must reproduce
    # audio_prompt_ref at corr>0.99. Reference = the generated TTS wav (a real speech clip, self-
    # contained). Also dump the raw samples (voice_ref_audio) so the C# probe feeds the SAME input.
    with torch.no_grad():
        ref_wav = audio.reshape(-1).to(torch.float32)            # [S] 24 kHz mono
        save("voice_ref_audio", ref_wav)                         # [S]
        enc = model.mimi.encode_to_latent(ref_wav.view(1, 1, -1))  # [1,32,T]
        save("voice_ref_latents", enc[0].transpose(0, 1))        # [T,32] (pre speaker_proj, localization)
        conditioning = torch.nn.functional.linear(
            enc.transpose(-1, -2).to(torch.float32), model.flow_lm.speaker_proj_weight)  # [1,T,1024]
        save("audio_prompt_ref", conditioning[0])                # [T,1024] the cloned voice prefix
        print(f"P8 clone: ref_wav {ref_wav.shape[0]} samples ({ref_wav.shape[0]/model.sample_rate:.2f}s) "
              f"-> latents {list(enc.shape)} -> audio_prompt {list(conditioning[0].shape)}")

    # listen wav
    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(os.path.join(DUMP, "wav.wav"), model.sample_rate,
                               audio.reshape(-1).cpu().numpy())
    except Exception as e:
        print("wav write skipped:", e)

    meta = {
        "text": TEXT, "voice": VOICE, "voice_state": "audio_prompt-built (voice-cloning path)",
        "seed": SEED,
        "ldim": int(model.flow_lm.ldim), "dim": int(model.flow_lm.dim),
        "sample_rate": int(model.sample_rate),
        "frame_rate": 12.5, "seanet_ratios": [6, 5, 4], "hop_length": 120,
        "mimi_steps_per_latent": 16, "flow_depth": 6, "flow_dim": 512,
        "transformer_layers": 6, "transformer_heads": 16, "transformer_ffn": 4096,
        "rope_theta": 10000, "lsd_decode_steps": int(model.lsd_decode_steps),
        "audio_samples": int(audio.reshape(-1).shape[0]),
    }
    with open(os.path.join(DUMP, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"DONE. {len(raw_lat)} latents → {meta['audio_samples']} samples "
          f"({meta['audio_samples']/model.sample_rate:.2f}s). dump/ ready.")


if __name__ == "__main__":
    main()
