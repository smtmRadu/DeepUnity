#!/usr/bin/env python3
"""CosyVoice3 reference dumps + default-voice baking for the DeepUnity port (A0).

Runs ONE deterministic zero-shot synthesis (greedy LM, fixed flow noise) through the
official pipeline with forward hooks on every stage boundary, and writes:
  validation/dump/*.npy                          — per-stage parity tensors
  <weights_dir>/voices/default/*.bin (+manifest) — the baked default voice

Run (WSL):
  conda activate cosyvoice
  python dump_reference.py \
    --model_dir ~/cosyvoice_work/pretrained_models/Fun-CosyVoice3-0.5B \
    --repo      ~/cosyvoice_work/CosyVoice \
    --weights   /mnt/c/dev/DeepUnity/Assets/Resources/Weights/weights_cosyvoice3_fp16 \
    --out       /mnt/c/dev/DeepUnity/Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/validation/dump
"""
import argparse, os, sys
import numpy as np

# NOTE: the official ZH example — short EN text after the ZH prompt degenerates on the RL
# checkpoint (LM stops instantly), while this speaks ~8s. Parity is language-agnostic; the
# EN listen-quality test comes later with an EN-prompted baked voice (make_voice.py).
TTS_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.model_dir, a.repo = os.path.expanduser(a.model_dir), os.path.expanduser(a.repo)
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.repo)
    sys.path.insert(0, os.path.join(a.repo, "third_party/Matcha-TTS"))
    import torch
    torch.manual_seed(1986)

    # frontend needs spk2info.pt to exist in some repo revisions — provide an empty one
    spk2info = os.path.join(a.model_dir, "spk2info.pt")
    if not os.path.exists(spk2info):
        torch.save({}, spk2info)

    from cosyvoice.cli.cosyvoice import CosyVoice3
    cv = CosyVoice3(a.model_dir, fp16=False)     # llm.pt is a symlink to llm.rl.pt (exported variant)
    m = cv.model

    dumps = {}
    def save(name, t):
        arr = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
        dumps[name] = arr
        np.save(os.path.join(a.out, name + ".npy"), arr)
        print(f"[dump] {name:26s} {list(arr.shape)}")

    # ---- LM sampling: wrap sampling_ids — capture step-0 logp + the emitted sequence.
    # Greedy COLLAPSES to silent tokens on this model (like Chatterbox T3), so the reference
    # keeps the real seeded RAS sampling; Unity grades the LM on llm_logp_step0 and the
    # flow/hift stages on the INJECTED sampled token sequence (chatterbox parity recipe).
    gen_tokens = []
    orig_ids = m.llm.sampling_ids
    def ids_wrap(weighted_scores, decoded_tokens, sampling, ignore_eos=False):
        if len(gen_tokens) == 0 and "llm_logp_step0" not in dumps:
            save("llm_logp_step0", weighted_scores)   # log_softmax over 6761 at step 0
        top = orig_ids(weighted_scores, decoded_tokens, sampling, ignore_eos)
        gen_tokens.append(int(top if not torch.is_tensor(top) else top.reshape(-1)[0].item()))
        return top
    m.llm.sampling_ids = ids_wrap

    # ---- hooks ----------------------------------------------------------------------------
    state = {"llm_step": 0, "est_step": 0}
    def h_llm_decoder(mod, inp, out):
        if state["llm_step"] == 0:
            save("llm_logits_step0", out[:, -1] if out.dim() == 3 else out)
        state["llm_step"] += 1
    m.llm.llm_decoder.register_forward_hook(h_llm_decoder)

    def h_lookahead(mod, inp, out):
        if "flow_h_lookahead" not in dumps: save("flow_h_lookahead", out)
    m.flow.pre_lookahead_layer.register_forward_hook(h_lookahead)

    def h_estimator(mod, inp, out):
        if state["est_step"] < 1:   # step-0 CFG batch: row0 = cond, row1 = uncond
            x, mask, mu, t, spks, cond = inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]
            save("dit_in_x_step0", x); save("dit_in_mu_step0", mu)
            save("dit_in_t_step0", t); save("dit_in_spks_step0", spks)
            save("dit_in_cond_step0", cond); save("dit_dxdt_step0", out)
        state["est_step"] += 1
    m.flow.decoder.estimator.register_forward_hook(h_estimator)

    def h_f0(mod, inp, out):
        if "hift_f0" not in dumps: save("hift_f0", out)
    m.hift.f0_predictor.register_forward_hook(h_f0)

    def h_source(mod, inp, out):
        if "hift_source" not in dumps: save("hift_source", out[0])
    m.hift.m_source.register_forward_hook(h_source)

    flow_out = {}
    orig_flow_inf = m.flow.inference
    def flow_inf(*args, **kw):
        feat, cache = orig_flow_inf(*args, **kw)
        if "flow_mel" not in dumps: save("flow_mel", feat)
        return feat, cache
    m.flow.inference = flow_inf

    # ---- capture frontend outputs (they double as the baked default voice) -----------------
    fz = cv.frontend.frontend_zero_shot
    captured = {}
    def fz_wrap(tts_text, prompt_text, prompt_wav, resample_rate, spk_id):
        mi = fz(tts_text, prompt_text, prompt_wav, resample_rate, spk_id)
        if not captured:
            captured.update(mi)
            save("text_tokens", mi["text"] if "text" in mi else mi.get("tts_text_token", list(mi.values())[0]))
        return mi
    cv.frontend.frontend_zero_shot = fz_wrap

    prompt_wav = os.path.join(a.repo, "asset/zero_shot_prompt.wav")
    # CosyVoice3LM.inference asserts <|endofprompt|> (id 151646) is present in the text —
    # the caller appends it to the prompt transcript (SPEC §4).
    prompt_text = "希望你以后能够做的比我还好呦。<|endofprompt|>"

    wavs = [out["tts_speech"] for out in cv.inference_zero_shot(TTS_TEXT, prompt_text, prompt_wav, stream=False)]
    wav = torch.cat(wavs, dim=1)
    stops = [t for t in gen_tokens if t >= m.llm.speech_token_size]
    speech = [t for t in gen_tokens if t < m.llm.speech_token_size]
    assert len(speech) > 0, "LM generated zero speech tokens — greedy patch failed again"
    save("speech_tokens", np.array(speech, dtype=np.int64))
    print(f"[lm] generated {len(speech)} speech tokens (+{len(stops)} stop)")
    save("wav", wav)
    import torchaudio
    torchaudio.save(os.path.join(a.out, "reference.wav"), wav, cv.sample_rate)

    # remaining frontend tensors (names differ across revisions — resolve generically)
    def pick(*names):
        for n in names:
            if n in captured: return captured[n]
        raise KeyError(names)
    save("prompt_text_tokens",   pick("prompt_text"))
    save("prompt_speech_tokens", pick("llm_prompt_speech_token", "flow_prompt_speech_token"))
    save("prompt_feat",          pick("prompt_speech_feat"))
    save("embedding",            pick("llm_embedding", "flow_embedding"))

    # ---- bake the default voice into the exported weights manifest -------------------------
    vdir = os.path.join(a.weights, "voices", "default")
    os.makedirs(vdir, exist_ok=True)
    entries = []
    def bake(rel, arr, dtype):
        arr = np.asarray(arr)
        path = os.path.join(vdir, rel + ".bin")
        (arr.astype(np.int32) if dtype == "i32" else arr.astype(np.float16)).tofile(path)
        shape = ",".join(map(str, arr.shape))
        entries.append(f"voices/default/{rel}\tvoices/default/{rel}.bin\t{dtype}\t{arr.size}\t{shape}\n")
    bake("prompt_text_tokens",   dumps["prompt_text_tokens"].squeeze(0),   "i32")
    bake("prompt_speech_tokens", dumps["prompt_speech_tokens"].squeeze(0), "i32")
    bake("prompt_feat",          dumps["prompt_feat"].squeeze(0),          "f16")
    bake("embedding",            dumps["embedding"].squeeze(0),            "f16")
    man = os.path.join(a.weights, "manifest.tsv")
    existing = open(man, encoding="utf-8").read()
    with open(man, "a", encoding="utf-8") as f:
        for e in entries:
            if e.split("\t")[0] + "\t" not in existing:
                f.write(e)
    print(f"[voice] default voice baked into {vdir} (+manifest)")
    print("DUMP_REFERENCE DONE")

if __name__ == "__main__":
    main()
