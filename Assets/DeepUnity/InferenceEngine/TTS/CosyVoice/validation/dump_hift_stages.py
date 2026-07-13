#!/usr/bin/env python3
"""A1 debug: per-stage CausalHiFT intermediates for bisecting the Unity port.

Standalone (hift.pt only — no LM/flow). Injects the reference NSF source from
dump/hift_source.npy (same as the Unity probe), so every stage is deterministic.

Run (WSL):
  conda activate cosyvoice
  python dump_hift_stages.py \
    --model_dir ~/cosyvoice_work/pretrained_models/Fun-CosyVoice3-0.5B \
    --repo      ~/cosyvoice_work/CosyVoice \
    --dump      /mnt/c/dev/DeepUnity/Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/validation/dump
"""
import argparse, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dump", required=True)
    a = ap.parse_args()
    a.model_dir, a.repo = os.path.expanduser(a.model_dir), os.path.expanduser(a.repo)
    out = os.path.join(a.dump, "hift_stages")
    os.makedirs(out, exist_ok=True)

    sys.path.insert(0, a.repo)
    sys.path.insert(0, os.path.join(a.repo, "third_party/Matcha-TTS"))
    import torch
    from hyperpyyaml import load_hyperpyyaml

    with open(os.path.join(a.model_dir, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    hift = configs["hift"]
    sd = torch.load(os.path.join(a.model_dir, "hift.pt"), map_location="cpu")
    sd = {k.replace("generator.", ""): v for k, v in sd.items()}
    hift.load_state_dict(sd, strict=True)
    hift.eval()

    mel = torch.from_numpy(np.load(os.path.join(a.dump, "flow_mel.npy"))).float()      # [1,80,T]
    src = torch.from_numpy(np.load(os.path.join(a.dump, "hift_source.npy"))).float()   # [1,S,1]

    # inject the dumped source exactly like the Unity probe (removes SineGen2 RNG)
    hift.m_source.forward = lambda x: (src, None, None)

    def save(name, t):
        arr = t.detach().float().cpu().numpy()
        np.save(os.path.join(out, name + ".npy"), arr)
        print(f"[stage] {name:14s} {list(arr.shape)}")

    hooks = []
    def cap(name, mod):
        def h(m, i, o):
            if not os.path.exists(os.path.join(out, name + ".npy")):
                save(name, o if torch.is_tensor(o) else o[0])
        hooks.append(mod.register_forward_hook(h))

    cap("conv_pre", hift.conv_pre)
    for i in range(3):
        cap(f"up{i}", hift.ups[i])
        cap(f"sdown{i}", hift.source_downs[i])
        cap(f"srb{i}", hift.source_resblocks[i])
    for n in range(9):
        cap(f"rb{n}", hift.resblocks[n])
    cap("conv_post", hift.conv_post)

    # source spectrum exactly as decode builds it
    with torch.no_grad():
        re, im = hift._stft(src.transpose(1, 2).squeeze(1))
        save("sstft", torch.cat([re, im], dim=1))
        wav, _ = hift.inference(speech_feat=mel)
    save("wav_stages", wav)

    ref_wav = np.load(os.path.join(a.dump, "wav.npy"))
    w = wav.detach().numpy()
    n = min(w.shape[-1], ref_wav.shape[-1])
    c = np.corrcoef(w.reshape(-1)[:n], ref_wav.reshape(-1)[:n])[0, 1]
    print(f"[check] standalone wav vs pipeline wav corr = {c:.6f} (sanity: should be ~1.0)")
    print("DUMP_HIFT_STAGES DONE")

if __name__ == "__main__":
    main()
