#!/usr/bin/env python3
"""A3 debug: per-stage CosyVoice3LM intermediates for bisecting the Unity port.

Assembles lm_input exactly like Qwen2LM.inference (llm.py:474-494) from the existing
dumps, runs ONE full prefill forward, and saves: the assembled embeddings, every
layer's output, the final-norm last hidden, and the llm_decoder logits.

Run (WSL): conda activate cosyvoice; python dump_lm_stages.py
"""
import os, sys
import numpy as np

MODEL_DIR = os.path.expanduser("~/cosyvoice_work/pretrained_models/Fun-CosyVoice3-0.5B")
REPO = os.path.expanduser("~/cosyvoice_work/CosyVoice")
DUMP = "/mnt/c/dev/DeepUnity/Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/validation/dump"
OUT = os.path.join(DUMP, "lm_stages")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party/Matcha-TTS"))
import torch

from cosyvoice.cli.cosyvoice import CosyVoice3
cv = CosyVoice3(MODEL_DIR, fp16=False)
lm = cv.model.llm
lm.eval()

dev = next(lm.parameters()).device

def L(name):  # dumped token tensors are float32 npys
    return torch.from_numpy(np.load(os.path.join(DUMP, name + ".npy"))).long().to(dev)

prompt_text = L("prompt_text_tokens")
text = L("text_tokens")
prompt_speech = L("prompt_speech_tokens")

def save(name, t):
    arr = t.detach().float().cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    print(f"[stage] {name:16s} {list(arr.shape)}")

with torch.inference_mode():
    full_text = torch.concat([prompt_text, text], dim=1)
    text_emb = lm.llm.model.model.embed_tokens(full_text)
    sos_emb = lm.speech_embedding.weight[lm.sos].reshape(1, 1, -1)
    task_emb = lm.speech_embedding.weight[lm.task_id].reshape(1, 1, -1)
    prompt_emb = lm.speech_embedding(prompt_speech)
    lm_input = torch.concat([sos_emb, text_emb, task_emb, prompt_emb], dim=1)
    save("lm_embeds", lm_input)

    outs = {}
    hooks = []
    def cap(i, mod):
        def h(m, inp, out):
            key = f"lm_layer{i}"
            if key not in outs:
                outs[key] = True
                save(key, out[0] if isinstance(out, tuple) else out)
        hooks.append(mod.register_forward_hook(h))
    for i, layer in enumerate(lm.llm.model.model.layers):
        cap(i, layer)

    masks = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool)
    y_pred, cache = lm.llm.forward_one_step(lm_input, masks=masks, cache=None)
    save("lm_hidden_last", y_pred[:, -1])
    logits = lm.llm_decoder(y_pred[:, -1])
    save("lm_logits", logits)
    logp = logits.log_softmax(dim=-1)
    ref = np.load(os.path.join(DUMP, "llm_logp_step0.npy"))
    c = np.corrcoef(logp.numpy().ravel(), ref.ravel())[0, 1]
    print(f"[check] standalone logp vs pipeline llm_logp_step0 corr = {c:.6f} (sanity ~1.0)")
    print(f"[check] argmax standalone {int(logits.argmax())} vs pipeline {int(ref.argmax())}")
print("DUMP_LM_STAGES DONE")
