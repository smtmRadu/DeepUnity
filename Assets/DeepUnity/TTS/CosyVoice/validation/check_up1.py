#!/usr/bin/env python3
"""A1 debug: isolate the up1 divergence — exported weight vs dispatch math.

1. Folds ups.{0,1,2} weight-norm from hift.pt and diffs vs the exported .bin.
2. Recomputes up1 from the reference rb0..rb2 stage dumps with torch ops
   (mean -> leaky 0.1 -> nearest x5 -> pad-left 10 -> conv1d) vs up1.npy.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = os.path.expanduser("~/cosyvoice_work/pretrained_models/Fun-CosyVoice3-0.5B")
W = "/mnt/c/dev/DeepUnity/Assets/Resources/Weights/weights_cosyvoice3_fp16"
D = "/mnt/c/dev/DeepUnity/Assets/DeepUnity/TTS/CosyVoice/validation/dump/hift_stages"

manifest = {}
for line in open(os.path.join(W, "manifest.tsv"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 5:
        manifest[p[0]] = (p[1], p[2], int(p[3]), [int(x) for x in p[4].split(",")])

def load_bin(name):
    f, dt, numel, shape = manifest[name]
    raw = np.fromfile(os.path.join(W, f), dtype=np.float16 if dt == "f16" else np.int32)
    return torch.from_numpy(raw.astype(np.float32)).reshape(shape)

sd = torch.load(os.path.join(MODEL_DIR, "hift.pt"), map_location="cpu")
sd = {k.replace("generator.", ""): v for k, v in sd.items()}

print("== 1. export fold check (ups.*) ==")
for i in range(3):
    g = sd[f"ups.{i}.parametrizations.weight.original0"].float()
    v = sd[f"ups.{i}.parametrizations.weight.original1"].float()
    w_ref = g * v / v.norm(dim=(1, 2), keepdim=True)
    w_exp = load_bin(f"hift/ups.{i}.weight")
    d = (w_ref - w_exp).abs()
    print(f"ups.{i}: ref {list(w_ref.shape)} exp {list(w_exp.shape)} "
          f"maxAbs {d.max().item():.6f} mean {d.mean().item():.8f}")
    b_ref = sd[f"ups.{i}.bias"].float()
    b_exp = load_bin(f"hift/ups.{i}.bias")
    print(f"        bias maxAbs {(b_ref-b_exp).abs().max().item():.6f}")

print("== 2. recompute up1 from reference rb0..rb2 ==")
rb = [torch.from_numpy(np.load(os.path.join(D, f"rb{j}.npy"))).float() for j in range(3)]
x = (rb[0] + rb[1] + rb[2]) / 3.0                        # [1,256,3216]
x = F.leaky_relu(x, 0.1)
x = F.interpolate(x, scale_factor=5, mode="nearest")     # [1,256,16080]
x = F.pad(x, (10, 0))
g = sd["ups.1.parametrizations.weight.original0"].float()
v = sd["ups.1.parametrizations.weight.original1"].float()
w = g * v / v.norm(dim=(1, 2), keepdim=True)
y = F.conv1d(x, w, sd["ups.1.bias"].float())             # [1,128,16080]
ref = torch.from_numpy(np.load(os.path.join(D, "up1.npy"))).float()
d = (y - ref).abs()
c = np.corrcoef(y.numpy().ravel(), ref.numpy().ravel())[0, 1]
print(f"recomputed up1 vs ref: maxAbs {d.max().item():.6f} MAE {d.mean().item():.6f} corr {c:.6f}")

# same but with the EXPORTED fp16 weight (isolates fp16 rounding vs structure)
y2 = F.conv1d(x, load_bin("hift/ups.1.weight"), load_bin("hift/ups.1.bias"))
d2 = (y2 - ref).abs()
c2 = np.corrcoef(y2.numpy().ravel(), ref.numpy().ravel())[0, 1]
print(f"exported-weight up1 vs ref: maxAbs {d2.max().item():.6f} MAE {d2.mean().item():.6f} corr {c2:.6f}")
print("CHECK_UP1 DONE")
