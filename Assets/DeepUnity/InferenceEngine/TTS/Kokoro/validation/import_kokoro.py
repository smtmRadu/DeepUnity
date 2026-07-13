#!/usr/bin/env python3
"""
Kokoro-82M -> DeepUnity weight exporter (fp16 manifest.tsv + .bin, ChatterboxWeights format).

Turns kokoro-v1_0.pth (+ voicepacks + config.json vocab) into the binary params folder the
DeepUnity Kokoro runtime streams via the generic manifest loader (same TSV schema as
Assets/DeepUnity/InferenceEngine/TTS/Chatterbox/ChatterboxWeights.cs: name\tfile\tdtype\tnumel\tshape-csv,
fp16 packed 2-per-uint on the GPU side). SEPARATE from Assets/DeepUnity/InferenceEngine/import_params.py
on purpose (that file is owned by the LLM workstream).

USAGE (WSL, conda env `kokoro` — torch needed for the .pth/.pt pickles):
    conda activate kokoro
    python import_kokoro.py [--staging /mnt/c/dev/_model_staging/kokoro/hf] [--out DIR]
                            [--quant fp16|int8]

Inputs (staging dir, downloaded from HF hexgrad/Kokoro-82M — NEVER placed under Assets):
    kokoro-v1_0.pth   config.json   voices/*.pt

Output (default): Assets/Resources/Weights/weights_kokoro_<quant>/
    manifest.tsv + manifest.json + vocab.txt (line i = phoneme symbol for id i; id 0 = '$')
    + per-tensor .bin files (fp16; weight-norm folded; bert pooler skipped).

Conventions (see ../SPEC.md section 10):
  - old-style weight norm folded: w = g * v / ||v||_2(dims != 0)
  - LSTMs: 8 tensors each: <p>/wih, whh, bih, bhh, wih_r, whh_r, bih_r, bhh_r  (gates i,f,g,o)
  - ALBERT: the ONE shared layer exported once under bert/layer/*
  - voicepacks [510,1,256] -> voices/<name> [510,256]
  - completeness: every source key must be consumed exactly once or the export FAILS.
  - --quant int8: ONLY the GPU LinearBias matmul weights (QUANT_TARGETS below — the names
    KokoroModel.Linear() dispatches; the LinearBiasQ8 kernel reads them) go symmetric int8,
    one fp16 scale per OUTPUT ROW (scale_r = max|w_r|/127), same scheme as
    Assets/DeepUnity/InferenceEngine/import_params.py quantize_int8/TTSExporter.mat: manifest dtype "q8"
    (int8 packed 4-per-uint, low byte = element 0) file <name>.int8.bin + a SIBLING f16
    manifest entry "<name>.scales" file <name>.scales.bin. Everything else — convs,
    embeddings, norms, biases, LSTMs, voicepacks (the KokoroCPU-side families) — stays fp16,
    so the int8 folder is fully self-contained.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STAGING = "/mnt/c/dev/_model_staging/kokoro/hf" if os.name != "nt" \
    else "C:/dev/_model_staging/kokoro/hf"


# ----------------------------------------------------------------------------- int8 quant
# The ONLY int8 targets: weights the GPU runs through KokoroCS LinearBias(Q8) — i.e. every
# name KokoroModel.cs passes to its Linear() helper (directly or via StyleFc in AdainBlock /
# SnakeResBlock). All have in_dim % 4 == 0 (128 / 768 / 2048). Convs, embeddings, norms,
# projections and every KokoroCPU-consumed family (LSTMs, durenc adaln FCs, dur_proj,
# nsf_linear, voicepacks) stay fp16.
_ADAIN_BLOCKS = ([f"pred/{fam}_{i}" for fam in ("F0", "N") for i in range(3)]
                 + ["dec/encode"] + [f"dec/decode{i}" for i in range(4)])
_SNAKE_BLOCKS = [f"dec/gen/noise_res{i}" for i in range(2)] + [f"dec/gen/rb{r}" for r in range(6)]
QUANT_TARGETS = frozenset(
    ["bert/map.w", "benc.w"]
    + [f"bert/layer/{n}.w" for n in ("attn_q", "attn_k", "attn_v", "attn_o", "ffn", "ffn_out")]
    + [f"{b}/norm{j}_fc.w" for b in _ADAIN_BLOCKS for j in (1, 2)]
    + [f"{b}/ada{a}_{j}_fc.w" for b in _SNAKE_BLOCKS for a in (1, 2) for j in range(3)]
)  # 2 + 6 + 22 + 48 = 78 tensors

Q8_ERRS = []  # (max_abs_err, name) across all parts — reported at the end


def quantize_int8(w):  # fp32 [rows, cols] -> int8 + fp16 per-row scales + (max_err, mean_err)
    # copied from Assets/DeepUnity/InferenceEngine/import_params.py (keep byte-identical behavior)
    s = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-8)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    rec = q.astype(np.float32) * s[:, None].astype(np.float32)
    return q, s.astype(np.float16), (float(np.abs(rec - w).max()), float(np.abs(rec - w).mean()))


# ----------------------------------------------------------------------------- exporter core
class Exporter:
    """fp16 manifest exporter + consumed-key tracking over a flat {key: tensor} dict.
    quant == "int8": tensors named in QUANT_TARGETS divert to q8 (everything else fp16)."""

    def __init__(self, out_dir, src, quant="fp16"):
        self.out = out_dir
        self.src = src                      # {key: np.ndarray fp32}
        self.quant = quant
        self.consumed = set()
        self.manifest = {}                  # name -> {file, shape, dtype}
        self.bytes_written = 0

    def take(self, key):
        if key not in self.src:
            sys.exit(f"ERROR: checkpoint missing expected key '{key}'")
        if key in self.consumed:
            sys.exit(f"ERROR: key consumed twice: '{key}'")
        self.consumed.add(key)
        return self.src[key]

    def skip(self, key):
        if key in self.src:
            self.consumed.add(key)

    def f16(self, name, arr):
        if self.quant == "int8" and name in QUANT_TARGETS:
            self.q8(name, arr)
            return
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).astype(np.float16))
        rel = name + ".bin"
        path = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr.tofile(path)
        self.bytes_written += os.path.getsize(path)
        self.manifest[name] = {"file": rel, "shape": [int(d) for d in arr.shape], "dtype": "f16"}

    def q8(self, name, arr):
        """Symmetric int8, one fp16 scale per OUTPUT ROW (TTSExporter.mat convention):
        <name>.int8.bin (dtype q8, packed 4-per-uint GPU-side, low byte = element 0)
        + sibling manifest entry <name>.scales -> <name>.scales.bin (f16 [rows])."""
        w = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        assert w.ndim == 2 and w.shape[1] % 4 == 0, \
            f"{name}: q8 needs a 2-D [rows, cols] weight with cols % 4 == 0, got {w.shape}"
        q, s, err = quantize_int8(w)
        qrel, srel = name + ".int8.bin", name + ".scales.bin"
        qp, sp = os.path.join(self.out, qrel), os.path.join(self.out, srel)
        os.makedirs(os.path.dirname(qp), exist_ok=True)
        q.tofile(qp)
        s.tofile(sp)
        self.bytes_written += os.path.getsize(qp) + os.path.getsize(sp)
        self.manifest[name] = {"file": qrel, "shape": [int(d) for d in w.shape], "dtype": "q8"}
        self.manifest[name + ".scales"] = {"file": srel, "shape": [int(w.shape[0])], "dtype": "f16"}
        Q8_ERRS.append((err[0], name))

    # -- structured helpers ----------------------------------------------------
    def lin(self, dst, src, bias=True):
        """Plain Linear/Conv (no weight norm): <src>.weight/.bias -> <dst>.w/.b"""
        self.f16(dst + ".w", self.take(src + ".weight"))
        if bias:
            self.f16(dst + ".b", self.take(src + ".bias"))

    def wn(self, dst, src, bias=True):
        """Old-style weight-normed conv: fold weight_g [out,1,..]*weight_v/||v|| -> <dst>.w"""
        g = self.take(src + ".weight_g").astype(np.float32)
        v = self.take(src + ".weight_v").astype(np.float32)
        axes = tuple(range(1, v.ndim))
        n = np.sqrt((v * v).sum(axis=axes, keepdims=True))
        self.f16(dst + ".w", g * v / np.maximum(n, 1e-12))
        if bias:
            self.f16(dst + ".b", self.take(src + ".bias"))

    def lstm(self, dst, src):
        """Bidirectional 1-layer LSTM -> 8 tensors (torch gate order i,f,g,o)."""
        for suf, out in (("", ""), ("_reverse", "_r")):
            self.f16(f"{dst}/wih{out}", self.take(f"{src}.weight_ih_l0{suf}"))
            self.f16(f"{dst}/whh{out}", self.take(f"{src}.weight_hh_l0{suf}"))
            self.f16(f"{dst}/bih{out}", self.take(f"{src}.bias_ih_l0{suf}"))
            self.f16(f"{dst}/bhh{out}", self.take(f"{src}.bias_hh_l0{suf}"))

    def adain_blk(self, dst, src):
        """AdainResBlk1d (istftnet.py): conv1/conv2 (wn), norm1.fc/norm2.fc, optional conv1x1
        (wn, no bias) + pool (wn depthwise ConvT, upsample blocks only)."""
        self.wn(f"{dst}/conv1", f"{src}.conv1")
        self.wn(f"{dst}/conv2", f"{src}.conv2")
        self.lin(f"{dst}/norm1_fc", f"{src}.norm1.fc")
        self.lin(f"{dst}/norm2_fc", f"{src}.norm2.fc")
        if f"{src}.conv1x1.weight_v" in self.src:
            self.wn(f"{dst}/conv1x1", f"{src}.conv1x1", bias=False)
        if f"{src}.pool.weight_v" in self.src:
            self.wn(f"{dst}/pool", f"{src}.pool")

    def snake_blk(self, dst, src):
        """AdaINResBlock1 (generator): convs1/convs2 ×3 (wn), adain1/adain2 fc ×3, alpha1/2 ×3."""
        for j in range(3):
            self.wn(f"{dst}/c1_{j}", f"{src}.convs1.{j}")
            self.wn(f"{dst}/c2_{j}", f"{src}.convs2.{j}")
            self.lin(f"{dst}/ada1_{j}_fc", f"{src}.adain1.{j}.fc")
            self.lin(f"{dst}/ada2_{j}_fc", f"{src}.adain2.{j}.fc")
            self.f16(f"{dst}/a1_{j}", self.take(f"{src}.alpha1.{j}").reshape(-1))
            self.f16(f"{dst}/a2_{j}", self.take(f"{src}.alpha2.{j}").reshape(-1))

    def save_manifest(self):
        with open(os.path.join(self.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)
        with open(os.path.join(self.out, "manifest.tsv"), "w", encoding="utf-8", newline="\n") as f:
            for name, m in self.manifest.items():
                numel = 1
                for d in m["shape"]:
                    numel *= d
                f.write(f"{name}\t{m['file']}\t{m['dtype']}\t{numel}\t{','.join(map(str, m['shape']))}\n")


# ----------------------------------------------------------------------------- sections
def export_bert(ex):
    """PLBERT (ALBERT): embeddings + hidden mapping + the ONE shared layer. Pooler skipped."""
    e = "embeddings."
    ex.lin("bert/emb/word", e + "word_embeddings", bias=False)
    ex.lin("bert/emb/pos", e + "position_embeddings", bias=False)
    ex.lin("bert/emb/tok", e + "token_type_embeddings", bias=False)
    ex.lin("bert/emb/ln", e + "LayerNorm")
    ex.lin("bert/map", "encoder.embedding_hidden_mapping_in")
    L = "encoder.albert_layer_groups.0.albert_layers.0."
    for dst, src in (("attn_q", "attention.query"), ("attn_k", "attention.key"),
                     ("attn_v", "attention.value"), ("attn_o", "attention.dense"),
                     ("attn_ln", "attention.LayerNorm"), ("ffn", "ffn"),
                     ("ffn_out", "ffn_output"), ("ln", "full_layer_layer_norm")):
        ex.lin(f"bert/layer/{dst}", L + src)
    ex.skip("pooler.weight")
    ex.skip("pooler.bias")


def export_predictor(ex):
    # DurationEncoder: lstms.[0,2,4] = biLSTM, lstms.[1,3,5] = AdaLayerNorm
    for i, src_i in enumerate((0, 2, 4)):
        ex.lstm(f"pred/durenc/lstm{i}", f"text_encoder.lstms.{src_i}")
    for i, src_i in enumerate((1, 3, 5)):
        ex.lin(f"pred/durenc/adaln{i}_fc", f"text_encoder.lstms.{src_i}.fc")
    ex.lstm("pred/lstm", "lstm")
    ex.lin("pred/dur_proj", "duration_proj.linear_layer")
    ex.lstm("pred/shared", "shared")
    for fam in ("F0", "N"):
        for i in range(3):
            ex.adain_blk(f"pred/{fam}_{i}", f"{fam}.{i}")
        ex.lin(f"pred/{fam}_proj", f"{fam}_proj")


def export_text_encoder(ex):
    ex.lin("tenc/embedding", "embedding", bias=False)
    for i in range(3):
        ex.wn(f"tenc/cnn{i}/conv", f"cnn.{i}.0")
        ex.f16(f"tenc/cnn{i}/ln.w", ex.take(f"cnn.{i}.1.gamma"))
        ex.f16(f"tenc/cnn{i}/ln.b", ex.take(f"cnn.{i}.1.beta"))
    ex.lstm("tenc/lstm", "lstm")


def export_decoder(ex):
    ex.adain_blk("dec/encode", "encode")
    for i in range(4):
        ex.adain_blk(f"dec/decode{i}", f"decode.{i}")
    ex.wn("dec/F0_conv", "F0_conv")
    ex.wn("dec/N_conv", "N_conv")
    ex.wn("dec/asr_res", "asr_res.0")
    G = "generator."
    ex.lin("dec/gen/nsf_linear", G + "m_source.l_linear")
    for i in range(2):
        ex.lin(f"dec/gen/noise_conv{i}", G + f"noise_convs.{i}")
        ex.snake_blk(f"dec/gen/noise_res{i}", G + f"noise_res.{i}")
        ex.wn(f"dec/gen/ups{i}", G + f"ups.{i}")
    for r in range(6):
        ex.snake_blk(f"dec/gen/rb{r}", G + f"resblocks.{r}")
    ex.wn("dec/gen/conv_post", G + "conv_post")


EXPECTED_SHAPES = {  # spot checks against SPEC.md (exporter fails loudly on drift)
    "bert/emb/word.w": (178, 128),
    "bert/map.w": (768, 128),
    "bert/layer/ffn.w": (2048, 768),
    "benc.w": (512, 768),
    "pred/dur_proj.w": (50, 512),
    "pred/F0_proj.w": (1, 256, 1),
    "tenc/embedding.w": (178, 512),
    "dec/gen/ups0.w": (512, 256, 20),   # ConvTranspose1d [in, out, k]
    "dec/gen/conv_post.w": (22, 128, 7),
    "dec/gen/nsf_linear.w": (1, 9),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--staging", default=DEFAULT_STAGING,
                    help="dir with kokoro-v1_0.pth + config.json + voices/*.pt")
    ap.add_argument("--out", default=None, help="override output folder")
    ap.add_argument("--quant", choices=("fp16", "int8"), default="fp16",
                    help="int8: QUANT_TARGETS (GPU LinearBiasQ8 matmuls) go q8 + per-row "
                         "f16 scales; everything else stays fp16")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("ERROR: torch required to unpickle .pth/.pt (run in the WSL `kokoro` conda env).")

    pth = os.path.join(args.staging, "kokoro-v1_0.pth")
    cfg_path = os.path.join(args.staging, "config.json")
    voices_dir = os.path.join(args.staging, "voices")
    for p in (pth, cfg_path):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: missing {p} (download from HF hexgrad/Kokoro-82M into staging first)")

    # Assets/DeepUnity/InferenceEngine/TTS/Kokoro/validation -> Assets/Resources/Weights/...
    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "..", "..", "Resources", "Weights",
        f"weights_kokoro_{args.quant}"))
    os.makedirs(out, exist_ok=True)
    print(f"[out]    {out}")

    sd = torch.load(pth, map_location="cpu", weights_only=True)
    parts = {"bert", "bert_encoder", "predictor", "text_encoder", "decoder"}
    assert set(sd.keys()) == parts, f"unexpected top-level keys: {set(sd.keys())}"

    for part, fn in (("bert", export_bert), ("predictor", export_predictor),
                     ("text_encoder", export_text_encoder), ("decoder", export_decoder)):
        flat = {(k[7:] if k.startswith("module.") else k): v.numpy().astype(np.float32)
                for k, v in sd[part].items()}
        ex_part = Exporter(out, flat, args.quant)
        ex_part.manifest = main.manifest  # shared manifest across parts
        ex_part.bytes_written = main.bytes_written
        fn(ex_part)
        left = set(flat) - ex_part.consumed
        if left:
            sys.exit(f"ERROR: {part}: {len(left)} unconsumed keys, e.g. {sorted(left)[:5]}")
        main.bytes_written = ex_part.bytes_written
        print(f"[{part}] {len(ex_part.consumed)} keys -> exported")

    # bert_encoder is a bare Linear: keys module.weight / module.bias
    be = {(k[7:] if k.startswith("module.") else k): v.numpy().astype(np.float32)
          for k, v in sd["bert_encoder"].items()}
    ex = Exporter(out, be, args.quant)
    ex.manifest = main.manifest
    ex.bytes_written = main.bytes_written
    ex.f16("benc.w", ex.take("weight"))
    ex.f16("benc.b", ex.take("bias"))
    main.bytes_written = ex.bytes_written
    print(f"[bert_encoder] 2 keys -> exported")

    # voicepacks [510,1,256] -> voices/<name> [510,256]
    n_voices = 0
    if os.path.isdir(voices_dir):
        for f in sorted(os.listdir(voices_dir)):
            if not f.endswith(".pt"):
                continue
            pack = torch.load(os.path.join(voices_dir, f), map_location="cpu", weights_only=True)
            a = pack.numpy().astype(np.float32)
            assert a.shape == (510, 1, 256), f"unexpected voicepack shape {a.shape} in {f}"
            ex.f16(f"voices/{f[:-3]}", a.reshape(510, 256))
            n_voices += 1
    if n_voices == 0:
        print("WARNING: no voicepacks exported (put voices/*.pt in staging).")

    # shape spot-checks
    for name, shape in EXPECTED_SHAPES.items():
        got = tuple(ex.manifest[name]["shape"])
        if got != shape:
            sys.exit(f"ERROR: {name}: exported shape {got}, SPEC expects {shape}")

    # vocab.txt: line i = symbol for id i (id 0 = '$'); DO NOT trim lines when parsing (id 16 = ' ')
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    inv = {int(i): s for s, i in cfg["vocab"].items()}
    inv[0] = "$"
    n_token = int(cfg["n_token"])
    with open(os.path.join(out, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        for i in range(n_token):
            f.write(inv.get(i, "") + "\n")

    # int8 completeness: every QUANT_TARGET must have landed as q8, and nothing else
    if args.quant == "int8":
        q8_names = {n for n, m in ex.manifest.items() if m["dtype"] == "q8"}
        if q8_names != QUANT_TARGETS:
            missing, extra = QUANT_TARGETS - q8_names, q8_names - QUANT_TARGETS
            sys.exit(f"ERROR: q8 target mismatch — missing {sorted(missing)[:5]}, "
                     f"extra {sorted(extra)[:5]}")
        worst = max(Q8_ERRS)
        print(f"[int8] {len(q8_names)} tensors q8 (+{len(q8_names)} .scales siblings); "
              f"worst |err| {worst[0]:.5f} @ {worst[1]}")

    ex.save_manifest()
    mb = main.bytes_written / 1024 / 1024
    print(f"\nDone - {len(ex.manifest)} tensors ({n_voices} voicepacks), {mb:.0f} MB -> {out}")
    print("Manifest: manifest.tsv (name\\tfile\\tdtype\\tnumel\\tshape) + manifest.json + vocab.txt")


main.manifest = {}
main.bytes_written = 0

if __name__ == "__main__":
    main()
