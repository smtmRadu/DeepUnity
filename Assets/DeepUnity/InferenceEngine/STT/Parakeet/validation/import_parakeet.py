#!/usr/bin/env python3
"""
DeepUnity Parakeet-TDT weight exporter — turns an HF-format Parakeet-TDT checkpoint folder
(config.json + model.safetensors + tokenizer.json) into the manifest.tsv + .bin params folder
the DeepUnity loader streams at runtime (ChatterboxWeights/CosyVoiceWeights contract).

USAGE
    python import_parakeet.py <hf_folder> [--quant fp16|int8] [--out DIR]

    python import_parakeet.py C:/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v3-hf
    python import_parakeet.py C:/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v2-hf --quant int8

  v3 is HF-native on the hub (nvidia/parakeet-tdt-0.6b-v3). v2 ships .nemo-only — convert it once
  with validation/convert_v2_nemo.py (wraps transformers' official convert_nemo_to_hf), then feed
  the converted folder here. Both variants flow through identical HF tensor names.

OUTPUT (self-describing; variant auto-detected from config.json vocab_size)
    Assets/Resources/Weights/weights_parakeet_tdt_0.6b_{v2,v3}_{quant}/
        manifest.tsv                 name\tfile\tdtype\tnumel\tshape-csv   (+ manifest.json twin)
        frontend/mel_filters.bin     [128,257] slaney mel bank (baked; vendored numpy == librosa)
        frontend/window.bin          [400] symmetric hann
        sub/*, layer_{0..23}/*, dec/*, joint/*    per SPEC.md §10
        tokenizer/vocab.txt          line i = token string for id i
        tokenizer/specials.tsv       id\tcontent\tspecial\tbyte

QUANT   fp16 = packed 2-per-uint reference format (dtype f16)
        int8 = ONLY the big encoder matmuls (ff linears, q/k/v/o, rel-pos proj, conv pointwise,
               subsampling linear, enc_proj) as q8 + fp16 per-output-row .scales — norms, biases,
               depthwise/2-D convs, embedding, LSTM, pred_proj and joint head stay fp16
               (repo convention; dec/* + joint/head are CPU-side anyway).

BatchNorm is FOLDED at export (eval running stats -> conv.bn.scale/.shift). The decoder embedding
blank row is asserted ~0 (NeMo blank_as_pad padding_idx invariant the C# loop relies on).

Requires: numpy only.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# What ParakeetConfig.cs will hard-code (config.json keys, both variants).
EXPECTED_ENCODER = {
    "hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8,
    "num_key_value_heads": 8, "intermediate_size": 4096, "conv_kernel_size": 9,
    "num_mel_bins": 128, "subsampling_factor": 8, "subsampling_conv_channels": 256,
    "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
    "attention_bias": False, "convolution_bias": False, "scale_input": False,
    "hidden_act": "silu", "max_position_embeddings": 5000,
}
EXPECTED_TOP = {"decoder_hidden_size": 640, "num_decoder_layers": 2,
                "durations": [0, 1, 2, 3, 4], "hidden_act": "relu", "max_symbols_per_step": 10}
VOCAB_TO_VARIANT = {8193: "v3", 1025: "v2"}


# ----------------------------------------------------------------------------- safetensors reader
class Safetensors:
    """Single/multi-file safetensors reader: header parsing + per-key fp32 loads, no torch."""

    def __init__(self, files):
        self.map = {}
        for path in files:
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len).decode("utf-8"))
            base = 8 + header_len
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                self.map[key] = (path, meta["dtype"], tuple(meta["shape"]), meta["data_offsets"], base)

    def keys(self):
        return self.map.keys()

    def __contains__(self, key):
        return key in self.map

    def load(self, key):  # -> fp32 ndarray
        if key not in self.map:
            raise KeyError(f"safetensors: missing tensor '{key}'")
        path, dtype, shape, (a, b), base = self.map[key]
        raw = np.memmap(path, dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,)).tobytes()
        if dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if dtype == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            return (u16.astype(np.uint32) << 16).view(np.float32).copy()
        raise ValueError(f"unsupported dtype {dtype} for {key}")


# ----------------------------------------------------------------------------- frontend synthesis
def hann_symmetric(n=400):
    i = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))).astype(np.float32)


def mel_filters_slaney(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0):
    """librosa.filters.mel(norm='slaney', htk=False) reimplemented in numpy (verified equal)."""
    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        mel = (f - 0.0) / (200.0 / 3)
        min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / (200.0 / 3), np.log(6.4) / 27.0
        if mel.ndim:
            log = f >= min_log_hz
            mel[log] = min_log_mel + np.log(f[log] / min_log_hz) / logstep
        elif f >= min_log_hz:
            mel = min_log_mel + np.log(f / min_log_hz) / logstep
        return mel

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f = 0.0 + (200.0 / 3) * m
        min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / (200.0 / 3), np.log(6.4) / 27.0
        log = m >= min_log_mel
        f[log] = min_log_hz * np.exp(logstep * (m[log] - min_log_mel))
        return f

    n_bins = 1 + n_fft // 2
    fftfreqs = np.linspace(0, sr / 2.0, n_bins)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    weights = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]  # slaney area norm
    return weights.astype(np.float32)  # [n_mels, n_bins]


# ----------------------------------------------------------------------------- quant + exporter
def quantize_int8(w):
    s = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-8)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    rec = q.astype(np.float32) * s[:, None].astype(np.float32)
    return q, s.astype(np.float16), (float(np.abs(rec - w).max()), float(np.abs(rec - w).mean()))


class Exporter:
    """Manifest exporter (ChatterboxWeights contract): f16 always-precise, mat() per --quant."""

    def __init__(self, out_dir, quant):
        self.out, self.quant = out_dir, quant
        self.manifest = {}
        self.bytes_written = 0
        self.worst = (0.0, "-")

    def _reg(self, rel, arr, ext, dtype):
        self.manifest[rel] = {"file": rel + ext, "shape": [int(d) for d in arr.shape], "dtype": dtype}

    def _track(self, path, err=None, name=None):
        self.bytes_written += os.path.getsize(path)
        if err and err[0] > self.worst[0]:
            self.worst = (err[0], name)

    def f16(self, rel, arr):
        arr = np.asarray(arr)
        path = os.path.join(self.out, rel + ".bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.ascontiguousarray(arr.astype(np.float16)).tofile(path)
        self._track(path)
        self._reg(rel, arr, ".bin", "f16")

    def mat(self, rel, arr):
        arr = np.asarray(arr, dtype=np.float32)
        if self.quant == "fp16":
            self.f16(rel, arr)
            return
        os.makedirs(os.path.dirname(os.path.join(self.out, rel)), exist_ok=True)
        q, s, err = quantize_int8(arr)
        qp, sp = os.path.join(self.out, rel + ".int8.bin"), os.path.join(self.out, rel + ".scales.bin")
        q.tofile(qp)
        s.tofile(sp)
        self._track(qp, err, rel)
        self._track(sp)
        self._reg(rel, arr, ".int8.bin", "q8")
        self.manifest[rel + ".scales"] = {"file": rel + ".scales.bin", "shape": [int(arr.shape[0])], "dtype": "f16"}

    def save_manifest(self):
        with open(os.path.join(self.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)
        with open(os.path.join(self.out, "manifest.tsv"), "w", encoding="utf-8", newline="\n") as f:
            for name, m in self.manifest.items():
                numel = int(np.prod(m["shape"]))
                f.write(f"{name}\t{m['file']}\t{m['dtype']}\t{numel}\t{','.join(map(str, m['shape']))}\n")


# ----------------------------------------------------------------------------- checks
def check_config(cfg):
    enc = cfg["encoder_config"]
    bad = [f"  encoder.{k}: checkpoint={enc.get(k)!r}, expected {v!r}"
           for k, v in EXPECTED_ENCODER.items() if enc.get(k) != v]
    bad += [f"  {k}: checkpoint={cfg.get(k)!r}, expected {v!r}"
            for k, v in EXPECTED_TOP.items() if cfg.get(k) != v]
    if bad:
        print("\n" + "!" * 88)
        print("WARNING: checkpoint config differs from the constants SPEC.md/ParakeetConfig.cs assume:")
        print("\n".join(bad))
        print("!" * 88 + "\n")
    v = cfg["vocab_size"]
    if v not in VOCAB_TO_VARIANT:
        sys.exit(f"ERROR: unexpected vocab_size={v} (known: {VOCAB_TO_VARIANT}) — not a 0.6b v2/v3 TDT?")
    if cfg.get("blank_token_id") != v - 1:
        sys.exit(f"ERROR: blank_token_id={cfg.get('blank_token_id')} != vocab_size-1={v - 1}")
    return VOCAB_TO_VARIANT[v]


# ----------------------------------------------------------------------------- export passes
def export_frontend(ex):
    ex.f16("frontend/mel_filters", mel_filters_slaney())  # [128,257]
    ex.f16("frontend/window", hann_symmetric())            # [400]


def export_subsampling(reader, ex):
    # layers ModuleList: 0 Conv2d(1->256) / 1 ReLU / 2 dw / 3 pw / 4 ReLU / 5 dw / 6 pw / 7 ReLU
    p = "encoder.subsampling."
    ex.f16("sub/conv0.w", reader.load(p + "layers.0.weight"))            # [256,1,3,3]
    ex.f16("sub/conv0.b", reader.load(p + "layers.0.bias"))
    for j, (dw, pw) in enumerate([(2, 3), (5, 6)], start=1):
        cv = reader.load(p + f"layers.{dw}.weight")                       # [256,1,3,3]
        ex.f16(f"sub/conv{j}_dw.w", cv.reshape(cv.shape[0], 3, 3))
        ex.f16(f"sub/conv{j}_dw.b", reader.load(p + f"layers.{dw}.bias"))
        pww = reader.load(p + f"layers.{pw}.weight")                      # [256,256,1,1]
        ex.f16(f"sub/conv{j}_pw.w", pww.reshape(pww.shape[0], pww.shape[1]))
        ex.f16(f"sub/conv{j}_pw.b", reader.load(p + f"layers.{pw}.bias"))
    ex.mat("sub/linear.w", reader.load(p + "linear.weight"))              # [1024,4096]
    ex.f16("sub/linear.b", reader.load(p + "linear.bias"))


def export_encoder_layer(reader, ex, i):
    kp, lp = f"encoder.layers.{i}.", f"layer_{i}/"

    def ln(dst, src):
        ex.f16(lp + dst + ".w", reader.load(kp + src + ".weight"))
        ex.f16(lp + dst + ".b", reader.load(kp + src + ".bias"))

    for ff, n in (("ff1", "norm_feed_forward1"), ("ff2", "norm_feed_forward2")):
        ln(ff + ".ln", n)
        src = "feed_forward1" if ff == "ff1" else "feed_forward2"
        ex.mat(lp + ff + ".lin1.w", reader.load(kp + src + ".linear1.weight"))   # [4096,1024] no bias
        ex.mat(lp + ff + ".lin2.w", reader.load(kp + src + ".linear2.weight"))   # [1024,4096] no bias

    ln("attn.ln", "norm_self_att")
    for t in ("q", "k", "v", "o"):
        ex.mat(lp + f"attn.{t}.w", reader.load(kp + f"self_attn.{t}_proj.weight"))
    ex.mat(lp + "attn.pos.w", reader.load(kp + "self_attn.relative_k_proj.weight"))
    ex.f16(lp + "attn.bias_u", reader.load(kp + "self_attn.bias_u"))             # [8,128]
    ex.f16(lp + "attn.bias_v", reader.load(kp + "self_attn.bias_v"))

    ln("conv.ln", "norm_conv")
    pw1 = reader.load(kp + "conv.pointwise_conv1.weight")                        # [2048,1024,1]
    ex.mat(lp + "conv.pw1.w", pw1.reshape(pw1.shape[0], pw1.shape[1]))
    dw = reader.load(kp + "conv.depthwise_conv.weight")                          # [1024,1,9]
    ex.f16(lp + "conv.dw.w", dw.reshape(dw.shape[0], dw.shape[-1]))
    # fold BatchNorm1d eval stats -> per-channel scale/shift
    g = reader.load(kp + "conv.norm.weight")
    b = reader.load(kp + "conv.norm.bias")
    mean = reader.load(kp + "conv.norm.running_mean")
    var = reader.load(kp + "conv.norm.running_var")
    scale = g / np.sqrt(var + 1e-5)
    ex.f16(lp + "conv.bn.scale", scale)
    ex.f16(lp + "conv.bn.shift", b - mean * scale)
    pw2 = reader.load(kp + "conv.pointwise_conv2.weight")                        # [1024,1024,1]
    ex.mat(lp + "conv.pw2.w", pw2.reshape(pw2.shape[0], pw2.shape[1]))

    ln("out_ln", "norm_out")


def export_decoder_joint(reader, ex, vocab):
    emb = reader.load("decoder.embedding.weight")                                # [V,640]
    blank_row_max = float(np.abs(emb[vocab - 1]).max())
    print(f"[check]  decoder embedding blank row max|w| = {blank_row_max:.2e} "
          f"({'OK, ~zero' if blank_row_max < 1e-6 else 'NOT ZERO — C# must special-case start token!'})")
    ex.f16("dec/embedding", emb)
    for l in (0, 1):
        for t, n in (("wih", "weight_ih"), ("whh", "weight_hh"), ("bih", "bias_ih"), ("bhh", "bias_hh")):
            ex.f16(f"dec/lstm.{t}{l}", reader.load(f"decoder.lstm.{n}_l{l}"))
    ex.f16("dec/pred_proj.w", reader.load("decoder.decoder_projector.weight"))
    ex.f16("dec/pred_proj.b", reader.load("decoder.decoder_projector.bias"))
    ex.mat("joint/enc_proj.w", reader.load("encoder_projector.weight"))          # [640,1024]
    ex.f16("joint/enc_proj.b", reader.load("encoder_projector.bias"))
    ex.f16("joint/head.w", reader.load("joint.head.weight"))                     # [V+5,640] CPU-side
    ex.f16("joint/head.b", reader.load("joint.head.bias"))


def export_tokenizer(src_dir, out_dir, vocab_size):
    tok_path = os.path.join(src_dir, "tokenizer.json")
    if not os.path.isfile(tok_path):
        print(f"[tok]    WARNING: {tok_path} missing — tokenizer export skipped (converted v2 should have it)")
        return
    with open(tok_path, encoding="utf-8") as f:
        tok = json.load(f)
    if tok["model"]["type"] != "BPE":
        sys.exit(f"ERROR: expected BPE tokenizer.json, got {tok['model']['type']}")
    by_id = {}
    for s, i in tok["model"]["vocab"].items():
        by_id[i] = s
    added = {a["id"]: a["content"] for a in tok.get("added_tokens", [])}
    # only added tokens FLAGGED special are skip-on-decode: v3 carries plain digits '0'-'9' as
    # added tokens with special:false and they must survive decoding (HF skip_special_tokens rule)
    specials = {a["id"]: a["content"] for a in tok.get("added_tokens", []) if a.get("special", True)}
    by_id.update(added)  # added tokens override/extend (e.g. <blank> at vocab_size-1)
    n = max(by_id) + 1
    if n != vocab_size:
        print(f"[tok]    NOTE: tokenizer ids 0..{n - 1} vs model vocab {vocab_size} (blank may be model-only)")
    tdir = os.path.join(out_dir, "tokenizer")
    os.makedirs(tdir, exist_ok=True)
    n_bytes = 0
    with open(os.path.join(tdir, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f, \
         open(os.path.join(tdir, "specials.tsv"), "w", encoding="utf-8", newline="\n") as g:
        for i in range(max(n, vocab_size)):
            s = by_id.get(i, "<unused>")
            assert "\n" not in s and "\t" not in s, f"token {i} contains newline/tab"
            f.write(s + "\n")
            is_byte = int(len(s) == 6 and s.startswith("<0x") and s.endswith(">"))
            n_bytes += is_byte
            g.write(f"{i}\t{s}\t{int(i in specials)}\t{is_byte}\n")
    print(f"[tok]    vocab.txt ({max(n, vocab_size)} ids), specials={len(specials)}, "
          f"byte-fallback tokens={n_bytes} {'(C# needs byte accumulation!)' if n_bytes else '(none — plain string decode)'}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Export an HF Parakeet-TDT folder into DeepUnity params.")
    ap.add_argument("src", help="HF checkpoint folder (config.json + model.safetensors [+ tokenizer.json])")
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = os.path.join(args.src, "config.json")
    st_path = os.path.join(args.src, "model.safetensors")
    if not (os.path.isfile(cfg_path) and os.path.isfile(st_path)):
        sys.exit(f"ERROR: {args.src} must contain config.json + model.safetensors "
                 "(for v2, run validation/convert_v2_nemo.py first).")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "parakeet_tdt":
        sys.exit(f"ERROR: model_type={cfg.get('model_type')!r}, expected 'parakeet_tdt'")

    variant = check_config(cfg)
    vocab = cfg["vocab_size"]
    reader = Safetensors([st_path])
    n_layers = cfg["encoder_config"]["num_hidden_layers"]

    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "..", "..", "Resources", "Weights",
        f"weights_parakeet_tdt_0.6b_{variant}_{args.quant}"))
    os.makedirs(out, exist_ok=True)
    print(f"[src]    {args.src}\n[variant] {variant} (vocab {vocab}, blank {vocab - 1})"
          f"\n[out]    {out}\n[quant]  {args.quant}\n")

    ex = Exporter(out, args.quant)
    export_frontend(ex)
    export_subsampling(reader, ex)
    for i in range(n_layers):
        export_encoder_layer(reader, ex, i)
        print(f"\r[layers] {i + 1}/{n_layers}", end="", flush=True)
    print()
    export_decoder_joint(reader, ex, vocab)
    ex.save_manifest()
    export_tokenizer(args.src, out, vocab)

    # leftover-tensor audit: everything in the checkpoint should be consumed or knowingly skipped
    consumed_prefixes = ("encoder.subsampling.", "encoder.layers.", "decoder.", "encoder_projector.", "joint.head.")
    skipped = [k for k in reader.keys() if not k.startswith(consumed_prefixes)]
    if skipped:
        print(f"[audit]  {len(skipped)} unconsumed tensors (expect none): {skipped[:8]}")
    else:
        print("[audit]  all checkpoint tensors consumed")

    print(f"\nDone - {ex.bytes_written / 1024 / 1024:.0f} MB, {len(ex.manifest)} manifest entries ->\n  {out}")
    if args.quant != "fp16":
        print(f"Worst per-element int8 reconstruction error: {ex.worst[0]:.6f} ({ex.worst[1]})")


if __name__ == "__main__":
    main()
