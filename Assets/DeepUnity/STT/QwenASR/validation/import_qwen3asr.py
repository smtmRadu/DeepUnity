#!/usr/bin/env python3
"""
Qwen3-ASR -> DeepUnity weight exporter (D0, standalone — deliberately does NOT import or edit
Assets/DeepUnity/LLM/import_params.py, which another workstream owns; packing/quant conventions
are mirrored from it 1:1).

USAGE
    python import_qwen3asr.py <checkpoint_dir> [--quant fp16|int8] [--out DIR]

    python import_qwen3asr.py C:/dev/_model_staging/qwen3asr/Qwen3-ASR-0.6B-hf
    python import_qwen3asr.py C:/dev/_model_staging/qwen3asr/Qwen3-ASR-1.7B-hf --quant int8

INPUT   a local clone of Qwen/Qwen3-ASR-{0.6B,1.7B}-hf (config.json + model.safetensors +
        tokenizer.json). Size is auto-detected from audio_config.d_model (896 -> 0.6b, 1024 -> 1.7b).

OUTPUT  Assets/Resources/Weights/weights_qwen3asr_<size>_<quant>/
        manifest.tsv lines `name\tfile\tdtype\tnumel\tshape-csv` + one .bin per tensor —
        the exact contract ChatterboxWeights.cs already streams (f16 packed 2-per-uint on GPU,
        q8 4-per-uint + sibling `<name>.scales` f16 per-output-row scales, i32 raw).
        Also emits tokenizer/vocab.txt + merges.txt + specials.tsv for the C# BPE side.

QUANT   fp16 (reference) | int8 (symmetric per-output-row, scale = max|w_r|/127 — the repo
        convention). int8 quantizes ONLY the big matmul weights (encoder attn/ffn + conv_out,
        projector, decoder q/k/v/o + gate/up/down). Norms, biases, conv2d kernels, pos_emb,
        mel_filters and the tied embedding/lm_head ALWAYS stay fp16 (quantizing them collapses
        small models — see LLM/OPTIMIZATIONS.md).

Synthesized tensors (not in the checkpoint):
    frontend/mel_filters [201,128]  slaney mel filterbank (vendored transformers math, so the
                                    C# frontend never reimplements slaney)
    enc/pos_emb          [13,d]     Whisper-style [sin|cos] sinusoids (SinusoidsPositionEmbedding)

Requires: numpy only.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED_CHUNKS = 16  # 151936 % 16 == 0 -> 9496-row shards


# ----------------------------------------------------------------------------- safetensors
class Safetensors:
    """Single/multi-shard safetensors reader: header parse + per-key fp16 loads, no torch."""

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

    def load_fp16(self, key):
        if key not in self.map:
            raise KeyError(f"safetensors: missing tensor '{key}'")
        path, dtype, shape, (a, b), base = self.map[key]
        raw = np.memmap(path, dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
        buf = raw.tobytes()
        if dtype == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
            return ((u16.astype(np.uint32) << 16).view(np.float32)).astype(np.float16)
        if dtype == "F16":
            return np.frombuffer(buf, dtype=np.float16).reshape(shape).copy()
        if dtype == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(shape).astype(np.float16)
        raise ValueError(f"unsupported dtype {dtype} for {key}")


# ----------------------------------------------------------------------------- synthesized tensors
def slaney_mel_filter_bank(num_frequency_bins=201, num_mel_filters=128, min_frequency=0.0,
                           max_frequency=8000.0, sampling_rate=16000):
    """Vendored transformers.audio_utils.mel_filter_bank(norm='slaney', mel_scale='slaney')."""
    def hz_to_mel(freq):
        freq = np.asarray(freq, dtype=np.float64)
        mels = 3.0 * freq / 200.0
        log_region = freq >= 1000.0
        mels = np.where(log_region, 15.0 + np.log(np.maximum(freq, 1e-10) / 1000.0) * (27.0 / np.log(6.4)), mels)
        return mels

    def mel_to_hz(mels):
        mels = np.asarray(mels, dtype=np.float64)
        freq = 200.0 * mels / 3.0
        log_region = mels >= 15.0
        freq = np.where(log_region, 1000.0 * np.exp(np.log(6.4) / 27.0 * (mels - 15.0)), freq)
        return freq

    mel_freqs = np.linspace(hz_to_mel(min_frequency), hz_to_mel(max_frequency), num_mel_filters + 2)
    filter_freqs = mel_to_hz(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))

    enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    fb *= np.expand_dims(enorm, 0)
    if (fb.max(axis=0) == 0.0).any():
        print("WARNING: at least one mel filter is all-zero (unexpected for 201x128 @16k)")
    return fb.astype(np.float32)  # [201, 128]


def sinusoids_pos_emb(length=13, channels=896, max_timescale=10000):
    """SinusoidsPositionEmbedding.compute_default_singular_positional_embedding, numpy twin."""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2, dtype=np.float32))
    scaled_time = np.arange(length, dtype=np.float32)[:, None] * inv_timescales[None, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)  # [13, channels]


# ----------------------------------------------------------------------------- exporter
def quantize_int8(w):  # fp32 [rows, cols] -> int8 + fp16 per-row scales + (max_err, mean_err)
    s = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-8)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    rec = q.astype(np.float32) * s[:, None].astype(np.float32)
    return q, s.astype(np.float16), (float(np.abs(rec - w).max()), float(np.abs(rec - w).mean()))


class Exporter:
    """Manifest exporter, ChatterboxWeights contract (name\tfile\tdtype\tnumel\tshape-csv)."""

    def __init__(self, out_dir, quant):
        self.out, self.quant = out_dir, quant
        self.manifest = {}
        self.bytes_written = 0
        self.worst = (0.0, "-")

    def _reg(self, name, file, dtype, shape):
        self.manifest[name] = {"file": file, "dtype": dtype, "shape": [int(d) for d in shape]}
        self.bytes_written += os.path.getsize(os.path.join(self.out, file))

    def _path(self, rel):
        p = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def f16(self, name, arr):
        arr = np.asarray(arr)
        rel = name + ".bin"
        np.ascontiguousarray(arr.astype(np.float16, copy=False)).tofile(self._path(rel))
        self._reg(name, rel, "f16", arr.shape)

    def mat(self, name, arr):  # big matmul weight [rows, cols]
        arr = np.asarray(arr, dtype=np.float32)
        if self.quant == "fp16":
            self.f16(name, arr)
            return
        q, s, err = quantize_int8(arr)
        q.tofile(self._path(name + ".int8.bin"))
        s.tofile(self._path(name + ".scales.bin"))
        self._reg(name, name + ".int8.bin", "q8", arr.shape)
        self._reg(name + ".scales", name + ".scales.bin", "f16", (arr.shape[0],))
        if err[0] > self.worst[0]:
            self.worst = (err[0], name)

    def embedding(self, name, embed):  # tied embedding/lm_head -> 16 fp16 row shards, ALWAYS fp16
        vocab, hidden = embed.shape
        assert vocab % EMBED_CHUNKS == 0, f"vocab {vocab} not divisible by {EMBED_CHUNKS}"
        rows = vocab // EMBED_CHUNKS
        for k in range(EMBED_CHUNKS):
            self.f16(f"{name}/part_{k}", embed[k * rows:(k + 1) * rows])

    def save_manifest(self):
        with open(os.path.join(self.out, "manifest.tsv"), "w", encoding="utf-8", newline="\n") as f:
            for name, m in self.manifest.items():
                numel = 1
                for d in m["shape"]:
                    numel *= d
                f.write(f"{name}\t{m['file']}\t{m['dtype']}\t{numel}\t{','.join(map(str, m['shape']))}\n")


# ----------------------------------------------------------------------------- export passes
def export_encoder(r, ex, cfg_a):
    n_layers, d = cfg_a["encoder_layers"], cfg_a["d_model"]
    ex.f16("frontend/mel_filters", slaney_mel_filter_bank(
        num_frequency_bins=1 + 400 // 2, num_mel_filters=cfg_a["num_mel_bins"]))
    ex.f16("enc/pos_emb", sinusoids_pos_emb(cfg_a["max_position_embeddings"], d))

    for c in ("conv2d1", "conv2d2", "conv2d3"):  # conv kernels stay fp16 in every mode
        ex.f16(f"enc/{c}.w", r.load_fp16(f"model.audio_tower.{c}.weight"))
        ex.f16(f"enc/{c}.b", r.load_fp16(f"model.audio_tower.{c}.bias"))
    ex.mat("enc/conv_out.w", r.load_fp16("model.audio_tower.conv_out.weight"))  # [d,7680], no bias

    for i in range(n_layers):
        kp, lp = f"model.audio_tower.layers.{i}.", f"enc/layer_{i}/"
        ex.f16(lp + "ln1.w", r.load_fp16(kp + "self_attn_layer_norm.weight"))
        ex.f16(lp + "ln1.b", r.load_fp16(kp + "self_attn_layer_norm.bias"))
        for src, dst in (("q_proj", "attn_q"), ("k_proj", "attn_k"),
                         ("v_proj", "attn_v"), ("out_proj", "attn_out")):
            ex.mat(lp + dst + ".w", r.load_fp16(kp + f"self_attn.{src}.weight"))
            ex.f16(lp + dst + ".b", r.load_fp16(kp + f"self_attn.{src}.bias"))
        ex.f16(lp + "ln2.w", r.load_fp16(kp + "final_layer_norm.weight"))
        ex.f16(lp + "ln2.b", r.load_fp16(kp + "final_layer_norm.bias"))
        ex.mat(lp + "fc1.w", r.load_fp16(kp + "fc1.weight"))
        ex.f16(lp + "fc1.b", r.load_fp16(kp + "fc1.bias"))
        ex.mat(lp + "fc2.w", r.load_fp16(kp + "fc2.weight"))
        ex.f16(lp + "fc2.b", r.load_fp16(kp + "fc2.bias"))
    ex.f16("enc/ln_post.w", r.load_fp16("model.audio_tower.ln_post.weight"))
    ex.f16("enc/ln_post.b", r.load_fp16("model.audio_tower.ln_post.bias"))


def export_projector(r, ex):
    for j in (1, 2):
        ex.mat(f"proj/linear_{j}.w", r.load_fp16(f"model.multi_modal_projector.linear_{j}.weight"))
        ex.f16(f"proj/linear_{j}.b", r.load_fp16(f"model.multi_modal_projector.linear_{j}.bias"))


def export_decoder(r, ex, cfg_t):
    n = cfg_t["num_hidden_layers"]
    ex.embedding("dec/embed_tokens", r.load_fp16("model.language_model.embed_tokens.weight"))
    ex.f16("dec/norm", r.load_fp16("model.language_model.norm.weight"))
    for i in range(n):
        kp, lp = f"model.language_model.layers.{i}.", f"dec/layer_{i}/"
        ex.f16(lp + "input_ln", r.load_fp16(kp + "input_layernorm.weight"))
        ex.f16(lp + "post_attn_ln", r.load_fp16(kp + "post_attention_layernorm.weight"))
        ex.f16(lp + "q_norm", r.load_fp16(kp + "self_attn.q_norm.weight"))
        ex.f16(lp + "k_norm", r.load_fp16(kp + "self_attn.k_norm.weight"))
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ex.mat(lp + p, r.load_fp16(kp + f"self_attn.{p}.weight"))
        for p in ("gate_proj", "up_proj", "down_proj"):
            ex.mat(lp + "mlp_" + p.split("_")[0], r.load_fp16(kp + f"mlp.{p}.weight"))
        print(f"  dec layer {i + 1}/{n}", end="\r")
    print()


def export_tokenizer(ckpt_dir, out_dir):
    """tokenizer.json -> vocab.txt (line i = token for id i) + merges.txt + specials.tsv."""
    with open(os.path.join(ckpt_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok["model"]["vocab"]                      # token -> id (byte-level BPE strings)
    added = {t["content"]: t["id"] for t in tok["added_tokens"]}
    merges = tok["model"]["merges"]                    # list of "a b" or [a, b]
    size = max(max(vocab.values()), max(added.values())) + 1
    table = [""] * size
    for s, i in vocab.items():
        table[i] = s
    for s, i in added.items():
        table[i] = s
    d = os.path.join(out_dir, "tokenizer")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        for s in table:
            f.write(s + "\n")
    with open(os.path.join(d, "merges.txt"), "w", encoding="utf-8", newline="\n") as f:
        for m in merges:
            f.write((m if isinstance(m, str) else " ".join(m)) + "\n")
    with open(os.path.join(d, "specials.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for s, i in sorted(added.items(), key=lambda kv: kv[1]):
            f.write(f"{i}\t{s}\n")
    print(f"[tok]    vocab {len(vocab)} + {len(added)} added -> {d}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Export Qwen3-ASR-{0.6B,1.7B}-hf into DeepUnity params.")
    ap.add_argument("checkpoint", help="local folder with config.json + model.safetensors (+tokenizer.json)")
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--out", default=None, help="override the output folder")
    args = ap.parse_args()

    ckpt = args.checkpoint
    with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "qwen3_asr":
        sys.exit(f"ERROR: not a qwen3_asr checkpoint (model_type={cfg.get('model_type')})")
    cfg_a, cfg_t = cfg["audio_config"], cfg["text_config"]

    size = {896: "0.6b", 1024: "1.7b"}.get(cfg_a["d_model"])
    if size is None:
        sys.exit(f"ERROR: unknown audio d_model={cfg_a['d_model']} (expected 896 or 1024)")
    # sanity: dims the Unity config will hardcode
    expect = {"0.6b": (18, 14, 3584, 1024, 3072), "1.7b": (24, 16, 4096, 2048, 6144)}[size]
    got = (cfg_a["encoder_layers"], cfg_a["encoder_attention_heads"], cfg_a["encoder_ffn_dim"],
           cfg_t["hidden_size"], cfg_t["intermediate_size"])
    if got != expect:
        sys.exit(f"ERROR: dims {got} != expected {expect} for {size} — update SPEC.md/Unity config first")
    if cfg_t["vocab_size"] != 151936 or cfg_t["num_hidden_layers"] != 28:
        sys.exit("ERROR: decoder vocab/layers unexpected — re-verify checkpoint")

    files = [os.path.join(ckpt, "model.safetensors")]
    if not os.path.isfile(files[0]):
        idx = os.path.join(ckpt, "model.safetensors.index.json")
        with open(idx, encoding="utf-8") as f:
            files = [os.path.join(ckpt, n) for n in sorted(set(json.load(f)["weight_map"].values()))]
    r = Safetensors(files)

    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "..", "..", "Resources", "Weights",
        f"weights_qwen3asr_{size}_{args.quant}"))
    os.makedirs(out, exist_ok=True)
    print(f"[src]    {ckpt}\n[size]   {size}\n[out]    {out}\n[quant]  {args.quant}\n")

    ex = Exporter(out, args.quant)
    export_encoder(r, ex, cfg_a)
    export_projector(r, ex)
    export_decoder(r, ex, cfg_t)
    ex.save_manifest()
    export_tokenizer(ckpt, out)

    n_ckpt = len(r.map)
    n_out = len([k for k in ex.manifest if not k.endswith(".scales")])
    print(f"\nDone - {ex.bytes_written / 1024 / 1024:.0f} MB, {len(ex.manifest)} manifest entries "
          f"({n_out} tensors) from {n_ckpt} checkpoint tensors (+2 synthesized, embed as 16 shards).")
    if args.quant == "int8":
        print(f"Worst per-element int8 reconstruction error: {ex.worst[0]:.6f} ({ex.worst[1]})")


if __name__ == "__main__":
    main()
