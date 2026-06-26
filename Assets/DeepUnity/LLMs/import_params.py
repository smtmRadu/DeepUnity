#!/usr/bin/env python3
"""
DeepUnity LLM weight exporter - turns a HuggingFace checkpoint (hub id or local folder)
into the binary params folder the DeepUnity inference pipeline streams at runtime.

USAGE
    python import_params.py <model> [--quant fp16|int8|int4] [--out DIR] [--arch gemma3|qwen3_5]

    python import_params.py google/gemma-3-270m-it
    python import_params.py Qwen/Qwen3.5-0.8B --quant int8
    python import_params.py D:/checkpoints/my-finetuned-qwen --quant int8

SUPPORTED MODELS                                                fp16   int8   int4
    gemma3    google/gemma-3-270m-it (and 270m mirrors)          OK     OK     OK    (*)
    qwen3_5   Qwen/Qwen3.5-0.8B, unsloth/Qwen3.5-0.8B            OK     OK     OK    (*)

    (*)  int4 (GGUF Q4_0, groups of 32) trades quality for ~quarter the VRAM/disk and was
         measured LOSSY on these small models (story quality visibly drops, and it decodes
         slower than int8) - prefer int8; see LLMQuant docs in LLM.cs for the measured numbers.

    Other sizes of the same architectures export fine, but the Unity-side configs
    (Gemma3Config.cs / Qwen3_5Config.cs) are compile-time constants - the script compares
    the checkpoint dims against what Unity expects and warns loudly on mismatch.

QUANTIZATION (weight-only - activations/KV stay fp32 at runtime; formulas are the
DeepUnity convention documented on LLMQuant in LLM.cs)
    fp16   packed 2-per-uint reference format
    int8   symmetric, ONE fp16 scale per OUTPUT ROW: scale_r = max|w_r| / 127   (~lossless)
    int4   GGUF Q4_0-style, GROUPS OF 32 per row: d = w[argmax|w|] / -8, nibbles store q+8
    Norm gammas, conv1d, dt_bias, A_log and in_proj_a/b always stay fp16 (tiny or sensitive).

OUTPUT LAYOUT (the unified convention every DeepUnity LLM loader reads; the C# resolves
this Resources location first and falls back to the legacy Assets/DeepUnity/LLMs/ one)
    Assets/Resources/DeepUnity/LLMs/<Arch>/weights_<model>_<size>_<quant>/
        e.g.  .../LLMs/Qwen3_5/weights_qwen3.5_0.8B_int8/   .../LLMs/Gemma3/weights_gemma3_270M_fp16/
        norm.bin                                  final RMSNorm gamma (fp16)
        embed_tokens/part_{0..15}[.intN].bin      tied embedding / lm_head, 16 row-aligned shards
        embed_tokens/scales.bin                   (quantized modes only)
        layer_{i}/<tensor>.bin                    fp16 tensors
        layer_{i}/<tensor>.intN.bin + .scales.bin quantized matmul weights

Requires: numpy; huggingface_hub (only for hub ids); tqdm (optional, nicer progress).
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # plain fallback so tqdm stays optional
    def tqdm(it, **kw):
        return it

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED_CHUNKS = 16
G4 = 32  # int4 group size

# What the Unity-side *Config.cs constants currently expect (config.json key -> value).
EXPECTED_DIMS = {
    "gemma3": {"hidden_size": 640, "num_hidden_layers": 18, "num_attention_heads": 4,
               "num_key_value_heads": 1, "head_dim": 256, "intermediate_size": 2048,
               "vocab_size": 262144},
    "qwen3_5": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8,
                "num_key_value_heads": 2, "head_dim": 256, "intermediate_size": 3584,
                "vocab_size": 248320},
}
ARCH_FOLDER = {"gemma3": "Gemma3", "qwen3_5": "Qwen3_5"}
# Human model name + size designation baked into the self-describing output folder name
# (weights_<model>_<size>_<quant>, e.g. weights_qwen3.5_0.8B_int8). Add a row per new size.
MODEL_LABEL = {"gemma3": ("gemma3", "270M"), "qwen3_5": ("qwen3.5", "0.8B")}


# ----------------------------------------------------------------------------- model source
class Safetensors:
    """Multi-shard safetensors reader: header parsing + per-key fp16 loads, no torch."""

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
                self.map[key] = (path, meta["dtype"], tuple(meta["shape"]),
                                 meta["data_offsets"], base)

    def __contains__(self, key):
        return key in self.map

    def keys(self):
        return self.map.keys()

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


def resolve_model(model):
    """model = local folder or HF hub id -> (config dict, Safetensors reader)."""
    if os.path.isdir(model):
        print(f"[source] local folder: {model}")
        cfg_path = os.path.join(model, "config.json")
        if not os.path.isfile(cfg_path):
            sys.exit(f"ERROR: {model} has no config.json")
        files = [os.path.join(model, f) for f in sorted(os.listdir(model)) if f.endswith(".safetensors")]
        if not files:
            sys.exit(f"ERROR: no .safetensors files in {model}")
    else:
        print(f"[source] HuggingFace hub: {model}  (downloads cache under ~/.cache/huggingface)")
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(model, "config.json")
        try:
            files = [hf_hub_download(model, "model.safetensors")]
        except Exception:
            idx_path = hf_hub_download(model, "model.safetensors.index.json")
            with open(idx_path, encoding="utf-8") as f:
                shard_names = sorted(set(json.load(f)["weight_map"].values()))
            print(f"[source] sharded checkpoint: {len(shard_names)} files")
            files = [hf_hub_download(model, name) for name in shard_names]

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)  # multimodal repos nest the text model's config
    print(f"[source] {len(files)} safetensors shard(s) ready")
    return cfg, Safetensors(files)


def detect_arch(cfg, reader, override):
    if override:
        return override
    mt = (cfg.get("model_type") or "").lower()
    archs = " ".join(cfg.get("architectures") or []).lower()
    if "gemma3" in mt or "gemma3" in archs:
        return "gemma3"
    if "qwen3" in mt or "qwen3" in archs:
        return "qwen3_5"
    sys.exit(f"ERROR: unrecognized architecture (model_type='{mt}'). Pass --arch gemma3|qwen3_5.")


def find_prefix(reader):
    """Locate the transformer root ('model.' / 'model.language_model.' / ...)."""
    for key in reader.keys():
        if key.endswith("embed_tokens.weight"):
            return key[: -len("embed_tokens.weight")]
    sys.exit("ERROR: no embed_tokens.weight in checkpoint - is this a causal LM?")


def check_dims(arch, cfg):
    bad = []
    for key, expected in EXPECTED_DIMS[arch].items():
        got = cfg.get(key)
        if got is not None and got != expected:
            bad.append(f"  {key}: checkpoint={got}, Unity expects {expected}")
    if bad:
        print("\n" + "!" * 88)
        print(f"WARNING: checkpoint dims differ from the constants in {ARCH_FOLDER[arch]}Config.cs —")
        print("the export will finish, but Unity can NOT run it until those constants are updated:")
        print("\n".join(bad))
        print("!" * 88 + "\n")


# ----------------------------------------------------------------------------- quantizers
def quantize_int8(w):  # fp32 [rows, cols] -> int8 + fp16 per-row scales + (max_err, mean_err)
    s = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-8)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    rec = q.astype(np.float32) * s[:, None].astype(np.float32)
    return q, s.astype(np.float16), (float(np.abs(rec - w).max()), float(np.abs(rec - w).mean()))


def quantize_int4(w):  # fp32 [rows, cols] (cols%32==0) -> packed nibbles + fp16 group scales + err
    rows, cols = w.shape
    gr = w.reshape(rows, cols // G4, G4)
    idx = np.abs(gr).argmax(axis=-1)
    maxv = np.take_along_axis(gr, idx[..., None], axis=-1)[..., 0]
    d = maxv / -8.0
    d = np.where(np.abs(d) < 1e-10, 1e-10, d).astype(np.float32)
    q = np.clip(np.rint(gr / d[..., None]) + 8.0, 0, 15).astype(np.uint8)
    rec = (q.astype(np.float32) - 8.0) * d[..., None]
    err = (float(np.abs(rec - gr).max()), float(np.abs(rec - gr).mean()))
    qf = q.reshape(rows, cols)
    packed = (qf[:, 0::2] | (qf[:, 1::2] << 4)).astype(np.uint8)  # low nibble = even column
    return packed, d.astype(np.float16), err


class Exporter:
    def __init__(self, out_dir, quant):
        self.out, self.quant = out_dir, quant
        self.bytes_written = 0
        self.worst = (0.0, "-")

    def _track(self, path, err=None, name=None):
        self.bytes_written += os.path.getsize(path)
        if err and err[0] > self.worst[0]:
            self.worst = (err[0], name)

    def fp16(self, rel, arr):  # always-fp16 tensors (norms etc.) regardless of quant mode
        path = os.path.join(self.out, rel + ".bin")
        np.ascontiguousarray(arr.astype(np.float16, copy=False)).tofile(path)
        self._track(path)

    def weight(self, rel, w):  # big matmul weight [rows, cols] - quantized per --quant
        w = w.astype(np.float32)
        if self.quant == "fp16":
            self.fp16(rel, w)
            return
        if self.quant == "int8":
            q, s, err = quantize_int8(w)
            ext = ".int8.bin"
        else:
            assert w.shape[1] % G4 == 0, f"{rel}: cols {w.shape[1]} not divisible by {G4}"
            q, s, err = quantize_int4(w)
            ext = ".int4.bin"
        qp, sp = os.path.join(self.out, rel + ext), os.path.join(self.out, rel + ".scales.bin")
        q.tofile(qp)
        s.tofile(sp)
        self._track(qp, err, rel)
        self._track(sp)

    # tied embedding/lm_head [vocab, hidden] -> 16 row-aligned shards (+ one scales file)
    def embedding(self, embed):
        vocab, hidden = embed.shape
        assert vocab % EMBED_CHUNKS == 0, f"vocab {vocab} not divisible by {EMBED_CHUNKS}"
        rows = vocab // EMBED_CHUNKS
        d = os.path.join(self.out, "embed_tokens")
        os.makedirs(d, exist_ok=True)
        scales = None
        for k in tqdm(range(EMBED_CHUNKS), desc="embed_tokens", unit="shard"):
            w = embed[k * rows:(k + 1) * rows].astype(np.float32)
            if self.quant == "fp16":
                path = os.path.join(d, f"part_{k}.bin")
                w.astype(np.float16).tofile(path)
                self._track(path)
                continue
            q, s, err = (quantize_int8(w) if self.quant == "int8" else quantize_int4(w))
            if scales is None:
                scales = np.empty((EMBED_CHUNKS,) + s.shape, dtype=np.float16)
            scales[k] = s
            path = os.path.join(d, f"part_{k}.int8.bin" if self.quant == "int8" else f"part_{k}.int4.bin")
            q.tofile(path)
            self._track(path, err, "embed_tokens")
        if scales is not None:
            path = os.path.join(d, "scales.bin")
            scales.tofile(path)
            self._track(path)


# ----------------------------------------------------------------------------- arch exports
def export_qwen3_5(reader, cfg, ex):
    pf = find_prefix(reader)  # 'model.language_model.' on the official checkpoints
    layer_types = cfg.get("layer_types") or (["linear_attention"] * 3 + ["full_attention"]) * 6
    n = cfg.get("num_hidden_layers", len(layer_types))

    ex.fp16("norm", reader.load_fp16(pf + "norm.weight"))
    ex.embedding(reader.load_fp16(pf + "embed_tokens.weight"))

    for i in tqdm(range(n), desc="layers", unit="layer"):
        lp = f"layer_{i}"
        os.makedirs(os.path.join(ex.out, lp), exist_ok=True)
        kp = f"{pf}layers.{i}."
        ex.fp16(f"{lp}/input_layernorm", reader.load_fp16(kp + "input_layernorm.weight"))
        ex.fp16(f"{lp}/post_attention_layernorm", reader.load_fp16(kp + "post_attention_layernorm.weight"))
        for src, dst in [("mlp.gate_proj.weight", "mlp_gate_proj"),
                         ("mlp.up_proj.weight", "mlp_up_proj"),
                         ("mlp.down_proj.weight", "mlp_down_proj")]:
            ex.weight(f"{lp}/{dst}", reader.load_fp16(kp + src))

        if layer_types[i] == "full_attention":
            for src, dst in [("self_attn.q_proj.weight", "self_attn_q_proj"),
                             ("self_attn.k_proj.weight", "self_attn_k_proj"),
                             ("self_attn.v_proj.weight", "self_attn_v_proj"),
                             ("self_attn.o_proj.weight", "self_attn_o_proj")]:
                ex.weight(f"{lp}/{dst}", reader.load_fp16(kp + src))
            ex.fp16(f"{lp}/self_attn_q_norm", reader.load_fp16(kp + "self_attn.q_norm.weight"))
            ex.fp16(f"{lp}/self_attn_k_norm", reader.load_fp16(kp + "self_attn.k_norm.weight"))
        else:
            ex.weight(f"{lp}/linear_in_proj_qkv", reader.load_fp16(kp + "linear_attn.in_proj_qkv.weight"))
            ex.weight(f"{lp}/linear_in_proj_z", reader.load_fp16(kp + "linear_attn.in_proj_z.weight"))
            ex.weight(f"{lp}/linear_out_proj", reader.load_fp16(kp + "linear_attn.out_proj.weight"))
            # small / exp()-sensitive DeltaNet tensors stay fp16 in every mode
            ex.fp16(f"{lp}/linear_in_proj_a", reader.load_fp16(kp + "linear_attn.in_proj_a.weight"))
            ex.fp16(f"{lp}/linear_in_proj_b", reader.load_fp16(kp + "linear_attn.in_proj_b.weight"))
            ex.fp16(f"{lp}/linear_dt_bias", reader.load_fp16(kp + "linear_attn.dt_bias"))
            ex.fp16(f"{lp}/linear_A_log", reader.load_fp16(kp + "linear_attn.A_log"))
            ex.fp16(f"{lp}/linear_norm", reader.load_fp16(kp + "linear_attn.norm.weight"))
            cv = reader.load_fp16(kp + "linear_attn.conv1d.weight")  # [conv_dim, 1, k] -> squeeze
            ex.fp16(f"{lp}/linear_conv1d", cv.reshape(cv.shape[0], cv.shape[-1]))


def export_gemma3(reader, cfg, ex):
    pf = find_prefix(reader)  # 'model.' on google/gemma-3-270m-it
    n = cfg.get("num_hidden_layers", 18)

    ex.fp16("norm", reader.load_fp16(pf + "norm.weight"))
    ex.embedding(reader.load_fp16(pf + "embed_tokens.weight"))  # tied lm_head

    for i in tqdm(range(n), desc="layers", unit="layer"):
        lp = f"layer_{i}"
        os.makedirs(os.path.join(ex.out, lp), exist_ok=True)
        kp = f"{pf}layers.{i}."
        for src, dst in [("self_attn.q_proj.weight", "self_attn_q_proj"),
                         ("self_attn.k_proj.weight", "self_attn_k_proj"),
                         ("self_attn.v_proj.weight", "self_attn_v_proj"),
                         ("self_attn.o_proj.weight", "self_attn_o_proj")]:
            ex.weight(f"{lp}/{dst}", reader.load_fp16(kp + src))
        for src, dst in [("self_attn.q_norm.weight", "self_attn_q_norm"),
                         ("self_attn.k_norm.weight", "self_attn_k_norm"),
                         ("input_layernorm.weight", "input_layernorm"),
                         ("post_attention_layernorm.weight", "post_attention_layernorm"),
                         ("pre_feedforward_layernorm.weight", "pre_feedforward_layernorm"),
                         ("post_feedforward_layernorm.weight", "post_feedforward_layernorm")]:
            ex.fp16(f"{lp}/{dst}", reader.load_fp16(kp + src))
        for src, dst in [("mlp.gate_proj.weight", "mlp_gate_proj"),
                         ("mlp.up_proj.weight", "mlp_up_proj"),
                         ("mlp.down_proj.weight", "mlp_down_proj")]:
            ex.weight(f"{lp}/{dst}", reader.load_fp16(kp + src))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Export a HF checkpoint into DeepUnity params (see module docstring).")
    ap.add_argument("model", help="HF hub id (e.g. Qwen/Qwen3.5-0.8B) or local checkpoint folder")
    ap.add_argument("--quant", choices=["fp16", "int8", "int4"], default="fp16")
    ap.add_argument("--arch", choices=["gemma3", "qwen3_5"], default=None,
                    help="override architecture auto-detection")
    ap.add_argument("--out", default=None, help="override the output folder")
    args = ap.parse_args()

    cfg, reader = resolve_model(args.model)
    arch = detect_arch(cfg, reader, args.arch)
    print(f"[arch]   {arch}  (model_type='{cfg.get('model_type')}')")

    check_dims(arch, cfg)

    mdl, sz = MODEL_LABEL[arch]
    folder = f"weights_{mdl}_{sz}_{args.quant}"   # e.g. weights_qwen3.5_0.8B_int8
    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "Resources", "DeepUnity", "LLMs", ARCH_FOLDER[arch], folder))
    os.makedirs(out, exist_ok=True)
    print(f"[out]    {out}\n[quant]  {args.quant}\n")

    ex = Exporter(out, args.quant)
    (export_gemma3 if arch == "gemma3" else export_qwen3_5)(reader, cfg, ex)

    print(f"\nDone - {ex.bytes_written / 1024 / 1024:.0f} MB written to:\n  {out}")
    print("Layout: norm.bin + embed_tokens/part_0..15 + layer_i/<tensor> "
          + ("(.bin fp16)" if args.quant == "fp16" else f"(.{args.quant}.bin + .scales.bin, fp16 passthrough for norms etc.)"))
    if args.quant != "fp16":
        print(f"Worst per-element reconstruction error: {ex.worst[0]:.6f} ({ex.worst[1]})")
    print("\nUse it in Unity (the loaders resolve this Resources folder automatically):")
    q = {"fp16": "LLMQuant.FP16", "int8": "LLMQuant.INT8", "int4": "LLMQuant.INT4"}[args.quant]
    if arch == "gemma3":
        print(f"  var llm = new Gemma3ForCausalLM({q});")
    else:
        print(f"  var llm = new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, {q});")


if __name__ == "__main__":
    main()
