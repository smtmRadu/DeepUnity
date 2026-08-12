#!/usr/bin/env python3
"""
DeepUnity LLM weight exporter - turns a HuggingFace checkpoint (hub id or local folder)
into the binary params folder the DeepUnity inference pipeline streams at runtime.

USAGE
    python import_params.py <model> [--quant fp16|int8|int4] [--out DIR] [--arch gemma3|qwen3_5|chatterbox]

    python import_params.py google/gemma-3-270m-it
    python import_params.py Qwen/Qwen3.5-0.8B --quant int8
    python import_params.py D:/checkpoints/my-finetuned-qwen --quant int8
    python import_params.py ResembleAI/chatterbox-turbo          # TTS -> Resources/Weights/
                                                                 # (fp16 only; conds.pt export needs torch)
    python import_params.py pocket-tts --quant int8              # TTS (DEFAULT NPC TTS) -> Resources/Weights/

SUPPORTED MODELS                                                fp16   int8   int4
    gemma3    google/gemma-3-270m-it (and 270m mirrors)          OK     OK     OK    (*)
    qwen3_5   Qwen/Qwen3.5-0.8B, unsloth/Qwen3.5-0.8B            OK     OK     OK    (*)
    qwen3_5   Qwen/Qwen3.5-2B (size auto-detected by hidden dim) OK     OK     OK    (*)
    minicpm5  openbmb/MiniCPM5-1B (or -SFT; vanilla llama arch)  OK     OK     OK    (*)

    TTS (dedicated export paths in THIS file — no standalone scripts)
    chatterbox  ResembleAI/chatterbox-turbo                      OK     OK     -     (conds.pt needs torch)
    cosyvoice3  FunAudioLLM/Fun-CosyVoice3-0.5B-2512             OK     OK     -     (needs torch; .pt pickles)
    pockettts   kyutai/pocket-tts — the DEFAULT NPC TTS          OK     OK     -     (needs torch + sentencepiece;
                -> weights_pockettts_english_<quant>; merged here from the former
                TTS/PocketTTS/validation/import_pocket_tts.py; --voice/--include-encoder flags)

    Kokoro + the STT models (qwen3asr, parakeet) still export via their workstream
    validation/import_*.py scripts — registered in the pool below as pointers only.

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
    Norm gammas, conv1d, dt_bias, A_log, in_proj_a/b AND the tied embedding/lm_head ALWAYS stay
    fp16 in every mode. Only the transformer-block LINEAR weights (attention q/k/v/o, MLP
    gate/up/down, DeltaNet in/out projections) are quantized — quantizing norms or the lm_head
    poisons every logit and collapses small models (GPTQ/AWQ/QLoRA/llama.cpp all keep them high).

OUTPUT LAYOUT (the unified convention every DeepUnity LLM loader reads; the C# resolves
this Resources location first and falls back to the legacy Assets/DeepUnity/InferenceEngine/LLM/ one)
    Assets/Resources/Weights/weights_<model>_<size>_<quant>/
        e.g.  .../LLM/Qwen3_5/weights_qwen3.5_0.8B_int8/   .../LLM/Gemma3/weights_gemma3_270M_fp16/
        norm.bin                                  final RMSNorm gamma (fp16)
        embed_tokens/part_{0..15}.bin             tied embedding / lm_head, 16 fp16 shards (ALWAYS fp16)
        layer_{i}/<tensor>.bin                    fp16 tensors
        layer_{i}/<tensor>.intN.bin + .scales.bin quantized matmul weights

Requires: numpy; huggingface_hub (only for hub ids); tqdm (optional, nicer progress).
"""
import argparse
import json
import os
import shutil
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
# qwen3_5 lists only the dims shared by every exported size; hidden/intermediate are
# size-dependent and validated via QWEN3_5_SIZES below.
EXPECTED_DIMS = {
    "gemma3": {"hidden_size": 640, "num_hidden_layers": 18, "num_attention_heads": 4,
               "num_key_value_heads": 1, "head_dim": 256, "intermediate_size": 2048,
               "vocab_size": 262144},
    "qwen3_5": {"num_hidden_layers": 24, "num_attention_heads": 8,
                "num_key_value_heads": 2, "head_dim": 256, "vocab_size": 248320},
    "minicpm5": {"hidden_size": 1536, "num_hidden_layers": 24, "num_attention_heads": 16,
                 "num_key_value_heads": 2, "head_dim": 128, "intermediate_size": 4608,
                 "vocab_size": 130560},
}
# Qwen3.5 exported sizes, keyed by the checkpoint's hidden_size:
# hidden_size -> (folder size label, expected intermediate_size).
# Must mirror Qwen3_5Config.ApplySize in Unity. Add a row per new size.
QWEN3_5_SIZES = {1024: ("0.8B", 3584), 2048: ("2B", 6144)}
ARCH_FOLDER = {"gemma3": "Gemma3", "qwen3_5": "Qwen3_5", "minicpm5": "MiniCPM5"}
# Human model name + default size designation baked into the self-describing output folder
# name (weights_<model>_<size>_<quant>). qwen3_5's size is resolved per-checkpoint by
# resolve_size(); the entry here is just the model-name half.
MODEL_LABEL = {"gemma3": ("gemma3", "270M"), "qwen3_5": ("qwen3.5", "0.8B"),
               "minicpm5": ("minicpm5", "1B")}


def resolve_size(arch, cfg):
    """Folder size label for the checkpoint; validates size-dependent dims where they vary."""
    if arch == "qwen3_5":
        hs = cfg.get("hidden_size")
        if hs not in QWEN3_5_SIZES:
            sys.exit(f"ERROR: unsupported Qwen3.5 hidden_size={hs}; known sizes: "
                     + ", ".join(f"{k}->{v[0]}" for k, v in sorted(QWEN3_5_SIZES.items()))
                     + ". Add the variant to QWEN3_5_SIZES here AND Qwen3_5Config.ApplySize in Unity.")
        sz, inter = QWEN3_5_SIZES[hs]
        got = cfg.get("intermediate_size")
        if got is not None and got != inter:
            sys.exit(f"ERROR: Qwen3.5-{sz} expects intermediate_size={inter}, checkpoint has {got}")
        return sz
    return MODEL_LABEL[arch][1]


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
    # MiniCPM5 checkpoints ship plain LlamaForCausalLM configs — the only llama-arch family we
    # export, so map llama -> minicpm5 (check_dims screams if the dims aren't MiniCPM5-1B's).
    if "llama" in mt or "llama" in archs or "minicpm" in mt:
        return "minicpm5"
    sys.exit(f"ERROR: unrecognized architecture (model_type='{mt}'). Pass --arch gemma3|qwen3_5|minicpm5.")


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

    # tied embedding/lm_head [vocab, hidden] -> 16 row-aligned fp16 shards.
    # ALWAYS fp16, in EVERY quant mode: this tensor doubles as the lm_head, so its error lands
    # directly on every one of the 248k-262k output logits and collapses small models. Only the
    # transformer-block LINEAR weights are quantized; embeddings/lm_head + norms stay fp16 — the
    # GPTQ/AWQ/QLoRA/llama.cpp convention (see OPTIMIZATIONS.md "What stays higher-precision").
    def embedding(self, embed, folder="embed_tokens"):
        # folder="lm_head" exports an UNTIED output head with the same 16-shard fp16 layout.
        vocab, hidden = embed.shape
        assert vocab % EMBED_CHUNKS == 0, f"vocab {vocab} not divisible by {EMBED_CHUNKS}"
        rows = vocab // EMBED_CHUNKS
        d = os.path.join(self.out, folder)
        os.makedirs(d, exist_ok=True)
        for k in tqdm(range(EMBED_CHUNKS), desc=folder, unit="shard"):
            w = embed[k * rows:(k + 1) * rows].astype(np.float16)
            path = os.path.join(d, f"part_{k}.bin")
            w.tofile(path)
            self._track(path)


# ----------------------------------------------------------------------------- arch exports
def export_minicpm5(reader, cfg, ex):
    """MiniCPM5 = vanilla llama: q/k/v/o + gate/up/down + 2 norms per layer, UNTIED lm_head."""
    pf = find_prefix(reader)  # 'model.' on the official checkpoints
    n = cfg.get("num_hidden_layers", 24)

    # NORM CONVENTION SHIM: the shared RmsNorm kernels (Gemma3CS.compute) compute
    # x_hat * (1 + gamma) — Gemma's convention. Llama applies x_hat * gamma, so every llama norm
    # gamma is exported as (gamma - 1); the kernel's (1 + g') then reconstructs gamma EXACTLY.
    # (fp16 precision actually improves: values near 1 move near 0, where fp16 is densest.)
    def norm_m1(t):
        return t.astype(np.float32) - 1.0

    ex.fp16("norm", norm_m1(reader.load_fp16(pf + "norm.weight")))
    ex.embedding(reader.load_fp16(pf + "embed_tokens.weight"))
    # Untied output head — same always-fp16 16-shard layout, own folder. lm_head.weight lives at
    # the checkpoint root (outside the transformer prefix).
    ex.embedding(reader.load_fp16("lm_head.weight"), folder="lm_head")

    for i in tqdm(range(n), desc="layers", unit="layer"):
        lp = f"layer_{i}"
        os.makedirs(os.path.join(ex.out, lp), exist_ok=True)
        kp = f"{pf}layers.{i}."
        ex.fp16(f"{lp}/input_layernorm", norm_m1(reader.load_fp16(kp + "input_layernorm.weight")))
        ex.fp16(f"{lp}/post_attention_layernorm", norm_m1(reader.load_fp16(kp + "post_attention_layernorm.weight")))
        for src, dst in [("self_attn.q_proj.weight", "self_attn_q_proj"),
                         ("self_attn.k_proj.weight", "self_attn_k_proj"),
                         ("self_attn.v_proj.weight", "self_attn_v_proj"),
                         ("self_attn.o_proj.weight", "self_attn_o_proj"),
                         ("mlp.gate_proj.weight", "mlp_gate_proj"),
                         ("mlp.up_proj.weight", "mlp_up_proj"),
                         ("mlp.down_proj.weight", "mlp_down_proj")]:
            ex.weight(f"{lp}/{dst}", reader.load_fp16(kp + src))


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


# ----------------------------------------------------------------------------- chatterbox-turbo TTS
# Exports HF ResembleAI/chatterbox-turbo into DeepUnity TTS params (fp16, or int8 for T3's
# four per-layer matmul families — s3gen/embeddings/head/norms stay fp16 in both modes).
# Output: Assets/Resources/Weights/weights_chatterbox_turbo_<quant>/
#   t3/         GPT2-medium T3 (HF Conv1D weights TRANSPOSED to [out,in])
#   s3gen/enc   UpsampleConformerEncoder + flow embeddings/projections
#   s3gen/est   ConditionalDecoder meanflow estimator
#   s3gen/voc   HiFTGenerator vocoder (weight-norm FOLDED: w = g*v/||v||, dims!=0)
#   conds/      baked default voice from conds.pt (requires torch to unpickle)
#   manifest.json  name -> {file, shape, dtype} for the C# loader
# Tensor layouts kept native-torch unless noted: Linear [out,in]; Conv1d [out,in,k];
# ConvTranspose1d [in,out,k]. See Assets/DeepUnity/InferenceEngine/TTS/SPEC.md for the full port spec.
CHATTERBOX_REPO = "ResembleAI/chatterbox-turbo"


class TTSExporter(Exporter):
    """Manifest exporter (name -> file/shape/dtype): f16/i32 always-precise, mat() per --quant."""

    def __init__(self, out_dir, quant="fp16"):
        super().__init__(out_dir, quant)
        self.manifest = {}

    def _reg(self, rel, arr, ext, dtype):
        self.manifest[rel] = {"file": rel + ext, "shape": list(arr.shape), "dtype": dtype}

    def mat(self, rel, arr):
        """Big matmul weight [rows, cols] — int8 under --quant int8 (4-per-uint + one fp16 scale
        per OUTPUT ROW, same scheme as the LLMs), fp16 otherwise. int8 lands as dtype 'q8' plus a
        sibling '<rel>.scales' f16 manifest entry, so the C# loader needs no special casing."""
        arr = np.asarray(arr)
        if self.quant == "fp16":
            self.f16(rel, arr)
            return
        os.makedirs(os.path.dirname(os.path.join(self.out, rel)), exist_ok=True)
        q, s, err = quantize_int8(arr.astype(np.float32))
        qp = os.path.join(self.out, rel + ".int8.bin")
        sp = os.path.join(self.out, rel + ".scales.bin")
        q.tofile(qp)
        s.tofile(sp)
        self._track(qp, err, rel)
        self._track(sp)
        self._reg(rel, arr, ".int8.bin", "q8")
        self.manifest[rel + ".scales"] = {"file": rel + ".scales.bin",
                                          "shape": [int(arr.shape[0])], "dtype": "f16"}

    def f16(self, rel, arr):
        os.makedirs(os.path.dirname(os.path.join(self.out, rel)), exist_ok=True)
        self.fp16(rel, np.asarray(arr))
        self._reg(rel, np.asarray(arr), ".bin", "f16")

    def i32(self, rel, arr):
        os.makedirs(os.path.dirname(os.path.join(self.out, rel)), exist_ok=True)
        path = os.path.join(self.out, rel + ".bin")
        np.ascontiguousarray(np.asarray(arr).astype(np.int32)).tofile(path)
        self._track(path)
        self._reg(rel, np.asarray(arr), ".bin", "i32")

    def save_manifest(self):
        with open(os.path.join(self.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)
        # .tsv twin for the C# loader (no JSON parser needed): name\tfile\tdtype\tnumel\tshape-csv
        with open(os.path.join(self.out, "manifest.tsv"), "w", encoding="utf-8") as f:
            for name, m in self.manifest.items():
                numel = 1
                for d in m["shape"]:
                    numel *= d
                f.write(f"{name}\t{m['file']}\t{m['dtype']}\t{numel}\t{','.join(map(str, m['shape']))}\n")


def fold_weight_norm(reader, key):
    """New-style torch parametrization: weight = g * v / ||v|| (norm over all dims except 0)."""
    g = reader.load_fp16(key + ".parametrizations.weight.original0").astype(np.float32)
    v = reader.load_fp16(key + ".parametrizations.weight.original1").astype(np.float32)
    axes = tuple(range(1, v.ndim))
    n = np.sqrt((v * v).sum(axis=axes, keepdims=True))
    return (g * v / np.maximum(n, 1e-12))


def _lin(ex, reader, dst, src, bias=True):
    ex.f16(dst + ".w", reader.load_fp16(src + ".weight"))
    if bias:
        ex.f16(dst + ".b", reader.load_fp16(src + ".bias"))


def export_chatterbox_t3(reader, ex):
    """t3_turbo_v1.safetensors -> t3/ (GPT2-medium; Conv1D weights transposed to [out,in])."""
    def convT(key):  # HF GPT2 Conv1D stores (in, out) -> transpose to (out, in)
        return reader.load_fp16(key).astype(np.float32).T
    assert reader.map["tfmr.wpe.weight"][2] == (8196, 1024), "unexpected T3 wpe shape"
    assert reader.map["speech_emb.weight"][2] == (6563, 1024), "unexpected speech vocab"

    ex.f16("t3/text_emb", reader.load_fp16("text_emb.weight"))          # [50276,1024]
    ex.f16("t3/speech_emb", reader.load_fp16("speech_emb.weight"))      # [6563,1024]
    ex.f16("t3/wpe", reader.load_fp16("tfmr.wpe.weight"))               # [8196,1024]
    _lin(ex, reader, "t3/spkr_enc", "cond_enc.spkr_enc")                # 256->1024
    _lin(ex, reader, "t3/speech_head", "speech_head")                   # 1024->6563 (untied, bias)
    _lin(ex, reader, "t3/ln_f", "tfmr.ln_f")
    for i in tqdm(range(24), desc="t3 layers", unit="layer"):
        kp, lp = f"tfmr.h.{i}.", f"t3/layer_{i}/"
        ex.f16(lp + "ln_1.w", reader.load_fp16(kp + "ln_1.weight")); ex.f16(lp + "ln_1.b", reader.load_fp16(kp + "ln_1.bias"))
        ex.mat(lp + "qkv.w", convT(kp + "attn.c_attn.weight"))          # [3072,1024]
        ex.f16(lp + "qkv.b", reader.load_fp16(kp + "attn.c_attn.bias"))
        ex.mat(lp + "attn_out.w", convT(kp + "attn.c_proj.weight"))     # [1024,1024]
        ex.f16(lp + "attn_out.b", reader.load_fp16(kp + "attn.c_proj.bias"))
        ex.f16(lp + "ln_2.w", reader.load_fp16(kp + "ln_2.weight")); ex.f16(lp + "ln_2.b", reader.load_fp16(kp + "ln_2.bias"))
        ex.mat(lp + "fc.w", convT(kp + "mlp.c_fc.weight"))              # [4096,1024]
        ex.f16(lp + "fc.b", reader.load_fp16(kp + "mlp.c_fc.bias"))
        ex.mat(lp + "mlp_out.w", convT(kp + "mlp.c_proj.weight"))       # [1024,4096]
        ex.f16(lp + "mlp_out.b", reader.load_fp16(kp + "mlp.c_proj.bias"))
    # skipped: tfmr.wte.weight (deleted after load in reference), text_head.weight (unused at inference)


def _rel_attn(ex, reader, dst, src):
    """RelPositionMultiHeadedAttention tensors."""
    for p in ("linear_q", "linear_k", "linear_v", "linear_out"):
        _lin(ex, reader, f"{dst}.{p}", f"{src}.{p}")
    ex.f16(dst + ".linear_pos.w", reader.load_fp16(src + ".linear_pos.weight"))  # no bias
    ex.f16(dst + ".pos_bias_u", reader.load_fp16(src + ".pos_bias_u"))
    ex.f16(dst + ".pos_bias_v", reader.load_fp16(src + ".pos_bias_v"))


def _enc_layer(ex, reader, dst, src):
    _rel_attn(ex, reader, dst + ".attn", src + ".self_attn")
    _lin(ex, reader, dst + ".ff.w1", src + ".feed_forward.w_1")
    _lin(ex, reader, dst + ".ff.w2", src + ".feed_forward.w_2")
    _lin(ex, reader, dst + ".norm_mha", src + ".norm_mha")
    _lin(ex, reader, dst + ".norm_ff", src + ".norm_ff")


def _btb(ex, reader, dst, src):
    """diffusers BasicTransformerBlock (no cross-attn, plain LN)."""
    _lin(ex, reader, dst + ".norm1", src + ".norm1")
    for p in ("to_q", "to_k", "to_v"):
        ex.f16(f"{dst}.{p}.w", reader.load_fp16(f"{src}.attn1.{p}.weight"))      # [512,256] no bias
    _lin(ex, reader, dst + ".to_out", src + ".attn1.to_out.0")                   # [256,512]+bias
    _lin(ex, reader, dst + ".norm3", src + ".norm3")
    _lin(ex, reader, dst + ".ff_in", src + ".ff.net.0.proj")                     # [1024,256]+bias (GELU proj)
    _lin(ex, reader, dst + ".ff_out", src + ".ff.net.2")                         # [256,1024]+bias


def _causal_resnet(ex, reader, dst, src):
    """CausalResnetBlock1D: block{1,2} = Seq(CausalConv1d k3, LN), time mlp.1, res_conv k1."""
    for b in ("block1", "block2"):
        _lin(ex, reader, f"{dst}.{b}.conv", f"{src}.{b}.block.0")                # [out,in,3]
        _lin(ex, reader, f"{dst}.{b}.ln", f"{src}.{b}.block.2")
    _lin(ex, reader, dst + ".tmlp", src + ".mlp.1")                              # [out,1024]
    _lin(ex, reader, dst + ".res_conv", src + ".res_conv")                       # [out,in,1]


def export_chatterbox_s3gen(reader, ex):
    """s3gen_meanflow.safetensors -> s3gen/{enc,est,voc}. speaker_encoder.*/tokenizer.* SKIPPED (baked voice v1)."""
    # ---- flow-level embeddings/projections
    ex.f16("s3gen/enc/input_embedding", reader.load_fp16("flow.input_embedding.weight"))  # [6561,512]
    _lin(ex, reader, "s3gen/enc/spk_affine", "flow.spk_embed_affine_layer")               # 192->80
    _lin(ex, reader, "s3gen/enc/encoder_proj", "flow.encoder_proj")                       # 512->80
    # ---- UpsampleConformerEncoder
    for stem, hf in (("embed", "flow.encoder.embed"), ("up_embed", "flow.encoder.up_embed")):
        _lin(ex, reader, f"s3gen/enc/{stem}.linear", hf + ".out.0")
        _lin(ex, reader, f"s3gen/enc/{stem}.ln", hf + ".out.1")
    _lin(ex, reader, "s3gen/enc/prelook.conv1", "flow.encoder.pre_lookahead_layer.conv1")  # [512,512,4]
    _lin(ex, reader, "s3gen/enc/prelook.conv2", "flow.encoder.pre_lookahead_layer.conv2")  # [512,512,3]
    for i in range(6):
        _enc_layer(ex, reader, f"s3gen/enc/enc{i}", f"flow.encoder.encoders.{i}")
    _lin(ex, reader, "s3gen/enc/up_layer.conv", "flow.encoder.up_layer.conv")              # [512,512,5]
    for i in range(4):
        _enc_layer(ex, reader, f"s3gen/enc/upenc{i}", f"flow.encoder.up_encoders.{i}")
    _lin(ex, reader, "s3gen/enc/after_norm", "flow.encoder.after_norm")
    # ---- meanflow estimator
    est, E = "flow.decoder.estimator", "s3gen/est"
    _lin(ex, reader, f"{E}/time_mlp1", f"{est}.time_mlp.linear_1")                        # [1024,320]
    _lin(ex, reader, f"{E}/time_mlp2", f"{est}.time_mlp.linear_2")                        # [1024,1024]
    ex.f16(f"{E}/time_mixer.w", reader.load_fp16(f"{est}.time_embed_mixer.weight"))       # [1024,2048] no bias
    blocks = [("down0", f"{est}.down_blocks.0")] + \
             [(f"mid{i}", f"{est}.mid_blocks.{i}") for i in range(12)] + \
             [("up0", f"{est}.up_blocks.0")]
    for dst, src in tqdm(blocks, desc="estimator blocks", unit="block"):
        _causal_resnet(ex, reader, f"{E}/{dst}.res", f"{src}.0")
        for j in range(4):
            _btb(ex, reader, f"{E}/{dst}.tfmr{j}", f"{src}.1.{j}")
    _lin(ex, reader, f"{E}/down0.conv", f"{est}.down_blocks.0.2")                          # k3 causal, stride 1
    _lin(ex, reader, f"{E}/up0.conv", f"{est}.up_blocks.0.2")
    # final_block is a single CausalBlock1D: conv + LN
    _lin(ex, reader, f"{E}/final_block.conv", f"{est}.final_block.block.0")
    _lin(ex, reader, f"{E}/final_block.ln", f"{est}.final_block.block.2")
    _lin(ex, reader, f"{E}/final_proj", f"{est}.final_proj")                               # [80,256,1]
    # ---- HiFTGenerator vocoder (fold weight-norm)
    V, voc = "s3gen/voc", "mel2wav"
    def wn(dst, src):
        ex.f16(dst + ".w", fold_weight_norm(reader, src))
        ex.f16(dst + ".b", reader.load_fp16(src + ".bias"))
    wn(f"{V}/conv_pre", f"{voc}.conv_pre")                                                 # [512,80,7]
    for i in range(3):
        wn(f"{V}/ups{i}", f"{voc}.ups.{i}")                                                # ConvT [in,out,k]
        _lin(ex, reader, f"{V}/sdown{i}", f"{voc}.source_downs.{i}")                       # plain conv (no wn)
    for r in range(9):   # main resblocks: stage i=r//3 (ch 256/128/64), kernel [3,7,11][r%3]
        for c in range(3):
            wn(f"{V}/rb{r}.c1_{c}", f"{voc}.resblocks.{r}.convs1.{c}")
            wn(f"{V}/rb{r}.c2_{c}", f"{voc}.resblocks.{r}.convs2.{c}")
            ex.f16(f"{V}/rb{r}.a1_{c}", reader.load_fp16(f"{voc}.resblocks.{r}.activations1.{c}.alpha"))
            ex.f16(f"{V}/rb{r}.a2_{c}", reader.load_fp16(f"{voc}.resblocks.{r}.activations2.{c}.alpha"))
    for r in range(3):   # source resblocks (k 7,7,11)
        for c in range(3):
            wn(f"{V}/srb{r}.c1_{c}", f"{voc}.source_resblocks.{r}.convs1.{c}")
            wn(f"{V}/srb{r}.c2_{c}", f"{voc}.source_resblocks.{r}.convs2.{c}")
            ex.f16(f"{V}/srb{r}.a1_{c}", reader.load_fp16(f"{voc}.source_resblocks.{r}.activations1.{c}.alpha"))
            ex.f16(f"{V}/srb{r}.a2_{c}", reader.load_fp16(f"{voc}.source_resblocks.{r}.activations2.{c}.alpha"))
    wn(f"{V}/conv_post", f"{voc}.conv_post")                                               # [18,64,7]
    for j, idx in enumerate((0, 2, 4, 6, 8)):                                              # f0 condnet convs
        ex.f16(f"{V}/f0.conv{j}.w", fold_weight_norm(reader, f"{voc}.f0_predictor.condnet.{idx}"))
        ex.f16(f"{V}/f0.conv{j}.b", reader.load_fp16(f"{voc}.f0_predictor.condnet.{idx}.bias"))
    _lin(ex, reader, f"{V}/f0.cls", f"{voc}.f0_predictor.classifier")                      # [1,512]
    _lin(ex, reader, f"{V}/nsf_linear", f"{voc}.m_source.l_linear")                        # [1,9]


def export_chatterbox_conds(conds_path, ex):
    """conds.pt (torch pickle) -> conds/ baked default voice."""
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: exporting conds.pt requires torch (pip install torch, CPU build is fine).")
    d = torch.load(conds_path, map_location="cpu", weights_only=True)
    t3, gen = d["t3"], d["gen"]
    def np_of(x):
        return x.detach().cpu().float().numpy()
    ex.f16("conds/t3_speaker_emb", np_of(t3["speaker_emb"]).reshape(-1))                    # [256]
    ex.i32("conds/t3_prompt_tokens", np_of(t3["cond_prompt_speech_tokens"]).reshape(-1))    # [375]
    ex.i32("conds/prompt_token", np_of(gen["prompt_token"]).reshape(-1))                    # [P]
    pf = np_of(gen["prompt_feat"])                                                          # [1,2P,80]
    ex.f16("conds/prompt_feat", pf.reshape(pf.shape[-2], pf.shape[-1]))
    ex.f16("conds/embedding", np_of(gen["embedding"]).reshape(-1))                          # [192]
    P = ex.manifest["conds/prompt_token"]["shape"][0]
    F = ex.manifest["conds/prompt_feat"]["shape"][0]
    print(f"[conds]  prompt tokens={P}, prompt mel frames={F} (expect F == 2P: {'OK' if F == 2 * P else 'MISMATCH!'})")
    print(f"[conds]  t3 prompt tokens={ex.manifest['conds/t3_prompt_tokens']['shape'][0]}")


def build_chatterbox_tokenizer_json(files, out_path):
    """vocab.json + merges.txt + added_tokens.json -> single HF-style tokenizer json for C#."""
    with open(files["vocab.json"], encoding="utf-8") as f:
        vocab = json.load(f)
    merges = []
    with open(files["merges.txt"], encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#version"):
                merges.append(line)
    with open(files["added_tokens.json"], encoding="utf-8") as f:
        added = json.load(f)
    tok = {
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "added_tokens": [{"id": i, "content": s, "special": False} for s, i in sorted(added.items(), key=lambda kv: kv[1])],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)
    # Plain-text twins for the C# tokenizer (no JSON parser in DeepUnity): vocab.txt line i =
    # token string for id i (byte-level BPE tokens never contain newlines), merges.txt as-is.
    base = out_path[:-5] if out_path.endswith(".json") else out_path
    by_id = sorted(vocab.items(), key=lambda kv: kv[1]) + sorted(added.items(), key=lambda kv: kv[1])
    with open(base + ".vocab.txt", "w", encoding="utf-8", newline="\n") as f:
        for tok_str, tid in by_id:
            f.write(tok_str + "\n")
    with open(base + ".merges.txt", "w", encoding="utf-8", newline="\n") as f:
        for m in merges:
            f.write(m + "\n")
    print(f"[tok]    {out_path} (+.vocab.txt/.merges.txt)  (vocab {len(vocab)} + {len(added)} added tokens)")


def export_chatterbox(args):
    if args.quant == "int4":
        sys.exit("ERROR: chatterbox export supports --quant fp16 or int8 (int4 untested on TTS).")
    model = args.model if args.model != "chatterbox" else CHATTERBOX_REPO
    names = ["t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors", "conds.pt",
             "vocab.json", "merges.txt", "added_tokens.json"]
    if os.path.isdir(model):
        files = {n: os.path.join(model, n) for n in names}
        missing = [n for n, p in files.items() if not os.path.isfile(p)]
        if missing:
            sys.exit(f"ERROR: {model} is missing {missing}")
    else:
        from huggingface_hub import hf_hub_download
        print(f"[source] HuggingFace hub: {model}")
        files = {n: hf_hub_download(model, n) for n in tqdm(names, desc="download", unit="file")}

    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "Resources", "Weights", f"weights_chatterbox_turbo_{args.quant}"))
    os.makedirs(out, exist_ok=True)
    # int8 quantizes ONLY T3's four per-layer matmul families (via TTSExporter.mat); embeddings,
    # wpe, speech_head, norms, biases and ALL of s3gen stay fp16 (s3gen int8 is a separate pass).
    print(f"[out]    {out}\n[quant]  {args.quant}\n")
    ex = TTSExporter(out, args.quant)

    export_chatterbox_t3(Safetensors([files["t3_turbo_v1.safetensors"]]), ex)
    export_chatterbox_s3gen(Safetensors([files["s3gen_meanflow.safetensors"]]), ex)
    export_chatterbox_conds(files["conds.pt"], ex)
    ex.save_manifest()

    tok_out = os.path.normpath(os.path.join(HERE, "TTS", "Chatterbox", "ChatterboxTokenizer.json"))
    os.makedirs(os.path.dirname(tok_out), exist_ok=True)
    build_chatterbox_tokenizer_json(files, tok_out)

    print(f"\nDone - {ex.bytes_written / 1024 / 1024:.0f} MB written to:\n  {out}")
    print("Use it in Unity:  var tts = new ChatterboxTTS();   // resolves the Resources folder automatically")


# ----------------------------------------------------------------------------- cosyvoice3 TTS
COSYVOICE3_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def _fold_new_weight_norm(sd):
    """Fold torch>=2 parametrized weight norm (name.parametrizations.weight.original0/1)
    into plain '<name>.weight' entries: w = g * v / ||v||  (norm over all dims but 0)."""
    import torch
    out, done = {}, set()
    for k in sd:
        if ".parametrizations.weight.original0" in k:
            base = k.split(".parametrizations.weight.original0")[0]
            g = sd[f"{base}.parametrizations.weight.original0"].float()
            v = sd[f"{base}.parametrizations.weight.original1"].float()
            norm = v.norm(dim=tuple(range(1, v.dim())), keepdim=True)
            out[base + ".weight"] = g * v / norm
            done.update({f"{base}.parametrizations.weight.original0",
                         f"{base}.parametrizations.weight.original1"})
    for k, t in sd.items():
        if k not in done and ".parametrizations." not in k:
            out[k] = t
    return out


def _walk_export(sd, ex, prefix, mat_pred=None, skip=()):
    """Export every tensor under manifest name '<prefix>/<key>'; mat_pred(key) True -> quantizable."""
    n = 0
    for k, t in sd.items():
        if any(s in k for s in skip):
            continue
        arr = t.detach().float().cpu().numpy()
        rel = f"{prefix}/{k}"
        if mat_pred is not None and arr.ndim == 2 and mat_pred(k):
            ex.mat(rel, arr)
        else:
            ex.f16(rel, arr)
        n += 1
    return n


def export_cosyvoice3(args):
    """Fun-CosyVoice3-0.5B-2512 -> Resources/Weights/weights_cosyvoice3_<quant>/.
    Names mirror the checkpoint tree under llm/ flow/ hift/ roots (see TTS/CosyVoice/SPEC.md §6).
    Requires torch (checkpoints are .pt pickles) — run inside the WSL 'cosyvoice' env."""
    import torch
    if args.quant == "int4":
        sys.exit("ERROR: cosyvoice3 export supports --quant fp16 or int8.")
    model = args.model if args.model.lower() in ("cosyvoice3", "cosyvoice3-0.5b") else args.model
    llm_file = "llm.rl.pt" if args.llm_variant == "rl" else "llm.pt"
    names = [llm_file, "flow.pt", "hift.pt",
             "CosyVoice-BlankEN/vocab.json", "CosyVoice-BlankEN/merges.txt",
             "CosyVoice-BlankEN/tokenizer_config.json"]
    if os.path.isdir(model):
        files = {n: os.path.join(model, n) for n in names}
        missing = [n for n, p in files.items() if not os.path.isfile(p)]
        if missing:
            sys.exit(f"ERROR: {model} is missing {missing}")
    else:
        from huggingface_hub import hf_hub_download
        repo = COSYVOICE3_REPO if not os.path.sep in model and "/" not in model else model
        print(f"[source] HuggingFace hub: {repo}")
        files = {n: hf_hub_download(repo, n) for n in tqdm(names, desc="download", unit="file")}

    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "Resources", "Weights", f"weights_cosyvoice3_{args.quant}"))
    os.makedirs(out, exist_ok=True)
    print(f"[out]    {out}\n[quant]  {args.quant}  [llm]  {llm_file}\n")
    ex = TTSExporter(out, args.quant)

    # ---- LM: Qwen2.5-0.5B backbone + speech_embedding + llm_decoder ------------------
    sd = torch.load(files[llm_file], map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    # locate the qwen tree root (e.g. 'llm.model.model.'), keep names below it
    root = next(k[:-len("embed_tokens.weight")] for k in sd if k.endswith("embed_tokens.weight"))
    qwen = {k[len(root):]: t for k, t in sd.items() if k.startswith(root)}
    # embed_tokens sharded like the LLM exporters (16 fp16 parts) to cap loader buffers
    emb = qwen.pop("embed_tokens.weight").float().numpy()
    rows = emb.shape[0] // EMBED_CHUNKS
    for i in range(EMBED_CHUNKS):
        ex.f16(f"llm/embed_tokens/part_{i}", emb[i * rows:(i + 1) * rows])
    big = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
           "gate_proj.weight", "up_proj.weight", "down_proj.weight")
    n = _walk_export(qwen, ex, "llm", mat_pred=lambda k: k.endswith(big),
                     skip=("lm_head.weight", "rotary_emb",))  # lm_head unused (untied speech head instead)
    for extra in ("speech_embedding.weight", "llm_decoder.weight"):
        key = next(k for k in sd if k.endswith(extra))
        ex.f16(f"llm/{extra.rsplit('.', 1)[0]}", sd[key].detach().float().cpu().numpy())
    print(f"[llm]    {n + 2 + EMBED_CHUNKS} tensors (root '{root}')")

    # ---- Flow: input emb + lookahead + DiT estimator ----------------------------------
    sd = torch.load(files["flow.pt"], map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = _fold_new_weight_norm(sd)
    dit_big = ("to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight",
               "ff.ff.0.0.weight", "ff.ff.2.weight", "attn_norm.linear.weight")
    n = _walk_export(sd, ex, "flow", mat_pred=lambda k: k.endswith(dit_big))
    # the CFM's FIXED noise buffer (CausalConditionalCFM: set_all_random_seed(0) -> randn)
    torch.manual_seed(0)
    ex.f16("flow/rand_noise", torch.randn([1, 80, 50 * 300]).numpy()[0])  # stored [80, 15000]
    print(f"[flow]   {n + 1} tensors")

    # ---- HiFT vocoder ------------------------------------------------------------------
    sd = torch.load(files["hift.pt"], map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = _fold_new_weight_norm(sd)
    n = _walk_export(sd, ex, "hift")   # all fp16 (small; quant not worth it)
    print(f"[hift]   {n} tensors")

    ex.save_manifest()

    # ---- tokenizer files next to the C# tokenizer --------------------------------------
    tok_dir = os.path.normpath(os.path.join(HERE, "TTS", "CosyVoice"))
    os.makedirs(tok_dir, exist_ok=True)
    for src, dst in (("CosyVoice-BlankEN/vocab.json", "CosyVoiceTokenizer.vocab.json"),
                     ("CosyVoice-BlankEN/merges.txt", "CosyVoiceTokenizer.merges.txt"),
                     ("CosyVoice-BlankEN/tokenizer_config.json", "CosyVoiceTokenizer.config.json")):
        shutil.copyfile(files[src], os.path.join(tok_dir, dst))
    print(f"\nDone - {ex.bytes_written / 1024 / 1024:.0f} MB written to:\n  {out}")
    print("Voices are baked separately: TTS/CosyVoice/validation/make_voice.py")
    print("Use it in Unity:  var tts = new CosyVoiceTTS();")


# ----------------------------------------------------------------------------- pocket-tts TTS
# Merged from the former TTS/PocketTTS/validation/import_pocket_tts.py (2026-07-13) — byte-identical
# output, now served by TTSExporter like the other TTS paths.
POCKETTTS_REPO = "kyutai/pocket-tts"
POCKETTTS_REPO_OPEN = "kyutai/pocket-tts-without-voice-cloning"  # tokenizer.model lives here (open)


def _pockettts_is_matmul(name, shape):
    """Only 2D '.weight' linears are int8-able; conditioner.embed.weight is an embedding table
    [4001,1024] — kept fp16 (lookup, not matmul). Norms/biases/conv kernels are never 2D .weight."""
    if len(shape) != 2 or not name.endswith(".weight"):
        return False
    return not name.endswith("conditioner.embed.weight")


def _pockettts_export_tokenizer_vocab(model_path, out_json):
    """Dump the SentencePiece Unigram vocab (piece, score, type per id) to JSON so the C# encoder
    can run a Viterbi best-segmentation + byte-fallback WITHOUT parsing the protobuf at runtime.
    type: 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 6=BYTE (SentencePiece ModelProto.SentencePiece.Type).
    Also records unk/bos/eos/pad ids and the byte-piece base so the C# maps a raw byte -> its <0xXX> id.
    NO Unicode normalization is applied by this tokenizer (verified: ligatures/fullwidth survive as
    bytes, double spaces preserved) — the C# only needs add_dummy_prefix + space->U+2581 + Viterbi."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(str(model_path))
    n = sp.vocab_size()
    pieces = []
    byte_base = None
    for i in range(n):
        piece = sp.id_to_piece(i)
        t = 1
        if sp.is_unknown(i): t = 2
        elif sp.is_control(i): t = 3
        elif sp.is_byte(i): t = 6
        elif sp.is_unused(i): t = 5
        if t == 6 and byte_base is None:
            byte_base = i          # first byte piece; SentencePiece lays <0x00>..<0xFF> contiguously
        pieces.append({"piece": piece, "score": float(sp.get_score(i)), "type": t})
    obj = {
        "vocab_size": n,
        "unk_id": sp.unk_id(), "bos_id": sp.bos_id(), "eos_id": sp.eos_id(), "pad_id": sp.pad_id(),
        "byte_base_id": byte_base,           # id of <0x00>; byte b -> byte_base_id + b
        "pieces": pieces,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"[tok]    tokenizer.vocab.json: {n} pieces, byte_base_id={byte_base}, "
          f"unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()}")


def export_pockettts(args):
    """kyutai/pocket-tts -> Resources/Weights/weights_pockettts_english_<quant>/.
    Arch (frozen in TTS/PocketTTS/SPEC.md): flow_lm (SentencePiece 4001->dim1024 conditioner;
    6L/1024d/16h/FFN4096 RoPE-causal StreamingMHA transformer; SimpleMLPAdaLN flow head dim512/
    6 res_blocks, 1 Euler step; latent ldim32) + mimi (SEANet decoder + 2L/512d/8h
    decoder_transformer). Voice = audio_prompt [1,125,1024] prefix, baked per --voice.
    Torch names kept verbatim (module path dots -> slashes, trailing param stays a dot-leaf:
    `mimi.decoder.model.0.conv.weight` -> `mimi/decoder/model/0/conv.weight`) so the manifest is
    self-documenting and the C# Conv/Linear helpers do Get(path+".weight").
    The Mimi ENCODER + downsample are needed only for RUNTIME VOICE CLONING (P8: reference wav ->
    audio_prompt): exported into the SAME dir under --include-encoder, skipped otherwise; the
    runtime loader reads them lazily on CloneVoice() only.
    Requires torch (safetensors 'pt' framework) + sentencepiece; HF access to kyutai/pocket-tts."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    if args.quant not in ("fp16", "int8"):
        sys.exit("ERROR: pocket-tts export supports --quant fp16 or int8.")
    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "Resources", "Weights", f"weights_pockettts_english_{args.quant}"))
    os.makedirs(out, exist_ok=True)
    ex = TTSExporter(out, args.quant)
    # kyutai/pocket-tts is GATED; the open mirror ships the same checkpoint MINUS mimi.encoder/
    # downsample — exactly what the default export skips, so it is byte-identical output. Only
    # --include-encoder (runtime voice cloning) truly needs the gated repo.
    repo = POCKETTTS_REPO
    try:
        model_path = hf_hub_download(repo, "languages/english/model.safetensors")
    except Exception:
        if args.include_encoder:
            sys.exit(f"ERROR: --include-encoder needs the gated {POCKETTTS_REPO} "
                     "(hf auth login + accept its gate) — the open mirror ships no mimi encoder.")
        repo = POCKETTTS_REPO_OPEN
        print(f"[source] {POCKETTTS_REPO} gated/unreachable -> open mirror {repo}")
        model_path = hf_hub_download(repo, "languages/english/model.safetensors")
    print(f"[source] {model_path}")
    print(f"[out]    {out}  (quant={args.quant}, include_encoder={args.include_encoder})")

    n_mat = n_f16 = n_skip = n_enc = 0
    with safe_open(model_path, "pt") as f:
        for k in f.keys():
            if k.startswith("mimi.encoder") or k.startswith("mimi.downsample"):
                if not args.include_encoder:
                    n_skip += 1
                    continue
                n_enc += 1
            arr = f.get_tensor(k).float().cpu().numpy()
            parts = k.split(".")
            name = "/".join(parts[:-1]) + "." + parts[-1] if len(parts) > 1 else k
            if _pockettts_is_matmul(k, arr.shape):
                if not args.dry:
                    ex.mat(name, arr)
                n_mat += 1
            else:
                if not args.dry:
                    ex.f16(name, arr)
                n_f16 += 1

    # voice embedding (audio_prompt [1,125,1024]) — bake the requested prebuilt voice into the dir
    try:
        vp = hf_hub_download(repo, f"embeddings/{args.voice}.safetensors")
        with safe_open(vp, "pt") as vf:
            for vk in vf.keys():
                varr = vf.get_tensor(vk).float().cpu().numpy()
                if not args.dry:
                    ex.f16(f"voices/{args.voice}/{vk}", varr)
        print(f"[voice]  baked: {args.voice}  {list(varr.shape)}")
    except Exception as e:
        print(f"WARN voice {args.voice}: {str(e)[:120]}")

    # tokenizer.model (SentencePiece, from the open repo) + a plain JSON dump of pieces/scores/types
    # for the C# Unigram encoder (protobuf is painful to parse at runtime in C#).
    if not args.dry:
        tok = hf_hub_download(POCKETTTS_REPO_OPEN, "languages/english/tokenizer.model")
        shutil.copy(tok, os.path.join(out, "tokenizer.model"))
        _pockettts_export_tokenizer_vocab(tok, os.path.join(out, "tokenizer.vocab.json"))
        ex.save_manifest()
    print(f"exported: {n_mat} matmul + {n_f16} f16 (incl. {n_enc} encoder), skipped {n_skip} "
          f"(encoder/downsample kept-out). tokenizer.model copied.")
    if args.quant == "int8" and ex.worst[0] > 0:
        print(f"[quant]  worst int8 |err| {ex.worst[0]:.4f} on {ex.worst[1]}")
    print(f"\nDone - {ex.bytes_written / 1024 / 1024:.0f} MB written to:\n  {out}")
    print("Use it in Unity:  var tts = new PocketTTS();")


# ----------------------------------------------------------------------------- model pool
# Downloadable pool: `python import_params.py <name> [--quant ...]` — no HF id needed.
MODEL_POOL = {
    # LLMs
    "qwen3.5-0.8b":       ("llm", "Qwen/Qwen3.5-0.8B",            None),
    "qwen3.5-2b":         ("llm", "Qwen/Qwen3.5-2B",              None),
    "gemma3-270m":        ("llm", "google/gemma-3-270m-it",       None),
    "minicpm5-1b":        ("llm", "openbmb/MiniCPM5-1B",          None),
    # TTS
    "chatterbox-turbo":   ("tts", CHATTERBOX_REPO,                "chatterbox"),
    "cosyvoice3-0.5b":    ("tts", COSYVOICE3_REPO,                "cosyvoice3"),
    "pocket-tts":         ("tts", POCKETTTS_REPO,                 "pockettts"),   # DEFAULT NPC TTS

    "kokoro-82m":         ("tts", "hexgrad/Kokoro-82M",           "kokoro"),      # exporter: TTS/Kokoro/validation/import_kokoro.py (merging)
    # STT (exporters land with the STT workstreams)
    "qwen3asr-0.6b":      ("stt", "Qwen/Qwen3-ASR-0.6B",          "qwen3asr"),
    "qwen3asr-1.7b":      ("stt", "Qwen/Qwen3-ASR-1.7B",          "qwen3asr"),
    "parakeet-tdt-0.6b-v2": ("stt", "nvidia/parakeet-tdt-0.6b-v2", "parakeet"),
    "parakeet-tdt-0.6b-v3": ("stt", "nvidia/parakeet-tdt-0.6b-v3", "parakeet"),
}


def print_pool():
    print("Available models (python import_params.py <name> [--quant fp16|int8|int4]):")
    for kind in ("llm", "tts", "stt"):
        print(f"  {kind.upper()}:")
        for name, (k, repo, arch) in MODEL_POOL.items():
            if k == kind:
                note = "   (exporter merging from its workstream)" if (k == "stt" or arch == "kokoro") else ""
                print(f"    {name:<22} -> {repo}{note}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Export a HF checkpoint into DeepUnity params (see module docstring).")
    ap.add_argument("model", nargs="?", default=None,
                    help="pool name (see --list), HF hub id, or local checkpoint folder")
    ap.add_argument("--quant", choices=["fp16", "int8", "int4"], default="fp16")
    ap.add_argument("--arch", choices=["gemma3", "qwen3_5", "minicpm5", "chatterbox", "cosyvoice3", "pockettts"],
                    default=None, help="override architecture auto-detection")
    ap.add_argument("--out", default=None, help="override the output folder")
    ap.add_argument("--llm-variant", choices=["rl", "base"], default="rl",
                    help="cosyvoice3 only: llm.rl.pt (RL-tuned, default) or llm.pt")
    ap.add_argument("--voice", default="jean", help="pocket-tts only: prebuilt voice embedding to bake in")
    ap.add_argument("--dry", action="store_true", help="pocket-tts only: walk + classify tensors, write nothing")
    ap.add_argument("--include-encoder", action="store_true",
                    help="pocket-tts only: also export mimi.encoder* + mimi.downsample (P8 runtime voice cloning)")
    ap.add_argument("--list", action="store_true", help="print the downloadable model pool and exit")
    args = ap.parse_args()

    if args.list or args.model is None:
        print_pool()
        return

    # pool-name resolution
    key = args.model.lower()
    if key in MODEL_POOL:
        kind, repo, arch = MODEL_POOL[key]
        if arch == "kokoro" or kind == "stt":
            sys.exit(f"'{key}': exporter ships with its workstream "
                     f"(TTS/Kokoro or STT/*) — run that module's validation/import_*.py for now.")
        args.model, args.arch = repo, arch or args.arch

    # dedicated TTS paths (no HF config.json layout)
    if args.arch == "cosyvoice3" or "cosyvoice3" in args.model.lower():
        export_cosyvoice3(args)
        return
    if args.arch == "chatterbox" or "chatterbox" in args.model.lower():
        export_chatterbox(args)
        return
    if args.arch == "pockettts" or "pocket-tts" in args.model.lower():
        export_pockettts(args)
        return

    cfg, reader = resolve_model(args.model)
    arch = detect_arch(cfg, reader, args.arch)
    print(f"[arch]   {arch}  (model_type='{cfg.get('model_type')}')")

    check_dims(arch, cfg)

    mdl = MODEL_LABEL[arch][0]
    sz = resolve_size(arch, cfg)
    folder = f"weights_{mdl}_{sz}_{args.quant}"   # e.g. weights_qwen3.5_0.8B_int8, weights_qwen3.5_2B_int8
    out = args.out or os.path.normpath(os.path.join(
        HERE, "..", "..", "Resources", "Weights", folder))
    os.makedirs(out, exist_ok=True)
    print(f"[out]    {out}\n[quant]  {args.quant}\n")

    ex = Exporter(out, args.quant)
    {"gemma3": export_gemma3, "qwen3_5": export_qwen3_5, "minicpm5": export_minicpm5}[arch](reader, cfg, ex)

    print(f"\nDone - {ex.bytes_written / 1024 / 1024:.0f} MB written to:\n  {out}")
    print("Layout: norm.bin + embed_tokens/part_0..15 + layer_i/<tensor> "
          + ("(.bin fp16)" if args.quant == "fp16" else f"(.{args.quant}.bin + .scales.bin; norms + embeddings/lm_head stay fp16)"))
    if args.quant != "fp16":
        print(f"Worst per-element reconstruction error: {ex.worst[0]:.6f} ({ex.worst[1]})")
    print("\nUse it in Unity (the loaders resolve this Resources folder automatically):")
    q = {"fp16": "LLMQuant.FP16", "int8": "LLMQuant.INT8", "int4": "LLMQuant.INT4"}[args.quant]
    if arch == "gemma3":
        print(f"  var llm = new Gemma3ForCausalLM({q});")
    elif arch == "minicpm5":
        print(f"  var llm = new MiniCPM5ForCausalLM({q});")
    else:
        size_enum = "Qwen3_5Size.B2" if sz == "2B" else "Qwen3_5Size.B0_8"
        print(f"  var llm = new Qwen3_5ForCausalLM({size_enum}, {q});")


if __name__ == "__main__":
    main()
