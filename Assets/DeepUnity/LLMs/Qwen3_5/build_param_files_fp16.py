"""Convert Qwen3.5-0.8B (text-only) into FP16 binary files for the DeepUnity
inference pipeline.

This version reads the safetensors shard directly via `safetensors` + `numpy`.
No `torch`, no `transformers` model loading. The point: avoid the
`nn.Parameter` model-dtype contamination that previously caused F32-stored
weights (RMSNormGated `norm.weight`) to be silently rounded through BF16
before reaching FP16. Each tensor is cast to FP16 directly from its native
storage dtype:
    BF16 source:  u16 bits -> shift << 16 -> fp32 -> fp16  (lossless wrt bf16)
    F16  source:  view as fp16
    F32  source:  view as fp32 -> fp16   (the case the old script broke)

Output layout under params_it/ is unchanged; see the original docstring below.

##   norm.bin
##   embed_tokens/part_{k}.bin            -- chunked, also serves as tied lm_head
##   layer_{i}/input_layernorm.bin
##   layer_{i}/post_attention_layernorm.bin
##   For full-attention layers:
##     self_attn_q_proj.bin               -- [num_attn_heads * head_dim * 2, hidden]   (Q + gate)
##     self_attn_k_proj.bin               -- [num_kv_heads  * head_dim, hidden]
##     self_attn_v_proj.bin
##     self_attn_o_proj.bin               -- [hidden, num_attn_heads * head_dim]
##     self_attn_q_norm.bin               -- [head_dim]
##     self_attn_k_norm.bin
##   For linear-attention (Gated DeltaNet) layers:
##     linear_in_proj_qkv.bin             -- [key_dim*2 + value_dim, hidden]
##     linear_in_proj_z.bin               -- [value_dim, hidden]
##     linear_in_proj_a.bin               -- [num_v_heads, hidden]
##     linear_in_proj_b.bin               -- [num_v_heads, hidden]
##     linear_conv1d.bin                  -- [conv_dim, kernel_size] (depthwise)
##     linear_dt_bias.bin                 -- [num_v_heads]
##     linear_A_log.bin                   -- [num_v_heads]
##     linear_norm.bin                    -- [head_v_dim]
##     linear_out_proj.bin                -- [hidden, value_dim]
##   mlp_gate_proj.bin / mlp_up_proj.bin / mlp_down_proj.bin
"""
import json
import os
import struct
import sys

import numpy as np
from huggingface_hub import hf_hub_download

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
SAFETENSORS_FILENAME = "model.safetensors-00001-of-00001.safetensors"
NUM_LAYERS = 24
NUM_EMBED_CHUNKS = 16
# Layer-type pattern: L L L F repeated 6 times (full_attention every 4th layer).
LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 6
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "params_it")


def open_safetensors():
    path = hf_hub_download(MODEL_NAME, SAFETENSORS_FILENAME)
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return path, header, 8 + header_len


def load_as_fp16(path, header, base, key):
    if key not in header:
        raise KeyError(f"safetensors: missing key {key}")
    meta = header[key]
    a, b = meta["data_offsets"]
    shape = tuple(meta["shape"])
    dtype = meta["dtype"]
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
    buf = raw.tobytes()
    if dtype == "BF16":
        u16 = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.astype(np.float16)
    if dtype == "F16":
        return np.frombuffer(buf, dtype=np.float16).reshape(shape).copy()
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(shape).astype(np.float16)
    raise ValueError(f"unsupported dtype {dtype} for {key}")


def dump_fp16(path, arr):
    np.ascontiguousarray(arr.astype(np.float16, copy=False)).tofile(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    safet_path, header, base = open_safetensors()
    print(f"safetensors: {safet_path}")

    # ---- final norm ----
    dump_fp16(os.path.join(OUT_DIR, "norm.bin"),
              load_as_fp16(safet_path, header, base, "model.language_model.norm.weight"))

    # ---- embedding (tied LM head) — split into 16 contiguous chunks of the flat tensor ----
    embed = load_as_fp16(safet_path, header, base, "model.language_model.embed_tokens.weight")
    flat = embed.reshape(-1)
    embed_dir = os.path.join(OUT_DIR, "embed_tokens")
    os.makedirs(embed_dir, exist_ok=True)
    for k, chunk in enumerate(np.array_split(flat, NUM_EMBED_CHUNKS)):
        dump_fp16(os.path.join(embed_dir, f"part_{k}.bin"), chunk)

    # ---- per layer ----
    for i in range(NUM_LAYERS):
        ld = os.path.join(OUT_DIR, f"layer_{i}")
        os.makedirs(ld, exist_ok=True)
        prefix = f"model.language_model.layers.{i}"

        dump_fp16(os.path.join(ld, "input_layernorm.bin"),
                  load_as_fp16(safet_path, header, base, f"{prefix}.input_layernorm.weight"))
        dump_fp16(os.path.join(ld, "post_attention_layernorm.bin"),
                  load_as_fp16(safet_path, header, base, f"{prefix}.post_attention_layernorm.weight"))

        if LAYER_TYPES[i] == "full_attention":
            ap = f"{prefix}.self_attn"
            for src, fn in [
                ("q_proj.weight", "self_attn_q_proj.bin"),
                ("k_proj.weight", "self_attn_k_proj.bin"),
                ("v_proj.weight", "self_attn_v_proj.bin"),
                ("o_proj.weight", "self_attn_o_proj.bin"),
                ("q_norm.weight", "self_attn_q_norm.bin"),
                ("k_norm.weight", "self_attn_k_norm.bin"),
            ]:
                dump_fp16(os.path.join(ld, fn),
                          load_as_fp16(safet_path, header, base, f"{ap}.{src}"))
        elif LAYER_TYPES[i] == "linear_attention":
            ap = f"{prefix}.linear_attn"
            for src, fn in [
                ("in_proj_qkv.weight", "linear_in_proj_qkv.bin"),
                ("in_proj_z.weight",   "linear_in_proj_z.bin"),
                ("in_proj_a.weight",   "linear_in_proj_a.bin"),
                ("in_proj_b.weight",   "linear_in_proj_b.bin"),
                ("dt_bias",            "linear_dt_bias.bin"),
                ("A_log",              "linear_A_log.bin"),
                ("norm.weight",        "linear_norm.bin"),
                ("out_proj.weight",    "linear_out_proj.bin"),
            ]:
                dump_fp16(os.path.join(ld, fn),
                          load_as_fp16(safet_path, header, base, f"{ap}.{src}"))
            # conv1d.weight is stored as [conv_dim, 1, kernel_size] — squeeze the dummy dim.
            cv = load_as_fp16(safet_path, header, base, f"{ap}.conv1d.weight")
            dump_fp16(os.path.join(ld, "linear_conv1d.bin"), cv.squeeze(1))
        else:
            raise ValueError(f"unknown layer_type at {i}: {LAYER_TYPES[i]}")

        mp = f"{prefix}.mlp"
        for src, fn in [
            ("gate_proj.weight", "mlp_gate_proj.bin"),
            ("up_proj.weight",   "mlp_up_proj.bin"),
            ("down_proj.weight", "mlp_down_proj.bin"),
        ]:
            dump_fp16(os.path.join(ld, fn),
                      load_as_fp16(safet_path, header, base, f"{mp}.{src}"))

        print(f"  layer {i:2d}  {LAYER_TYPES[i]}")

    print("Done.")


if __name__ == "__main__":
    main()
