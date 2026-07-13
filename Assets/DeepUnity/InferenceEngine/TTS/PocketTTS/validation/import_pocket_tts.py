"""
Pocket-TTS (Kyutai) weight exporter → DeepUnity format. STANDALONE (like import_kokoro.py /
import_qwen3asr.py), but reuses the SAME manifest.tsv + packing conventions as
InferenceEngine/import_params.py so the C# PocketTTSWeights loader needs no special casing:
  - manifest.tsv line:  name<TAB>file<TAB>dtype<TAB>numel<TAB>shape-csv
  - fp16 packed 2-per-uint (.bin, dtype f16); int8 = per-output-row q8 4-per-uint (.int8.bin,
    dtype q8) + sibling '<name>.scales' f16 (only for big matmuls under --quant int8).
  - norms / biases / embeddings / small vectors / conv & convtranspose kernels ALWAYS fp16.

Usage (WSL conda env 'pocket', HF access to kyutai/pocket-tts):
    python import_pocket_tts.py --quant fp16          # -> Assets/Resources/Weights/weights_pockettts_english_fp16
    python import_pocket_tts.py --quant int8
    python import_pocket_tts.py --voice jean          # also bake one voice embedding into the dir

Arch (frozen in ../SPEC.md): flow_lm (SentencePiece 4001→dim1024 conditioner; 6L/1024d/16h/FFN4096
RoPE-causal StreamingMHA transformer; SimpleMLPAdaLN flow head dim512/6 res_blocks, 1 Euler step;
latent ldim32) + mimi (SEANet decoder + 2L/512d/8h decoder_transformer). Voice = audio_prompt
[1,125,1024] prefix. Mimi ENCODER is offline-only (voice baking) — exported under mimi/encoder* too
so make_voice can run fully in C#-free Python if needed, but the RUNTIME loader ignores encoder*.
"""
import argparse
import json
import os
import shutil

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO = "kyutai/pocket-tts"
REPO_OPEN = "kyutai/pocket-tts-without-voice-cloning"  # tokenizer.model lives here (open)
HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = .../InferenceEngine/TTS/PocketTTS/validation  →  up 5 (validation→PocketTTS→TTS→
# InferenceEngine→DeepUnity→Assets) then Resources/Weights
RES_WEIGHTS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "Resources", "Weights"))


# ---------------------------------------------------------------- packing (mirrors import_params.py)
# fp16 on disk = RAW little-endian float16, numel values, 2 bytes each (NOT uint-packed) — exactly
# what import_params.py's Exporter.fp16 writes and CosyVoiceWeights reads. The readH shader kernel
# reads the same bytes 2-per-uint; identical for even numel, and this raw form is correct for ODD
# numel too (uint-packing would pad odd arrays with 2 stray bytes → loader size mismatch).
def write_fp16(path, arr):
    np.ascontiguousarray(np.asarray(arr).astype(np.float16)).tofile(path)


def quantize_int8_perrow(w):
    """Weight-only symmetric int8, one fp16 scale PER OUTPUT ROW (w[rows,cols]); 4 int8 per uint.
    Same scheme as the LLM/TTS q8 path. Returns (packed_uint32, scales_fp16_raw, max_abs_err)."""
    w = w.astype(np.float32)
    rows, cols = w.shape
    scale = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
    q = np.round(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * scale[:, None] - w).max())
    # pack 4 int8 per uint32, row-major (matches readQ8)
    qflat = np.ascontiguousarray(q).ravel().astype(np.int32) & 0xFF
    if qflat.size % 4 != 0:
        qflat = np.concatenate([qflat, np.zeros(4 - qflat.size % 4, np.int32)])
    packed = (qflat[0::4] | (qflat[1::4] << 8) | (qflat[2::4] << 16) | (qflat[3::4] << 24)).astype(np.uint32)
    scales = np.ascontiguousarray(scale.astype(np.float16))  # raw fp16, one per row
    return packed, scales, err


class Exporter:
    def __init__(self, out_dir, quant):
        self.out = out_dir
        self.quant = quant
        self.manifest = {}
        os.makedirs(out_dir, exist_ok=True)

    def _reg(self, name, file, dtype, shape):
        self.manifest[name] = {"file": file, "dtype": dtype, "shape": [int(s) for s in shape]}

    def _ensure(self, rel):
        d = os.path.dirname(os.path.join(self.out, rel))
        if d:
            os.makedirs(d, exist_ok=True)

    def f16(self, name, arr):
        arr = np.asarray(arr)
        rel = name + ".bin"
        self._ensure(rel)
        write_fp16(os.path.join(self.out, rel), arr)
        self._reg(name, rel, "f16", arr.shape)

    def mat(self, name, arr):
        """Big matmul weight [out,in]. int8 under --quant int8, else fp16. Sibling .scales when q8."""
        arr = np.asarray(arr)
        if self.quant != "int8" or arr.ndim != 2:
            self.f16(name, arr)
            return
        self._ensure(name + ".int8.bin")
        packed, scales, err = quantize_int8_perrow(arr)
        packed.tofile(os.path.join(self.out, name + ".int8.bin"))
        scales.tofile(os.path.join(self.out, name + ".scales.bin"))
        self._reg(name, name + ".int8.bin", "q8", arr.shape)
        self._reg(name + ".scales", name + ".scales.bin", "f16", [arr.shape[0]])
        if err > 0.05:
            print(f"  WARN int8 err {err:.4f} on {name}")

    def save_manifest(self):
        with open(os.path.join(self.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)
        with open(os.path.join(self.out, "manifest.tsv"), "w", encoding="utf-8") as f:
            for name, m in self.manifest.items():
                numel = 1
                for d in m["shape"]:
                    numel *= d
                f.write(f"{name}\t{m['file']}\t{m['dtype']}\t{numel}\t{','.join(map(str, m['shape']))}\n")


# ---------------------------------------------------------------- name mapping (torch → DeepUnity)
# We keep the torch names verbatim (dots → slashes) so the manifest is self-documenting and the C#
# loader indexes by the same paths the SPEC lists. Only decision per tensor: mat() vs f16().
#   mat()  = 2D transformer/flow linear weights (int8-able)     — in_proj/out_proj/linear1/linear2,
#            flow_net input_proj/cond_embed/mlp/adaLN/final linear, conditioner.embed, out_eos.
#   f16()  = norms, biases, layer_scale, conv/convtr kernels (3D), time_embed freqs/alpha, latent
#            stat vectors, bos/speaker vectors.
MAT_SUFFIXES = (
    ".weight",  # candidate; filtered to 2D below
)


def is_matmul(name, shape):
    if len(shape) != 2:
        return False
    if not name.endswith(".weight"):
        return False
    # conditioner.embed.weight is an embedding table [4001,1024] — keep fp16 (lookup, not matmul)
    if name.endswith("conditioner.embed.weight"):
        return False
    return True


def export_tokenizer_vocab(model_path, out_json):
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
    print(f"tokenizer.vocab.json: {n} pieces, byte_base_id={byte_base}, "
          f"unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()}")


def export(quant, voice, dry=False, include_encoder=False):
    out = os.path.join(RES_WEIGHTS, f"weights_pockettts_english_{quant}")
    ex = Exporter(out, quant)
    model_path = hf_hub_download(REPO, "languages/english/model.safetensors")
    print(f"model: {model_path}")
    print(f"out:   {out}  (quant={quant}, include_encoder={include_encoder})")

    n_mat = n_f16 = n_skip = n_enc = 0
    with safe_open(model_path, "pt") as f:
        for k in f.keys():
            # The Mimi encoder + downsample are only needed for RUNTIME VOICE CLONING (P8:
            # reference wav -> audio_prompt). --include-encoder exports them into the SAME dir;
            # without it they're skipped (baked voices don't need them). The runtime loader reads
            # them ONLY when CloneVoice is called (LoadBlocking("mimi/encoder") / "mimi/downsample").
            if k.startswith("mimi.encoder") or k.startswith("mimi.downsample"):
                if not include_encoder:
                    n_skip += 1
                    continue
                n_enc += 1
            t = f.get_tensor(k)
            arr = t.float().cpu().numpy()
            # naming convention (matches the C# loader + CosyVoice family): slash the module PATH,
            # keep the trailing param name as a DOT leaf — torch `mimi.decoder.model.0.conv.weight`
            # → `mimi/decoder/model/0/conv.weight`. The C# Conv/Linear helpers do Get(path+".weight").
            parts = k.split(".")
            name = "/".join(parts[:-1]) + "." + parts[-1] if len(parts) > 1 else k
            if is_matmul(k, arr.shape):
                if not dry:
                    ex.mat(name, arr)
                n_mat += 1
            else:
                if not dry:
                    ex.f16(name, arr)
                n_f16 += 1

    # voice embedding (audio_prompt [1,125,1024]) — bake the requested prebuilt voice into the dir
    try:
        vp = hf_hub_download(REPO, f"embeddings/{voice}.safetensors")
        with safe_open(vp, "pt") as vf:
            for vk in vf.keys():
                varr = vf.get_tensor(vk).float().cpu().numpy()
                if not dry:
                    ex.f16(f"voices/{voice}/{vk}", varr)
        print(f"voice baked: {voice}  {list(varr.shape)}")
    except Exception as e:
        print(f"WARN voice {voice}: {str(e)[:120]}")

    # tokenizer.model (SentencePiece, from the open repo) + a plain JSON dump of pieces/scores/types
    # for the C# Unigram encoder (protobuf is painful to parse at runtime in C#).
    if not dry:
        tok = hf_hub_download(REPO_OPEN, "languages/english/tokenizer.model")
        shutil.copy(tok, os.path.join(out, "tokenizer.model"))
        export_tokenizer_vocab(tok, os.path.join(out, "tokenizer.vocab.json"))
        ex.save_manifest()
    print(f"exported: {n_mat} matmul + {n_f16} f16 (incl. {n_enc} encoder), skipped {n_skip} "
          f"(encoder/downsample kept-out). tokenizer.model copied.")
    print(f"manifest: {os.path.join(out, 'manifest.tsv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--voice", default="jean", help="prebuilt voice embedding to bake in")
    ap.add_argument("--dry", action="store_true", help="walk + classify tensors, write nothing")
    ap.add_argument("--include-encoder", action="store_true",
                    help="also export mimi.encoder* + mimi.downsample (P8 runtime voice cloning)")
    a = ap.parse_args()
    export(a.quant, a.voice, a.dry, a.include_encoder)
