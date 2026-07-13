#!/usr/bin/env python3
"""
One-shot v2 conversion: nvidia/parakeet-tdt-0.6b-v2 ships .nemo-only — convert the LOCAL staged
.nemo into an HF-format ParakeetForTDT folder so v2 flows through the exact same exporter +
reference-dump path as v3.

USAGE (WSL, env with transformers>=5.6 [ParakeetForTDT] + torch + tokenizers + pyyaml):
    python convert_v2_nemo.py /mnt/c/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v2-nemo/parakeet-tdt-0.6b-v2.nemo \
                              /mnt/c/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v2-hf

This vendors the mapping/config logic of transformers' official
`models/parakeet/convert_nemo_to_hf.py` (HF main, 2026-07-11) with a local-path source instead of
`cached_file` (avoids re-downloading 2.47 GB), no hub pushes, and an added config sanity dump.
Mapping dicts are verbatim from the official script.
"""
import json
import os
import re
import sys
import tarfile
import tempfile

import torch
import yaml
from tokenizers import AddedToken

from transformers import ParakeetEncoderConfig, ParakeetForTDT, ParakeetTDTConfig, ParakeetTokenizer
from transformers.convert_slow_tokenizer import ParakeetConverter

# --- verbatim from transformers/models/parakeet/convert_nemo_to_hf.py -------------------------
NEMO_TO_HF_WEIGHT_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"decoder\.decoder_layers\.0\.(weight|bias)": r"ctc_head.\1",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
}
NEMO_TDT_WEIGHT_MAPPING = {
    r"decoder\.prediction\.embed\.": r"decoder.embedding.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    r"joint\.enc\.": r"encoder_projector.",
    r"joint\.pred\.": r"decoder.decoder_projector.",
    r"joint\.joint_net\.2\.": r"joint.head.",
}
ENCODER_KEYS_IGNORE = [
    "att_context_size", "causal_downsampling", "stochastic_depth_start_layer", "feat_out",
    "stochastic_depth_drop_prob", "_target_", "ff_expansion_factor", "untie_biases",
    "att_context_style", "self_attention_model", "conv_norm_type", "subsampling",
    "stochastic_depth_mode", "conv_context_size", "dropout_pre_encoder", "reduction",
    "reduction_factor", "reduction_position",
]
ENCODER_KEYS_MAP = {
    "d_model": "hidden_size", "n_heads": "num_attention_heads", "n_layers": "num_hidden_layers",
    "feat_in": "num_mel_bins", "conv_kernel_size": "conv_kernel_size",
    "subsampling_factor": "subsampling_factor", "subsampling_conv_channels": "subsampling_conv_channels",
    "pos_emb_max_len": "max_position_embeddings", "dropout": "dropout",
    "dropout_emb": "dropout_positions", "dropout_att": "attention_dropout",
    "xscaling": "scale_input", "use_bias": "attention_bias",
}
# ------------------------------------------------------------------------------------------------


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def main(nemo_path, out_dir):
    workdir = tempfile.mkdtemp(prefix="parakeet_v2_nemo_")
    print(f"Extracting {nemo_path} -> {workdir}")
    with tarfile.open(nemo_path, "r", encoding="utf-8") as tar:
        tar.extractall(workdir)
    files = {os.path.basename(p): os.path.join(r, p) for r, _, fs in os.walk(workdir) for p in fs}
    print("Archive contents:", sorted(files))
    cfg_path = files.get("model_config.yaml")
    ckpt_path = files.get("model_weights.ckpt")
    sp_model = next((p for n, p in files.items() if n.endswith(".model")), None)
    assert cfg_path and ckpt_path and sp_model, "missing model_config.yaml / model_weights.ckpt / tokenizer .model"

    nemo_config = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
    # sanity dump for the E0 report: the fields SPEC.md marked [VERIFY]
    print("\n--- v2 config ground truth ---")
    print("labels:", len(nemo_config["labels"]), "-> blank id", len(nemo_config["labels"]))
    print("durations:", nemo_config["decoding"].get("durations"))
    print("encoder use_bias:", nemo_config["encoder"].get("use_bias"),
          "xscaling:", nemo_config["encoder"].get("xscaling"),
          "n_layers:", nemo_config["encoder"].get("n_layers"),
          "d_model:", nemo_config["encoder"].get("d_model"))
    print("prednet:", nemo_config["decoder"].get("prednet"))
    print("preprocessor:", {k: v for k, v in nemo_config["preprocessor"].items() if k != "_target_"})
    print("------------------------------\n")

    # encoder config
    enc_kwargs = {}
    for key, value in nemo_config["encoder"].items():
        if key in ENCODER_KEYS_IGNORE:
            continue
        if key in ENCODER_KEYS_MAP:
            enc_kwargs[ENCODER_KEYS_MAP[key]] = value
            if key == "use_bias":
                enc_kwargs["convolution_bias"] = value
        else:
            raise ValueError(f"Unhandled NeMo encoder key: {key}")
    encoder_config = ParakeetEncoderConfig(**enc_kwargs)

    # TDT config
    labels = nemo_config["labels"]
    prednet = nemo_config["decoder"].get("prednet", {})
    model_config = ParakeetTDTConfig(
        vocab_size=len(labels) + 1,
        decoder_hidden_size=prednet.get("pred_hidden", 640),
        num_decoder_layers=prednet.get("pred_rnn_layers", 2),
        durations=nemo_config["decoding"].get("durations", [0, 1, 2, 3, 4]),
        hidden_act="relu",
        max_symbols_per_step=10,
        encoder_config=encoder_config.to_dict(),
        pad_token_id=labels.index("<pad>") if "<pad>" in labels else 0,
        blank_token_id=len(labels),
    )

    # weights
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    all_mappings = {**NEMO_TO_HF_WEIGHT_MAPPING, **NEMO_TDT_WEIGHT_MAPPING}
    converted = {}
    for key, value in state_dict.items():
        if key.endswith("featurizer.window") or key.endswith("featurizer.fb"):
            continue
        converted[convert_key(key, all_mappings)] = value

    with torch.device("meta"):
        model = ParakeetForTDT(model_config)
    missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
    assert not missing, "conversion produced missing keys — mapping drifted, DO NOT SHIP"
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys ignored")
    model.generation_config.decoder_start_token_id = model.config.blank_token_id
    model.generation_config.suppress_tokens = list(
        range(model.config.vocab_size, model.config.vocab_size + len(model.config.durations)))
    model.save_pretrained(out_dir)

    # tokenizer (SentencePiece .model -> tokenizers BPE), same as official script
    tok = ParakeetTokenizer(tokenizer_object=ParakeetConverter(sp_model).converted(),
                            clean_up_tokenization_spaces=False)
    if tok.convert_tokens_to_ids("<unk>") is None:
        tok.add_tokens([AddedToken("<unk>", normalized=False, special=True)])
    if tok.convert_tokens_to_ids("<pad>") is None:
        tok.add_tokens([AddedToken("<pad>", normalized=False, special=True)])
    tok.add_tokens([AddedToken("<blank>", normalized=False, special=True)])
    tok.add_special_tokens({"pad_token": AddedToken("<pad>", normalized=False, special=True),
                            "unk_token": AddedToken("<unk>", normalized=False, special=True)})
    tok.save_pretrained(out_dir)
    print("blank id in tokenizer:", tok.convert_tokens_to_ids("<blank>"),
          "(model blank:", model.config.blank_token_id, ")")

    # feature-extractor / processor config for AutoProcessor loading in dump_reference.py
    pre = nemo_config["preprocessor"]
    sr = pre["sample_rate"]
    feat = {
        "feature_extractor_type": "ParakeetFeatureExtractor",
        "feature_size": pre["features"], "sampling_rate": sr,
        "hop_length": int(pre["window_stride"] * sr), "win_length": int(pre["window_size"] * sr),
        "n_fft": pre["n_fft"], "preemphasis": pre.get("preemph", 0.97),
        "padding_side": "right", "padding_value": 0.0, "return_attention_mask": True,
    }
    with open(os.path.join(out_dir, "preprocessor_config.json"), "w") as f:
        json.dump(feat, f, indent=2)
    with open(os.path.join(out_dir, "processor_config.json"), "w") as f:
        json.dump({"processor_class": "ParakeetProcessor", "blank_token": "<blank>",
                   "feature_extractor": feat}, f, indent=2)

    print(f"\nDone -> {out_dir}")
    print("Reload check...")
    ParakeetForTDT.from_pretrained(out_dir, dtype=torch.float32)
    print("Reload OK.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
