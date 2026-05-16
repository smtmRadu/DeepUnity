## This script is used to transform google/embeddinggemma-300m into fp16 bin
## files that will be imported in Unity for the Gemma3ForEmbeddings inference
## pipeline (24-layer Gemma3 trunk + mean-pool + 2-layer dense head + L2-norm).

from sentence_transformers import SentenceTransformer

import os
import torch
import numpy as np
from tqdm import tqdm

model = SentenceTransformer("google/embeddinggemma-300m")

# Sentence-transformers structure for embedding-gemma:
#   model[0] = Transformer       -> .auto_model is the HF Gemma3TextModel trunk
#   model[1] = Pooling           -> mean-pool, no weights
#   model[2] = Dense             -> Linear(hidden -> 3072), .linear.weight
#   model[3] = Dense             -> Linear(3072 -> hidden), .linear.weight
#   model[4] = Normalize         -> L2-norm, no weights
trunk = model[0].auto_model
dense_1 = model[2].linear
dense_2 = model[3].linear

os.makedirs("params_embedding", exist_ok=True)

norm = trunk.norm
with open(f"params_embedding/norm.bin", "wb")as f:
    f.write(norm.weight.detach().cpu().float().numpy().astype(np.float16).tobytes())


embed_tokens = trunk.embed_tokens.weight.detach()
embed_flat = embed_tokens.flatten()

os.makedirs("params_embedding/embed_tokens", exist_ok=True)

num_chunks = 16
chunks = torch.chunk(embed_flat, num_chunks)

for idx, chunk in enumerate(chunks):
    np_chunk = chunk.cpu().float().numpy().astype(np.float16)

    with open(f"params_embedding/embed_tokens/part_{idx}.bin", "wb") as f:
        f.write(np_chunk.tobytes())

    print(f"Saved chunk {idx} with {np_chunk.size} weights")


# ============================================================== Dense head =====================================================
with open(f"params_embedding/dense_1.bin", "wb")as f:
    f.write(dense_1.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())
with open(f"params_embedding/dense_2.bin", "wb")as f:
    f.write(dense_2.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())




import os
import numpy as np
from tqdm import tqdm
for idx, layer in tqdm(enumerate(trunk.layers)):
    self_attn = layer.self_attn
    mlp = layer.mlp
    input_layernorm = layer.input_layernorm
    post_attention_layernorm = layer.post_attention_layernorm
    pre_feedforward_layernorm = layer.pre_feedforward_layernorm
    post_feedforward_layernorm = layer.post_feedforward_layernorm

    os.makedirs(f"params_embedding/layer_{idx}", exist_ok = True)

    # ================================================================ GQA =====================================================
    with open(f"params_embedding/layer_{idx}/self_attn_q_proj.bin", "wb")as f:
        f.write(self_attn.q_proj.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())

    with open(f"params_embedding/layer_{idx}/self_attn_k_proj.bin", "wb")as f:
        f.write(self_attn.k_proj.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())

    with open(f"params_embedding/layer_{idx}/self_attn_v_proj.bin", "wb")as f:
        f.write(self_attn.v_proj.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())
    with open(f"params_embedding/layer_{idx}/self_attn_o_proj.bin", "wb")as f:
        f.write(self_attn.o_proj.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())

    with open(f"params_embedding/layer_{idx}/self_attn_q_norm.bin", "wb")as f:
        f.write(self_attn.q_norm.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())
    with open(f"params_embedding/layer_{idx}/self_attn_k_norm.bin", "wb")as f:
        f.write(self_attn.k_norm.weight.detach().flatten().cpu().float().numpy().astype(np.float16).tobytes())


    # =============================================================== MLP ======================================================
    with open(f"params_embedding/layer_{idx}/mlp_gate_proj.bin", "wb")as f:
        f.write(mlp.gate_proj.weight.detach().cpu().flatten().float().numpy().astype(np.float16).tobytes())
    with open(f"params_embedding/layer_{idx}/mlp_up_proj.bin", "wb")as f:
        f.write(mlp.up_proj.weight.detach().cpu().flatten().float().numpy().astype(np.float16).tobytes())
    with open(f"params_embedding/layer_{idx}/mlp_down_proj.bin", "wb")as f:
        f.write(mlp.down_proj.weight.detach().cpu().flatten().float().numpy().astype(np.float16).tobytes())

    # ================================================================ RMS =====================================================
    with open(f"params_embedding/layer_{idx}/input_layernorm.bin", "wb")as f:
        f.write(input_layernorm.weight.detach().cpu().flatten().float().numpy().astype(np.float16).tobytes())
    with open(f"params_embedding/layer_{idx}/post_attention_layernorm.bin", "wb")as f:
        f.write(post_attention_layernorm.weight.detach().cpu().flatten().float().numpy().astype(np.float16).tobytes())
    with open(f"params_embedding/layer_{idx}/pre_feedforward_layernorm.bin", "wb")as f:
        f.write(pre_feedforward_layernorm.weight.detach().cpu().flatten().float().numpy().astype(np.float16).tobytes())
    with open(f"params_embedding/layer_{idx}/post_feedforward_layernorm.bin", "wb")as f:
        f.write(post_feedforward_layernorm.weight.detach().cpu().flatten().float().numpy().astype(np.float16).tobytes())

    # print(vars(layer))
print("Done")
