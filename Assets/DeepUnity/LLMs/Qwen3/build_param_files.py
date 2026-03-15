from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")


import math
import os
import torch
import numpy as np
from tqdm import tqdm
model._modules["model"]

if not os.path.exists("params"):
    os.mkdir("params")
norm = model._modules["model"].norm
with open(f"params/norm.bin", "wb")as f:
    f.write(norm.weight.detach().cpu().numpy().astype(np.float32).tobytes())


lm_head = model._modules["lm_head"].weight.detach()
lm_head_flat = lm_head.flatten()

os.makedirs("params/lm_head", exist_ok=True)

num_chunks = 12
chunks = torch.chunk(lm_head_flat, num_chunks)

for idx, chunk in enumerate(chunks):
    # Convert to float32 numpy array
    np_chunk = chunk.cpu().numpy().astype('float32')
    
    # Write raw binary
    with open(f"params/lm_head/part_{idx}.bin", "wb") as f:
        f.write(np_chunk.tobytes())
    
    print(f"Saved chunk {idx} with {np_chunk.size} weights")



import os
import numpy as np
from tqdm import tqdm
for idx, layer in tqdm(enumerate(model._modules["model"].layers)):
    self_attn = layer.self_attn
    mlp = layer.mlp
    input_layernorm = layer.input_layernorm
    post_attention_layernorm = layer.post_attention_layernorm
    #pre_feedforward_layernorm = layer.pre_feedforward_layernorm
    #post_feedforward_layernorm = layer.post_feedforward_layernorm
    
    os.makedirs(f"params/layer_{idx}", exist_ok = True)
    
    # ================================================================ GQA =====================================================
    with open(f"params/layer_{idx}/self_attn_q_proj.bin", "wb")as f:
        f.write(self_attn.q_proj.weight.detach().flatten().cpu().numpy().astype(np.float32).tobytes())
        
    with open(f"params/layer_{idx}/self_attn_k_proj.bin", "wb")as f:
        f.write(self_attn.k_proj.weight.detach().flatten().cpu().numpy().astype(np.float32).tobytes())
        
    with open(f"params/layer_{idx}/self_attn_v_proj.bin", "wb")as f:
        f.write(self_attn.v_proj.weight.detach().flatten().cpu().numpy().astype(np.float32).tobytes())
    with open(f"params/layer_{idx}/self_attn_o_proj.bin", "wb")as f:
        f.write(self_attn.o_proj.weight.detach().flatten().cpu().numpy().astype(np.float32).tobytes())
        
    with open(f"params/layer_{idx}/self_attn_q_norm.bin", "wb")as f:
        f.write(self_attn.q_norm.weight.detach().flatten().cpu().numpy().astype(np.float32).tobytes())
    with open(f"params/layer_{idx}/self_attn_k_norm.bin", "wb")as f:
        f.write(self_attn.k_norm.weight.detach().flatten().cpu().numpy().astype(np.float32).tobytes())
        
    
    # =============================================================== MLP ======================================================
    with open(f"params/layer_{idx}/mlp_gate_proj.bin", "wb")as f:
        f.write(mlp.gate_proj.weight.detach().cpu().flatten().numpy().astype(np.float32).tobytes())
    with open(f"params/layer_{idx}/mlp_up_proj.bin", "wb")as f:
        f.write(mlp.up_proj.weight.detach().cpu().flatten().numpy().astype(np.float32).tobytes())
    with open(f"params/layer_{idx}/mlp_down_proj.bin", "wb")as f:
        f.write(mlp.down_proj.weight.detach().cpu().flatten().numpy().astype(np.float32).tobytes())
        
    # ================================================================ RMS =====================================================
    with open(f"params/layer_{idx}/input_layernorm.bin", "wb")as f:
        f.write(input_layernorm.weight.detach().cpu().flatten().numpy().astype(np.float32).tobytes())
    with open(f"params/layer_{idx}/post_attention_layernorm.bin", "wb")as f:
        f.write(post_attention_layernorm.weight.detach().cpu().flatten().numpy().astype(np.float32).tobytes())
    # with open(f"params/layer_{idx}/pre_feedforward_layernorm.bin", "wb")as f:
    #     f.write(pre_feedforward_layernorm.weight.detach().cpu().flatten().numpy().astype(np.float32).tobytes())  
    # with open(f"params/layer_{idx}/post_feedforward_layernorm.bin", "wb")as f:
    #     f.write(post_feedforward_layernorm.weight.detach().cpu().flatten().numpy().astype(np.float32).tobytes())
        
    # print(vars(layer))