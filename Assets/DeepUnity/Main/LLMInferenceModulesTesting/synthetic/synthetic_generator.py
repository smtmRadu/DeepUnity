from flashml.llm import vllm_chat_openai_entrypoint, GPT_OSS_120B_HIGH_VLLM_CONFIG
from flashml import log_json 

with open("synthetic_conversation_prompt.txt", "r") as f:
    prompt = f.read()
    
    

NUM_GEN = 100

input_batch = []

for i in range(NUM_GEN):
    input_batch.append([
    {
        "role":"user",
        "content": prompt
    }])
    
print(input_batch[0])
    
output = vllm_chat_openai_entrypoint(input_batch, GPT_OSS_120B_HIGH_VLLM_CONFIG)

for o in output:
    log_json(o, "generated_conversations.jsonl")
    