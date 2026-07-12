from flashml.llm import vllm_chat_openai_entrypoint, GPT_OSS_120B_HIGH_VLLM_CONFIG, QWEN3_4B_THINKING_2507_VLLM_CONFIG
from flashml import log_json
NUM_SAMPLES_TO_GENERATE = 4


with open("roleplay_dataset_prompt.txt", "r") as f:
    USER_PROMPT = f.read()

inputs = []

for i in range(NUM_SAMPLES_TO_GENERATE):
    inputs.append({
        "role": "user",
        "content": USER_PROMPT
    })


outputs = vllm_chat_openai_entrypoint(inputs, QWEN3_4B_THINKING_2507_VLLM_CONFIG)

for i in outputs:
    log_json(
        {
            "output": i["response"]["body"]["choices"][0]["message"]['content']

        },
        path="roleplay_dataset.jsonl",
        add_timestamp=True
    )


