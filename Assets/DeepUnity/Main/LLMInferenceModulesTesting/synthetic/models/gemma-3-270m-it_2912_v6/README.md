---
base_model: unsloth/gemma-3-270m-it-unsloth-bnb-4bit
library_name: transformers
model_name: gemma-3-270m-it_2912_v6
tags:
- generated_from_trainer
- sft
- unsloth
- trl
licence: license
---

# Model Card for gemma-3-270m-it_2912_v6

This model is a fine-tuned version of [unsloth/gemma-3-270m-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3-270m-it-unsloth-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.22.2
- Transformers: 4.56.0
- Pytorch: 2.8.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```