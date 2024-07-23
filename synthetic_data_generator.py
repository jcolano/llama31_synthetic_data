"""
Created by Juan C Olano on 7/23/2024
Purpose: This class generates synthetic conversational data using the Llama 3.1 model.

Overview:
The Llama family of models can generate text by following specific templates. In this case, we use a template to simulate a conversation between a user and an assistant. The process involves two main steps:

1. **User Prompt Generation**: Given a starting prompt for the user, the model generates a random user prompt.
2. **Assistant Response Generation**: This user prompt is then used as input to generate the assistant's response, creating a complete conversational exchange.

By running this process in a loop, we can create a dataset of synthetic conversations. The quality and coherence of the generated dataset depend on the underlying model's performance and training data.

Key Features:
- The script uses the Llama 3.1 model to generate synthetic data.
- It includes functions to load the model, generate text, and sample tokens using top-p sampling.
- The generated dataset is saved in a JSONL file format, with each line containing a user-assistant conversation pair.

Usage:
1. Initialize the `SyntheticDataGenerator` with the directory containing the model checkpoints.
2. Use the `synthesize_data` method to generate the desired number of conversation pairs and save them to a file.

Example:
```python
generator = SyntheticDataGenerator("Meta-Llama-3.1-8B-Instruct")
generator.synthesize_data(n=10)

"""

import os
import torch
import torch.nn.functional as F
from torch import nn
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from contextlib import nullcontext

from model import Transformer
from args import ModelArgs
from tokenizer import Tokenizer

class SyntheticDataGenerator:
    def __init__(self, ckpt_dir: str, max_seq_len: int = 512, max_batch_size: int = 8):
        self.ckpt_dir = ckpt_dir
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.load_model()

    def load_model(self):
        start_time = time.time()

        checkpoints = sorted(Path(self.ckpt_dir).glob("*.pth"))
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        with open(f"{self.ckpt_dir}/params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            **params,
        )
        self.tokenizer = Tokenizer(model_path=f"{self.ckpt_dir}/tokenizer.model")

        assert model_args.vocab_size == self.tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.model = Transformer(model_args)

        self.model.load_state_dict(checkpoint, strict=False)
        self.model = self.model.half()
        self.model.to(self.device)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

    @staticmethod
    def sample_top_p(probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    @torch.inference_mode()
    def generate(self, prompt: str, max_gen_len: int = 128, temperature: float = 0.6, 
                 top_p: float = 0.9, logprobs: bool = False, stop_token_id: int = 128009) -> Tuple[List[int], Optional[List[float]]]:
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        prompt_len = len(prompt_tokens)
        total_len = min(self.max_seq_len, max_gen_len + prompt_len)
        
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((total_len,), pad_id, dtype=torch.long, device=self.device)
        tokens[:prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)
        
        if logprobs:
            token_logprobs = torch.zeros(total_len, dtype=torch.float, device=self.device)

        stop_token_ids = [91, 408, 3659, 4424, 91]
        stop_token_len = len(stop_token_ids)
        
        start_header_ids = [91, 2527, 8932, 851, 91]
        start_header_len = len(start_header_ids)

        last_tokens = []
        
        actual_gen_len = 0
        for cur_pos in range(prompt_len, total_len):
            input_tokens = tokens.unsqueeze(0)[:, :cur_pos]
            logits = self.model.forward(input_tokens, 0)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            tokens[cur_pos] = next_token
            actual_gen_len += 1

            if logprobs:
                token_logprobs[cur_pos] = -F.cross_entropy(
                    input=logits[:, -1].unsqueeze(1),
                    target=next_token.unsqueeze(0),
                    reduction="none",
                    ignore_index=pad_id,
                ).squeeze()

            last_tokens.append(next_token.item())
            if len(last_tokens) > stop_token_len:
                last_tokens.pop(0)

            if last_tokens == stop_token_ids or last_tokens == start_header_ids or next_token == 128008 or next_token == 128009:
                tokens = tokens[:cur_pos - stop_token_len]
                break

        out_tokens = tokens[prompt_len:prompt_len + max_gen_len].tolist()
        out_tokens = [t for t in out_tokens if t != pad_id]
        
        if logprobs:
            out_logprobs = token_logprobs[prompt_len:prompt_len + max_gen_len].tolist()
            out_logprobs = [p for p in out_logprobs if p != float('inf')]
        else:
            out_logprobs = None

        return out_tokens, out_logprobs

    def synthesize_data(self, n=100, output_file='synthetic_data.jsonl'):
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for i in range(n):
                user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                gentokens1, _ = self.generate(user_prompt, max_gen_len=128)
                generated_text1 = self.tokenizer.decode(gentokens1)
                
                assistant_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{generated_text1}<|end_of_text|><|start_header_id|>assistant<|end_header_id|>"
                gentokens2, _ = self.generate(assistant_prompt, max_gen_len=256)
                generated_text2 = self.tokenizer.decode(gentokens2)
                
                data = {
                    "user": generated_text1,
                    "assistant": generated_text2
                }
                
                json.dump(data, jsonl_file)
                jsonl_file.write('\n')
                jsonl_file.flush()
                
                print(f"Iteration {i+1}/{n} completed and saved.")

# Example usage:
# generator = SyntheticDataGenerator("Meta-Llama-3.1-8B-Instruct")
# generator.synthesize_data(n=10)