import os
import math
from typing import List, Tuple, Optional, Any, Dict, Union
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

class ModelHandler:

    def __init__(self, model_path, config, max_tokens=3072, device="cuda:0", rank=0):

        self.max_tokens = max_tokens
        dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            config=config,
            device_map={"": rank},
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.device = device
        self.rank = rank

        # extra check - whether to use top_k settings from qwen3
        if "deepseek" in model_path.lower():
            self.deepseek = True
        else:
            self.deepseek = False

    def generate_answer(
        self,
        prompt: Union[str, List[str]],
        **kwargs
    ) -> str:

        torch.cuda.set_device(self.rank)
        # 1. keep role structure
        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # 2. build prompt with chat template
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = self.device
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(device)

        # 3. generate
        ctx_len = inputs.input_ids.shape[-1]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        if self.deepseek:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                use_cache=True
            )[0]

        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                do_sample=True,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                use_cache=True
            )[0]

        # 4. decode only the newly generated tokens
        text = self.tokenizer.decode(outputs[ctx_len:], skip_special_tokens=True)
        return text
