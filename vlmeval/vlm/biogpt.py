import torch
import inspect
import re
from collections import OrderedDict
from PIL import Image
from vlmeval.vlm.base import BaseModel
import os

class BioGPT(BaseModel):
    def __init__(
        self,
        path: str = "microsoft/biogpt",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs
    ):
        
        from transformers import BioGptTokenizer, BioGptForCausalLM, pipeline
        super().__init__(**kwargs)

        # ---- Resolve device / dtype safely ----
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self.device = torch.device(device)
        else:
            self.device = device

        self.torch_dtype = dtype
        self.dtype = self.torch_dtype

        self.model_path = path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = BioGptTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = BioGptForCausalLM.from_pretrained(
            path,
            torch_dtype=self.torch_dtype if self.torch_dtype != torch.float32 else None,
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)

        # Move model to device/dtype and eval
        self.model.to(device=self.device)
        self.model.eval()

    def generate_inner(self, message, dataset=None):
        text, img_path = self.message_to_promptimg(message, dataset=dataset)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        with torch.no_grad():
              output = self.model.generate(
                    input_ids,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
              )
        prompt_len = input_ids.shape[1]
        new_tokens = output[0, prompt_len:]  # slice off the prompt
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return completion
