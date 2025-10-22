import os
import torch
from typing import List, Dict, Any
from PIL import Image

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
try:
    from HuatuoGPTVision.cli import HuatuoChatbot
except:
    pass

from .base import BaseModel

class HuatuoVision7b(BaseModel):
    def __init__(self,
                 path: str = "FreedomIntelligence/HuatuoGPT-Vision-7B",
                 device: str = "cuda",
                 dtype: str = "fp16",
                 max_new_tokens: int = 128,
                 min_new_tokens: int = 1,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)

        local_path = os.path.expanduser(f"~/.cache/modelscope/hub/models/{path}")
        if os.path.isdir(local_path):
            # print(f"[INFO] Using local ModelScope cache at: {local_path}")
            self.model_id = local_path
        else:
            self.model_id = path

        self.bot = HuatuoChatbot(self.model_id)
        self.bot.gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'temperature': temperature,
            'top_p' : top_p

        }
        self.torch_dtype = torch.float16 if dtype.lower() in {"fp16","float16"} else torch.bfloat16

    def generate(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        outputs = self.bot.inference(prompt, [image_path])
        return outputs[0]

class HuatuoVision34b(BaseModel):
    def __init__(self,
                 path: str = "FreedomIntelligence/HuatuoGPT-Vision-34B",
                 device: str = "cuda",
                 dtype: str = "fp16",
                 max_new_tokens: int = 128,
                 min_new_tokens: int = 1,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)

        local_path = os.path.expanduser(f"~/.cache/modelscope/hub/models/{path}")
        if os.path.isdir(local_path):
            # print(f"[INFO] Using local ModelScope cache at: {local_path}")
            self.model_id = local_path
        else:
            self.model_id = path

        self.bot = HuatuoChatbot(self.model_id)
        self.bot.gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'temperature': temperature,
            'top_p' : top_p

        }
        self.torch_dtype = torch.float16 if dtype.lower() in {"fp16","float16"} else torch.bfloat16

    def generate(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        outputs = self.bot.inference(prompt, [image_path])
        return outputs[0]