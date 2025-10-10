import re
import torch
from PIL import Image
from pathlib import Path
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
try:
    from peft import PeftModel
except:
    pass
from vlmeval.vlm.cogvlm import CogVlm
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoImageProcessor, AutoProcessor

class VividMed(BaseModel):
    def __init__(self, device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs):
        super().__init__(**kwargs)

        adapter_dir = Path('./adapter_dir')
        self.download_adapter_assets(adapter_dir)

        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self.device = torch.device(device)
        else:
            self.device = device

        # If user passes a string like "bf16"
        if isinstance(dtype, str):
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
                dtype.lower(), torch.float16
            )
        if self.device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        self.torch_dtype = dtype
        self.dtype = self.torch_dtype

        self.device = device
        # cogvlm = CogVlm()
        # self.model = cogvlm.model
        # self.tokenizer = cogvlm.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained("THUDM/cogvlm-chat-hf", trust_remote_code=True, torch_dtype="auto")
        try:
            self.processor = AutoProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
        except:
            self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', trust_remote_code=True, use_fast=True)
            # self.image_processor = AutoImageProcessor.from_pretrained("zai-org/cogvlm-chat-hf", trust_remote_code=True)
            self.processor = None

        self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))


        # Good practice to set pad/eos tokens
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"

        self.model.eval()
        

    # 2) Download matching adapter files
    def download_adapter_assets(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://api.github.com/repos/function2-llx/MMMM/releases/tags/s2"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        assets = data.get("assets", [])
        assets = [(a["name"], a["browser_download_url"]) for a in assets]
        if not assets:
            raise RuntimeError(f"No assets found")

        wanted = {"adapter_config.json", "adapter_model.safetensors"}
        picked = []
        for name, url in assets:
            if name in wanted or re.match(r"^adapter_[\w\-.]+$", name):
                print(f"Downloading: {name}")
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                (out_dir / name).write_bytes(resp.content)
                picked.append(name)

        if not picked:
            raise RuntimeError(
                f"No adapter files found in release. Available assets: {[n for n,_ in assets]}"
            )
        return out_dir
    
    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path).convert('RGB')
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
        response = response.split(self.end_text_token)[0].strip()
        return response