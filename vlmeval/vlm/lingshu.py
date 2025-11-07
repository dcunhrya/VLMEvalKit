import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from vlmeval.vlm.base import BaseModel

class Lingshu7(BaseModel):
    """
    VLMEvalKit wrapper for Lingshu-Medical-Mllm/Lingshu-7B.
    """

    def __init__(self,
                 path: str = "lingshu-medical-mllm/Lingshu-7B",
                 device: str = "cuda",
                 dtype: str = "fp16",
                 max_new_tokens: int = 64,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.torch_dtype = torch.float16 if dtype.lower() in {"fp16", "float16"} else torch.bfloat16
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # load processor + model
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()

    def generate(self, message, dataset: str = ""):
        """
        message: list of dicts like:
        [
            {"type": "image", "value": "/path/to/image.jpg"},
            {"type": "text", "value": "Your question here"}
        ]
        """
        # convert VLM evalkit message -> HF chat template
        content = []
        for m in message:
            if m["type"] == "image":
                content.append({"type": "image", "url": m["value"]})
            elif m["type"] == "text":
                content.append({"type": "text", "text": m["value"]})

        messages = [{"role": "user", "content": content}]

        # apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # generation
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=(self.temperature > 0)
            )

        # decode only new tokens
        text = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        return text
    
class Lingshu32(BaseModel):
    """
    VLMEvalKit wrapper for Lingshu-Medical-Mllm/Lingshu-32B.
    """

    def __init__(self,
                 path: str = "lingshu-medical-mllm/Lingshu-32B",
                 device: str = "cuda",
                 dtype: str = "fp16",
                 max_new_tokens: int = 128,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.torch_dtype = torch.float16 if dtype.lower() in {"fp16", "float16"} else torch.bfloat16
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # load processor + model
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        ).eval()

    def generate(self, message, dataset: str = ""):
        """
        message: list of dicts like:
        [
            {"type": "image", "value": "/path/to/image.jpg"},
            {"type": "text", "value": "Your question here"}
        ]
        """
        # convert VLM evalkit message -> HF chat template
        content = []
        for m in message:
            if m["type"] == "image":
                content.append({"type": "image", "url": m["value"]})
            elif m["type"] == "text":
                content.append({"type": "text", "text": m["value"]})

        messages = [{"role": "user", "content": content}]

        # apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # generation
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=(self.temperature > 0)
            )

        # decode only new tokens
        text = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        return text
