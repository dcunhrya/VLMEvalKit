import os
from typing import Any, Dict, List, Optional
import re
import torch
try:
    from transformers import AutoProcessor, GenerationConfig
    from transformers import Qwen2VLForConditionalGeneration
except:
    pass
from qwen_vl_utils import process_vision_info

from vlmeval.vlm.base import BaseModel


QUESTION_TEMPLATE = """
{Question}
Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
3. No extra information or text outside of these tags.
""".strip()

class MedVLM_R1(BaseModel):
    """
    VLMEvalKit wrapper for JZPeterPan/MedVLM-R1 (Qwen2-VL).
    - Loads with device_map='auto' when multi-GPU / allowed (requires accelerate>=0.26).
    - Uses processor.apply_chat_template + process_vision_info to build inputs.
    - Trims prompt tokens before decoding.
    """

    def __init__(
        self,
        path: str = "JZPeterPan/MedVLM-R1",
        device: str = "cuda",            # "cuda" | "cuda:0" | "cpu" | "mps"
        dtype: str = "bf16",             # "bf16" | "fp16" | "fp32" | "auto"
        max_new_tokens: int = 1024,
        temperature: float = 0.0,        # paper/demo uses deterministic gen
        top_p: float = 1.0,
        use_device_map: bool = True,
        attn_impl: str = "sdpa",
        pad_token_id: Optional[int] = 151643,   # matches your example; tokenizer may override
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.model_id = path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_device_map = use_device_map
        self.pad_token_id = pad_token_id

        if dtype == "auto":
            torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and device.startswith("cuda")) else torch.float32
        elif dtype.lower() in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype.lower() in {"fp16", "float16"}:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        load_kwargs = dict(
            torch_dtype=self.torch_dtype,
            attn_implementation=attn_impl,
        )

        multi_gpu = (
            self.use_device_map
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and device.startswith("cuda")
        )
        if multi_gpu:
            load_kwargs.update(device_map="auto", low_cpu_mem_usage=True)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if not multi_gpu:
            if device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda:0")
            elif device.startswith("cuda") and torch.cuda.is_available():
                self.model.to(device)
            elif device == "mps" and torch.backends.mps.is_available():
                self.model.to("mps")
            else:
                self.model.to("cpu")

        self.gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=max(self.temperature, 1e-5) if self.temperature > 0 else 1.0,  # ignored when do_sample=False
            num_return_sequences=1,
            pad_token_id=self.pad_token_id if self.pad_token_id is not None else getattr(self.model.generation_config, "pad_token_id", None),
        )


    def _to_qwen2_chat(self, message: List[Dict]) -> List[Dict]:
        """
        VLMEvalKit message -> Qwen2-VL chat turns.
        Input message: [{"type":"image","value":...}, {"type":"text","value":...}, ...]
        We combine all text segments and append after image(s).
        """
        images = []
        texts = []
        for m in message:
            t = m.get("type")
            if t == "image":
                images.append((str(m["value"])))
            elif t == "text":
                texts.append(str(m["value"]))

        question = "\n".join([x.strip() for x in texts if x]).strip()
        if not question:
            question = "Answer the question based on the image(s)."

        content: List[Dict] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)})

        return [{"role": "user", "content": content}]

    def generate(self, message: List[Dict], dataset: str = "") -> List[str]:
        """
        Returns a single-element list with the decoded answer (VLMEvalKit convention).
        """
        chat = self._to_qwen2_chat(message)
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(chat)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        is_sharded = getattr(self.model, "hf_device_map", None) is not None
        if not is_sharded:
            dev = next(self.model.parameters()).device
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs = dict(
            use_cache=True,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=max(self.temperature, 1e-5) if self.temperature > 0 else None,
            top_p=self.top_p if self.temperature > 0 else None,
            generation_config=self.gen_cfg,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_ids = inputs["input_ids"]
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]

        # Decode
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        out = texts[0].strip() if texts else ""
        match = re.search(r"<answer>(.*?)</answer>", out, flags=re.IGNORECASE | re.DOTALL)
        if match:
            out = match.group(1).strip()
        return out