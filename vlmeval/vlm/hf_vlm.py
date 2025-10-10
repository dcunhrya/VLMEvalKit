# VLMEvalKit/vlmeval/vlm/hf_vlm.py
from typing import List, Dict, Any
from PIL import Image
import torch
import inspect 
from transformers import (
    AutoProcessor, AutoTokenizer, AutoImageProcessor,
    AutoModelForCausalLM, AutoModelForVision2Seq,
    PretrainedConfig,
)
from vlmeval.vlm.base import BaseModel

IMAGE_FAMILIES_NEEDING_TOKENS = ("gemma3", "llava", "idefics2", "minicpm", "qwen2_vl", "phi3_vision", "internvl")

class HuggingFaceVisionVLM(BaseModel):
    def __init__(self,
                 path: str,
                 device: str = "cuda",
                 dtype: str = "auto",
                 max_new_tokens: int = 128,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_id = path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # dtype
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        self.device = device

        # ---- Load config with remote code allowed (avoids KeyError) ----
        try:
            cfg = PretrainedConfig.from_pretrained(self.model_id, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", None)
        except Exception:
            cfg = None
            model_type = None

        # ---- Processor / tokenizer (LLaVA can have custom Processor) ----
        # Try AutoProcessor first; if it fails, fall back to separate token/image processors.
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
        except Exception:
            # Fallback: some repos only provide a tokenizer + (optional) image processor
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
            except Exception:
                self.tokenizer = None
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception:
                self.image_processor = None

        # ---- Model: try Vision2Seq, then CausalLM (covers most VLMs) ----
        self.model = None
        last_err = None
        for Loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
            try:
                self.model = Loader.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype
                )
                break
            except Exception as e:
                last_err = e
                continue
        if self.model is None:
            raise RuntimeError(f"Failed to load model for {self.model_id}: {last_err}")

        # device
        if device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.model.to("mps")
        else:
            self.model.to("cpu")
        self.model.eval()

    def _prepare_inputs(self, message):
        # 1) Split images/text
        images, texts = [], []
        for it in message:
            if it["type"] == "image":
                images.append(Image.open(it["value"]).convert("RGB"))
            elif it["type"] == "text":
                texts.append(str(it["value"]))
        user_text = "\n".join(t.strip() for t in texts if t).strip() or "Describe the image briefly."

        # 2) Make a single user turn: [image, image, ..., text]
        chat = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))]
                                        + [{"type": "text", "text": user_text}]}]

        model_type = getattr(getattr(self.model, "config", None), "model_type", "") or ""
        proc_name  = type(self.processor).__name__.lower()

        # 3) Prefer chat template if available (most modern HF VLMs)
        templated_text = None
        if hasattr(self.processor, "apply_chat_template"):
            try:
                templated_text = self.processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                templated_text = None  # fall back

        # 4) Fallback: inject <image> markers for families that require them
        if templated_text is None:
            text = user_text
            needs_tokens = any(k in (model_type + " " + proc_name) for k in IMAGE_FAMILIES_NEEDING_TOKENS)
            if images and needs_tokens:
                have = text.count("<image>")
                need = len(images)
                if have < need:
                    prefix = "\n".join(["<image>"] * (need - have))
                    text = prefix + ("\n" if text else "") + text
            templated_text = text

        # 5) Pack with processor (uniform across families)
        inputs = self.processor(
            images=images or None,
            text=templated_text,
            padding=True,
            return_tensors="pt"
        )

        # 6) Normalize device/dtype + attention_mask
        if hasattr(self, "device"):
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if hasattr(self, "torch_dtype"):
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=self.torch_dtype)
        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=self.device)
        return inputs

    def _align_inputs_with_model(self, inputs):
        """Move all input tensors to the model's device, with sane dtypes."""
        dev = next(self.model.parameters()).device  # e.g., cuda:0
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                # Float tensors (e.g., pixel_values) should match your inference dtype
                if v.dtype.is_floating_point:
                    dtype = getattr(self, "torch_dtype", None)
                    if dtype is None:
                        # default: bf16/half on CUDA, float32 on CPU
                        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
                    v = v.to(device=dev, dtype=dtype)
                else:
                    # int tensors like input_ids/attention_mask/token_type_ids
                    v = v.to(device=dev)
            out[k] = v
        return out

    def _dump_devices(self, label, inputs, model):
        print(f"== {label} ==")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                print(k, v.device, v.shape, v.dtype)
        try:
            print("model main device:", next(model.parameters()).device)
        except StopIteration:
            pass
        print("device_map:", getattr(model, "hf_device_map", None))

    def generate(self, message, dataset: str = "") -> str:
        inputs = self._prepare_inputs(message)
        inputs = self._align_inputs_with_model(inputs)
        # self._dump_devices("pre-generate", inputs, self.model)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            return_dict_in_generate=True
        )
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        seq = out_ids.sequences if hasattr(out_ids, "sequences") else out_ids
        if "lang_x" in inputs:
            in_len = inputs["lang_x"].shape[1]
        elif "input_ids" in inputs:
            in_len = inputs["input_ids"].shape[1]
        else:
            raise ValueError("Can't find text input ids (expected 'lang_x' or 'input_ids').")

        gen_tokens = seq[0, in_len:]
        decode = getattr(self.processor, "decode", None) or self.tokenizer.decode
        return decode(gen_tokens, skip_special_tokens=True).strip()
