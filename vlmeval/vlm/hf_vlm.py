# VLMEvalKit/vlmeval/vlm/hf_vlm.py
from typing import List, Dict, Any
from PIL import Image
import torch
import inspect

from transformers import (
    AutoProcessor, AutoTokenizer, AutoImageProcessor,
    AutoModelForCausalLM, AutoModelForVision2Seq, PretrainedConfig
)
from vlmeval.vlm.base import BaseModel

IMAGE_FAMILIES_NEEDING_TOKENS = ("gemma3", "llava", "idefics2", "minicpm", "qwen2_vl", "phi3_vision", "internvl")

def _has_multi_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 1

class HuggingFaceVisionVLM(BaseModel):
    def __init__(self,
                 path: str,
                 device: str = "cuda",        # "cuda" | "cuda:0" | "cpu" | "mps"
                 dtype: str = "auto",
                 max_new_tokens: int = 128,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 use_device_map: bool = True,  # turn off to force single-device .to(device)
                 **kwargs):
        super().__init__(**kwargs)
        self.model_id = path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.use_device_map = use_device_map

        # dtype
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and device.startswith("cuda")) else torch.float32
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        # ---- Config ----
        try:
            cfg = PretrainedConfig.from_pretrained(self.model_id, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", None)
        except Exception:
            cfg = None
            model_type = None

        # ---- Processor / tokenizer ----
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
            except Exception:
                self.tokenizer = None
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception:
                self.image_processor = None

        # ---- Model (Vision2Seq first, then CausalLM) ----
        self.model = None
        last_err = None

        # Decide loading strategy
        do_shard = (self.use_device_map and _has_multi_gpu() and device.startswith("cuda"))
        common_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if do_shard:
            common_kwargs.update(
                device_map="auto",             # <- Accelerate will shard across GPUs
                low_cpu_mem_usage=True,
            )

        for Loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
            try:
                self.model = Loader.from_pretrained(self.model_id, **common_kwargs).eval()
                break
            except Exception as e:
                last_err = e
                continue
        if self.model is None:
            raise RuntimeError(f"Failed to load model for {self.model_id}: {last_err}")

        # If NOT sharded, place on a single device explicitly.
        # IMPORTANT: never call .to() when using device_map="auto".
        if not do_shard:
            if device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda:0")
            elif device.startswith("cuda") and torch.cuda.is_available():
                self.model.to(device)
            elif device == "mps" and torch.backends.mps.is_available():
                self.model.to("mps")
            else:
                self.model.to("cpu")

        # basic generation config hygiene
        if hasattr(self.model, "generation_config"):
            gc = self.model.generation_config
            if getattr(gc, "pad_token_id", None) is None and getattr(gc, "eos_token_id", None) is not None:
                gc.pad_token_id = gc.eos_token_id

    def _prepare_inputs(self, message):
        # 1) Split images/text
        images, texts = [], []
        for it in message:
            if it["type"] == "image":
                images.append(Image.open(it["value"]).convert("RGB"))
            elif it["type"] == "text":
                texts.append(str(it["value"]))
        user_text = "\n".join(t.strip() for t in texts if t).strip() or "Describe the image briefly."

        # 2) Single user turn: [image*] + text
        chat = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))]
                                        + [{"type": "text", "text": user_text}]}]

        model_type = getattr(getattr(self.model, "config", None), "model_type", "") or ""
        proc_name  = type(self.processor).__name__.lower() if self.processor is not None else ""

        # 3) Chat template if available
        templated_text = None
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                templated_text = self.processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                templated_text = None

        # 4) Fallback: inject <image> markers if needed
        if templated_text is None:
            text = user_text
            needs_tokens = any(k in (model_type + " " + proc_name) for k in IMAGE_FAMILIES_NEEDING_TOKENS)
            if images and needs_tokens:
                need = len(images)
                have = text.count("<image>")
                if have < need:
                    text = ("\n".join(["<image>"] * (need - have))) + ("\n" if text else "") + text
            templated_text = text

        # 5) Pack with processor
        proc = self.processor if self.processor is not None else self._processor_fallback()
        inputs = proc(images=images or None, text=templated_text, padding=True, return_tensors="pt")

        # 6) Ensure attention_mask exists
        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

        return inputs

    def _processor_fallback(self):
        # Minimal fallback if AutoProcessor is unavailable
        if self.tokenizer is None:
            raise RuntimeError("No processor or tokenizer available.")
        def _call(images=None, text=None, padding=True, return_tensors="pt"):
            enc = self.tokenizer(text, padding=padding, return_tensors=return_tensors)
            if images is not None:
                if self.image_processor is None:
                    raise RuntimeError("Images provided but no image processor is available.")
                img_batch = self.image_processor(images=images, return_tensors=return_tensors)
                enc.update(img_batch)
            return enc
        return type("FallbackProcessor",(object,),{"__call__": staticmethod(_call), "decode": getattr(self.tokenizer, "decode", None)})

    def _align_inputs_with_model(self, inputs):
        """
        - If the model is sharded (has hf_device_map), DO NOT move tensors – leave them on CPU.
          Accelerate will route them.
        - If single-device, move tensors to that device and dtype.
        """
        is_sharded = getattr(self.model, "hf_device_map", None) is not None
        if is_sharded:
            # just normalize dtypes for floating tensors (keeping them on CPU is OK)
            out = {}
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    out[k] = v.to(dtype=self.torch_dtype)
                else:
                    out[k] = v
            return out

        # single-device path
        dev = next(self.model.parameters()).device
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if v.dtype.is_floating_point:
                    out[k] = v.to(device=dev, dtype=self.torch_dtype)
                else:
                    out[k] = v.to(device=dev)
            else:
                out[k] = v
        return out

    def generate(self, message, dataset: str = "") -> str:
        inputs = self._prepare_inputs(message)
        inputs = self._align_inputs_with_model(inputs)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            return_dict_in_generate=True
        )

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        seq = out.sequences if hasattr(out, "sequences") else out
        in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen_tokens = seq[0, in_len:]

        # processor.decode preferred; tokenizer.decode fallback
        decode_fn = getattr(self.processor, "decode", None) if self.processor is not None else None
        if decode_fn is None:
            decode_fn = getattr(self.tokenizer, "decode", None)
        if decode_fn is None:
            raise RuntimeError("No decode function available.")
        return decode_fn(gen_tokens, skip_special_tokens=True).strip()