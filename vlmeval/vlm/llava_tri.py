# vlmeval/vlm/llava_tri_local.py
import torch
import sys, os
from PIL import Image
from vlmeval.vlm.base import BaseModel

class LLaVATriPretrained(BaseModel):
    def __init__(self, path="yunfeixie/LLaVA-Tri-Pretrained",
                 device="cuda", dtype=torch.bfloat16, max_new_tokens=128, temperature=0.0, top_p=1.0, **kwargs):
        try:
            from vlmeval.vlm.llava_med import SafeKeywordsStoppingCriteria
            from llava.model.builder import load_pretrained_model 
            from llava.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX, process_images
            from llava.conversation import conv_templates
        except ModuleNotFoundError:
            raise('LLaVA not installed')
        super().__init__(**kwargs)
        self.model_id = path
        self.device = device
        self.torch_dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if dtype == "auto":
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif isinstance(dtype, str):
            if dtype.lower() in ("fp16", "float16"):
                self.torch_dtype = torch.float16
            elif dtype.lower() in ("bf16", "bfloat16"):
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = dtype

        # crucial: this loader comes from a LLaVA fork that knows llava_llama
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=path, model_base=None, model_name="llava_llama", device=device
        )
        if getattr(self, "tokenizer", None):
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(device=device, dtype=self.torch_dtype).eval()

    def _prepare_inputs_llavamed(self, message):
        # 1) Collect valid PIL images and text
        images, texts, bad = [], [], []
        for it in message:
            if it.get("type") == "image":
                p = it.get("value")
                if isinstance(p, str) and os.path.exists(p):
                    try:
                        images.append(Image.open(p).convert("RGB"))
                    except Exception as e:
                        bad.append((p, f"open_failed:{e}"))
                else:
                    bad.append((p, "missing_or_invalid_path"))
            elif it.get("type") == "text":
                texts.append(str(it["value"]))
        user_text = "\n".join(t.strip() for t in texts if t).strip()

        if not images:
            raise ValueError(f"LLaVA-Med: no usable image(s). Bad: {bad[:3]}")

        # 2) Conversation template (robust fallback)
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], "<image>\n" + user_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 3) Tokenize prompt (must insert IMAGE_TOKEN_INDEX)
        raw = tokenizer_image_token(
            prompt,
            self.tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        )
        input_ids = raw["input_ids"] if isinstance(raw, dict) else raw
        if input_ids.dim() == 1:  # normalize to [B,T]
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        # 4) Vision pipeline
        image_tensor = process_images(images, self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(self.device, dtype=self.torch_dtype) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=self.torch_dtype)

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "prompt_text": prompt, 
        }

    def _align_inputs_with_model_llavamed(self, inputs):
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

    def _dump_devices_llavamed(self, label, inputs):
        print(f"== {label} ==")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                print(k, v.device, tuple(v.shape), v.dtype)
        try:
            print("model main device:", next(self.model.parameters()).device)
        except StopIteration:
            pass

    def generate(self, message, dataset: str = "") -> str:
        pack = self._prepare_inputs_llavamed(message)
        pack = self._align_inputs_with_model_llavamed(pack)
        input_ids = pack["input_ids"]
        image_tensor = pack["images"]
        prompt = pack["prompt_text"]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        stop_strs = ["</s>", "###", "ASSISTANT:"]
        stopper = SafeKeywordsStoppingCriteria(stop_strs, self.tokenizer, input_ids, device=input_ids.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=0.0, #max(self.temperature, 1e-5),
            top_p=self.top_p,
            use_cache=True,
            stopping_criteria=[stopper],
        )

        with torch.inference_mode():
            out_ids = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                **gen_kwargs
            )

        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        # Optional: strip prompt echo
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return text