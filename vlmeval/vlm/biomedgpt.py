# vlmeval/vlm/biomedgpt_local.py
import os, torch
from .base import BaseModel
# from llava_med import SafeKeywordsStoppingCriteria
from PIL import Image


class BioMedGPT(BaseModel):
    SUPPORTED_ARCH = ("biomedgpt",)

    def __init__(self, path: str, device="cuda", dtype="bf16", **kw):
        try:
            from transformers import OFATokenizer, OFAModel
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token, process_images
            from llava.conversation import conv_templates
        except ModuleNotFoundError:
            raise('No module named biomedgpt')
        super().__init__(**kw)
        # keep integrations from pulling in conflicting deps
        os.environ.setdefault("TRANSFORMERS_NO_ADAPTERS", "1")
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        self.model_id = path
        self.tokenizer = OFATokenizer.from_pretrained(path, trust_remote_code=True)
        torch_dtype = (torch.bfloat16 if dtype in ("bf16", "auto") and torch.cuda.is_available()
                       else torch.float16 if dtype == "fp16" else torch.float32)
        self.model = OFAModel.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch_dtype
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.torch_dtype = torch_dtype

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
        conv = None
        for key in ("llava_v1", "llava_v1.5", "vicuna_v1", "chatml"):
            if key in conv_templates:
                conv = conv_templates[key].copy()
                break
        if conv is None:
            # Minimal manual prompt if your repo lacks templates
            prompt = (("<image>\n" + user_text) if user_text else "<image>") + "\n### ASSISTANT:"
        else:
            user_payload = ("<image>\n" + user_text) if user_text else "<image>"
            conv.append_message(conv.roles[0], user_payload)
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
            "prompt_text": prompt,  # for optional echo-stripping
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
            temperature=max(self.temperature, 1e-5),
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