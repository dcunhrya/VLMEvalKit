import torch
import inspect
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from vlmeval.vlm.base import BaseModel
import os

try: 
    from LLavaMed.llava.mm_utils import KeywordsStoppingCriteria as OrigStop
    from LLavaMed.llava.model.builder import load_pretrained_model
    from LLavaMed.llava.mm_utils import tokenizer_image_token, process_images
    from LLavaMed.llava.conversation import conv_templates
    from LLavaMed.llava.constants import IMAGE_TOKEN_INDEX

    class SafeKeywordsStoppingCriteria(OrigStop):
        def __init__(self, keywords, tokenizer, input_ids, device=None):
            super().__init__(keywords, tokenizer, input_ids)
            # normalize device now; fall back to input_ids.device
            self.device = device or (input_ids.device if hasattr(input_ids, "device") else torch.device("cpu"))
            # move all keyword tensors to the target device
            self.keyword_ids = [kid.to(self.device) for kid in self.keyword_ids]

        def call_for_batch(self, output_ids, scores):
            # ensure output and keywords are on same device (guard against odd HF behaviors)
            if output_ids.device != self.device:
                # move keywords instead of moving output_ids (cheaper)
                self.keyword_ids = [kid.to(output_ids.device) for kid in self.keyword_ids]
                self.device = output_ids.device

            gen_len = output_ids.shape[1]
            for keyword_id in self.keyword_ids:
                stop_len = keyword_id.shape[0]
                if gen_len >= stop_len:
                    # safe slicing even when lengths differ
                    if (output_ids[0, gen_len - stop_len : gen_len] == keyword_id).all():
                        return True
            return False

except:
    pass

class LLaVAMedLocal(BaseModel):
    def __init__(
        self,
        path: str = "microsoft/llava-med-v1.5-mistral-7b",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs
    ):
        
        super().__init__(**kwargs)

        # ---- Resolve device / dtype safely ----
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
        # Prefer bf16 on newer GPUs if requested but not available, fall back to fp16
        if self.device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        self.torch_dtype = dtype
        self.dtype = self.torch_dtype

        self.model_path = path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        last_err = None
        model_name = "llava-med-v1.5-mistral-7b"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path, None, model_name, device=str(self.device)
        )

        # self.tokenizer = AutoTokenizer("microsoft/llava-med-v1.5-mistral-7b")
        # self.model = AutoModelForCausalLM("microsoft/llava-med-v1.5-mistral-7b", torch_dtype="auto")
        # self.image_processor = AutoImageProcessor("microsoft/llava-med-v1.5-mistral-7b")
        # after you load self.tokenizer, self.model
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        # most VLMs expect right padding
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"
        if self.model is None:
            raise RuntimeError(f"Failed to load LLaVA-Med from '{self.model_path}': {last_err}")

        # Move model to device/dtype and eval
        self.model.to(device=self.device)
        self.model.eval()

        # ---- Tokenizer sanity: make sure pad/eos exist ----
        if getattr(self.tokenizer, "pad_token", None) is None:
            # Many LLaVA tokenizers use eos as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Image token index & conversation template ----
        # IMAGE_TOKEN_INDEX is typically set in llava.constants (e.g., 32000)
        self.image_token_index = IMAGE_TOKEN_INDEX
        # Choose a conv template present in your repo; fall back gracefully
        # self.conv_id = "llava_v1" if "llava_v1" in conv_templates else list(conv_templates.keys())[0]

        # Predefine stop strings for generation
        # self.stop_strs = ["</s>", "###", "ASSISTANT:"]

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

        if not user_text:
            raise ValueError("LLaVA-Med: empty message (no image, no text).")
        # prompt = user_text + "\n</s> ASSISTANT:"
        prompt = user_text

        # 2) Conversation template (robust fallback)
        conv = "vicuna_v1"
        conv = conv_templates[conv].copy()
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
            image_tensor = [t.to(self.device) for t in image_tensor]
            # image_tensor = [t.to(self.device, dtype=self.torch_dtype) for t in image_tensor]
        else:
            # image_tensor = image_tensor.to(self.device, dtype=self.torch_dtype)
            image_tensor = image_tensor.to(self.device)

        image_sizes = [img.size for img in images]

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "prompt_text": prompt,  # for optional echo-stripping
            # "image_sizes": image_sizes,
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
        # image_sizes = pack["image_sizes"]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        stop_strs = ["</s>", "###", "ASSISTANT:"]
        stopper = SafeKeywordsStoppingCriteria(stop_strs, self.tokenizer, input_ids, device=input_ids.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=2,
            do_sample=(self.temperature > 0),
            temperature=max(self.temperature, 0.2),
            top_p=self.top_p,
            use_cache=True,
            # stopping_criteria=[stopper],
        )

        with torch.inference_mode():
            out_ids = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                # image_sizes=image_sizes
                **gen_kwargs
            )

        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        return text
       
