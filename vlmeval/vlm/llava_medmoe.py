import torch
import inspect
from PIL import Image
from transformers import (
    AutoProcessor, AutoTokenizer, AutoImageProcessor,
    AutoModelForCausalLM, AutoModelForVision2Seq,
    PretrainedConfig,
)
try:
    from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from moellava.conversation import conv_templates, SeparatorStyle
    from moellava.model.builder import load_pretrained_model
    from moellava.utils import disable_torch_init
    from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
except:
    pass
from huggingface_hub import snapshot_download
from vlmeval.vlm.base import BaseModel
import os


class LLaVAPhi(BaseModel):
    def __init__(
        self,
        path: str = "JsST/Med-MoE-stage3-llavaphi-2.7b-medmoe",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

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

        self.model_path = path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        last_err = None
        repo_id = "JsST/Med-MoE"
        subfolder = "stage3/llavaphi-2.7b-medmoe"

        local_root = snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])
        local_path = os.path.join(local_root, subfolder)
        self.model_name = "llava-phi-2.7b"
        self.tokenizer, self.model, processor, _ = load_pretrained_model(
            local_path, None, self.model_name, load_8bit=False, load_4bit=False, device=device
        )
        self.image_processor = processor['image']

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
        self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()

        # ---- Tokenizer sanity: make sure pad/eos exist ----
        if getattr(self.tokenizer, "pad_token", None) is None:
            # Many LLaVA tokenizers use eos as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Image token index & conversation template ----
        # IMAGE_TOKEN_INDEX is typically set in llava.constants (e.g., 32000)
        self.image_token_index = IMAGE_TOKEN_INDEX
        # Choose a conv template present in your repo; fall back gracefully
        self.conv_id = "llava_v1" if "llava_v1" in conv_templates else list(conv_templates.keys())[0]

        # Predefine stop strings for generation
        self.stop_strs = ["</s>", "###", "ASSISTANT:"]

    def _prepare_inputs_llavamed(self, message):
        prompt_text, img_val = self.message_to_promptimg(message)
        # 1) Collect valid PIL images and text
        # images, texts, bad = [], [], []
        # for it in message:
        #     if it.get("type") == "image":
        #         p = it.get("value")
        #         if isinstance(p, str) and os.path.exists(p):
        #             try:
        #                 images.append(Image.open(p).convert("RGB"))
        #             except Exception as e:
        #                 bad.append((p, f"open_failed:{e}"))
        #         else:
        #             bad.append((p, "missing_or_invalid_path"))
        #     elif it.get("type") == "text":
        #         texts.append(str(it["value"]))
        # user_text = "\n".join(t.strip() for t in texts if t).strip()

        # if not images:
        #     raise ValueError(f"LLaVA-Med: no usable image(s). Bad: {bad[:3]}")

        user_text = prompt_text.strip()
        if not user_text:
            raise ValueError("LLaVA-Med: empty message (no image, no text).")
        # prompt = user_text + "\n### ASSISTANT:"
        conv_mode = "llava_v1"

        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        user_payload = ("<image>\n" + user_text) if user_text else "<image>"
        conv.append_message(conv.roles[0], user_payload)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


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
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        image = Image.open(img_val).convert('RGB')
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(self.model.device, dtype=torch.float16)
        # image_tensor = process_images(
        #     image,
        #     self.image_processor,
        #     self.model.config
        # ).to(self.model.device, dtype=torch.float16)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(self.device, dtype=self.torch_dtype) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=self.torch_dtype)

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "stop": stopping_criteria
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

    def generate(self, message, dataset: str = "") -> str:
        pack = self._prepare_inputs_llavamed(message)
        pack = self._align_inputs_with_model_llavamed(pack)
        input_ids = pack["input_ids"]
        image_tensor = pack["images"]
        stop_criteria = pack["stop"]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=32,
            do_sample=(self.temperature > 0),
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            use_cache=True,
            return_dict_in_generate=True,
            stopping_criteria=[stop_criteria],
        )

        with torch.inference_mode():
            out_ids = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                **gen_kwargs
            )
        seq = out_ids.sequences if hasattr(out_ids, "sequences") else out_ids

        gen_tokens = seq[0, input_ids.shape[1]:]      
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


class LLaVAStableLM(BaseModel):
    def __init__(
        self,
        path: str = "JsST/Med-MoE-stage3-llavastablelm-1.6b-medmoe",
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

        # ---- Load via LLaVA-Med's native loader (NOT HF AutoModel) ----
        last_err = None
        # repo_id = self.model_path
        # subfolder = 'stage3/llavaphi-2.7b-medmoe'
        # self.tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     repo_id, subfolder=subfolder, trust_remote_code=True, torch_dtype="auto"
        # )
        # self.image_processor = AutoImageProcessor.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
        repo_id = "JsST/Med-MoE"
        subfolder = "stage3/llavastablelm-1.6b-medmoe"

        local_root = snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])
        local_path = os.path.join(local_root, subfolder)
        self.model_name = "llavastablelm-1.6b"
        self.tokenizer, self.model, processor, _ = load_pretrained_model(
            local_path, None, self.model_name, load_8bit=False, load_4bit=False, device=device
        )
        self.image_processor = processor['image']

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
        self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()

        # ---- Tokenizer sanity: make sure pad/eos exist ----
        if getattr(self.tokenizer, "pad_token", None) is None:
            # Many LLaVA tokenizers use eos as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Image token index & conversation template ----
        # IMAGE_TOKEN_INDEX is typically set in llava.constants (e.g., 32000)
        self.image_token_index = IMAGE_TOKEN_INDEX
        # Choose a conv template present in your repo; fall back gracefully
        self.conv_id = "llava_v1" if "llava_v1" in conv_templates else list(conv_templates.keys())[0]

        # Predefine stop strings for generation
        self.stop_strs = ["</s>", "###", "ASSISTANT:"]

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
        conv_mode = "stablelm"

        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        conv = None
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
        image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(self.model.device, dtype=torch.float16)
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

        # stop_strs = ["</s>", "###", "ASSISTANT:"]
        stop_strs = ["<|endoftext>"]
        stopper = KeywordsStoppingCriteria(stop_strs, self.tokenizer, input_ids)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=1,
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
        text = self.tokenizer.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=False).strip()
        print(f'TEXT IS {text}')
        # Optional: strip prompt echo
        # if text.startswith(prompt):
        #     text = text[len(prompt):].lstrip()
        return text
       
