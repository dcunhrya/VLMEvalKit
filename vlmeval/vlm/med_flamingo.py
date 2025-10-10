import os, torch
from PIL import Image
from functools import partial
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from vlmeval.vlm.base import BaseModel

try:
    from open_flamingo.src.factory import create_vision_backbone
    from open_flamingo.src.flamingo import Flamingo 
except:
    pass

# A light, VLMEvalKit-friendly wrapper for Med-Flamingo (OpenFlamingo family).
class MedFlamingo(BaseModel):
    """
    Args:
        lm_path: HF id of the language model (e.g., 'meta-llama/Llama-2-7b-hf' or Vicuna).
        clip_name: CLIP vision tower id (e.g., 'openai/clip-vit-large-patch14').
        med_ckpt: path to the Med-Flamingo checkpoint (OpenFlamingo format).
        dtype: 'bf16' | 'fp16' | 'fp32' | 'auto'
    """
    def __init__(self,
                 lm_path: str,
                 clip_name: str = "openai/clip-vit-large-patch14",
                 med_ckpt: str | None = None,
                 device: str = "cuda",
                 dtype: str = "bf16",
                 max_new_tokens: int = 128,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        if dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        elif dtype == "fp32":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        vision_encoder, image_processor = create_vision_backbone(
            clip_vision_encoder_path = "ViT-L-14",
            clip_vision_encoder_pretrained = "openai"
        )
        llama_path = "meta-llama/Llama-2-7b-hf"
        cfg = AutoConfig.from_pretrained(llama_path, trust_remote_code=True)
        with init_empty_weights():
            empty_lm = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
        flamingo = Flamingo(
            vision_encoder=vision_encoder,
            lang_encoder=empty_lm,
            cross_attn_every_n_layers=4
        )

        max_mem = {"cuda:0":"46GiB","cpu":"26GiB"}
        # Quantized load (bitsandbytes), honors device_map & max_memory
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        lm = AutoModelForCausalLM.from_pretrained(
            llama_path,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            max_memory=max_mem,
        )
        flamingo.lang_encoder = lm
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 6) (Optional) Load Med-Flamingo checkpoint on top
        # if load_med_flamingo_ckpt:
        #     ckpt = hf_hub_download(med_flamingo_repo, med_flamingo_filename)
        #     sd = torch.load(ckpt, map_location="cpu")
        #     flamingo.load_state_dict(sd, strict=False)

        # 7) Move wrapper to GPU (LM already sharded/quantized appropriately)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flamingo.to(device).eval()

        self.model.eval()

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    # ---- helpers ----
    def _pack(self, message):
        """Turn VLMEvalKit message list into (images, prompt_text)."""
        images, texts = [], []
        for it in message:
            if it.get("type") == "image":
                img = Image.open(it["value"]).convert("RGB")
                images.append(img)
            elif it.get("type") == "text":
                texts.append(str(it["value"]).strip())
        prompt_text = "\n".join(t for t in texts if t)

        # OpenFlamingo expects interleaved tokens; VLMEvalKit already puts images separately.
        # For single-turn VQA, use one <image> then the question text.
        # IMAGE_PLACEHOLDER is "<image>" in VLMEvalKit. If absent, just prepend "<image>".
        IMAGE_PLACEHOLDER = "<image-placeholder>"
        if IMAGE_PLACEHOLDER not in prompt_text and images:
            prompt_text = (IMAGE_PLACEHOLDER + "\n" + prompt_text).strip()

        return images, prompt_text

    def generate(self, message, dataset: str = "") -> str:
        images, prompt_text = self._pack(message)

        # 1) Preprocess images to CLIP format
        if not images:
            raise ValueError("MedFlamingoLocal: no image provided for a vision-language prompt.")
        vision_inputs = [self.image_processor(img) for img in images]
        # Stack to [B=1, T=images, C, H, W] or as list (OpenFlamingo accepts a list of vision inputs per sequence)
        # Keep it simple: one image per sample here.
        vision_inputs = torch.stack(vision_inputs).unsqueeze(0).to(self.device, dtype=self.torch_dtype)

        # 2) Tokenize text
        # OpenFlamingo uses the LM tokenizer directly.
        tok = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            use_cache=True
        )

        with torch.inference_mode():
            # OpenFlamingo models typically expect vision inputs via `vision_x`,
            # and language tokens via standard LM kwargs.
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_x=vision_inputs,
                **gen_kwargs
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # If the model echoes the prompt, strip it
        if text.startswith(prompt_text):
            text = text[len(prompt_text):].lstrip()

        return text.strip()