import os
import torch
from typing import List, Dict, Any
from PIL import Image

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
try:
    from LLavaMed.llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
    from LLavaMed.llava.mm_utils import process_images
except Exception:
    LlavaQwen2ForCausalLM = None
try:
    from HuatuoGPTVision.cli import HuatuoChatbot
except:
    pass

from .base import BaseModel

class HuatuoVision7b(BaseModel):
    def __init__(self,
                 path: str = "FreedomIntelligence/HuatuoGPT-Vision-7B",
                 device: str = "cuda",
                 dtype: str = "fp16",
                 max_new_tokens: int = 128,
                 min_new_tokens: int = 1,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.bot = HuatuoChatbot(path)
        self.bot.gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'temperature': temperature,
            'top_p' : top_p

        }
        self.torch_dtype = torch.float16 if dtype.lower() in {"fp16","float16"} else torch.bfloat16

    def generate(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        outputs = self.bot.inference(prompt, [image_path])
        return outputs[0]


class HuatuoVision34b(BaseModel):
    def __init__(self,
                 path: str = "FreedomIntelligence/HuatuoGPT-Vision-34B",
                 device: str = "cuda",
                 dtype: str = "bf16",
                 max_new_tokens: int = 128,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p

        repo_dir = snapshot_download(path, local_files_only=False)

        rel_tower = "vit/clip_vit_large_patch14_336"
        abs_tower = os.path.join(repo_dir, rel_tower)
        if not os.path.isdir(abs_tower):
            raise FileNotFoundError(f"Expected vision tower folder not found: {abs_tower}")

        cfg = AutoConfig.from_pretrained(repo_dir, trust_remote_code=True)
        setattr(cfg, "mm_vision_tower", abs_tower)
        setattr(cfg, "vision_tower", abs_tower)
        if not hasattr(cfg, "use_sliding_window"):
            cfg.use_sliding_window = False              # default in Qwen2Config
        if not hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "sdpa"          
        if not hasattr(cfg, "sliding_window"):
            cfg.sliding_window = None 

        per_gpu = "44GiB"
        max_memory = {i: per_gpu for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "120GiB"  

        self.model, _ = LlavaQwen2ForCausalLM.from_pretrained(
            repo_dir,
            config=cfg,
            init_vision_encoder_from_ckpt=True,
            output_loading_info=True,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            max_memory=max_memory,
        )

        # 5) Tokenizer + generation defaults
        self.tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        # 6) Ensure tower is loaded/moved; get an image processor
        vt = self.model.get_vision_tower()
        if vt is None:
            raise RuntimeError("Vision tower is None; check cfg.mm_vision_tower/vision_tower")
        if not getattr(vt, "is_loaded", False):
            vt.load_model()
            if getattr(vt, "vision_tower", None) is None:
                try:
                    vt.vision_tower = vt.vision_tower.from_pretrained(abs_tower, trust_remote_code=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
        # vt.to(dtype=self.torch_dtype, device=device)

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(abs_tower, trust_remote_code=True)
        except Exception:
            self.image_processor = getattr(vt, "image_processor", None) or AutoImageProcessor.from_pretrained(repo_dir, trust_remote_code=True)

        # self.model = self.model.to(device=device, dtype=self.torch_dtype).eval()
        self.model.eval()

    def generate(self, message, dataset: str = "") -> str:
        """
        Huatuo (LLaVA-Qwen2) compatible generation:
        - prompt is plain text ending with 'ASSISTANT:'
        - images are provided via `images=`
        - inputs/attention_mask are lists of 1D tensors
        - returns a non-empty string (falls back to 'INVALID')
        """

        # -----------------------
        # 1) Gather text + image
        # -----------------------
        images, texts = [], []
        for m in message:
            t = m.get("type")
            if t == "image":
                images.append(Image.open(m["value"]).convert("RGB"))
            elif t == "text":
                texts.append(str(m["value"]))

        question_text = "\n".join(s.strip() for s in texts if s).strip()
        # IMPORTANT: no <image> token in text for Huatuo; image comes through `images=`
        # prompt = (
        #     f"{question_text}\n"
        #     # "Answer with a single one of the option's letter from the given options directly.\n"
        #     # "ASSISTANT:"
        # )
        prompt = (
            "USER: You are given an image and a question. Answer concisely.\n"
            f"Question: {question_text}\n"
            "ASSISTANT:"
        )

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        enc = self.tokenizer(prompt, return_tensors="pt", padding=True, add_special_tokens=True)

        ids  = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(ids))

        # Huatuo forks often expect lists of 1D tensors
        # inputs_list = [ids[0].to(self.device)]
        # attn_list = [attn[0].to(self.device)]
        dev = next(self.model.parameters()).device
        inputs_list = ids.to(dev)
        attn_list = attn.to(dev)
        # in_len = inputs_list[0].shape[0]

        # -------------------------------------
        # 3) Process images to model device/dt
        # -------------------------------------
        dev = next(self.model.parameters()).device
        dt  = getattr(self, "torch_dtype", (torch.bfloat16 if dev.type == "cuda" else torch.float32))
        pix = process_images(images, self.image_processor, self.model.config)
        if isinstance(pix, list):
            pix = [p.to(dev, dtype=dt) for p in pix]
        else:
            pix = pix.to(dev, dtype=dt)

        # -----------------------------
        # 4) First deterministic attempt
        # -----------------------------
        gen_kwargs = dict(
            max_new_tokens=128,
            # min_new_tokens=2,            # coax at least some emission
            do_sample=False,             # deterministic
            temperature=0.0,            # ignored with do_sample=False
            top_p=1.0,
            # use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
            attention_mask=attn_list
            # return_dict_in_generate=True,
            # intentionally DO NOT set eos_token_id first try
        )


        with torch.inference_mode():
            out = self.model.generate(
                inputs=inputs_list,
                images=pix,
                **gen_kwargs
            )

        # ---------- 5) decode only the continuation ----------
        seq = out.sequences if hasattr(out, "sequences") else out
        gen_tokens = seq[0]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if not text:
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
        return text or "INVALID"