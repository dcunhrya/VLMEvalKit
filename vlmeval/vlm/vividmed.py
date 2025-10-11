import re
import torch
from PIL import Image
from pathlib import Path
import cytoolz
from jsonargparse import class_from_function
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
try:
    from peft import PeftModel
except:
    pass
from vlmeval.vlm.cogvlm import CogVlm
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoImageProcessor, AutoProcessor, AutoConfig

class VividMed(BaseModel):
    def __init__(self, device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs):
        super().__init__(**kwargs)

        adapter_dir = Path('./adapter_dir')
        self.download_adapter_assets(adapter_dir)

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

        self.device = device
        # cogvlm = CogVlm()
        # self.model = cogvlm.model
        # self.tokenizer = cogvlm.tokenizer
        tok = MMMMTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
        cfg = AutoConfig.from_pretrained("THUDM/cogvlm-chat-hf", trust_remote_code=True)
        # Apply vividmed-style overrides
        vision_updates = {
            "patch_size": 16,
            "pos_embed_shape": [8, 32, 32],
            "pt_pos_embed_shape": [35, 35],
        }
        if hasattr(cfg, "vision_config"):
            vc = cfg.vision_config
            if isinstance(vc, dict):
                vc.update(vision_updates)
                cfg.vision_config = vc
            else:
                # attribute-style config
                for k, v in vision_updates.items():
                    setattr(vc, k, v)
        else:
            # some repos keep it top-level
            for k, v in vision_updates.items():
                setattr(cfg, k, v)
        self.model = AutoModelForCausalLM.from_pretrained("THUDM/cogvlm-chat-hf", 
                config=cfg, 
                trust_remote_code=True, 
                torch_dtype="auto",)
        # try:
        #     self.processor = AutoProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
        # except:
        #     # self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', trust_remote_code=True, use_fast=True)
        #     # self.image_processor = AutoImageProcessor.from_pretrained("zai-org/cogvlm-chat-hf", trust_remote_code=True)
        #     self.processor = None

        self.model.resize_token_embeddings(len(tok))
        self.model = PeftModel.from_pretrained(self.model, str(adapter_dir), is_trainable=False)


        # Good practice to set pad/eos tokens
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"

        self.model.eval()
        

    # 2) Download matching adapter files
    def download_adapter_assets(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://api.github.com/repos/function2-llx/MMMM/releases/tags/s2"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        assets = data.get("assets", [])
        assets = [(a["name"], a["browser_download_url"]) for a in assets]
        if not assets:
            raise RuntimeError(f"No assets found")

        wanted = {"adapter_config.json", "adapter_model.safetensors"}
        picked = []
        for name, url in assets:
            if name in wanted or re.match(r"^adapter_[\w\-.]+$", name):
                print(f"Downloading: {name}")
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                (out_dir / name).write_bytes(resp.content)
                picked.append(name)

        if not picked:
            raise RuntimeError(
                f"No adapter files found in release. Available assets: {[n for n,_ in assets]}"
            )
        return out_dir
    
    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path).convert('RGB')
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
        response = response.split(self.end_text_token)[0].strip()
        return response
    
class MMMMTokenizer(LlamaTokenizer):
    sys_token = '<sys>'
    sys_token_id: int
    usr_token = '<usr>'
    usr_token_id: int
    # enable grounding
    grd_token = '<grd>'
    grd_token_id: int
    # disable grounding
    ngrd_token = '<ngrd>'
    ngrd_token_id: int
    # begin of phrase
    bop_token = '<p>'
    bop_token_id: int
    eop_token = '</p>'
    eop_token_id: int
    # begin of negative phrase, not actually used by model
    bonp_token = '<np>'
    bonp_token_id: int
    eonp_token = '</np>'
    eonp_token_id: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_vocab_size = self.vocab_size

        special_token_names = [*map(
            lambda name: f'{name}_token',
            ['sys', 'usr', 'grd', 'ngrd', 'bop', 'eop', 'bonp', 'eonp'],
        )]
        special_tokens = [*map(self.__getattribute__, special_token_names)]
        self.add_tokens(special_tokens, special_tokens=True)
        special_token_ids = self.convert_tokens_to_ids(special_tokens)
        for token_name, special_token_id in zip(special_token_names, special_token_ids):
            setattr(self, f'{token_name}_id', special_token_id)

    @classmethod
    def build(cls, hf_model_path, use_seg_token: bool = False, share_seg_token: bool = True):
        # no type hint (https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/tokenization_utils_base.py#L1827)
        # will cause jsonargparse fail (https://github.com/omni-us/jsonargparse/issues/454).
        return cls.from_pretrained(
            hf_model_path, use_seg_token=use_seg_token, share_seg_token=share_seg_token,
        )

    def _parse_targets(self, token_ids: list[int]) -> list[str] | None:
        ret = []
        last_bop: int | None = None
        for i, token_id in enumerate(token_ids):
            match token_id:
                case self.bop_token_id:
                    if last_bop is not None:
                        return None
                    last_bop = i
                case self.eop_token_id:
                    if last_bop is None:
                        return None
                    ret.append(self.decode(token_ids[last_bop + 1:i - 1]))
                    last_bop = None
        return ret

    def parse_targets(self, token_ids: torch.LongTensor) -> list[list[str] | None]:
        return [
            self._parse_targets(token_ids[i].tolist())
            for i in range(token_ids.shape[0])
        ]

    def build_classes_index(self, names: set[str]):
        """This method is useful only when not self.share_seg_token"""
        self.class_to_idx = {name: i for i, name in enumerate(sorted(names))}

    @cytoolz.curry
    def wrap_name(self, name: str, pos: bool):
        if pos:
            bop_token, eop_token = self.bop_token, self.eop_token
        else:
            bop_token, eop_token = self.bonp_token, self.eonp_token
        ret = f'{bop_token} {name}{eop_token}'
        return ret

