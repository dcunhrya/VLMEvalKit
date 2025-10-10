import sys, glob
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import *
from huggingface_hub import snapshot_download, hf_hub_download
from accelerate import dispatch_model, load_checkpoint_and_dispatch


class OpenFlamingo(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 name,
                 mpt_pth=None,
                 ckpt_pth=None,
                 dtype="bf16",
                 device_map="auto",
                 device="cuda",
                 offload_folder="/home/ubuntu/hf_offload",
                 max_memory={"cuda:0": "38GiB", "cpu": "64GiB"},
                 **kwargs):

        if mpt_pth is None:
            raise ValueError(
                'Please set `mpt_pth` to the directory of MPT-7B, which is cloned from here: '
                'https://huggingface.co/mosaicml/mpt-7b. '
            )
            raise ValueError
        if ckpt_pth is None:
            raise ValueError(
                'Please set `ckpt_pth` to the openflamingo ckpt, which is the `checkpoint.pt` file downloaded '
                'from: https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b/tree/main. '
            )
        else:
            if 'med-flamingo' in ckpt_pth:
                ckpt_pth = osp.join(ckpt_pth, '/tree/main/model.pt')
            elif osp.exists(ckpt_pth):
                if ckpt_pth.endswith('checkpoint.pt'):
                    pass
                elif osp.isdir(ckpt_pth):
                    ckpt_pth = osp.join(ckpt_pth, 'checkpoint.pt')
                    if not osp.exists(ckpt_pth):
                        raise ValueError(f'File {ckpt_pth} does not exist. ')
            elif splitlen(ckpt_pth, '/') == 2:
                cache_path = get_cache_path(ckpt_pth)
                if cache_path is None:
                    snapshot_download(ckpt_pth)
                cache_path = get_cache_path(ckpt_pth)
                if cache_path is None:
                    raise ValueError(f'Directory {cache_path} does not exist. ')
                else:
                    ckpt_pth = osp.join(cache_path, 'checkpoint.pt')

        self.name = name
        assert name in ['v2']
        self.mpt_pth = mpt_pth
        try:
            from open_flamingo import create_model_and_transforms
        except Exception as e:
            logging.critical('Please first install open_flamingo to use OpenFlamingo')
            raise e
        
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype in ("fp16", "float16"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.makedirs(offload_folder, exist_ok=True)

        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self.device = torch.device(device)
        else:
            self.device = device

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path='ViT-L-14',
            clip_vision_encoder_pretrained='openai',
            lang_encoder_path=mpt_pth,
            tokenizer_path=mpt_pth,
            cross_attn_every_n_layers=4)
        # print('-------- MODEL LOADED IN ----------')
        # print(f'CKPT PATH IS {ckpt_pth}')
        try:
            ckpt = torch.load(ckpt_pth)
        except:
            if "med-flamingo" in ckpt_pth:
                hf_hub_download(repo_id="med-flamingo/med-flamingo", filename="model.pt")
        torch.cuda.empty_cache()
        model = model.to(dtype=torch_dtype).cuda()
        # dispatch_kwargs = dict(device_map=device_map)
        # if max_memory:
        #     dispatch_kwargs["max_memory"] = max_memory
        # model = dispatch_model(model,device_map="auto",offload_dir=offload_folder, **dispatch_kwargs)
        self.model = model.to(device=self.device, dtype=torch_dtype)
        torch.cuda.empty_cache()
        self.model = self.model.eval()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.tokenizer.eos_token = "<|endofchunk|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_proc = image_processor

        kwargs_default = dict(max_new_tokens=64, 
                              min_new_tokens=1,
                              temperature=0.0,
                              top_p=1.0,

                              num_beams=3)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        vision_x = []
        prompt = ''
        for msg in message:
            if msg['type'] == 'image':
                img = Image.open(msg['value'])
                vision_x.append(self.image_proc(img).unsqueeze(0))
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value']
        prompt += 'Answer: '
        vision_x = torch.cat(vision_x, dim=0) if len(vision_x) > 1 else vision_x[0]
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vdtype = next(self.model.vision_encoder.parameters()).dtype  # e.g., torch.bfloat16
        vision_x = vision_x.to(device="cuda", dtype=vdtype, non_blocking=True)      
        lang_x = self.tokenizer([prompt], return_tensors='pt')
        generated_text = self.model.generate(
            vision_x=vision_x.cuda(),
            lang_x=lang_x['input_ids'].cuda(),
            attention_mask=lang_x['attention_mask'].cuda(),
            **self.kwargs)
        generated_text = self.tokenizer.decode(generated_text[0])
        text = generated_text[len(prompt):].split('<|endofchunk|>')[0]
        return text
