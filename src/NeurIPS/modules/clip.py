from typing import List

import torch
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
       
class CLIPWrapper:
    
    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        device: torch.device = "cpu",
        clip_dtype: torch.dtype = torch.float16,
        out_dtype: torch.dtype = torch.float32,
        clamp_embs: float = 1.5,
        norm_embs: bool = True,
        pooling: bool = False,
    ) -> None:
        super().__init__()
        
        self.device = device
        self.clip_dtype = clip_dtype
        self.out_dtype = out_dtype
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.pooling = pooling
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model)
        self.image_encoder.requires_grad_(False).eval().to(device=device, dtype=clip_dtype)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model)
        self.text_encoder.requires_grad_(False).eval().to(device=device, dtype=clip_dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x (torch.Tensor): [bs, 3, h, w], value -1~1
        Output:
            x (torch.Tensor): [bs, 257, 768] (`pooling` = `False`) or [bs, 768] (`pooling` = `True`)
        """
        x = self.preprocess(x.to(device=self.device, dtype=self.clip_dtype))
        x = self.image_encoder(x).last_hidden_state
        x = self.image_encoder.vision_model.post_layernorm(x)
        x = self.image_encoder.visual_projection(x)
        
        if self.clamp_embs:
            x = torch.clamp(x, -self.clamp_embs, self.clamp_embs)
        
        if self.norm_embs:
            x = x / torch.norm(x[:, 0], dim=-1).reshape(-1, 1, 1)
        
        if self.pooling:
            return x.to(dtype=self.out_dtype)[:, 0, :]
        
        return x.to(dtype=self.out_dtype)
    
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Input:
            prompts (List[str]): length = bs, type = str
        Output:
            x (torch.Tensor): [bs, 77, 768] (`pooling` = `False`) or [bs, 768] (`pooling` = `True`)
        """
        input_ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        encoder_output = self.text_encoder(input_ids.to(device=self.device))
        
        if self.pooling:
            x = self.text_encoder.text_projection(encoder_output.text_embeds)
            
            if self.clamp_embs:
                x = torch.clamp(x, -self.clamp_embs, self.clamp_embs)
                
            if self.norm_embs:
                x = x / torch.norm(encoder_output.text_embeds, dim=-1).reshape(-1, 1)
                
            return x.to(dtype=self.out_dtype)
                
        else:    
            x = self.text_encoder.text_projection(encoder_output.last_hidden_state)
            
            if self.clamp_embs:
                x = torch.clamp(x, -self.clamp_embs, self.clamp_embs)
                
            if self.norm_embs:
                x = x / torch.norm(encoder_output.text_embeds, dim=-1).reshape(-1, 1, 1)
                
            return x.to(dtype=self.out_dtype)