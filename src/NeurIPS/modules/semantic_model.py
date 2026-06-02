from typing import List, Literal

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

from .encoders import fMRIEncoder

class fMRISemanticModel(ModelMixin, ConfigMixin):
    
    config_name = "config.json"
    
    @register_to_config
    def __init__(
        self,
        in_level: int = 6,
        functional_hidden_level: int = 3,
        functional_tokenizer_dims: List[int] = [64, 128, 256, 512],
        structural_hidden_level: int = 1,
        structural_tokenizer_dims: List[int] = [16, 32, 64, 128, 256, 512],
        in_dim: int = 1,
        embed_dim: int = 768,
        dropout_tokenizer: float = 0.0,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_intermediate_dim: int = 1024,
        encoder_dropout_attention: float = 0.0,
        moe_intermediate_dim: int = 512,
        num_routed_experts: int = 32,
        num_activated_experts: int = 6,
        num_shared_experts: int = 2,
        num_cls_tokens: int = 1,
        num_global_tokens: int = 1,
        mask_name: Literal["nsdgeneral", "hcp-mmp1.0-vision", "yeo17-vision", "none"] = "none"
    ):
        super().__init__()

        self.encoder = fMRIEncoder(
            in_level=in_level,
            functional_hidden_level=functional_hidden_level,
            functional_tokenizer_dims=functional_tokenizer_dims,
            structural_hidden_level=structural_hidden_level,
            structural_tokenizer_dims=structural_tokenizer_dims,
            in_dim=in_dim,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            intermediate_dim=encoder_intermediate_dim,
            moe_intermediate_dim=moe_intermediate_dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            num_shared_experts=num_shared_experts,
            dropout_tokenizer=dropout_tokenizer,
            dropout_attention=encoder_dropout_attention,
            num_cls_tokens=num_cls_tokens,
            num_global_tokens=num_global_tokens,
            mask_name=mask_name
        )
        
        hidden_dim = self.encoder.num_cls_tokens * embed_dim
        self.image_decoder = nn.Linear(hidden_dim, 768 * 257)
        self.text_decoder = nn.Linear(hidden_dim, 768 * 77)
    
    def forward(
        self,
        fmri_L: torch.Tensor,
        fmri_R: torch.Tensor,
        smri_L: torch.Tensor,
        smri_R: torch.Tensor,
    ):
        bs = fmri_L.shape[0]
        hidden_state = self.encoder(fmri_L, fmri_R, smri_L, smri_R)
        hidden_state = hidden_state[:, :self.encoder.num_cls_tokens, :]
        hidden_state = hidden_state.reshape(bs, -1)
        image_emb = self.image_decoder(hidden_state)
        text_emb = self.text_decoder(hidden_state)
        return image_emb, text_emb