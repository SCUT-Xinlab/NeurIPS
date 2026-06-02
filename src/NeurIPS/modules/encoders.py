from typing import List

import torch
import torch.nn as nn

from .moe import ConditionalMoEBlock, MLP, SelfAttention
from .tokenizers import FunctionalSphereTokenizer, StructuralSphereTokenizer
    
class fMRIEncoder(nn.Module):
    
    def __init__(
        self,
        in_level: int = 6,
        functional_hidden_level: int = 3,
        functional_tokenizer_dims: List[int] = [64, 128, 256, 512],
        structural_hidden_level: int = 1,
        structural_tokenizer_dims: List[int] = [16, 32, 64, 128, 256, 512],
        in_dim: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        moe_intermediate_dim: int = 512,
        num_routed_experts: int = 32,
        num_activated_experts: int = 6,
        num_shared_experts: int = 2,
        dropout_tokenizer: float = 0.0,
        dropout_attention: float = 0.0,
        num_cls_tokens: int = 1,
        num_global_tokens: int = 1,
        mask_name: str = "none"
    ) -> None:
        super().__init__()
        
        self.functional_tokenizer = FunctionalSphereTokenizer(
            in_level=in_level,
            out_level=functional_hidden_level,
            dims=functional_tokenizer_dims,
            in_dim=in_dim,
            embed_dim=embed_dim,
            dropout=dropout_tokenizer,
            num_cls_tokens=num_cls_tokens,
            num_global_tokens=num_global_tokens,
            mask_name=mask_name
        )
              
        self.structural_tokenizer = StructuralSphereTokenizer(
            in_level=in_level,
            out_level=structural_hidden_level,
            dims=structural_tokenizer_dims,
            in_dim=4,
            embed_dim=embed_dim,
            dropout=dropout_tokenizer,
            mask_name=mask_name
        )
        
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ConditionalMoEBlock(
                dim=embed_dim,
                n_heads=num_heads,
                dropout=dropout_attention,
                inter_dim=intermediate_dim,
                moe_inter_dim=moe_intermediate_dim,
                n_routed_experts=num_routed_experts,
                n_activated_experts=num_activated_experts,
                n_shared_experts=num_shared_experts,
                score_func="softmax",
                route_scale=1
            ))
        
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_tokens = self.functional_tokenizer.num_tokens
        self.num_cls_tokens = self.functional_tokenizer.num_cls_tokens
        
        self.functional_tokenizer.apply(self._init_weights_)
        self.structural_tokenizer.apply(self._init_weights_)
        self.blocks.apply(self._init_weights_transformer_)
    
    def _init_weights_(self, module: nn.Module):
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _init_weights_transformer_(self, module: nn.Module):
        
        if isinstance(module, SelfAttention):
            in_proj_std = (module.embed_dim ** -0.5) * ((2 * self.depth) ** -0.5)
            out_proj_std = module.embed_dim ** -0.5
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
    
    def forward(
        self,
        fmri_L: torch.Tensor,
        fmri_R: torch.Tensor,
        smri_L: torch.Tensor,
        smri_R: torch.Tensor
    ) -> torch.Tensor:
        
        subj_tokens = self.structural_tokenizer(smri_L, smri_R)
        hidden_state = self.functional_tokenizer(fmri_L, fmri_R)
        
        for block in self.blocks:
            hidden_state = block(hidden_state, subj_tokens)
        
        return hidden_state