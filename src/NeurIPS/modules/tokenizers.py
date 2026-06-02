from typing import List, Literal

import torch
import torch.nn as nn

from .sphere.conv import SphereConv, SphereAveragePooling

class HemiSphereTokenizer(nn.Module):
    
    def __init__(
        self,
        in_level: int,
        out_level: int,
        dims: List[int],
        in_dim: int,
        out_dim: int,
        dropout: float,
        hemi: Literal["L", "R"],
        mask_name: Literal["nsdgeneral", "hcp-mmp1.0-vision", "yeo17-vision", "none"] = "none"
    ) -> None:
        super().__init__()
        
        levels = list(range(in_level, out_level, -1))
        num_levels = len(levels)
        assert len(dims) == num_levels + 1
        
        self.proj_in = nn.Linear(in_dim, dims[0])
        
        self.blocks = nn.ModuleList()
        for i in range(num_levels):
            self.blocks.append(nn.Sequential(
                SphereConv(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    bias=True,
                    hemi=hemi,
                    level=levels[i],
                    mask_name=mask_name
                ),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
                SphereAveragePooling(
                    hemi=hemi, 
                    in_level=levels[i], 
                    mask_name=mask_name
                )
            ))
        
        self.proj_out = nn.Linear(dims[-1], out_dim)
        
        self.in_num_voxels = self.blocks[0][0].in_voxels
        self.out_num_voxels = self.blocks[-1][-1].out_voxels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        return x

class FunctionalSphereTokenizer(nn.Module):
    
    def __init__(
        self,
        in_level: int,
        out_level: int,
        dims: List[int],
        in_dim: int,
        embed_dim: int,
        dropout: float,
        num_cls_tokens: int = 1,
        num_global_tokens: int = 1,
        mask_name: Literal["nsdgeneral", "hcp-mmp1.0-vision", "yeo17-vision", "none"] = "none"
    ) -> None:
        super().__init__()
        
        self.embedder_L = HemiSphereTokenizer(
            in_level=in_level,
            out_level=out_level,
            dims=dims,
            in_dim=in_dim,
            out_dim=embed_dim,
            dropout=dropout,
            hemi="L",
            mask_name=mask_name
        )
        
        self.embedder_R = HemiSphereTokenizer(
            in_level=in_level,
            out_level=out_level,
            dims=dims,
            in_dim=in_dim,
            out_dim=embed_dim,
            dropout=dropout,
            hemi="R",
            mask_name=mask_name
        )
        
        self.class_embedding = nn.Parameter(torch.randn(1, num_cls_tokens, embed_dim))
        gloabl_proj_in_dim = self.embedder_L.in_num_voxels + self.embedder_R.in_num_voxels
        global_proj_out_dim = num_global_tokens * embed_dim
        self.global_proj = nn.Linear(gloabl_proj_in_dim, global_proj_out_dim)
        num_positions = num_cls_tokens + self.embedder_L.out_num_voxels + self.embedder_R.out_num_voxels + num_global_tokens
        self.position_embedding = nn.Embedding(num_positions, embed_dim)
        self.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)
        
        self.num_tokens = num_positions
        self.num_global_tokens = num_global_tokens
        self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
    
    def forward(self, fmri_L: torch.Tensor, fmri_R: torch.Tensor) -> torch.Tensor:
        
        batch_size = fmri_L.shape[0]
        
        global_tokens = self.global_proj(torch.cat([fmri_L, fmri_R], dim=1).squeeze(-1))
        global_tokens = global_tokens.view(batch_size, self.num_global_tokens, self.embed_dim)
        
        fmri_L = self.embedder_L(fmri_L)
        fmri_R = self.embedder_R(fmri_R)

        class_embeds = self.class_embedding.repeat(batch_size, 1, 1)
        
        embeddings = torch.cat([class_embeds, global_tokens, fmri_L, fmri_R], dim=1)
            
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings
   
class StructuralSphereTokenizer(nn.Module):
    
    def __init__(
        self,
        in_level: int,
        out_level: int,
        dims: List[int],
        in_dim: int,
        embed_dim: int,
        dropout: float,
        mask_name: Literal["nsdgeneral", "hcp-mmp1.0-vision", "yeo17-vision", "none"] = "none"
    ) -> None:
        super().__init__()
        
        self.embedder_L = HemiSphereTokenizer(
            in_level=in_level,
            out_level=out_level,
            dims=dims,
            in_dim=in_dim,
            out_dim=embed_dim,
            dropout=dropout,
            hemi="L",
            mask_name=mask_name
        )
        
        self.embedder_R = HemiSphereTokenizer(
            in_level=in_level,
            out_level=out_level,
            dims=dims,
            in_dim=in_dim,
            out_dim=embed_dim,
            dropout=dropout,
            hemi="R",
            mask_name=mask_name
        )
        
        num_out_voxels = self.embedder_L.out_num_voxels + self.embedder_R.out_num_voxels
        self.proj_out = nn.Linear(num_out_voxels * embed_dim, embed_dim)
    
    def forward(self, smri_L: torch.Tensor, smri_R: torch.Tensor) -> torch.Tensor:

        smri_L = self.embedder_L(smri_L)
        smri_R = self.embedder_R(smri_R)
        
        embedding = torch.cat([smri_L.flatten(1), smri_R.flatten(1)], dim=1)
        embedding = self.proj_out(embedding)
        
        return embedding