from typing import Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.autoencoders.vae import Decoder

class fMRIPerceptionModel(ModelMixin, ConfigMixin):
    
    config_name = "config.json"
    mlp_hidden_shape = (64, 16, 16)
    vae_hidden_shape = (4, 64, 64)
    
    @register_to_config
    def __init__(
        self,
        input_dim: int = 9488,
        hidden_dim: int = 4096,
        num_blocks: int = 4,
        dropout_input: float = 0.5,
        dropout_block: float = 0.25,
        upsampler_channels: Tuple[int] = (64, 128, 256),
        upsampler_layer_per_block: int = 1,
        use_maps_proj: bool = True,
        maps_proj_dim: int = 512
    ):
        super().__init__()
        
        self.proj1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_input)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_block)
            ) for _ in range(num_blocks)
        ])
        
        vae_dim = self.mlp_hidden_shape[0] * self.mlp_hidden_shape[1] * self.mlp_hidden_shape[2]
        self.proj2 = nn.Linear(hidden_dim, vae_dim)
        self.norm = nn.GroupNorm(1, 64)
        
        self.upsampler = Decoder(
            in_channels=self.mlp_hidden_shape[0],
            out_channels=self.vae_hidden_shape[0],
            up_block_types=("UpDecoderBlock2D", ) * 3,
            block_out_channels=upsampler_channels,
            layers_per_block=upsampler_layer_per_block
        )
        
        if use_maps_proj:
            self.maps_proj = nn.Sequential(
                nn.Conv2d(self.mlp_hidden_shape[0], maps_proj_dim, 1, bias=False),
                nn.GroupNorm(1, maps_proj_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(maps_proj_dim, maps_proj_dim, 1, bias=False),
                nn.GroupNorm(1, maps_proj_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(maps_proj_dim, maps_proj_dim, 1, bias=True)
            )
        else:
            self.maps_proj = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        bs = x.shape[0]
        x = self.proj1(x)
        
        res = x
        for block in self.blocks:
            x = block(x)
            x += res
            res = x
        
        x = self.proj2(x)
        x = x.view(bs, *self.mlp_hidden_shape).contiguous()
        x = self.norm(x)
        
        if self.maps_proj is not None:
            x_map = self.maps_proj(x)
            x_map = x_map.flatten(2).permute(0, 2, 1)
        else:
            x_map = None
        
        x = self.upsampler(x)
        
        return x, x_map