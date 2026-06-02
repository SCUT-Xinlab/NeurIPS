from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from .utils import get_masked_neighbors_same_level, get_masked_neighbors_downsample

class SphereConv(nn.Module):
    """
    Input:
        x (torch.Tensor): [bs, n, dim1]
    Output:
        x (torch.Tensor): [bs, n, dim2]
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bias: bool,
        *,
        hemi: Literal["L", "R"],
        level: int,
        mask_name: Literal["nsdgeneral", "hcp-mmp1.0-vision", "yeo17-vision", "none"] = "none"
    ) -> None:
        super().__init__()
        
        masked_neighbors = get_masked_neighbors_same_level(level, mask_name, hemi)
        self.mask = masked_neighbors.mask            # [N]
        self.neighbors = masked_neighbors.neighbors  # [n, k]
        self.pad = masked_neighbors.pad              # [n, k]
        
        self.k = 7
        self.in_voxels = self.neighbors.shape[0]
        self.in_voxels_full = self.mask.shape[0]
        self.in_channels = in_channels
        self.out_voxels = self.neighbors.shape[0]
        self.out_voxels_full = self.mask.shape[0]
        self.out_channels = out_channels
        
        self.conv = nn.Linear(in_channels * self.k, out_channels, bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_pad = torch.zeros((x.shape[0], self.in_voxels_full, x.shape[2]), device=x.device, dtype=x.dtype)
        x_pad[:, self.mask, :] = x
        # mask:  [N], with n `True`
        # x:     [bs, n, dim1]
        # x_pad: [bs, N, dim1]
        
        x = x_pad[:, self.neighbors, :]
        x[:, self.pad, :] = 0.0
        # x:     [bs, n, k, dim2]
        
        x = rearrange(x, "b n k d -> b n (k d)")
        x = self.conv(x)
        # x:     [bs, n, dim2]
        
        return x

class SphereAveragePooling(nn.Module):
    """
    Input:
        x (torch.Tensor): [bs, n1, dim]
    Output:
        x (torch.Tensor): [bs, n2, dim]
    """
    def __init__(
        self, 
        hemi: Literal["L", "R"],
        in_level: int,
        mask_name: Literal["nsdgeneral", "hcp-mmp1.0-vision", "yeo17-vision", "none"] = "none"
    ) -> None:
        super().__init__()
        
        masked_neighbors = get_masked_neighbors_same_level(in_level, mask_name, hemi)
        self.in_mask = masked_neighbors.mask         # [N1]
        self.in_voxels_full = self.in_mask.shape[0]
        
        masked_neighbors = get_masked_neighbors_downsample(in_level, mask_name, hemi)
        self.out_mask = masked_neighbors.mask        # [N2]
        self.neighbors = masked_neighbors.neighbors  # [n2, k]
        self.pad = masked_neighbors.pad              # [n2, k]
        self.out_voxels_full = self.out_mask.shape[0]
        self.out_voxels = self.neighbors.shape[0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_pad = torch.zeros((x.shape[0], self.in_voxels_full, x.shape[2]), device=x.device, dtype=x.dtype)
        x_pad[:, self.in_mask, :] = x
        # in_mask:  [N1], with n1 `True`
        # x:        [bs, n1, dim]
        # x_pad:    [bs, N1, dim]
        
        x = x_pad[:, self.neighbors, :]
        x[:, self.pad, :] = 0.0
        # x:        [bs, n2, k, dim]
        
        x = torch.mean(x, dim=2)
        # x:        [bs, n2, dim]
        
        return x