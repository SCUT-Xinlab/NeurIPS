import dataclasses
import logging
import os
from typing import List, Literal

import numpy as np
import torch

from .neighbors import get_neighbors, get_deconv_neighbors

logger = logging.getLogger(__name__)

mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npy", "mask")

NUM_VOXELS: List[int] = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
NSD_GENERAL_L: List[torch.BoolTensor] = []
NSD_GENERAL_R: List[torch.BoolTensor] = []
HCP_MMP1_VISION_L: List[torch.BoolTensor] = []
HCP_MMP1_VISION_R: List[torch.BoolTensor] = []
YEO17_VISION_L: List[torch.BoolTensor] = []
YEO17_VISION_R: List[torch.BoolTensor] = []
NONE_L: List[torch.BoolTensor] = []
NONE_R: List[torch.BoolTensor] = []

for num_voxel in NUM_VOXELS:
    
    # nsdgeneral
    nsdgeneral_L_path = os.path.join(mask_path, "nsdgeneral", f"{num_voxel}", "lh.nsdgeneral.npy")
    NSD_GENERAL_L.append(torch.from_numpy(np.load(nsdgeneral_L_path)).to(torch.bool)[:, 0])
    logger.debug(f"Load {nsdgeneral_L_path}")
    nsdgeneral_R_path = os.path.join(mask_path, "nsdgeneral", f"{num_voxel}", "rh.nsdgeneral.npy")
    NSD_GENERAL_R.append(torch.from_numpy(np.load(nsdgeneral_R_path)).to(torch.bool)[:, 0])
    logger.debug(f"Load {nsdgeneral_R_path}")
    
    # hcp-mmp1.0-vision
    hcp_mmp_vis_L_path = os.path.join(mask_path, "hcp-mmp1.0", f"{num_voxel}", "lh.visual.npy")
    HCP_MMP1_VISION_L.append(torch.from_numpy(np.load(hcp_mmp_vis_L_path)).to(torch.bool)[:, 0])
    logger.debug(f"Load {hcp_mmp_vis_L_path}")
    hcp_mmp_vis_R_path = os.path.join(mask_path, "hcp-mmp1.0", f"{num_voxel}", "rh.visual.npy")
    HCP_MMP1_VISION_R.append(torch.from_numpy(np.load(hcp_mmp_vis_R_path)).to(torch.bool)[:, 0])
    logger.debug(f"Load {hcp_mmp_vis_L_path}")
    
    # yeo17-vision
    yeo17_vis_L_path = os.path.join(mask_path, "yeo17", f"{num_voxel}", "lh.visual.npy")
    YEO17_VISION_L.append(torch.from_numpy(np.load(yeo17_vis_L_path)).to(torch.bool)[:, 0])
    logger.debug(f"Load {yeo17_vis_L_path}")
    yeo17_vis_R_path = os.path.join(mask_path, "yeo17", f"{num_voxel}", "rh.visual.npy")
    YEO17_VISION_R.append(torch.from_numpy(np.load(yeo17_vis_R_path)).to(torch.bool)[:, 0])
    logger.debug(f"Load {yeo17_vis_R_path}")
    
    # none
    NONE_L.append(torch.ones((num_voxel,)).to(torch.bool))
    NONE_R.append(torch.ones((num_voxel,)).to(torch.bool))

ALL_MASK = {
    "nsdgeneral": (NSD_GENERAL_L, NSD_GENERAL_R),
    "hcp-mmp1.0-vision": (HCP_MMP1_VISION_L, HCP_MMP1_VISION_R),
    "yeo17-vision": (YEO17_VISION_L, YEO17_VISION_R),
    "none": (NONE_L, NONE_R)
}

def get_mask(mask_name: str, hemi: Literal["L", "R"], level: int) -> torch.BoolTensor:
    if mask_name not in ALL_MASK.keys():
        raise ValueError(f"no mask name '{mask_name}', supports: {ALL_MASK.keys()}")
    mask_L, mask_R = ALL_MASK[mask_name]
    masks = mask_L if hemi == "L" else mask_R
    return masks[level]

@dataclasses.dataclass
class MaskedNeighborsOutput:
    mask: torch.BoolTensor
    neighbors: torch.LongTensor
    pad: torch.BoolTensor

def get_masked_neighbors_same_level(
    level: int,
    mask_name: str,
    hemi: Literal["L", "R"]
) -> "MaskedNeighborsOutput":
    mask = get_mask(mask_name, hemi, level)
    selected_voxels = torch.where(mask)[0]
    neighbors = get_neighbors(level)
    selected_items = torch.isin(neighbors, selected_voxels)
    neighbors[~selected_items] = -1
    neighbors = neighbors[selected_voxels, :]
    pad = neighbors == -1
    return MaskedNeighborsOutput(mask=mask, neighbors=neighbors, pad=pad)

def get_masked_neighbors_downsample(
    in_level: int,
    mask_name: str,
    hemi: Literal["L", "R"]
) -> "MaskedNeighborsOutput":
    in_mask = get_mask(mask_name, hemi, in_level)
    selected_voxels = torch.where(in_mask)[0]
    next_num_voxels = NUM_VOXELS[in_level - 1]
    neighbors = get_neighbors(in_level)[:next_num_voxels, :]
    selected_items = torch.isin(neighbors, selected_voxels)
    neighbors[~selected_items] = -1
    out_mask = get_mask(mask_name, hemi, in_level - 1)
    neighbors = neighbors[out_mask, :]
    pad = neighbors == -1
    return MaskedNeighborsOutput(mask=out_mask, neighbors=neighbors, pad=pad)

def get_masked_neighbors_upsample(
    in_level: int,
    mask_name: str,
    hemi: Literal["L", "R"]
) -> "MaskedNeighborsOutput":
    in_mask = get_mask(mask_name, hemi, in_level)
    selected_voxels = torch.where(in_mask)[0]
    neighbors = get_deconv_neighbors(in_level)
    selected_items = torch.isin(neighbors, selected_voxels)
    neighbors[~selected_items] = -1
    nei_col0 = torch.where(neighbors[:, 0] == -1, neighbors[:, 1], neighbors[:, 0])
    nei_col1 = torch.where(neighbors[:, 1] == -1, neighbors[:, 0], neighbors[:, 1])
    neighbors = torch.stack([nei_col0, nei_col1], dim=1)
    out_mask = get_mask(mask_name, hemi, in_level + 1)
    neighbors = neighbors[out_mask, :]
    pad = neighbors == -1
    return MaskedNeighborsOutput(mask=out_mask, neighbors=neighbors, pad=pad)