import logging
import os
from typing import List, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

neighbors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npy", "neighbors")

NUM_VOXELS: List[int] = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
NUM_LEVELS: int = len(NUM_VOXELS)

########## conv neighbors ##########

NEIGHBORS: List[torch.LongTensor] = []

for num_voxel in NUM_VOXELS:
    index_path = os.path.join(neighbors_path, f"{num_voxel}.one_ring.npy")
    NEIGHBORS.append(torch.from_numpy(np.load(index_path)))
    logger.debug(f"Load {index_path}")

def get_neighbors(level: int) -> torch.LongTensor:
    return NEIGHBORS[level]

########## deconv neighbors ##########

DECONV_NEIGHBORS: List[torch.LongTensor] = []

for in_level in range(NUM_LEVELS - 1):
    down_n_voxels = NUM_VOXELS[in_level]
    up_n_voxels = NUM_VOXELS[in_level + 1]
    up_neighbor = NEIGHBORS[in_level + 1]
    deconv_neighbor = [[] for _ in range(up_n_voxels)]
    for i in range(down_n_voxels):
        for j in up_neighbor[i]:
            deconv_neighbor[j].append(i)
    deconv_neighbor = torch.from_numpy(np.array([vs + vs if len(vs) == 1 else vs for vs in deconv_neighbor]))
    DECONV_NEIGHBORS.append(deconv_neighbor)

def get_deconv_neighbors(in_level: int) -> torch.LongTensor:
    return DECONV_NEIGHBORS[in_level]

for level in range(NUM_LEVELS):
    logger.info(f"init spconv neighbor. level: {level}    num_voxels: {NUM_VOXELS[level]:<6}        shape: {NEIGHBORS[level].shape}")

for level in range(NUM_LEVELS - 1):
    logger.info(f"init deconv neighbor. level: {level}->{level + 1} num_voxels: {NUM_VOXELS[level]:>5}->{NUM_VOXELS[level + 1]:<6} shape: {DECONV_NEIGHBORS[level].shape}")
