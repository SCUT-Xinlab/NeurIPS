import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import sphericalunet
from sphericalunet.utils import utils as sphericalunet_utils
from sphericalunet.utils.interp_torch import resampleSphereSurf

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description="preprocess smri")
parser.add_argument('raw_nsd_path', type=str)
parser.add_argument('--target_root_path', type=str, default=None)
args = parser.parse_args()

raw_root_path = args.raw_nsd_path
target_root_path = args.target_root_path if args.target_root_path is not None else str(ROOT_PATH / ".outputs" / "nsd" / "preprocess")


def get_target_surf(num_voxels: int, rotated: int) -> torch.Tensor:
    surf_file = os.path.join(
        sphericalunet.SURF_PATH,
        f"sphere_{num_voxels}_rotated_{rotated}.vtk"
    )
    return torch.from_numpy(sphericalunet_utils.read_vtk(surf_file)["vertices"])


def get_neights_order(num_voxels: int, rotated: int) -> torch.Tensor:
    nights_order_file = os.path.join(
        sphericalunet.NEIGH_INDICES_PATH,
        f"adj_mat_order_{num_voxels}_rotated_{rotated}.npy"
    )
    return torch.from_numpy(np.load(nights_order_file))


def resample(
    data: torch.Tensor,
    source_surf: torch.Tensor,
    target_surf: torch.Tensor,
    neighs_order: torch.Tensor,
) -> np.ndarray:
    resample_data = resampleSphereSurf(
        fixed_xyz=source_surf,
        moving_xyz=target_surf,
        fixed_sulc=data,
        neigh_orders=neighs_order,
        device="cpu"
    ).cpu().numpy()
    resample_data = np.nan_to_num(resample_data)
    return resample_data


def process_one_subj(surf_path: os.PathLike, save_path: os.PathLike):
    os.makedirs(save_path, exist_ok=True)

    left_path = os.path.join(surf_path, "lh.sphere")
    right_path = os.path.join(surf_path, "rh.sphere")
    left_surf, _ = nib.freesurfer.read_geometry(left_path)
    right_surf, _ = nib.freesurfer.read_geometry(right_path)
    left_surf, right_surf = torch.from_numpy(left_surf), torch.from_numpy(right_surf)

    for num_voxels in [163842, 40962, 10242, 2562, 642, 162, 42, 12]:

        target_surf = get_target_surf(num_voxels, 0)
        neighs_order = get_neights_order(num_voxels, 0)

        for struc in ["curv", "area", "sulc", "thickness"]:
            data = nib.freesurfer.read_morph_data(os.path.join(surf_path, f"lh.{struc}")).astype(np.float32)
            data = torch.from_numpy(np.nan_to_num(data)).unsqueeze(1)
            data = resample(data, left_surf, target_surf, neighs_order).astype(np.float32)
            np.save(os.path.join(save_path, f"{struc}.{num_voxels}.lh.npy"), data)
            print(f"saved: {struc}.{num_voxels}.lh.npy, shape: {data.shape}")

            data = nib.freesurfer.read_morph_data(os.path.join(surf_path, f"rh.{struc}")).astype(np.float32)
            data = torch.from_numpy(np.nan_to_num(data)).unsqueeze(1)
            data = resample(data, right_surf, target_surf, neighs_order).astype(np.float32)
            np.save(os.path.join(save_path, f"{struc}.{num_voxels}.rh.npy"), data)
            print(f"saved: {struc}.{num_voxels}.rh.npy, shape: {data.shape}")


def preprocess_nsd(raw_path: os.PathLike, target_path: os.PathLike):
    for subj in range(1, 9):
        subj_raw_path = os.path.join(raw_path, "nsddata", "freesurfer", f"subj0{subj}", "surf")
        subj_target_path = os.path.join(target_path, "surf", f"subj0{subj}")
        process_one_subj(subj_raw_path, subj_target_path)


if __name__ == "__main__":
    print(f"[preprocess_smri] raw_nsd_path: {raw_root_path}")
    print(f"[preprocess_smri] target_root_path: {target_root_path}")
    print(f"[preprocess_smri] Starting preprocessing...")
    preprocess_nsd(raw_root_path, target_root_path)
    print(f"[preprocess_smri] Done!")