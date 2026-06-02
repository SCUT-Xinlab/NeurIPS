import argparse
import json
import os
import shutil
import typing
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

import sphericalunet
from sphericalunet.utils import utils as sphericalunet_utils
from sphericalunet.utils.interp_torch import resampleSphereSurf
from sphericalunet.utils.vtk import resample_label

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description="preprocess fmri")
parser.add_argument('raw_nsd_path', type=str)
parser.add_argument('--target_root_path', type=str, default=None)
parser.add_argument('--temp_path', type=str, default=None)
parser.add_argument('--num_voxels', type=int, default=40962)
parser.add_argument('--rotated', type=int, default=0)
parser.add_argument('--use_fsaverage', type=bool, default=True)
parser.add_argument('--clean_temp', type=bool, default=False)
args = parser.parse_args()

raw_path = args.raw_nsd_path
target_root_path = args.target_root_path if args.target_root_path is not None else str(ROOT_PATH / ".outputs" / "nsd" / "preprocess")
temp_path = args.temp_path if args.temp_path is not None else str(ROOT_PATH / ".outputs" / "temp")
num_voxels = args.num_voxels
rotated = args.rotated
use_fsaverage = args.use_fsaverage
clean_temp = args.clean_temp

os.makedirs(target_root_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)


def get_neights_order(num_voxels: int, rotated: typing.Literal[0, 1, 2]) -> torch.Tensor:
    nights_order_file = os.path.join(
        sphericalunet.NEIGH_INDICES_PATH,
        f"adj_mat_order_{num_voxels}_rotated_{rotated}.npy"
    )
    nights_order = torch.from_numpy(np.load(nights_order_file))
    return nights_order


def get_source_surf(dataset_path: os.PathLike, subj: int, fsaverage: bool = True) -> torch.Tensor:
    if fsaverage:
        path = os.path.join(dataset_path, "nsddata", "freesurfer", "fsaverage", "surf")
    else:
        path = os.path.join(dataset_path, "nsddata", "freesurfer", f"subj0{subj}", "surf")
    left_path = os.path.join(path, "lh.sphere")
    right_path = os.path.join(path, "rh.sphere")
    left_surf, _ = nib.freesurfer.read_geometry(left_path)
    right_surf, _ = nib.freesurfer.read_geometry(right_path)
    return torch.from_numpy(left_surf), torch.from_numpy(right_surf)


def get_target_surf(num_voxels: int, rotated: typing.Literal[0, 1, 2]) -> torch.Tensor:
    surf_file = os.path.join(
        sphericalunet.SURF_PATH,
        f"sphere_{num_voxels}_rotated_{rotated}.vtk"
    )
    return torch.from_numpy(sphericalunet_utils.read_vtk(surf_file)["vertices"])


def read_one_session_file(
    dataset_path: os.PathLike,
    subj: int,
    session: int,
    semi_brain: typing.Literal["lh", "rh"],
    fsaverage: bool = True
):
    if fsaverage:
        path = os.path.join(
            dataset_path,
            "nsddata_betas",
            "ppdata",
            f"subj0{subj}",
            "fsaverage",
            "betas_fithrf_GLMdenoise_RR",
            f"{semi_brain}.betas_session{session:02}.mgh"
        )
        data = nib.load(path).get_fdata()[:, 0, 0, :]
        return torch.from_numpy(data)
    else:
        raise NotImplementedError


def resample_one_session(
    data_tensor: torch.Tensor,
    source_surf_tensor: torch.Tensor,
    target_surf_tensor: torch.Tensor,
    neighs_order_tensor: torch.Tensor
):
    data_tensor  # (750, 163842)
    resample = resampleSphereSurf(
        fixed_xyz=source_surf_tensor,
        moving_xyz=target_surf_tensor,
        fixed_sulc=data_tensor,
        neigh_orders=neighs_order_tensor
    ).cpu().numpy().transpose()
    resample = np.nan_to_num(resample)
    return resample


@torch.no_grad()
def resample_nsd_data(
    raw_path: os.PathLike,
    target_path: os.PathLike,
    fsaverage: bool = True,
    num_voxels: int = 40962,
    rotated: int = 0,
    temp_path: os.PathLike = None
) -> os.PathLike:
    neighs_order = get_neights_order(num_voxels=num_voxels, rotated=rotated)
    target_surf = get_target_surf(num_voxels=num_voxels, rotated=rotated)
    save_path = os.path.join(
        temp_path if temp_path else target_path,
        "temp",
        f"resample_{num_voxels}_{'fsaverage' if fsaverage else 'nativesurface'}"
    )
    max_sessions = [40, 40, 32, 30, 40, 32, 40, 30]
    process_bar = tqdm(range(sum(max_sessions) * 2), desc=f"resample nsd data to {num_voxels}")
    for subj in range(1, 9):
        left_source_surf, right_source_surf = get_source_surf(raw_path, subj=subj, fsaverage=fsaverage)
        subj_path = os.path.join(save_path, f"subj0{subj}")
        os.makedirs(subj_path, exist_ok=True)
        left_index = 0
        right_index = 0
        for session in range(1, max_sessions[subj - 1] + 1):
            for semi_brain in ["lh", "rh"]:
                data = read_one_session_file(
                    dataset_path=raw_path,
                    subj=subj,
                    session=session,
                    semi_brain=semi_brain,
                    fsaverage=fsaverage
                )
                resample_data = resampleSphereSurf(
                    fixed_xyz=left_source_surf if semi_brain == "lh" else right_source_surf,
                    moving_xyz=target_surf,
                    fixed_sulc=data,
                    neigh_orders=neighs_order,
                    device="cpu"
                ).cpu().numpy().transpose()
                resample_data = np.nan_to_num(resample_data).astype(np.float32)
                index = left_index if semi_brain == "lh" else right_index
                for i in range(resample_data.shape[0]):
                    np.save(os.path.join(subj_path, f"{semi_brain}.{index}.npy"), resample_data[i, :])
                    index += 1
                if semi_brain == "lh":
                    left_index = index
                else:
                    right_index = index
                process_bar.update(1)
    return save_path


def load_all_train_samples(target_path: os.PathLike, subj: int) -> typing.List[int]:
    with open(os.path.join(target_path, "split", f"subj0{subj}.json"), "r") as file:
        split = json.load(file)
    split = split["train"]
    ids = []
    for _, vs in split.items():
        for v in vs:
            ids.append(v)
    return ids


def compute_mean_and_std(target_path: os.PathLike, temp_path: os.PathLike, num_voxels: int = 40962):
    max_sessions = [40, 40, 32, 30, 40, 32, 40, 30]
    process_bar = tqdm(range(sum(max_sessions) * 1500), desc=f"compute mean/std for training set")
    for subj in range(1, 9):
        subj_path = os.path.join(temp_path, f"subj0{subj}")
        left_sum = np.zeros(num_voxels, dtype=np.float64)
        left_sum_sq = np.zeros(num_voxels, dtype=np.float64)
        right_sum = np.zeros(num_voxels, dtype=np.float64)
        right_sum_sq = np.zeros(num_voxels, dtype=np.float64)
        total_samples = 0

        all_train_ids = load_all_train_samples(target_path, subj)
        for index in range(max_sessions[subj - 1] * 750):
            if index in all_train_ids:
                data = np.load(os.path.join(subj_path, f"lh.{index}.npy"))
                left_sum += data
                left_sum_sq += data ** 2
                data = np.load(os.path.join(subj_path, f"rh.{index}.npy"))
                right_sum += data
                right_sum_sq += data ** 2
                total_samples += 1
            process_bar.update(1)

        left_mean = left_sum / total_samples
        left_std = np.sqrt(left_sum_sq / total_samples - left_mean ** 2)
        right_mean = right_sum / total_samples
        right_std = np.sqrt(right_sum_sq / total_samples - right_mean ** 2)

        np.save(os.path.join(subj_path, "lh.mean.npy"), left_mean)
        np.save(os.path.join(subj_path, "lh.std.npy"), left_std)
        np.save(os.path.join(subj_path, "rh.mean.npy"), right_mean)
        np.save(os.path.join(subj_path, "rh.std.npy"), right_std)


def zero_score(target_path: os.PathLike, temp_path: os.PathLike, subdir: str):
    max_sessions = [40, 40, 32, 30, 40, 32, 40, 30]
    process_bar = tqdm(range(sum(max_sessions) * 750), desc=f"normalization")
    target_path = os.path.join(target_path, "tfMRI", subdir)
    for subj in range(1, 9):
        subj_path = os.path.join(temp_path, f"subj0{subj}")
        subj_output_path = os.path.join(target_path, f"subj0{subj}")
        os.makedirs(subj_output_path, exist_ok=True)
        left_mean = np.load(os.path.join(subj_path, "lh.mean.npy"))
        left_std = np.load(os.path.join(subj_path, "lh.std.npy"))
        right_mean = np.load(os.path.join(subj_path, "rh.mean.npy"))
        right_std = np.load(os.path.join(subj_path, "rh.std.npy"))
        for index in range(max_sessions[subj - 1] * 750):
            data = np.load(os.path.join(subj_path, f"lh.{index}.npy"))
            data = (data - left_mean) / left_std
            data = np.nan_to_num(data)
            np.save(os.path.join(subj_output_path, f"lh.{index}.npy"), data.astype(np.float32))
            data = np.load(os.path.join(subj_path, f"rh.{index}.npy"))
            data = (data - right_mean) / right_std
            data = np.nan_to_num(data)
            np.save(os.path.join(subj_output_path, f"rh.{index}.npy"), data.astype(np.float32))
            process_bar.update(1)


def resample_one_label(
    label: np.ndarray,
    source_surf: torch.Tensor,
    target_surf: torch.Tensor,
) -> np.ndarray:
    resampled_label = resample_label(source_surf, target_surf, label)
    resampled_label = np.nan_to_num(resampled_label).astype(np.int32)
    return resampled_label


if __name__ == "__main__":
    temp_path = os.path.join(temp_path, "resample", f"resample_{num_voxels}_{'fsaverage' if use_fsaverage else 'nativesurface'}")

    print("Step 1/4: Resampling fMRI data...")
    temp_path = resample_nsd_data(raw_path, target_root_path, fsaverage=use_fsaverage, num_voxels=num_voxels, rotated=rotated, temp_path=temp_path)

    print("Step 2/4: Computing mean and std...")
    compute_mean_and_std(target_root_path, temp_path, num_voxels=num_voxels)

    print("Step 3/4: Normalizing with z-score...")
    zero_score(target_root_path, temp_path, subdir=f"sphere{num_voxels}")

    if clean_temp:
        print("Step 4/4: Cleaning temp files...")
        shutil.rmtree(os.path.dirname(temp_path))
    else:
        print(f"Temp files kept at: {os.path.dirname(temp_path)}")

    print("Done!")