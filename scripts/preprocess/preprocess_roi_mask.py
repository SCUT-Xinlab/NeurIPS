import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista
import torch

from sphericalunet.utils.vtk import resample_label

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

SURFACE_NUM_VOXELS = [163842, 40962, 10242, 2562, 642, 162, 42, 12]
surf_path = ROOT_PATH / "src" / "sphericalunet" / "surf"
surface_convs = []
for num_voxels in SURFACE_NUM_VOXELS:
    mesh = pyvista.read(str(surf_path / f"sphere_{num_voxels}_rotated_0.vtk"))
    surface_convs.append(torch.from_numpy(mesh.points))


def get_fsaverage_surface(raw_path):
    surface_164k_L = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "surf", "lh.sphere")
    surface_164k_R = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "surf", "rh.sphere")
    surface_164k_L = nib.freesurfer.io.read_geometry(surface_164k_L)[0]
    surface_164k_R = nib.freesurfer.io.read_geometry(surface_164k_R)[0]
    return surface_164k_L, surface_164k_R


def resample_roi_mask(raw_path, target_root_path, name, label_L, label_R):
    surface_164k_L, surface_164k_R = get_fsaverage_surface(raw_path)

    for i, voxel in enumerate(SURFACE_NUM_VOXELS):
        output_dir = os.path.join(target_root_path, "mask", name, str(voxel))
        os.makedirs(output_dir, exist_ok=True)

        label_L_resampled = resample_label(
            vertices_fix=surface_164k_L,
            vertices_inter=surface_convs[i].numpy(),
            label=label_L
        )
        label_R_resampled = resample_label(
            vertices_fix=surface_164k_R,
            vertices_inter=surface_convs[i].numpy(),
            label=label_R
        )

        label_L_resampled = np.nan_to_num(label_L_resampled).astype(bool)
        label_R_resampled = np.nan_to_num(label_R_resampled).astype(bool)

        if name == "nsdgeneral":
            np.save(os.path.join(output_dir, "lh.nsdgeneral.npy"), label_L_resampled)
            np.save(os.path.join(output_dir, "rh.nsdgeneral.npy"), label_R_resampled)
        else:
            np.save(os.path.join(output_dir, "lh.visual.npy"), label_L_resampled)
            np.save(os.path.join(output_dir, "rh.visual.npy"), label_R_resampled)

        print(f"[{name}] saved {voxel}")


def preprocess_yeo17(raw_path, target_root_path):
    NAME = "yeo17"
    selected_labels = [1, 2]

    label_L_path = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "label", "lh.Yeo2011_17Networks_N1000.annot")
    label_L = nib.freesurfer.io.read_annot(label_L_path)[0].astype(int)
    label_L = np.isin(label_L, selected_labels)

    label_R_path = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "label", "rh.Yeo2011_17Networks_N1000.annot")
    label_R = nib.freesurfer.io.read_annot(label_R_path)[0].astype(int)
    label_R = np.isin(label_R, selected_labels)

    resample_roi_mask(raw_path, target_root_path, NAME, label_L, label_R)


def preprocess_hcp_mmp1(raw_path, target_root_path):
    import hcp_utils as hcp

    NAME = "hcp-mmp1.0"

    ROIs = [
        "V1",
        "V2", "V3", "V4",
        "IPS1", "V3A", "V3B", "V6", "V6A", "V7",
        "FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",
        "FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t"
    ]
    ROIs = [f"L_{ROI}" for ROI in ROIs] + [f"R_{ROI}" for ROI in ROIs]
    mmp = dict()
    for key, value in hcp.mmp.labels.items():
        mmp[value] = int(key)
    selected_labels = [mmp[ROI] for ROI in ROIs]

    label_L_path = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "label", "lh.HCP_MMP1.mgz")
    label_L = nib.load(label_L_path).get_fdata()[:, 0, 0].astype(int)
    label_L = np.isin(label_L, selected_labels)

    label_R_path = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "label", "rh.HCP_MMP1.mgz")
    label_R = nib.load(label_R_path).get_fdata()[:, 0, 0].astype(int)
    label_R = np.isin(label_R, selected_labels)

    resample_roi_mask(raw_path, target_root_path, NAME, label_L, label_R)


def preprocess_nsdgeneral(raw_path, target_root_path):
    NAME = "nsdgeneral"

    label_L_path = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "label", "lh.nsdgeneral.mgz")
    label_L = nib.load(label_L_path).get_fdata()[:, :, 0]
    label_R_path = os.path.join(raw_path, "nsddata", "freesurfer", "fsaverage", "label", "rh.nsdgeneral.mgz")
    label_R = nib.load(label_R_path).get_fdata()[:, :, 0]

    resample_roi_mask(raw_path, target_root_path, NAME, label_L, label_R)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess roi mask")
    parser.add_argument('raw_nsd_path', type=str)
    parser.add_argument('--target_root_path', type=str, default=None)
    args = parser.parse_args()

    raw_path = args.raw_nsd_path
    target_root_path = args.target_root_path if args.target_root_path is not None else str(ROOT_PATH / ".outputs" / "nsd" / "preprocess")

    os.makedirs(target_root_path, exist_ok=True)

    print("Processing Yeo17-Visual ROI...")
    preprocess_yeo17(raw_path, target_root_path)

    print("Processing HCP-MMP1.0-Visual ROI...")
    preprocess_hcp_mmp1(raw_path, target_root_path)

    print("Processing NSDGeneral ROI...")
    preprocess_nsdgeneral(raw_path, target_root_path)

    print("Done!")