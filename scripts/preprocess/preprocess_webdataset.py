import argparse
import json
import os
import tarfile
from pathlib import Path

import numpy as np

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description="make webdataset")
parser.add_argument('--preprocessed_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--temp_path', type=str, default=None)
parser.add_argument('--num_tar_samples', type=int, default=2000)
args = parser.parse_args()

preprocessed_path = args.preprocessed_path if args.preprocessed_path is not None else str(ROOT_PATH / ".outputs" / "nsd" / "preprocess")
output_path = args.output_path if args.output_path is not None else str(ROOT_PATH / ".outputs" / "nsd" / "nsd_sphere")
temp_path = args.temp_path if args.temp_path is not None else str(ROOT_PATH / ".outputs" / "temp")
num_tar_samples = args.num_tar_samples

os.makedirs(output_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

mask_L = np.load(os.path.join(preprocessed_path, "mask", "nsdgeneral", "40962", "lh.nsdgeneral.npy")).astype(bool)[:, 0]
mask_R = np.load(os.path.join(preprocessed_path, "mask", "nsdgeneral", "40962", "rh.nsdgeneral.npy")).astype(bool)[:, 0]

disc = []

sample_id = 0

for subj in range(1, 9):

    subj = f"subj0{subj}"
    with open(os.path.join(preprocessed_path, "split", f"{subj}.json"), "r") as file:
        split = json.load(file)
    meta_data = {
        "totals": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"])
        }
    }
    with open(os.path.join(output_path, f"metadata_{subj}.json"), "w") as file:
        json.dump(meta_data, file, indent=4)

    for type in ["train", "val", "test"]:
        output_dir = os.path.join(output_path, type)
        os.makedirs(output_dir, exist_ok=True)

        tar_idx = 0
        i = 0

        for key, value in split[type].items():

            tar_idx = i // num_tar_samples

            tar_path = os.path.join(output_dir, f"{type}_{subj}_{tar_idx}.tar")
            mode = "a" if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0 else "w"

            with tarfile.open(tar_path, mode) as tar:

                tar.add(
                    os.path.join(
                        preprocessed_path,
                        "stimuli",
                        "imgs",
                        f"{key}.npy"
                    ),
                    f"{subj}/sample_{sample_id:08}.image.npy"
                )

                tar.add(
                    os.path.join(
                        preprocessed_path,
                        "stimuli",
                        "captions",
                        f"{key}.json"
                    ),
                    f"{subj}/sample_{sample_id:08}.text.json"
                )

                fmri_L = []
                fmri_R = []
                for fmri_id in value:
                    fmri_L.append(np.load(os.path.join(
                        preprocessed_path,
                        "tfMRI",
                        "sphere40962",
                        subj,
                        f"lh.{fmri_id}.npy"
                    )))
                    fmri_R.append(np.load(os.path.join(
                        preprocessed_path,
                        "tfMRI",
                        "sphere40962",
                        subj,
                        f"rh.{fmri_id}.npy"
                    )))
                fmri_L = np.stack(fmri_L, axis=0)
                fmri_R = np.stack(fmri_R, axis=0)

                temp_npy = os.path.join(temp_path, "temp.npy")

                np.save(temp_npy, fmri_L)
                tar.add(temp_npy, f"{subj}/sample_{sample_id:08}.fmri.L.npy")

                np.save(temp_npy, fmri_R)
                tar.add(temp_npy, f"{subj}/sample_{sample_id:08}.fmri.R.npy")

                fmri_L = fmri_L[:, mask_L]
                fmri_R = fmri_R[:, mask_R]

                np.save(temp_npy, fmri_L)
                tar.add(temp_npy, f"{subj}/sample_{sample_id:08}.fmri.L.nsdgeneral.npy")

                np.save(temp_npy, fmri_R)
                tar.add(temp_npy, f"{subj}/sample_{sample_id:08}.fmri.R.nsdgeneral.npy")

                sample_id += 1
                i += 1

                disc.append({
                    "subj": subj,
                    "image_id": int(key),
                    "fmri_id": value,
                    "split": type,
                    "tar_file": tar_path
                })

                print(f"[{sample_id:08}]{subj} {type} {tar_path}")

with open(os.path.join(output_path, "info.json"), "w") as file:
    json.dump(disc, file, indent=4)

for subj in range(1, 9):

    smri_L = []
    smri_R = []

    for struc in ["area", "curv", "sulc", "thickness"]:

        data_L = np.load(os.path.join(preprocessed_path, "surf", f"subj0{subj}", f"{struc}.40962.lh.npy"))
        data_L = (data_L - data_L.mean()) / data_L.std()
        smri_L.append(data_L)

        data_R = np.load(os.path.join(preprocessed_path, "surf", f"subj0{subj}", f"{struc}.40962.rh.npy"))
        data_R = (data_R - data_R.mean()) / data_R.std()
        smri_R.append(data_R)

    smri_L = np.stack(smri_L, axis=0)
    smri_R = np.stack(smri_R, axis=0)
    outpath = os.path.join(output_path, "smri", f"subj0{subj}")
    os.makedirs(outpath, exist_ok=True)

    np.save(os.path.join(outpath, "smri.L.npy"), smri_L)
    np.save(os.path.join(outpath, "smri.R.npy"), smri_R)
    np.save(os.path.join(outpath, "smri.L.nsdgeneral.npy"), smri_L[:, mask_L])
    np.save(os.path.join(outpath, "smri.R.nsdgeneral.npy"), smri_R[:, mask_R])