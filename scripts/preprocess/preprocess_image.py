import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

NUM_STIMULI = [30000, 30000, 24000, 22500, 30000, 24000, 30000, 22500]

parser = argparse.ArgumentParser(description="preprocess image")
parser.add_argument('raw_nsd_path', type=str)
parser.add_argument('--target_root_path', type=str, default=None)
args = parser.parse_args()

raw_root_path = args.raw_nsd_path
target_root_path = args.target_root_path if args.target_root_path is not None else str(ROOT_PATH / ".outputs" / "nsd" / "preprocess")


def convert_stimuli_hdf5_to_npy(raw_root_path: os.PathLike, target_root_path: os.PathLike):
    raw_hdf5_path = os.path.join(raw_root_path, "nsddata_stimuli", "stimuli", "nsd", "nsd_stimuli.hdf5")
    target_path = os.path.join(target_root_path, "stimuli", "imgs")
    os.makedirs(target_path, exist_ok=True)
    with h5py.File(raw_hdf5_path, 'r') as hdf:
        num_images = len(hdf["imgBrick"])
        process_bar = tqdm(range(num_images), desc="Preprocessing Images")
        for i in range(num_images):
            img = hdf["imgBrick"][i].astype(np.uint8)
            np.save(os.path.join(target_path, f"{i}.npy"), img)
            process_bar.update(1)

def preprocess_infomations(raw_root_path: os.PathLike, target_root_path: os.PathLike):
    raw_path = os.path.join(raw_root_path, "nsddata", "experiments", "nsd")
    images_info_csv_path = os.path.join(raw_path, "nsd_stim_info_merged.csv")
    target_path = os.path.join(target_root_path, "stimuli")
    os.makedirs(target_path, exist_ok=True)
    images_info = pd.read_csv(images_info_csv_path, sep=',')
    cocoId = images_info["cocoId"].to_numpy()
    np.save(os.path.join(target_path, "cocoId.npy"), cocoId)
    share1000 = np.where(images_info["shared1000"].to_numpy())[0]
    np.save(os.path.join(target_path, "share1000.npy"), share1000)
    cocoSplit = images_info["cocoSplit"].to_numpy()
    np.save(os.path.join(target_path, "cocoSplit.npy"), cocoSplit, allow_pickle=True)
    images_info = scipy.io.loadmat(os.path.join(raw_path, "nsd_expdesign.mat"))
    masterordering = images_info["masterordering"] - 1
    subjectim = images_info["subjectim"]
    stimuli_ids = np.take_along_axis(subjectim, masterordering, axis=1) - 1
    np.save(os.path.join(target_path, "stimuli_ids.npy"), stimuli_ids)
    test_flags = np.where(masterordering[0, :] < 1000)[0]
    train_flags = np.where(masterordering[0, :] >= 1000)[0]
    np.save(os.path.join(target_path, "test_ids.npy"), test_flags)
    np.save(os.path.join(target_path, "train_ids.npy"), train_flags)
    split_path = os.path.join(target_root_path, "split")
    os.makedirs(split_path, exist_ok=True)
    for subj_id in range(stimuli_ids.shape[0]):
        stimuli_dict = {
            "train": {},
            "test": {}
        }
        for stimuli_id in range(stimuli_ids.shape[1]):
            if stimuli_id >= NUM_STIMULI[subj_id]:
                break
            image_id = stimuli_ids[subj_id, stimuli_id]
            if stimuli_id in test_flags:
                if str(image_id) in stimuli_dict["test"].keys():
                    stimuli_dict["test"][str(image_id)].append(stimuli_id)
                else:
                    stimuli_dict["test"][str(image_id)] = [stimuli_id]
            elif stimuli_id in train_flags:
                if str(image_id) in stimuli_dict["train"].keys():
                    stimuli_dict["train"][str(image_id)].append(stimuli_id)
                else:
                    stimuli_dict["train"][str(image_id)] = [stimuli_id]
        output = {}
        output["test"] = stimuli_dict["test"]
        output["train"] = {}
        output["val"] = {}
        val_ratio = 1 / 18
        num_val = int(len(stimuli_dict["train"]) * val_ratio)
        num_train = len(stimuli_dict["train"]) - num_val
        num = 0
        for key, value in stimuli_dict["train"].items():
            if num < num_train:
                output["train"][key] = value
            else:
                output["val"][key] = value
            num += 1
        with open(os.path.join(split_path, f"subj{subj_id + 1:02}.json"), "w") as file:
            json.dump(output, file, indent=2)


def preprocess_stimuli_all(raw_root_path: os.PathLike, target_root_path: os.PathLike):
    convert_stimuli_hdf5_to_npy(raw_root_path, target_root_path)

if __name__ == "__main__":
    print(f"[preprocess_image] raw_nsd_path: {raw_root_path}")
    print(f"[preprocess_image] target_root_path: {target_root_path}")
    print(f"[preprocess_image] Starting preprocessing...")
    preprocess_infomations(raw_root_path, target_root_path)
    preprocess_stimuli_all(raw_root_path, target_root_path)
    print(f"[preprocess_image] Done!")