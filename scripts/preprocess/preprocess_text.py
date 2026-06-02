import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description="preprocess text")
parser.add_argument('--target_root_path', type=str, default=None)
parser.add_argument('--temp_path', type=str, default=None)
args = parser.parse_args()

target_root_path = args.target_root_path if args.target_root_path is not None else str(ROOT_PATH / ".outputs" / "nsd" / "preprocess" / "stimuli")
temp_path = args.temp_path if args.temp_path is not None else str(ROOT_PATH / ".outputs" / "temp")

coco_annotations_dir = os.path.join(temp_path, "coco_annotations")
os.makedirs(coco_annotations_dir, exist_ok=True)

required_files = [
    "annotations/captions_val2017.json",
    "annotations/captions_train2017.json",
    "annotations/instances_val2017.json",
    "annotations/instances_train2017.json",
    "annotations/person_keypoints_val2017.json",
    "annotations/person_keypoints_train2017.json",
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(coco_annotations_dir, f))]

if missing_files:
    zip_path = os.path.join(coco_annotations_dir, "annotations_trainval2017.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading COCO annotations...")
        subprocess.run(["wget", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "-O", zip_path], check=True)
    print(f"Extracting COCO annotations...")
    subprocess.run(["unzip", "-o", zip_path], cwd=coco_annotations_dir, check=True)
    os.remove(zip_path)

def process_coco_annotations(nsd_preprocessed_stimuli_path: os.PathLike):

    val_capsFile = os.path.join(coco_annotations_dir, "annotations/captions_val2017.json")
    tra_capsFile = os.path.join(coco_annotations_dir, "annotations/captions_train2017.json")
    val_catsFile = os.path.join(coco_annotations_dir, "annotations/instances_val2017.json")
    tra_catsFile = os.path.join(coco_annotations_dir, "annotations/instances_train2017.json")
    val_kptsFile = os.path.join(coco_annotations_dir, "annotations/person_keypoints_val2017.json")
    tra_kptsFile = os.path.join(coco_annotations_dir, "annotations/person_keypoints_train2017.json")
    val_coco_caps = COCO(val_capsFile)
    tra_coco_caps = COCO(tra_capsFile)
    val_coco_cats = COCO(val_catsFile)
    tra_coco_cats = COCO(tra_catsFile)
    val_coco_kpts = COCO(val_kptsFile)
    tra_coco_kpts = COCO(tra_kptsFile)

    nsd_cocoIdsFile = os.path.join(nsd_preprocessed_stimuli_path, "cocoId.npy")
    nsd_cocoIds = np.load(nsd_cocoIdsFile)
    nsd_cocoSplitFile = os.path.join(nsd_preprocessed_stimuli_path, "cocoSplit.npy")
    nsd_cocoSplit = np.load(nsd_cocoSplitFile, allow_pickle=True)

    num_stimulis = len(nsd_cocoIds)
    process_bar = tqdm(range(num_stimulis), desc="Processing COCO Annotations")
    for nsd_stimuli_id in range(num_stimulis):
        nsd_cocoId = nsd_cocoIds[nsd_stimuli_id]
        coco_split = nsd_cocoSplit[nsd_stimuli_id]
        if coco_split == "val2017":
            coco_caps = val_coco_caps
            coco_cats = val_coco_cats
            coco_kpts = val_coco_kpts
        elif coco_split == "train2017":
            coco_caps = tra_coco_caps
            coco_cats = tra_coco_cats
            coco_kpts = tra_coco_kpts
        # 1. captions
        os.makedirs(os.path.join(nsd_preprocessed_stimuli_path, "captions"), exist_ok=True)
        annId = coco_caps.getAnnIds(imgIds=nsd_cocoId)
        anns = coco_caps.loadAnns(annId)
        captions = []
        for ann in anns:
            captions.append(ann["caption"])
        np.save(os.path.join(nsd_preprocessed_stimuli_path, "captions", f"{nsd_stimuli_id}.npy"), captions, allow_pickle=True)
        with open(os.path.join(nsd_preprocessed_stimuli_path, "captions", f"{nsd_stimuli_id}.json"), "w") as file:
            json.dump(captions, file, indent=4)
        # 2. categories
        os.makedirs(os.path.join(nsd_preprocessed_stimuli_path, "instances"), exist_ok=True)
        annId = coco_cats.getAnnIds(imgIds=nsd_cocoId)
        anns = coco_cats.loadAnns(annId)
        instance = []
        for ann in anns:
            cat = coco_cats.loadCats(ids=ann["category_id"])[0]
            supercategory = cat["supercategory"]
            category = cat["name"]
            ann.update({"supercategory": supercategory, "category": category})
            instance.append(ann)
        with open(os.path.join(nsd_preprocessed_stimuli_path, "instances", f"{nsd_stimuli_id}.json"), "w") as file:
            json.dump(instance, file, indent=4)
        # 3. keypoints
        os.makedirs(os.path.join(nsd_preprocessed_stimuli_path, "keypoints"), exist_ok=True)
        annId = coco_kpts.getAnnIds(imgIds=nsd_cocoId)
        anns = coco_kpts.loadAnns(annId)
        keypoint = []
        for ann in anns:
            cat = coco_cats.loadCats(ids=ann["category_id"])[0]
            supercategory = cat["supercategory"]
            category = cat["name"]
            ann.update({"supercategory": supercategory, "category": category})
            keypoint.append(ann)
        with open(os.path.join(nsd_preprocessed_stimuli_path, "keypoints", f"{nsd_stimuli_id}.json"), "w") as file:
            json.dump(keypoint, file, indent=4)
        process_bar.update(1)

import sys

if __name__ == "__main__":
    print(f"[preprocess_text] target_root_path: {target_root_path}", flush=True)
    print(f"[preprocess_text] temp_path: {temp_path}", flush=True)
    print(f"[preprocess_text] Starting preprocessing...", flush=True)
    process_coco_annotations(target_root_path)
    print(f"[preprocess_text] Done!", flush=True)