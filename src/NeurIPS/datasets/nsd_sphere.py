import math
import os
import random
import typing

import numpy as np
import torch
from webdataset import WebDataset, split_by_node
from torch.utils.data import DataLoader

FMRI_MIXUP_PROBABILITY = 0.5

def dict_collation_fn(samples):
    batch = {}
    
    for sample in samples:
        for key, value in sample.items():
            if key.startswith("__") or key.endswith(".npy"):
                continue
            if key not in batch:
                batch[key] = []
            if key == "text":
                batch[key].append(random.choice(value))
            elif key in ["fmri_L", "fmri_R"]:
                idx = np.random.choice(value.shape[0])
                batch[key].append(value[idx, :])
            else:
                batch[key].append(value)
    
    for key in batch:
        if isinstance(batch[key][0], np.ndarray):
            batch[key] = torch.from_numpy(np.stack(batch[key], axis=0))
    return batch

def collation_fn_train(samples):
    
    batch_size = len(samples)
    batch = dict(batch_size=batch_size)
    
    text = []
    image = []
    fmri_R = []
    fmri_L = []
    
    mixup = np.random.uniform(0, 1, batch_size) < FMRI_MIXUP_PROBABILITY
    
    for i, sample in enumerate(samples):
        
        text.append(random.choice(sample["text"]))
        image.append(sample["image"])
        
        num_scans = sample["fmri_L"].shape[0]
        if mixup[i]:
            mixup_weights = np.random.dirichlet((1,) * num_scans)
            fmri_L.append(np.dot(mixup_weights, sample["fmri_L"]))
            fmri_R.append(np.dot(mixup_weights, sample["fmri_R"]))
        else:
            selected_idx = np.random.choice(num_scans)
            fmri_L.append(sample["fmri_L"][selected_idx, :])
            fmri_R.append(sample["fmri_R"][selected_idx, :])
    
    batch["text"] = text
    batch["image"] = torch.from_numpy(np.stack(image, axis=0))
    batch["fmri_L"] = torch.from_numpy(np.stack(fmri_L, axis=0))
    batch["fmri_R"] = torch.from_numpy(np.stack(fmri_R, axis=0))

    return batch

def collation_fn_test(samples):
    
    batch_size = len(samples)
    batch = dict(batch_size=batch_size)
    
    text = []
    image = []
    fmri_R = []
    fmri_L = []
    
    for sample in samples:
        
        text.append(random.choice(sample["text"]))
        image.append(sample["image"])
        
        fmri_L.append(np.mean(sample["fmri_L"], axis=0))
        fmri_R.append(np.mean(sample["fmri_R"], axis=0))
    
    batch["text"] = text
    batch["image"] = torch.from_numpy(np.stack(image, axis=0))
    batch["fmri_L"] = torch.from_numpy(np.stack(fmri_L, axis=0))
    batch["fmri_R"] = torch.from_numpy(np.stack(fmri_R, axis=0))

    return batch

def get_nsd_dataloader(
    subj: int,
    spilt: typing.Literal["train", "val", "test"],
    dataset_path: os.PathLike,
    batch_size: int,
    load_image: bool = True,
    load_text: bool = True,
    load_whole_brain: bool = False,
    load_L_brain: bool = True,
    load_R_brain: bool = True,
    shuffle_size: int = 2000,
    seed: int = 0,
    cache_size: int = 10*1024**3,
) -> DataLoader:
    
    rename_dict = dict()
    if load_image:
        rename_dict["image"] = "image.npy"
    if load_text:
        rename_dict["text"] = "text.json"
    if load_whole_brain:
        if load_L_brain:
            rename_dict["fmri_L"] = "fmri.l.npy"
        if load_R_brain:
            rename_dict["fmri_R"] = "fmri.r.npy"
    else:
        if load_L_brain:
            rename_dict["fmri_L"] = "fmri.l.nsdgeneral.npy"
        if load_R_brain:
            rename_dict["fmri_R"] = "fmri.r.nsdgeneral.npy"
    
    path = os.path.join(dataset_path, spilt)
    tar_files = [file for file in os.listdir(path) if f"subj0{subj}" in file]
    tar_files = [os.path.join(path, file) for file in tar_files]
    is_train = spilt == "train"
    dataset = WebDataset(
        urls=tar_files, 
        resampled=is_train, 
        nodesplitter=split_by_node, 
        shardshuffle=is_train and (shuffle_size > 0),
        cache_dir="/dev/shm",
        cache_size=cache_size,
    )
    
    if is_train:
        dataset = dataset.shuffle(shuffle_size, rng=random.Random(seed))
    
    dataset = dataset.decode("torch", handler=lambda x: x).rename(**rename_dict)
    
    dataset = dataset.batched(
        batch_size,
        partial=True,
        collation_fn=collation_fn_train if is_train else collation_fn_test
    )
    
    if is_train:
        num_gpus = torch.cuda.device_count()
        num_batches = math.ceil(8500 / (batch_size * num_gpus))
        dataset = dataset.with_epoch(num_batches)

    dataloader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=None,
        pin_memory=True,
        shuffle=False
    )
    return dataloader

def get_nsd_smri(
    dataset_path: os.PathLike,
    subjs: typing.List[int],
    load_whole_brain: bool = False
) -> typing.Dict[str, typing.Dict[str, torch.Tensor]]:
    
    smri = dict()
    
    for subj in subjs:
        smri_L_path = os.path.join(dataset_path, "smri", f"subj0{subj}", "smri.L.npy" if load_whole_brain else "smri.L.nsdgeneral.npy")
        smri_R_path = os.path.join(dataset_path, "smri", f"subj0{subj}", "smri.R.npy" if load_whole_brain else "smri.R.nsdgeneral.npy")
    
        smri_L = torch.from_numpy(np.load(smri_L_path)).to(torch.float32)
        smri_R = torch.from_numpy(np.load(smri_R_path)).to(torch.float32)
        
        smri[str(subj)] = dict(smri_L=smri_L, smri_R=smri_R)
    
    return smri