import argparse
import itertools
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal

import accelerate
import kornia
import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import AutoencoderKL
from kornia.augmentation.container import AugmentationSequential
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm_type

from NeurIPS.modules import ConvnextXL, fMRIPerceptionModel
from NeurIPS.datasets import get_nsd_dataloader

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = os.path.join(ROOT_PATH, ".outputs")
DATA_PATH = os.path.join(ROOT_PATH, ".data")
CONFIGS_PATH = os.path.join(ROOT_PATH, "configs")
MODEL_PATH = os.path.join(ROOT_PATH, ".models")

def _soft_cont_loss(
    student_preds: torch.Tensor,
    teacher_preds: torch.Tensor,
    teacher_aug_preds: torch.Tensor,
    temp: float = 0.125
):
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T) / temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T) / temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T) / temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T) / temp
    
    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2) / 2
    return loss

def step_train(
    fmri_model: fMRIPerceptionModel,
    vae_model: AutoencoderKL,
    convnext_model: ConvnextXL,
    optimizer: torch.optim.AdamW,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: accelerate.Accelerator,
    fmri: torch.Tensor,
    image: torch.Tensor,
    image_aug: AugmentationSequential,
    max_grad_norm = None
) -> Dict[str, float]:
    
    optimizer.zero_grad()
    
    # 0. forward
    bs = fmri.shape[0]
    latent_gt = vae_model.encode(2 * image - 1).latent_dist.mode() * 0.18215
    latent_pred, feats_pred = fmri_model(fmri)
    if convnext_model is not None:
        _mean = torch.tensor([0.485, 0.456, 0.406]).to(fmri.device).reshape(1, 3, 1, 1)
        _std = torch.tensor([0.228, 0.224, 0.225]).to(fmri.device).reshape(1, 3, 1, 1)
        _, feats_gt = convnext_model((image - _mean) / _std)
        _, feats_aug = convnext_model((image_aug(image) - _mean) / _std)
    
    loss = 0.0
    reporte_loss = dict()
    
    # 1. condition loss
    if convnext_model is not None:
        cont_loss = _soft_cont_loss(
            torch.nn.functional.normalize(feats_pred.reshape(-1, feats_pred.shape[-1]), dim=-1),
            torch.nn.functional.normalize(feats_gt.reshape(-1, feats_gt.shape[-1]), dim=-1),
            torch.nn.functional.normalize(feats_aug.reshape(-1, feats_aug.shape[-1]), dim=-1),
            temp=0.075
        )
        loss += cont_loss
        reporte_loss["cont/train"] = cont_loss.item()
    
    # 2. latent loss
    latent_loss = torch.nn.functional.mse_loss(latent_pred, latent_gt)
    loss += latent_loss
    reporte_loss["latent/train"] = latent_loss.item()
    
    # 3. backward
    if torch.isnan(loss).any():
        raise RuntimeError(f"find NaN in loss")
    accelerator.backward(loss)
    if max_grad_norm is not None and accelerator.sync_gradients:
        accelerator.clip_grad_norm_(fmri_model.parameters(), max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    reporte_loss["loss/train"] = loss.item()
    
    return reporte_loss

@torch.no_grad()
def step_eval(
    fmri_model: fMRIPerceptionModel,
    vae_model: AutoencoderKL,
    convnext_model: ConvnextXL,
    fmri: torch.Tensor,
    image: torch.Tensor,
    image_aug: AugmentationSequential,
) -> Dict[str, float]:
    
    # 0. forward
    bs = fmri.shape[0]
    latent_gt = vae_model.encode(2 * image - 1).latent_dist.mode() * 0.18215
    latent_pred, feats_pred = fmri_model(fmri)
    if convnext_model is not None:
        _mean = torch.tensor([0.485, 0.456, 0.406]).to(fmri.device).reshape(1, 3, 1, 1)
        _std = torch.tensor([0.228, 0.224, 0.225]).to(fmri.device).reshape(1, 3, 1, 1)
        _, feats_gt = convnext_model((image - _mean) / _std)
        _, feats_aug = convnext_model((image_aug(image) - _mean) / _std)
    
    loss = 0.0
    reporte_loss = dict()
    
    # 1. condition loss
    if convnext_model is not None:
        cont_loss = _soft_cont_loss(
            torch.nn.functional.normalize(feats_pred.reshape(-1, feats_pred.shape[-1]), dim=-1),
            torch.nn.functional.normalize(feats_gt.reshape(-1, feats_gt.shape[-1]), dim=-1),
            torch.nn.functional.normalize(feats_aug.reshape(-1, feats_aug.shape[-1]), dim=-1),
            temp=0.075
        )
        loss += cont_loss
        reporte_loss["cont/val"] = cont_loss.item()
    
    # 2. latent loss
    latent_loss = torch.nn.functional.mse_loss(latent_pred, latent_gt)
    loss += latent_loss
    reporte_loss["latent/val"] = latent_loss.item()
    
    reporte_loss["loss/val"] = loss.item()
    
    return reporte_loss

def epoch_train(
    epoch: int,
    fmri_model: fMRIPerceptionModel, 
    vae_model: AutoencoderKL,
    convnext_model: ConvnextXL,
    optimizer: torch.optim.AdamW,
    lr_scheduler: torch.optim.lr_scheduler.OneCycleLR,
    train_dls: list,
    val_dls: list,
    subjs: List[int],
    accelerator: accelerate.Accelerator,
    epoch_interval_eval: int,
    epoch_interval_save: int,
    save_dir: os.PathLike,
    train_loss_csv_file: os.PathLike,
    eval_loss_csv_file: os.PathLike,
    image_aug: AugmentationSequential,
    dtype: torch.dtype,
    logger: logging.Logger,
    process_bar: tqdm_type,
    num_eval_samples: int,
    max_grad_norm = None
):
    logger.debug(f"epoch {epoch} start")
    
    if dtype is None:
        dtype = torch.float32
    
    process_bar.set_description(f"Training | epoch {epoch}")
    
    # 1. evaluate the model
    if epoch % epoch_interval_eval == 0:
        fmri_model.eval()
        epoch_eval(
            epoch=epoch,
            fmri_model=fmri_model,
            vae_model=vae_model,
            convnext_model=convnext_model,
            val_dls=val_dls,
            subjs=subjs,
            accelerator=accelerator,
            dtype=dtype,
            eval_loss_csv_file=eval_loss_csv_file,
            image_aug=image_aug,
            num_eval_samples=num_eval_samples
        )
        fmri_model.train()

    # 2. training loop
    results: Dict[str, List[float]] = dict()
    for batches in zip(*train_dls):
        
        # prepare batch data
        # NOTE: batches: [{"fmri_L": ..., "fmri_R": ..., "image": ..., "text": ...}, ...]
        fmri_L = torch.cat([b["fmri_L"] for b in batches], dim=0) \
            .to(device=accelerator.device, dtype=dtype, non_blocking=True)
        fmri_R = torch.cat([b["fmri_R"] for b in batches], dim=0) \
            .to(device=accelerator.device, dtype=dtype, non_blocking=True)
        fmri = torch.cat([fmri_L, fmri_R], dim=1)
        image = torch.cat([b["image"] for b in batches], dim=0) \
            .to(device=accelerator.device, dtype=dtype, non_blocking=True)
        image = image.permute((0, 3, 1, 2)).div(255.0)
        image = torch.nn.functional.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False)
        del batches
        
        # training step
        result = step_train(
            fmri_model=fmri_model,
            vae_model=vae_model,
            convnext_model=convnext_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            fmri=fmri,
            image=image,
            image_aug=image_aug,
            max_grad_norm=max_grad_norm
        )
        
        # report loss
        for k, v in result.items():
            if k not in results.keys():
                results[k] = [v]
            else:
                results[k].append(v)
        
        process_bar.update(1)
    
    # 3. log loss
    results = {
        k: accelerator.reduce(torch.tensor(np.mean(v), device=accelerator.device), reduction="mean").item()
        for k, v in results.items()
    }
    if accelerator.is_main_process:
        results["epoch"] = epoch
        results["lr"] = lr_scheduler.get_last_lr()[0]
        accelerator.log(results)
        results: pd.DataFrame = pd.DataFrame([results])
        csv_file_exists = os.path.exists(train_loss_csv_file)
        results.to_csv(
            train_loss_csv_file,
            mode="a" if csv_file_exists else "w",
            header=not csv_file_exists,
            index=False
        )
    
    # 4. save checkpoint
    if epoch % epoch_interval_save == 0:
        
        if accelerator.is_main_process:
            
            # save for resume train
            os.makedirs(os.path.join(save_dir, "resume"), exist_ok=True)
            accelerator.save_state(os.path.join(save_dir, "resume"))
            with open(os.path.join(save_dir, "resume", "epoch.json"), "w") as file:
                json.dump([epoch], file)
            
            # save checkpoint
            os.makedirs(os.path.join(save_dir, f"epoch-{epoch}"), exist_ok=True)
            fmri_model.save_pretrained(os.path.join(save_dir, f"epoch-{epoch}"))
            os.makedirs(os.path.join(save_dir, f"last"), exist_ok=True)
            fmri_model.save_pretrained(os.path.join(save_dir, f"last"))
    
    # 5. sync GPUs
    accelerator.wait_for_everyone()
        
@torch.no_grad()
def epoch_eval(
    epoch: int,
    fmri_model: fMRIPerceptionModel, 
    vae_model: AutoencoderKL,
    convnext_model: ConvnextXL,
    val_dls: list,
    subjs: List[int],
    accelerator: accelerate.Accelerator,
    dtype: torch.dtype,
    eval_loss_csv_file: os.PathLike,
    image_aug: AugmentationSequential,
    num_eval_samples: int
):
    results_total: Dict[str, List[float]] = dict()
    
    eval_process_bar = tqdm(
        range(num_eval_samples), 
        desc=f"Evaluation (epoch {epoch})", 
        leave=False,
        disable=not accelerator.is_main_process
    )
    
    for val_dl, subj in zip(val_dls, subjs):
        
        results: Dict[str, List[float]] = dict()
        for batch in val_dl:
            
            # preprocessing
            fmri_L = batch["fmri_L"] \
                .to(device=accelerator.device, dtype=dtype, non_blocking=True)
            fmri_R = batch["fmri_R"] \
                .to(device=accelerator.device, dtype=dtype, non_blocking=True)
            fmri = torch.cat([fmri_L, fmri_R], dim=1)
            image = batch["image"].permute((0, 3, 1, 2)).div(225.0) \
                .to(device=accelerator.device, dtype=dtype, non_blocking=True)
            image = torch.nn.functional.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False)
            
            # forward
            bs = fmri_L.shape[0]
            result = step_eval(
                fmri_model=fmri_model,
                vae_model=vae_model,
                convnext_model=convnext_model,
                fmri=fmri,
                image=image,
                image_aug=image_aug
            )
            
            # report loss
            for k, v in result.items():
                if k not in results.keys():
                    results[k] = [v]
                else:
                    results[k].append(v)
            
            eval_process_bar.update(bs)
        
        # log loss
        results = {
            k: accelerator.reduce(torch.tensor(np.mean(v), device=accelerator.device), reduction="mean").item()
            for k, v in results.items()
        }
        for k, v in results.items():
            if k not in results_total.keys():
                results_total[k] = [v]
            else:
                results_total[k].append(v)
        if accelerator.is_main_process:
            results["subj"] = subj
            results["epoch"] = epoch
            results: pd.DataFrame = pd.DataFrame([results])
            csv_file_exists = os.path.exists(eval_loss_csv_file)
            results.to_csv(
                eval_loss_csv_file,
                mode="a" if csv_file_exists else "w",
                header=not csv_file_exists,
                index=False
            )
    
    results_total = {k: np.mean(v) for k, v in results_total.items()}
    if accelerator.is_main_process:
        results_total["epoch"] = epoch
        accelerator.log(results_total)

def main(
    _config_path: os.PathLike,
    project: str,
    name: str,
    seed: int,
    dataset_path: os.PathLike = None,
    subjs: List[int] = [1, 2, 5, 7],
    pretrained_convnext_path: os.PathLike = None,
    
    # training config
    num_epochs: int = 600,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 0.01,
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no",
    dtype: Literal["mixed", "fp32", "fp16", "bf16"] = "fp32",
    gradient_accumulation_steps: int = 1,
    resume_training: bool = False,
    scale_learning_rate: bool = False,
    epoch_interval_eval: int = 1,
    epoch_interval_save: int = 50,
    max_grad_norm: float = None,
    
    # model config
    model_hidden_dim: int = 4096,
    model_num_blocks: int = 4,
    model_dropout_input: float = 0.5,
    model_dropout_block: float = 0.25,
    model_upsampler_layer_per_block: int = 1,
    model_maps_proj_dim: int = 512
):
    
    _args = locals()
    
    # =============================================================================== #
    # 0. check args
    
    if (mixed_precision != "no" and dtype != "mixed"):
        raise ValueError(f"`mixed_precision` is '{mixed_precision}' and `dtype` is {dtype}")
    
    # =============================================================================== #
    # 1. init experiment
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 1.1 init seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    accelerate.utils.set_seed(seed)
    
    # 1.2 init output_dir
    start_time = datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
    output_dir = os.path.join(OUTPUT_PATH, project, name, "last-run")
    if os.path.exists(os.path.join(output_dir, "run.time")) and not resume_training:
        with open(os.path.join(output_dir, "run.time"), "r") as file:
            last_time = file.read()
        os.rename(output_dir, os.path.join(OUTPUT_PATH, project, name, last_time))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run.time"), "w") as file:
        file.write(start_time)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "loss"), exist_ok=True)
    
    # 1.3 init acclerator
    accelerator = accelerate.Accelerator(
        device_placement=True,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        cpu=False,
        log_with="wandb"
    )
    
    # 1.4 init logger
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    logger_file = os.path.join(output_dir, "logs", f"rank{accelerator.local_process_index}.log")
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    del console_handler, file_handler
    
    # 1.5 init writer
    accelerator.init_trackers(
        project_name=project,
        init_kwargs={"wandb": {
            "name": name,
            "dir": output_dir,
            "config": _args,
            "mode": "offline"
        }}
    )
    
    # 1.6 save config
    config_file_name = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file_name):
        config_file_name = os.path.join(output_dir, f"{start_time}.config.json")
    with open(config_file_name, "w") as file:
        json.dump(_args, file, indent=4)
    del config_file_name
    
    # 1.7 save version
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    packages_dict = {}
    lines = result.stdout.split('\n')
    for line in lines[2:]:
        if line:
            package_name, version = line.split()[:2]
            packages_dict[package_name] = version
    imported_packages = set()
    for module_name in sys.modules.keys():
        if "." in module_name:
            module_name = module_name.split(".")[0]
        imported_packages.add(module_name)
    filtered_packages = {pkg: version for pkg, version in packages_dict.items() if pkg in imported_packages}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        git_commit = result.stdout.strip()
    except subprocess.CalledProcessError:
        git_commit = "N/A (due to not a git repository)"
    version_file_name = os.path.join(output_dir, "version.json")
    if os.path.exists(version_file_name):
        version_file_name = os.path.join(output_dir, f"{start_time}.version.json")
    with open(version_file_name, "w") as file:
        json.dump(
            obj={
                "datetime": start_time,
                "git_commit": git_commit,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": sys.version,
                "python_compiler": platform.python_compiler(),
                "python_packages": filtered_packages
            },
            fp=file,
            indent=4
        )
    del result, packages_dict, lines, package_name, version, imported_packages
    del module_name, filtered_packages, git_commit, version_file_name
    
    # 1.8 save Python script
    current_file = os.path.abspath(__file__)
    destination_file = os.path.join(output_dir, os.path.basename(current_file))
    if os.path.exists(destination_file):
        destination_file = os.path.join(
            output_dir, 
            f"{start_time}.{os.path.basename(current_file).replace('.py', '')}.py"
        )
    shutil.copy(current_file, destination_file)
    current_file = _config_path
    destination_file = os.path.join(output_dir, os.path.basename(current_file))
    if os.path.exists(destination_file):
        destination_file = os.path.join(
            output_dir, 
            f"{start_time}.{os.path.basename(current_file).replace('.yaml', '')}.yaml"
        )
    shutil.copy(current_file, destination_file)
    del current_file, destination_file
    
    del _args, start_time
    
    logger.info("successfully init experiment")
    
    # =============================================================================== #
    # 2. init dataloaders
    
    train_dls = []
    val_dls = []
    if dataset_path is None:
        dataset_path = os.path.join(DATA_PATH, "nsd_sphere")
    
    # 2.1 init train dataloaders
    num_samples = []
    for subj in subjs:
        train_dls.append(get_nsd_dataloader(
            subj,
            spilt="train",
            dataset_path=dataset_path,
            batch_size=train_batch_size,
            shuffle_size=2000,
            seed=seed
        ))
        with open(os.path.join(dataset_path, f"metadata_subj0{subj}.json"), "r") as file:
            num_samples.append(json.load(file)["totals"]["train"])
    max_num_samples = max(num_samples)
    num_steps_per_epoch = math.ceil(max_num_samples / (accelerator.num_processes * train_batch_size))
    
    # 2.2 init validation dataloders
    val_num_samples = []
    for subj in subjs:
        val_dls.append(get_nsd_dataloader(
            subj,
            spilt="val",
            dataset_path=dataset_path,
            batch_size=val_batch_size
        ))
        with open(os.path.join(dataset_path, f"metadata_subj0{subj}.json"), "r") as file:
            val_num_samples.append(json.load(file)["totals"]["val"])
    num_eval_samples = sum(val_num_samples)
    del val_num_samples
    
    logger.info("successfully init dataloader")
    
    # =============================================================================== #
    # 3. init model & optimizer & learning-rate scheduler
    
    # 3.1 init precision
    if mixed_precision == "no":
        if dtype == "fp32":
            precision = torch.float32
        elif dtype == "fp16":
            precision = torch.float16
        elif dtype == "bf16":
            precision = torch.bfloat16
        else:
            raise ValueError(f"unknown `dtype`: {dtype}")
    else:
        precision = None
    
    # 3.2 init vae model
    vae_model = AutoencoderKL.from_pretrained("shi-labs/versatile-diffusion", subfolder="vae", use_safetensors=False)
    vae_model: AutoencoderKL = vae_model.requires_grad_(False).eval()
    if precision is not None:
        vae_model.to(precision)
    
    # 3.3 init condition model
    if pretrained_convnext_path is None:
        pretrained_convnext_path = os.path.join(MODEL_PATH, "convnext_xlarge_alpha0.75_fullckpt.pth")
    convnext_model = ConvnextXL(pretrained_convnext_path)
    convnext_model.requires_grad_(False).eval()
    
    # 3.3 init fmri model
    fmri_model = fMRIPerceptionModel(
        input_dim=9488,
        hidden_dim=model_hidden_dim,
        num_blocks=model_num_blocks,
        dropout_input=model_dropout_input,
        dropout_block=model_dropout_block,
        upsampler_channels=(64, 128, 256),
        upsampler_layer_per_block=model_upsampler_layer_per_block,
        use_maps_proj=True,
        maps_proj_dim=model_maps_proj_dim
    )
    if precision is not None:
        fmri_model.to(precision)
        
    logger.info("successfully init model")
    
    # 3.4 init optimizer
    if scale_learning_rate:
        learning_rate = learning_rate * accelerator.num_processes * gradient_accumulation_steps
    no_wight_decay = ['bias', 'norm', 'class_embedding', 'queries']
    learnable_parameters = [
        {
            "params": [p for n, p in fmri_model.named_parameters() if not any(nwd in n for nwd in no_wight_decay)],
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in fmri_model.named_parameters() if any(nwd in n for nwd in no_wight_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = torch.optim.AdamW(learnable_parameters, lr=learning_rate)
    
    logger.info("successfully init optimizer")
    
    # 3.5 init learning-rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        total_steps=num_steps_per_epoch * num_epochs,
        final_div_factor=1000,
        last_epoch=-1, 
        pct_start= 2 / num_epochs,
    )
    
    logger.info("successfully init learning-rate scheduler")
    
    # =============================================================================== #
    # 4. prepare multi-GPUs
    
    # 4.1 prepare in acclerator
    fmri_model, vae_model, convnext_model, train_dls, val_dls = accelerator.prepare(
        fmri_model, vae_model, convnext_model, train_dls, val_dls
    )
    
    # 4.2 update train dataloaders
    train_dls = [
        itertools.cycle(dl) if n < max_num_samples else dl
        for dl, n in zip(train_dls, num_samples)
    ]
    del num_samples, max_num_samples
    
    logger.info("successfully prepare multi-GPUs")
    
    # =============================================================================== #
    # 5. init data agumentation
    
    image_aug = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.RandomSolarize(p=0.2),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
        kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
        data_keys=["input"],
    )
    
    # =============================================================================== #
    # 6. training!!! good luck :)
    
    process_bar = tqdm(
        range(num_epochs * num_steps_per_epoch), 
        desc="Training",
        disable=not accelerator.is_main_process
    )
    for epoch in range(num_epochs):
        epoch_train(
            epoch=epoch,
            fmri_model=fmri_model,
            vae_model=vae_model,
            convnext_model=convnext_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dls=train_dls,
            val_dls=val_dls,
            subjs=subjs,
            accelerator=accelerator,
            epoch_interval_eval=epoch_interval_eval,
            epoch_interval_save=epoch_interval_save,
            save_dir=os.path.join(output_dir, "checkpoints"),
            train_loss_csv_file=os.path.join(output_dir, "loss", "train.csv"),
            eval_loss_csv_file=os.path.join(output_dir, "loss", "val.csv"),
            image_aug=image_aug,
            dtype=precision,
            logger=logger,
            process_bar=process_bar,
            num_eval_samples=num_eval_samples,
            max_grad_norm=max_grad_norm
        )
    
    # =============================================================================== #
    # 7. save

    if accelerator.is_main_process:
        os.makedirs(os.path.join(output_dir, "checkpoints", f"final"), exist_ok=True)
        fmri_model.save_pretrained(os.path.join(output_dir, "checkpoints", f"final"))
        os.makedirs(os.path.join(output_dir, "checkpoints", f"last"), exist_ok=True)
        fmri_model.save_pretrained(os.path.join(output_dir, "checkpoints", f"last"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train perception model")
    base_path = os.path.dirname(__file__)
    parser.add_argument('--config_path', type=str, default=os.path.join(CONFIGS_PATH, "train_perception_model.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    main(_config_path=args.config_path, **config)