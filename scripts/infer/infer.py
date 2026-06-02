import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import accelerate
import numpy as np
import torch
import torchvision
import yaml
from diffusers.models import AutoencoderKL, DualTransformer2DModel, Transformer2DModel, UNet2DConditionModel
from diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet import UNetFlatConditionModel
from diffusers.schedulers import DDIMScheduler
from PIL import Image
from tqdm.auto import tqdm

from NeurIPS.modules import fMRISemanticModel, fMRIPerceptionModel
from NeurIPS.datasets import get_nsd_dataloader, get_nsd_smri

import diffusers.models.transformers.dual_transformer_2d as dual_trans_module
_original_dual_transformer_forward = dual_trans_module.DualTransformer2DModel.forward
def patched_dual_transformer_forward(self, hidden_states, encoder_hidden_states=None, *args, timestep=None, attention_mask=None, cross_attention_kwargs=None, return_dict=True, **kwargs):
    kwargs.pop('encoder_attention_mask', None)
    return _original_dual_transformer_forward(self, hidden_states, encoder_hidden_states, timestep, attention_mask, cross_attention_kwargs, return_dict)
dual_trans_module.DualTransformer2DModel.forward = patched_dual_transformer_forward

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = os.path.join(ROOT_PATH, ".outputs")
DATA_PATH = os.path.join(ROOT_PATH, ".data")
CONFIGS_PATH = os.path.join(ROOT_PATH, "configs")

def convert_to_dual_attention(image_unet: UNet2DConditionModel, text_unet: UNetFlatConditionModel):
    for name, module in image_unet.named_modules():
        if isinstance(module, Transformer2DModel):
            parent_name, index = name.rsplit(".", 1)
            index = int(index)

            image_transformer = image_unet.get_submodule(parent_name)[index]
            text_transformer = text_unet.get_submodule(parent_name)[index]

            config = image_transformer.config
            dual_transformer = DualTransformer2DModel(
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                in_channels=config.in_channels,
                num_layers=config.num_layers,
                dropout=config.dropout,
                norm_num_groups=config.norm_num_groups,
                cross_attention_dim=config.cross_attention_dim,
                attention_bias=config.attention_bias,
                sample_size=config.sample_size,
                num_vector_embeds=config.num_vector_embeds,
                activation_fn=config.activation_fn,
                num_embeds_ada_norm=config.num_embeds_ada_norm,
            )
            dual_transformer.transformers[0] = image_transformer
            dual_transformer.transformers[1] = text_transformer

            image_unet.get_submodule(parent_name)[index] = dual_transformer
            image_unet.register_to_config(dual_cross_attention=True)

    return image_unet

def set_transformer_params(image_unet: UNet2DConditionModel, mix_ratio: float = 0.5):
    condition_types = ("text", "image")
    for name, module in image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = mix_ratio

            for i, type in enumerate(condition_types):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer 
    
    return image_unet

def main(
    _config_path: os.PathLike,
    project_semantic: str,
    project_perception: str,
    name_semantic: str,
    name_perception: str,
    seed: int,
    
    dataset_path: os.PathLike = None,
    semantic_model_ckpt_name: str = "last",
    perception_model_ckpt_name: str = "last",
    reconstruction_name: str = "reconstructions",
    
    # inference config
    use_semantic_model: bool = True,
    use_perception_model: bool = True,
    num_inference_steps: int = 50,
    text_image_mixup_ratio: float = 0.5,
    guidance_scale: float = 7.5,
    batch_size: int = 64,
    latent_predict_ratio: float = 0.1
):
    
    _args = locals()
    
    # =============================================================================== #
    # 0. check args
    
    do_classifier_free_guidance = guidance_scale > 0.0
    
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
    experiment_dir = os.path.join(OUTPUT_PATH, project_semantic, name_semantic, "last-run")
    output_dir = os.path.join(experiment_dir, reconstruction_name)
    if os.path.exists(os.path.join(output_dir, "run.time")):
        with open(os.path.join(output_dir, "run.time"), "r") as file:
            last_time = file.read()
        os.rename(output_dir, os.path.join(experiment_dir, f"{reconstruction_name}-{last_time}"))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run.time"), "w") as file:
        file.write(start_time)
    
    # 1.3 save config
    config_file_name = os.path.join(output_dir, "config.json")
    with open(config_file_name, "w") as file:
        json.dump(_args, file, indent=4)
    del config_file_name
    
    # 1.4 save version
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
    
    # 1.5 save Python script
    current_file = os.path.abspath(__file__)
    destination_file = os.path.join(output_dir, os.path.basename(current_file))
    shutil.copy(current_file, destination_file)
    current_file = _config_path
    destination_file = os.path.join(output_dir, os.path.basename(current_file))
    shutil.copy(current_file, destination_file)
    del current_file, destination_file
    
    del _args, start_time
    
    # 1.6 load configs
    with open(os.path.join(experiment_dir, "config.json"), "r") as file:
        subjs = json.load(file)["subjs"]
    
    # =============================================================================== #
    # 2. init dataloaders
    
    test_dls = []
    num_samples = []
    if dataset_path is None:
        dataset_path = os.path.join(DATA_PATH, "nsd_sphere")
    
    for subj in subjs:
        test_dls.append(get_nsd_dataloader(
            subj,
            spilt="test",
            dataset_path=dataset_path,
            batch_size=batch_size
        ))
        with open(os.path.join(dataset_path, f"metadata_subj0{subj}.json"), "r") as file:
            num_samples.append(json.load(file)["totals"]["test"])
    
    num_samples = sum(num_samples)
    
    smri_data = get_nsd_smri(dataset_path, subjs)
    
    # =============================================================================== #
    # 3. init model & optimizer & learning-rate scheduler
    
    # 3.1 init fmri model
    if use_semantic_model:
        ckpt_path = os.path.join(experiment_dir, "checkpoints", semantic_model_ckpt_name)
        semantic_model = fMRISemanticModel.from_pretrained(ckpt_path).requires_grad_(False).eval().cuda()
    if use_perception_model:
        ckpt_path = os.path.join(OUTPUT_PATH, project_perception, name_perception, "last-run", "checkpoints", perception_model_ckpt_name)
        preception_model = fMRIPerceptionModel.from_pretrained(ckpt_path).requires_grad_(False).eval().cuda()
    
    # 3.2 init diffusion model
    diffusion_name = "shi-labs/versatile-diffusion"
    scheduler = DDIMScheduler.from_pretrained(diffusion_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(diffusion_name, subfolder="vae", use_safetensors=False).requires_grad_(False).eval().cuda()
    image_unet = UNet2DConditionModel.from_pretrained(diffusion_name, subfolder="image_unet", use_safetensors=False).requires_grad_(False).eval()
    text_unet = UNetFlatConditionModel.from_pretrained(diffusion_name, subfolder="text_unet", use_safetensors=False).requires_grad_(False).eval()
    scheduler.set_timesteps(num_inference_steps)
    
    # 3.2 prepare image-text dual guidance versatile-diffusion
    image_unet = convert_to_dual_attention(image_unet, text_unet)
    image_unet = set_transformer_params(image_unet, mix_ratio=text_image_mixup_ratio)
    text_unet = None
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_unet = image_unet.cuda()
    
    # =============================================================================== #
    # 6. inference!!! good luck :)
    
    process_bar = tqdm(range(num_samples), desc="Inference")
    
    for dl, subj in zip(test_dls, subjs):
        
        process_bar.set_description(f"Inference on subj {subj}")
        rec_dir = os.path.join(output_dir, f"subj0{subj}")
        os.makedirs(os.path.join(rec_dir, "pngs"), exist_ok=True)
        
        sample_idx = 0
        images_rec = []
        images_gt = []
        
        for batch in dl:
            
            fmri = torch.cat([batch["fmri_L"], batch["fmri_R"]], dim=1).cuda()
            fmri_L = batch["fmri_L"].unsqueeze(-1).cuda()
            fmri_R = batch["fmri_R"].unsqueeze(-1).cuda()
            bs = fmri_L.shape[0]
            subj_ids = [str(subj)] * bs
            smri_L = torch.stack([smri_data[subj]['smri_L'] for subj in subj_ids]).permute(0, 2, 1) \
                .to(device=fmri_L.device, dtype=fmri_L.dtype, non_blocking=True)
            smri_R = torch.stack([smri_data[subj]['smri_R'] for subj in subj_ids]).permute(0, 2, 1) \
                .to(device=fmri_L.device, dtype=fmri_L.dtype, non_blocking=True)
            
            
            # 6.1 get outputs of semantic model
            if use_semantic_model:
                image_emb, text_emb = semantic_model(fmri_L, fmri_R, smri_L, smri_R)
                image_emb = image_emb.reshape(bs, 257, 768)
                text_emb = text_emb.reshape(bs, 77, 768)
                image_emb = image_emb / torch.norm(image_emb[:, :1, :], dim=-1, keepdim=True)
                text_emb = text_emb / torch.norm(text_emb[:, :1, :], dim=-1, keepdim=True)

            # 6.2 prepare unconditional embeddings if needed
            if use_semantic_model and do_classifier_free_guidance:
                uncond_image_emb = torch.zeros_like(image_emb).cuda()
                uncond_text_emb = torch.zeros_like(text_emb).cuda()
                image_emb = torch.cat([uncond_image_emb, image_emb], dim=0)
                text_emb = torch.cat([uncond_text_emb, text_emb], dim=0)
            if use_semantic_model:
                dual_prompt_embeddings = torch.cat([text_emb, image_emb], dim=1)
            
            # 6.3 prepare latents
            c = image_unet.config.in_channels
            h = w = image_unet.config.sample_size * vae_scale_factor
            latents = torch.randn((bs, c, h // vae_scale_factor, w // vae_scale_factor)).cuda()
            if use_perception_model:
                latents_pred, _ = preception_model(fmri)
                if use_semantic_model:
                    latents = (1.0 - latent_predict_ratio) * latents + latent_predict_ratio * latents_pred
                else:
                    latents = latents_pred
            
            # 6.4 denoising loop
            if use_semantic_model:
                timesteps = scheduler.timesteps
                for t in timesteps:
                    latents_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                    latents_input = scheduler.scale_model_input(latents_input, t)
                    noise_pred = image_unet(latents_input, t, encoder_hidden_states=dual_prompt_embeddings).sample
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_fmri = noise_pred.chunk(2, dim=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_fmri - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # 6.5 decode
            image_pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            image_pred = (image_pred / 2 + 0.5).clamp(0, 1).cpu()
            image_gt = (batch["image"] / 255.0).permute(0, 3, 1, 2).cpu()
            image_gt = torch.nn.functional.interpolate(image_gt, size=(h, w), mode='bilinear', align_corners=False)

            torch.cuda.empty_cache()
            
            # 6.6 save and output
            for i in range(bs):
                sample_gt = image_gt[i, :, :, :]
                sample_pred = image_pred[i, :, :, :]
                sample = torch.stack([sample_gt, sample_pred], dim=0)
                grid = torchvision.utils.make_grid(sample, nrow=2, padding=2, normalize=False)
                grid = grid.permute(1, 2, 0).mul(255).byte().cpu().numpy()
                Image.fromarray(grid).save(os.path.join(rec_dir, "pngs", f"{sample_idx:04}.png"))
                sample_idx += 1
            image_pred = torch.nn.functional.interpolate(image_pred, size=(256, 256), mode='bilinear', align_corners=False)
            image_gt = torch.nn.functional.interpolate(image_gt, size=(256, 256), mode='bilinear', align_corners=False)
            images_rec.append(image_pred)
            images_gt.append(image_gt)
            
            process_bar.update(bs)
        
        images_rec = torch.cat(images_rec, dim=0).numpy()
        images_gt = torch.cat(images_gt, dim=0).numpy()
        np.save(os.path.join(rec_dir, "images_reconstruction.npy"), images_rec)
        np.save(os.path.join(rec_dir, "images_groundtruth.npy"), images_gt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    base_path = os.path.dirname(__file__)
    parser.add_argument('--config_path', type=str, default=os.path.join(CONFIGS_PATH, "infer.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    main(_config_path=args.config_path, **config)