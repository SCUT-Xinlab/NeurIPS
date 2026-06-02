import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import accelerate
import numpy as np
import scipy as sp
import pandas as pd
import torch
import yaml

from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = os.path.join(ROOT_PATH, ".outputs")
DATA_PATH = os.path.join(ROOT_PATH, ".data")
CONFIGS_PATH = os.path.join(ROOT_PATH, "configs")

@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True, device='cpu'):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1

def main(
    _config_path: os.PathLike,
    project: str,
    name: str,
    seed: int,
    rec_name: str = "reconstructions",
    device: str = "cuda:0"
):
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    accelerate.utils.set_seed(seed)
    
    experiment_dir = os.path.join(OUTPUT_PATH, project, name, "last-run")
    with open(os.path.join(experiment_dir, "config.json"), "r") as file:
        subjs = json.load(file)["subjs"]
        
    results = []
    
    for subj in subjs:
        
        # 1. init metrics
        subj_dir = os.path.join(experiment_dir, rec_name, f"subj0{subj}")
        start_time = datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
        with open(os.path.join(subj_dir, "metrics.run.time"), "w") as file:
            file.write(start_time)
        
        # 2. save Python script
        current_file = os.path.abspath(__file__)
        destination_file = os.path.join(subj_dir, os.path.basename(current_file))
        shutil.copy(current_file, destination_file)
        current_file = _config_path
        destination_file = os.path.join(subj_dir, os.path.basename(current_file))
        shutil.copy(current_file, destination_file)
        del current_file, destination_file
        
        # 3. init data
        all_images = np.load(os.path.join(subj_dir, "images_groundtruth.npy"))
        all_brain_recons = np.load(os.path.join(subj_dir, "images_reconstruction.npy"))
        all_images = torch.from_numpy(all_images)
        all_brain_recons = torch.from_numpy(all_brain_recons)
        number = all_images.shape[0]
        print("Images shape:", all_images.shape)
        print("Recons shape:", all_brain_recons.shape)
        print("Num images:", number)
        print("####################################################")

        # 4. PixCorr
        preprocess = transforms.Compose([
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
        all_brain_recons_flattened = preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()
        corrsum = 0
        for i in tqdm(range(number)):
            corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
        corrmean = corrsum / number
        pixcorr = corrmean
        print(f"PixCorr: {pixcorr:.4f}")
        del all_images_flattened, all_brain_recons_flattened
        torch.cuda.empty_cache()

        # 5. SSIM
        from skimage.color import rgb2gray
        from skimage.metrics import structural_similarity as ssim
        preprocess = transforms.Compose([
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
        ])
        img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
        recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
        ssim_score=[]
        for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
            ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
        _ssim = np.mean(ssim_score)
        print(f"SSIM: {_ssim:.4f}")
        del img_gray, recon_gray
        torch.cuda.empty_cache()

        # 6. AlexNet
        from torchvision.models import alexnet, AlexNet_Weights
        alex_weights = AlexNet_Weights.IMAGENET1K_V1
        alex_model = create_feature_extractor(
            alexnet(weights=alex_weights), return_nodes=['features.4','features.11']
        ).to(device)
        alex_model.eval().requires_grad_(False)
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                alex_model, preprocess, 'features.4', device=device)
        alexnet2 = np.mean(all_per_correct)
        print(f"AlexNet(2): {alexnet2:.4f}")
        all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                alex_model, preprocess, 'features.11', device=device)
        alexnet5 = np.mean(all_per_correct)
        print(f"AlexNet(5): {alexnet5:.4f}")
        del alex_model
        torch.cuda.empty_cache()

        # 7. InceptionV3
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                                return_nodes=['avgpool']).to(device=device, dtype=torch.float32)
        inception_model.eval().requires_grad_(False)
        preprocess = transforms.Compose([
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        all_brain_recons = all_brain_recons.to(torch.float32)
        all_images = all_images.to(torch.float32)
        all_per_correct = two_way_identification(all_brain_recons, all_images,
                                                inception_model, preprocess, 'avgpool', device=device)
        inception = np.mean(all_per_correct)
        print(f"InceptionV3: {inception:.4f}")
        del inception_model
        torch.cuda.empty_cache()

        # 8. CLIP
        import clip
        clip_model, preprocess = clip.load("ViT-L/14", device=device)
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        all_per_correct = two_way_identification(all_brain_recons, all_images,
                                                clip_model.encode_image, preprocess, None, device=device) # final layer
        clip_ = np.mean(all_per_correct)
        print(f"CLIP: {clip_:.4f}")
        del clip_model
        torch.cuda.empty_cache()

        # 9. Efficient Net
        from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
        weights = EfficientNet_B1_Weights.DEFAULT
        eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
        eff_model.eval().requires_grad_(False)
        preprocess = transforms.Compose([
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        gt = eff_model(preprocess(all_images).to(device))['avgpool']
        gt = gt.reshape(len(gt),-1).cpu().numpy()
        fake = eff_model(preprocess(all_brain_recons).to(device))['avgpool']
        fake = fake.reshape(len(fake),-1).cpu().numpy()
        effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
        print(f"EffNet-B: {effnet:.4f}")
        del eff_model
        torch.cuda.empty_cache()

        # 10. SwAV
        swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        swav_model = create_feature_extractor(swav_model, 
                                            return_nodes=['avgpool']).to(device)
        swav_model.eval().requires_grad_(False)
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        gt = swav_model(preprocess(all_images).to(device))['avgpool']
        gt = gt.reshape(len(gt),-1).cpu().numpy()
        fake = swav_model(preprocess(all_brain_recons).to(device))['avgpool']
        fake = fake.reshape(len(fake),-1).cpu().numpy()
        swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
        print("SwAV:",swav,"\n")
        del swav_model
        torch.cuda.empty_cache()

        # 11. save
        data = {
            "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"],
            "Value": [pixcorr, _ssim, alexnet2, alexnet5, inception, clip_, effnet, swav],
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(subj_dir, f'results_subj0{subj}_{number}samples.csv'), sep='\t', index=False)
        
        df = df.assign(subj=subj)
        results.append(df)
    
    results = pd.concat(results)
    results = results.pivot(index="subj", columns="Metric", values="Value").reset_index()
    columns_order = ["subj", "PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"]
    results = results[columns_order]
    mean_row = results.iloc[:, 1:].mean().values
    std_row = results.iloc[:, 1:].std().values
    mean_df = pd.DataFrame([["mean"] + list(mean_row)], columns=results.columns)
    std_df = pd.DataFrame([["std"] + list(std_row)], columns=results.columns)
    results = pd.concat([results, mean_df, std_df], ignore_index=True)
    results.to_csv(os.path.join(experiment_dir, rec_name, 'results.csv'), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval")
    base_path = os.path.dirname(__file__)
    parser.add_argument('--config_path', type=str, default=os.path.join(CONFIGS_PATH, "eval.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    main(_config_path=args.config_path, **config)