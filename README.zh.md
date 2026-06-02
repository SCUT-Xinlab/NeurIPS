<div align="center">
  <h1>NeurIPS: Neuro-anatomical Inductive Priors for Sphere-based Brain Decoding</h1>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2605.24993"><img alt="Paper"
    src="https://img.shields.io/badge/Paper-arXiv-DD0000?color=DD0000&logoColor=white"/></a>
  <a href="https://huggingface.co/datasets/yusijin/NSD-Sphere"><img alt="Dataset"
    src="https://img.shields.io/badge/Dataset-NSD--Sphere-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://huggingface.co/yusijin/NeurIPS"><img alt="Model"
    src="https://img.shields.io/badge/Model-NeurIPS-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://arxiv.org/pdf/2605.24993"><img alt="Paper PDF"
    src="https://img.shields.io/badge/PDF-arXiv-DD0000?color=DD0000&logoColor=white"/></a>
    <a href="README.md"><img alt="English"
    src="https://img.shields.io/badge/English-English-brightgreen?color=brightgreen&logoColor=white"/></a>
</div>


## 最新消息

🎉 2026/06/02 🎉 NeurIPS 已开源。

🎉 2026/04/30 🎉 NeurIPS 已被 **ICML 2026** 接收。

## 安装

1. 下载代码仓库：

```bash
git clone https://github.com/SCUT-Xinlab/NeurIPS.git
```

2. 创建 conda 环境并安装运行代码所需的包：

```bash
conda create -n NeurIPS python=3.12 -y
conda activate NeurIPS
pip install -r requirements.txt
```

3. 同意 Natural Scenes Dataset 的[条款和条件](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions)，并填写 [NSD 数据访问申请表](https://forms.gle/xue2bCdM9LaFNMeb7)。

4. 下载全部文件，直接运行：

```bash
bash downloads.sh
```

这会将预处理好的 NSD-Sphere 数据集下载到 `.data/nsd_sphere/`，将 ConvNeXt 预训练权重下载到 `.models`（用于 Perception 模型初始化），并将我们训练好的模型权重（SemanticModel 和 PerceptionModel）下载到 `.outputs/NeurIPS-official/`。你也可以在 [NSD-Sphere](https://huggingface.co/datasets/yusijin/NSD-Sphere) 和 [NeurIPS](https://huggingface.co/yusijin/NeurIPS) 上找到我们的预处理数据和训练好的模型权重。

> 如遇网络问题，可在 `downloads.sh` 中修改 `HF_ENDPOINT="hf-mirror.com"`。

5. （可选）如果你想从原始 NSD 数据自行预处理，从 [NSD dataset](https://natural-scenes-dataset.s3.amazonaws.com/index.html) 下载原始数据，编辑并运行：

```bash
# 先在 preprocess_all_in_one.sh 中设置你的 NSD 原始数据路径
bash scripts/preprocess/preprocess_all_in_one.sh
```

也可以单独运行每个脚本。详见 [preprocess guides](scripts/preprocess/README.md)。


## 训练

> 训练需要显存至少为 80GB 的 GPU。

1. 训练语义模型:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:$PYTHONPATH \
accelerate launch scripts/train/semantic_model.py
```

你可以在 [configs/train_semantic_model.yaml](configs/train_semantic_model.yaml) 中修改训练配置。

2. 训练感知模型:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:$PYTHONPATH \
accelerate launch scripts/train/perception_model.py
```

你可以在 [configs/train_perception_model.yaml](configs/train_perception_model.yaml) 中修改训练配置。

## 推理和评估

1. 推理:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:$PYTHONPATH \
accelerate launch scripts/infer/infer.py --config configs/infer.yaml
```

输出会保存到 `.outputs/NeurIPS-official/<project>/<name>/last-run/<rec_name>/`。

2. 评估

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:$PYTHONPATH \
accelerate launch scripts/infer/eval.py --config configs/eval.yaml
```

---

## 微调到新被试

1. 微调语义模型

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:$PYTHONPATH \
accelerate launch scripts/finetune/semantic_model.py
```

2. 微调感知模型

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:$PYTHONPATH \
accelerate launch scripts/finetune/perception_model.py
```

## 引用
```bibtex
@article{yu2026neurips,
  title={NeurIPS: Neuro-anatomical Inductive Priors for Sphere-based Brain Decoding},
  author={Yu, Sijin and Chen, Zijiao and Yang, Zhenyu and Tan, Zihao and Xu, Jiakun and Liu, Zhongliang and Chen, Shengxian and Wu, Wenxuan and Xu, Xiangmin and Zhang, Xin},
  journal={arXiv preprint arXiv:2605.24993},
  year={2026}
}
```