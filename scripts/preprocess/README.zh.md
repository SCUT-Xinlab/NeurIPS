# NSD 数据集配置指南

[English](README.md)

## 1. NSD 数据访问协议与下载

在下载 NSD 数据之前，您必须：

1. 同意 [Natural Scenes Dataset 条款和条件](https://naturalscenesdataset.org/)
2. 填写 [NSD 数据访问申请表](https://naturalscenesdataset.org/)

访问 [https://naturalscenesdataset.org/](https://naturalscenesdataset.org/) 了解详情。NSD 数据集体积庞大（约 2TB），我们在 [nsd_raw_files.json](nsd_raw_files.json) 中提供了最小必要文件列表以缩短下载时间。

## 2. 预处理流程

### 快速开始

编辑 [preprocess_all_in_one.sh](preprocess_all_in_one.sh) 并设置您的 NSD 原始数据路径：

```bash
NSD_RAW_PATH="/your/path/to/nsd/raw"
```

然后运行：

```bash
bash preprocess_all_in_one.sh
```

### 分步执行

您也可以单独运行每个脚本：

```bash
# 1. 预处理图像（刺激物）
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_image.py $NSD_RAW_PATH

# 2. 预处理文本
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_text.py

# 3. 预处理 fMRI 数据
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_fmri.py $NSD_RAW_PATH

# 4. 预处理结构 MRI (smri)
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_smri.py $NSD_RAW_PATH

# 5. 生成 ROI 掩码
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_roi_mask.py $NSD_RAW_PATH

# 6. 创建 webdataset tar 文件
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_webdataset.py
```

## 3. 预处理数据集

如果您不想从零开始预处理，我们在 HuggingFace 上提供了完整预处理版本：

**🔗 [HuggingFace 上的 NSD-Sphere](https://huggingface.co/datasets/yusijin/NSD-Sphere)**
