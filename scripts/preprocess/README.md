# NSD Dataset Setup Guide

[中文](README.zh.md)

## 1. NSD Data Access Agreement & Download

Before downloading NSD data, you must:

1. Agree to the [Natural Scenes Dataset Terms and Conditions](https://naturalscenesdataset.org/)
2. Fill out the [NSD Data Access Form](https://naturalscenesdataset.org/)

Visit [https://naturalscenesdataset.org/](https://naturalscenesdataset.org/) for details. The NSD dataset is very large (~2TB). We provide a minimal necessary file list at [nsd_raw_files.json](nsd_raw_files.json) to reduce download time.

## 2. Preprocessing Pipeline

### Quick Start

Edit [preprocess_all_in_one.sh](preprocess_all_in_one.sh) and set your NSD raw data path:

```bash
NSD_RAW_PATH="/your/path/to/nsd/raw"
```

Then run:

```bash
bash preprocess_all_in_one.sh
```

### Individual Steps

You can also run each script individually:

```bash
# 1. Preprocess images (stimuli)
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_image.py $NSD_RAW_PATH

# 2. Preprocess text (captions)
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_text.py

# 3. Preprocess fMRI data
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_fmri.py $NSD_RAW_PATH

# 4. Preprocess structural MRI (smri)
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_smri.py $NSD_RAW_PATH

# 5. Generate ROI masks
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_roi_mask.py $NSD_RAW_PATH

# 6. Create webdataset tar files
PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_webdataset.py
```

## 3. Preprocessed Dataset

If you don't want to preprocess from scratch, we provide a fully preprocessed version on HuggingFace:

**🔗 [NSD-Sphere on HuggingFace](https://huggingface.co/datasets/yusijin/NSD-Sphere)** 