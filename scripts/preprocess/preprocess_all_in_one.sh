#!/bin/bash

# Put your NSD raw data here
# Only need to download files listed in nsd_raw_files.json
NSD_RAW_PATH="/home/yusijin/datasets/nsd/raw"
PREPROCESSED_PATH=""  # default to .outputs/nsd/preprocess
TEMP_PATH=""          # default to .outputs/temp
FINAL_OUTPUT_PATH=""  # default to .outputs/nsd/nsd_sphere

# PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_image.py \
#     $NSD_RAW_PATH \
#     ${PREPROCESSED_PATH:+--target_root_path $PREPROCESSED_PATH} \
#     ${TEMP_PATH:+--temp_path $TEMP_PATH}

# PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_text.py \
#     ${PREPROCESSED_PATH:+--target_root_path $PREPROCESSED_PATH} \
#     ${TEMP_PATH:+--temp_path $TEMP_PATH}

# PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_fmri.py \
#     $NSD_RAW_PATH \
#     ${PREPROCESSED_PATH:+--target_root_path $PREPROCESSED_PATH} \
#     ${TEMP_PATH:+--temp_path $TEMP_PATH}

# PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_smri.py \
#     $NSD_RAW_PATH \
#     ${PREPROCESSED_PATH:+--target_root_path $PREPROCESSED_PATH}

# # Generates mask files (saved to src/NeurIPS/modules/sphere/utils/npy/mask)
# PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_roi_mask.py \
#     $NSD_RAW_PATH \
#     ${PREPROCESSED_PATH:+--target_root_path $PREPROCESSED_PATH}

PYTHONPATH=src:$PYTHONPATH python scripts/preprocess/preprocess_webdataset.py \
    ${PREPROCESSED_PATH:+--preprocessed_path $PREPROCESSED_PATH} \
    ${FINAL_OUTPUT_PATH:+--output_path $FINAL_OUTPUT_PATH} \
    ${TEMP_PATH:+--temp_path $TEMP_PATH}