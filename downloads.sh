#!/bin/bash

# NSD-Sphere Dataset Download Script
# Download from HuggingFace: https://huggingface.co/datasets/yusijin/NSD-Sphere

# ============ Configuration ============
# Download mirror: "huggingface.co" (main site) or "hf-mirror.com" (中国用户)
HF_ENDPOINT="hf-mirror.com"
# ======================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_ROOT/.data/nsd_sphere"

echo "[download] Project root: $PROJECT_ROOT"
echo "[download] Data directory: $DATA_DIR"
echo "[download] Using mirror: $HF_ENDPOINT"

HF_BASE="https://$HF_ENDPOINT"
REPO_ID="yusijin/NSD-Sphere"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[download] Downloading dataset files..."

DOWNLOADED=0
SKIPPED=0
ERRORS=0

# Metadata files
FILES=(
    "metadata_subj01.json"
    "metadata_subj02.json"
    "metadata_subj03.json"
    "metadata_subj04.json"
    "metadata_subj05.json"
    "metadata_subj06.json"
    "metadata_subj07.json"
    "metadata_subj08.json"
)

# Train tar files
TRAIN_FILES=(
    "train/train_subj01_0.tar"
    "train/train_subj01_1.tar"
    "train/train_subj01_2.tar"
    "train/train_subj01_3.tar"
    "train/train_subj01_4.tar"
    "train/train_subj02_0.tar"
    "train/train_subj02_1.tar"
    "train/train_subj02_2.tar"
    "train/train_subj02_3.tar"
    "train/train_subj02_4.tar"
    "train/train_subj03_0.tar"
    "train/train_subj03_1.tar"
    "train/train_subj03_2.tar"
    "train/train_subj03_3.tar"
    "train/train_subj03_4.tar"
    "train/train_subj04_0.tar"
    "train/train_subj04_1.tar"
    "train/train_subj04_2.tar"
    "train/train_subj04_3.tar"
    "train/train_subj05_0.tar"
    "train/train_subj05_1.tar"
    "train/train_subj05_2.tar"
    "train/train_subj05_3.tar"
    "train/train_subj05_4.tar"
    "train/train_subj06_0.tar"
    "train/train_subj06_1.tar"
    "train/train_subj06_2.tar"
    "train/train_subj06_3.tar"
    "train/train_subj06_4.tar"
    "train/train_subj07_0.tar"
    "train/train_subj07_1.tar"
    "train/train_subj07_2.tar"
    "train/train_subj07_3.tar"
    "train/train_subj07_4.tar"
    "train/train_subj08_0.tar"
    "train/train_subj08_1.tar"
    "train/train_subj08_2.tar"
    "train/train_subj08_3.tar"
)

# Val tar files
VAL_FILES=(
    "val/val_subj01_0.tar"
    "val/val_subj02_0.tar"
    "val/val_subj03_0.tar"
    "val/val_subj04_0.tar"
    "val/val_subj05_0.tar"
    "val/val_subj06_0.tar"
    "val/val_subj07_0.tar"
    "val/val_subj08_0.tar"
)

# Test tar files
TEST_FILES=(
    "test/test_subj01_0.tar"
    "test/test_subj02_0.tar"
    "test/test_subj03_0.tar"
    "test/test_subj04_0.tar"
    "test/test_subj05_0.tar"
    "test/test_subj06_0.tar"
    "test/test_subj07_0.tar"
    "test/test_subj08_0.tar"
)

# smri files
SMRI_FILES=(
    "smri/subj01/smri.L.npy"
    "smri/subj01/smri.R.npy"
    "smri/subj01/smri.L.nsdgeneral.npy"
    "smri/subj01/smri.R.nsdgeneral.npy"
    "smri/subj02/smri.L.npy"
    "smri/subj02/smri.R.npy"
    "smri/subj02/smri.L.nsdgeneral.npy"
    "smri/subj02/smri.R.nsdgeneral.npy"
    "smri/subj03/smri.L.npy"
    "smri/subj03/smri.R.npy"
    "smri/subj03/smri.L.nsdgeneral.npy"
    "smri/subj03/smri.R.nsdgeneral.npy"
    "smri/subj04/smri.L.npy"
    "smri/subj04/smri.R.npy"
    "smri/subj04/smri.L.nsdgeneral.npy"
    "smri/subj04/smri.R.nsdgeneral.npy"
    "smri/subj05/smri.L.npy"
    "smri/subj05/smri.R.npy"
    "smri/subj05/smri.L.nsdgeneral.npy"
    "smri/subj05/smri.R.nsdgeneral.npy"
    "smri/subj06/smri.L.npy"
    "smri/subj06/smri.R.npy"
    "smri/subj06/smri.L.nsdgeneral.npy"
    "smri/subj06/smri.R.nsdgeneral.npy"
    "smri/subj07/smri.L.npy"
    "smri/subj07/smri.R.npy"
    "smri/subj07/smri.L.nsdgeneral.npy"
    "smri/subj07/smri.R.nsdgeneral.npy"
    "smri/subj08/smri.L.npy"
    "smri/subj08/smri.R.npy"
    "smri/subj08/smri.L.nsdgeneral.npy"
    "smri/subj08/smri.R.nsdgeneral.npy"
)

# Combine all files
ALL_FILES=(
    "${FILES[@]}"
    "${TRAIN_FILES[@]}"
    "${VAL_FILES[@]}"
    "${TEST_FILES[@]}"
    "${SMRI_FILES[@]}"
)
TOTAL=${#ALL_FILES[@]}
echo "[download] Total files to download: $TOTAL"

for file in "${ALL_FILES[@]}"; do
    FILE_DIR=$(dirname "$file")
    if [ "$FILE_DIR" != "." ]; then
        mkdir -p "$FILE_DIR"
    fi

    URL="$HF_BASE/datasets/$REPO_ID/resolve/main/$file"
    TARGET_FILE="$file"

    echo ""
    echo "[download] Checking: $file"

    if [ -f "$TARGET_FILE" ]; then
        echo "[download] File exists, skipping: $file"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "[download] Downloading: $file"
        if wget -q --show-progress -O "$TARGET_FILE" "$URL"; then
            echo "[download] Downloaded: $file"
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            echo "[download] ERROR: Failed to download: $file"
            rm -f "$TARGET_FILE"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""
echo "=========================================="
echo "[download] Summary:"
echo "  Downloaded: $DOWNLOADED files"
echo "  Skipped (existing): $SKIPPED files"
echo "  Errors: $ERRORS files"
echo "  Data saved to: $DATA_DIR"
echo "=========================================="

if [ $ERRORS -gt 0 ]; then
    echo "[download] Some files failed to download."
    echo "[download] You may need to manually download from:"
    echo "  https://huggingface.co/datasets/yusijin/NSD-Sphere"
    exit 1
fi

echo "[download] All files downloaded successfully!"

# ============ Download ConvNeXt Weight ============
# Download from: https://huggingface.co/yusijin/NeurIPS
CONVNEXT_MODEL_DIR="$PROJECT_ROOT/.models"
CONVNEXT_FILES=(
    "convnext_xlarge_alpha0.75_fullckpt.pth"
)

mkdir -p "$CONVNEXT_MODEL_DIR"

echo ""
echo "=========================================="
echo "[download] Downloading ConvNeXt weights..."

for file in "${CONVNEXT_FILES[@]}"; do
    URL="https://$HF_ENDPOINT/yusijin/NeurIPS/resolve/main/$file"
    TARGET_FILE="$CONVNEXT_MODEL_DIR/$file"

    echo "[download] Checking: $file"

    if [ -f "$TARGET_FILE" ]; then
        echo "[download] File exists, skipping: $file"
    else
        echo "[download] Downloading: $file"
        if wget -q --show-progress -O "$TARGET_FILE" "$URL"; then
            echo "[download] Downloaded: $file"
        else
            echo "[download] ERROR: Failed to download: $file"
            rm -f "$TARGET_FILE"
            exit 1
        fi
    fi
done

echo "[download] ConvNeXt weight saved to: $CONVNEXT_MODEL_DIR"

# ============ Download Model Checkpoints ============
# Download from: https://huggingface.co/yusijin/NeurIPS
OUTPUT_DIR="$PROJECT_ROOT/.outputs/NeurIPS-official"

MODEL_DIRS=(
    "PerceptionModel-subj257"
    "PerceptionModel-subj1257"
    "SemanticModel-subj257"
    "SemanticModel-subj1257"
)

# Get all files for these directories via git ls-tree
echo ""
echo "=========================================="
echo "[download] Fetching file list from repository..."

# Use git ls-tree to get file list (lightweight operation)
FILE_LIST=$(cd /tmp && rm -rf NeurIPS_git 2>/dev/null; GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --no-checkout https://$HF_ENDPOINT/yusijin/NeurIPS.git NeurIPS_git 2>/dev/null && cd NeurIPS_git && git ls-tree -r HEAD --name-only | grep -E "^(${MODEL_DIRS[0]}|${MODEL_DIRS[1]}|${MODEL_DIRS[2]}|${MODEL_DIRS[3]})/")

echo "[download] File list retrieved."
echo "[download] Starting downloads to: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

DOWNLOADED=0
SKIPPED=0
ERRORS=0

while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    # Create target directory structure
    FILE_DIR=$(dirname "$file")
    TARGET_DIR="$OUTPUT_DIR/$FILE_DIR"
    TARGET_FILE="$OUTPUT_DIR/$file"

    if [ "$FILE_DIR" != "." ]; then
        mkdir -p "$TARGET_DIR"
    fi

    URL="https://$HF_ENDPOINT/yusijin/NeurIPS/resolve/main/$file"

    echo "[download] Checking: $file"

    if [ -f "$TARGET_FILE" ]; then
        echo "[download] File exists, skipping: $file"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "[download] Downloading: $file"
        if wget -q --show-progress -O "$TARGET_FILE" "$URL"; then
            echo "[download] Downloaded: $file"
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            echo "[download] ERROR: Failed to download: $file"
            rm -f "$TARGET_FILE"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done <<< "$FILE_LIST"

echo ""
echo "=========================================="
echo "[download] Model Download Summary:"
echo "  Downloaded: $DOWNLOADED files"
echo "  Skipped (existing): $SKIPPED files"
echo "  Errors: $ERRORS files"
echo "  Saved to: $OUTPUT_DIR"
echo "=========================================="

if [ $ERRORS -gt 0 ]; then
    echo "[download] Some files failed to download."
    exit 1
fi

echo "[download] All model files downloaded successfully!"