#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-./.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./.cache}"

split="${SPLIT:-warm_start_1_10}"
dataset="yamanishi_08"
data_root="${DATA_ROOT:-/kaggle/input/datasets/ngcaovn/kge-dti/data}"
kge_checkpoint_dir="${KGE_CHECKPOINT_DIR:-./output/kge_nfm_compgcn_${split}/model}"
output_dir="${OUTPUT_DIR:-./output/nfm_ablation_compgcn_${split}}"

python src/kge_nfm.py \
  --dataset "$dataset" \
  --data-root "$data_root" \
  --split "$split" \
  --folds 10 \
  --device auto \
  --nfm-only \
  --kge-checkpoint-dir "$kge_checkpoint_dir" \
  --descriptor-ablation all \
  --output-dir "$output_dir" \
  --protein-pca-components 100 \
  --kge-batch-size 8192 \
  --nfm-epochs 200 \
  --batch-size 20000 \
  --nfm-sparse-embedding-dim 50 \
  --nfm-lr 0.001 \
  --nfm-weight-decay 0.00001 \
  --nfm-hidden-units 128,128 \
  --nfm-patience 10 \
  --no-tqdm
