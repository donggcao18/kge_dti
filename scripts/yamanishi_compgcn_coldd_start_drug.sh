#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="./.cache"
export HF_DATASETS_CACHE="./.cache"

split="drug_coldstart"
dataset="yamanishi_08"
data_root="${DATA_ROOT:-/kaggle/input/datasets/ngcaovn/kge-dti/data}"
model="compgcn"

python src/kge_nfm.py \
  --dataset "$dataset" \
  --output-dir "./output/kge_nfm_compgcn_${split}" \
  --data-root "$data_root" \
  --split "$split" \
  --folds 5 \
  --device auto \
  --kge-model "$model" \
  --embedding-dim 200 \
  --kge-epochs 50 \
  --kge-batch-size 8192 \
  --kge-num-negs 5 \
  --compgcn-layers 2 \
  --compgcn-dropout 0.1 \
  --compgcn-composition mult \
  --nfm-epochs 200 \
  --batch-size 20000 \
  --nfm-field-embedding-dim 100 \
  --nfm-lr 0.001 \
  --nfm-weight-decay 1e-3 \
  --nfm-dropout 0.2 \
  --nfm-hidden-units 128,128 \
  --nfm-patience 10
