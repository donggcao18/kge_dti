#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME= ./.cache
export HF_DATASETS_CACHE= ./.cache


split="drug_coldstart"
dataset="yamanishi_08"
model=distmult
python src/kge_nfm.py \
  --dataset $dataset \
  --output-dir ./output/kge_nfm_$split \
  --data-root /kaggle/input/datasets/ngcaovn/kge-dti/data \
  --split $split \
  --kge-model $model \
  --folds 10 \
  --device auto \
  --embedding-dim 400 \
  --kge-epochs 50 \
  --nfm-epochs 200 \
  --kge-batch-size 8192 \
  --batch-size 20000 \
  --nfm-sparse-embedding-dim 50 \
  --nfm-lr 0.001 \
  --nfm-weight-decay 0.00001 \
  --nfm-hidden-units 128,128 \
  --nfm-patience 10 
