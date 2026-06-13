#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-./.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./.cache}"

split="${SPLIT:-warm_start_1_10}"
dataset="${DATASET:-yamanishi_08}"
data_root="${DATA_ROOT:-/kaggle/input/datasets/ngcaovn/kge-dti/data}"
variant="${ABLATION_VARIANT:-all}"
kge_checkpoint_dir="${KGE_CHECKPOINT_DIR:-./output/kge_nfm_compgcn_${split}/model}"
output_dir="${OUTPUT_DIR:-./output/descriptor_ablation_compgcn_${split}}"

python src/kge_nfm.py \
  --dataset "$dataset" \
  --data-root "$data_root" \
  --split "$split" \
  --folds "${FOLDS:-10}" \
  --device "${DEVICE:-auto}" \
  --nfm-only \
  --kge-checkpoint-dir "$kge_checkpoint_dir" \
  --descriptor-ablation "$variant" \
  --output-dir "$output_dir" \
  --protein-pca-components "${PROTEIN_PCA_COMPONENTS:-100}" \
  --kge-batch-size "${KGE_BATCH_SIZE:-8192}" \
  --nfm-epochs "${NFM_EPOCHS:-200}" \
  --batch-size "${NFM_BATCH_SIZE:-20000}" \
  --nfm-sparse-embedding-dim "${NFM_SPARSE_EMBEDDING_DIM:-50}" \
  --nfm-lr "${NFM_LR:-0.001}" \
  --nfm-weight-decay "${NFM_WEIGHT_DECAY:-0.00001}" \
  --nfm-dropout "${NFM_DROPOUT:-0.0}" \
  --nfm-hidden-units "${NFM_HIDDEN_UNITS:-128,128}" \
  --nfm-patience "${NFM_PATIENCE:-10}" \
  --seed "${SEED:-0}" \
  --no-tqdm
