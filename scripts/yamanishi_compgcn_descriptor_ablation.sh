#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-./.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./.cache}"

split="${SPLIT:-warm_start_1_10}"
dataset="${DATASET:-yamanishi_08}"
data_root="${DATA_ROOT:-/kaggle/input/datasets/ngcaovn/kge-dti/data}"
variant="${ABLATION_VARIANT:-without_descriptors}"
output_dir="${OUTPUT_DIR:-./output/descriptor_ablation_compgcn_${split}}"

kge_model="${KGE_MODEL:-compgcn}"
embedding_dim="${EMBEDDING_DIM:-200}"
kge_epochs="${KGE_EPOCHS:-50}"
kge_batch_size="${KGE_BATCH_SIZE:-8192}"
kge_lr="${KGE_LR:-0.001}"
kge_num_negs="${KGE_NUM_NEGS:-5}"
compgcn_layers="${COMPGCN_LAYERS:-1}"
compgcn_dropout="${COMPGCN_DROPOUT:-0.1}"
compgcn_composition="${COMPGCN_COMPOSITION:-mult}"

python src/kge_nfm.py \
  --dataset "$dataset" \
  --data-root "$data_root" \
  --split "$split" \
  --folds "${FOLDS:-10}" \
  --device "${DEVICE:-auto}" \
  --kge-model "$kge_model" \
  --embedding-dim "$embedding_dim" \
  --kge-epochs "$kge_epochs" \
  --kge-batch-size "$kge_batch_size" \
  --kge-lr "$kge_lr" \
  --kge-num-negs "$kge_num_negs" \
  --compgcn-layers "$compgcn_layers" \
  --compgcn-dropout "$compgcn_dropout" \
  --compgcn-composition "$compgcn_composition" \
  --descriptor-ablation "$variant" \
  --output-dir "$output_dir" \
  --protein-pca-components "${PROTEIN_PCA_COMPONENTS:-100}" \
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
