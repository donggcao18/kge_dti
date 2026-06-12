# KGE_NFM

A drug-target interaction (DTI) prediction repo based on knowledge graph embeddings and a neural factorization machine.

## Current Structure

- `src/kge_nfm.py`: main PyTorch experiment runner.
- `src/data_utils.py`: dataset paths, fold loading, KG loading, descriptor feature loading.
- `src/kge.py`: PyKEEN DistMult/CompGCN training, KGE scoring, entity embedding extraction.
- `src/nfm.py`: local PyTorch NFM model and training loop.
- `src/metrics_utils.py`: ROC-AUC and PR-AUC helpers.
- `src/utils.py`: seed, device, output-directory, and argument helpers.
- `data/`: benchmark datasets and precomputed folds/features.
- `output/`: experiment outputs.
- `logs/`: log output directory.

Legacy root scripts:

- `kge_nfm.py`: original TensorFlow 1 / Ampligraph / DeepCTR implementation.
- `kge_rf.py`: original RF and KGE_RF baseline.
- `deepdti.py`: original DeepPurpose baseline.

## Dependencies

Install the current PyTorch path with:

```bash
pip install -r requirements.txt
```

The implementation uses the PyTorch wheel build for NVIDIA CUDA 12.1
(`cu121`). To install the PyTorch packages directly from the official CUDA
12.1 wheel index, run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

In a Jupyter or Colab notebook, the equivalent command is:

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This command specifies the CUDA build (`cu121`) but does not pin a numeric
PyTorch release. The installed `torch`, `torchvision`, and `torchaudio`
versions therefore depend on the compatible packages available from the CUDA
12.1 index at installation time.

`requirements.txt` currently uses:

- PyTorch / TorchVision / TorchAudio from the CUDA 12.1 (`cu121`) wheel index
- `pykeen >= 1.10, < 1.12`
- `numpy >= 2.0, < 3`
- `pandas == 2.2.2`
- `scikit-learn >= 1.5, < 1.8`
- `tqdm >= 4.66, < 5`

Use Python 3.10+ in a clean virtual environment.

## Data

Supported by the current PyTorch runner:

- `yamanishi_08`
- `BioKG`
- `hetionet`

`luo's_dataset` is included in `data/`, but is not enabled in `src/kge_nfm.py` because this checkout does not include a KG triples file for that dataset.

## Run

Smoke test on Yamanishi08:

```bash
python src/kge_nfm.py \
  --dataset yamanishi_08 \
  --split warm_start_1_10 \
  --folds 1 \
  --kge-epochs 1 \
  --nfm-epochs 1 \
  --device auto \
  --no-tqdm
```

Full Yamanishi08 run:

```bash
python src/kge_nfm.py \
  --dataset yamanishi_08 \
  --split warm_start_1_10 \
  --folds 10 \
  --device auto \
  --embedding-dim 400 \
  --kge-pca-components 200 \
  --kge-epochs 50 \
  --nfm-epochs 2000 \
  --kge-batch-size 1024 \
  --batch-size 20000 \
  --nfm-field-embedding-dim 50 \
  --nfm-lr 0.001 \
  --nfm-weight-decay 0.00001 \
  --nfm-hidden-units 128,128 \
  --nfm-patience 10 \
  --output-dir output/kge_nfm_torch
```

CompGCN run:

```bash
python src/kge_nfm.py \
  --dataset yamanishi_08 \
  --split warm_start_1_10 \
  --folds 10 \
  --device auto \
  --kge-model compgcn \
  --compgcn-layers 2 \
  --compgcn-dropout 0.1 \
  --compgcn-composition mult \
  --embedding-dim 400 \
  --kge-epochs 50 \
  --nfm-epochs 2000 \
  --no-tqdm
```

`--compgcn-composition` accepts the paper's three entity-relation composition families: `sub`, `mult`, and `corr`.

## Outputs

The PyTorch runner writes fold-specific artifacts:

```text
output/kge_nfm_torch/model/kge_nfm_fold_0.pt
output/kge_nfm_torch/curve/roc/0.csv
output/kge_nfm_torch/curve/pr/0.csv
output/kge_nfm_torch/curve/roc_nfm/0.csv
output/kge_nfm_torch/curve/pr_nfm/0.csv
output/kge_nfm_torch/predictions/fold_0.csv
output/kge_nfm_torch/auc/kge_nfm_torch_auc.csv
```

Per fold, metrics include:

- `roc_auc`: standalone KGE ROC-AUC
- `pr_auc`: standalone KGE PR-AUC
- `roc_auc_nfm`: final NFM ROC-AUC
- `pr_auc_nfm`: final NFM PR-AUC

## Legacy Environment

The root-level original scripts are preserved, but they require the old stack:

- Python 3.6
- TensorFlow 1.15
- Ampligraph 1.3.2
- DeepCTR 0.8.4
- pandas 1.1.5
- numpy 1.18.4
- scikit-learn 0.24.1

Use the `src/` implementation for new experiments.
