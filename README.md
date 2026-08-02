<div align="center">

# CompGCN-NFM

### Relation-aware knowledge graph embeddings for drug-target interaction prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An improved version of **DistMult-NFM** that replaces its bilinear knowledge
graph encoder with **CompGCN**, enabling relation-aware message passing before
biological feature fusion and interaction prediction.

</div>

---

## Overview

Drug-target interaction (DTI) prediction benefits from both the relational
structure of a biomedical knowledge graph and the intrinsic properties of drugs
and proteins. This repository combines those complementary signals in a neural
factorization machine (NFM).

The original **DistMult-NFM** pipeline represents entities with DistMult, a
bilinear knowledge graph embedding model. The improved **CompGCN-NFM** pipeline
uses Composition-based Graph Convolutional Networks to propagate information
over neighboring entities and relations. The resulting graph representations
are fused with drug and protein descriptors and passed to the NFM classifier.

```text
Biomedical knowledge graph ──> CompGCN embeddings ──┐
                                                    ├──> Feature fusion ──> NFM ──> DTI probability
Drug + protein descriptors ─────────────────────────┘
```

### What is improved?

| Component | DistMult-NFM | CompGCN-NFM |
|---|---|---|
| Knowledge graph encoder | DistMult bilinear scoring | Relation-aware graph convolution |
| Graph context | Individual triples | Multi-hop neighborhood aggregation |
| Entity-relation interaction | Multiplicative | Configurable `sub`, `mult`, or `corr` composition |
| Downstream predictor | Neural Factorization Machine | Neural Factorization Machine |

## Results

The models were evaluated on **Yamanishi08** using **10-fold cross-validation**
under the warm-start protocol. Results are reported for a balanced setting
(1 positive : 1 negative) and a more realistic imbalanced setting
(1 positive : 10 negatives).

| Model | ROC-AUC (1:1) | PR-AUC (1:1) | ROC-AUC (1:10) | PR-AUC (1:10) |
|:---|---:|---:|---:|---:|
| Logistic Regression | 0.7139 | 0.6613 | 0.7255 | 0.1726 |
| Random Forest | 0.9024 | 0.9014 | 0.9486 | 0.7743 |
| XGBoost | 0.9051 | 0.8948 | 0.9225 | 0.6198 |
| DistMult-NFM | 0.9413 | 0.9390 | 0.9804 | 0.9315 |
| **CompGCN-NFM (ours)** | **0.9689** | **0.9681** | **0.9866** | **0.9362** |

Compared with DistMult-NFM, CompGCN-NFM improves every reported metric:

- **+0.0276 ROC-AUC** and **+0.0291 PR-AUC** in the balanced 1:1 setting.
- **+0.0062 ROC-AUC** and **+0.0047 PR-AUC** in the imbalanced 1:10 setting.

The strongest result is **0.9866 ROC-AUC / 0.9362 PR-AUC** under 1:10 class
imbalance, showing that relation-aware graph propagation improves on the old
DistMult representation while retaining strong positive-class retrieval.

> The table contains the final cross-validation results recorded in
> [`report/main_result.csv`](report/main_result.csv). PR-AUC is particularly
> informative in the 1:10 setting because the positive class is rare.

## Repository structure

```text
KGE_NFM/
├── src/
│   ├── kge_nfm.py       # Main PyTorch experiment runner
│   ├── kge.py           # DistMult and CompGCN training and embeddings
│   ├── nfm.py           # NFM model and training loop
│   ├── data_utils.py    # Datasets, folds, KG, and descriptor features
│   ├── metrics_utils.py # ROC-AUC and PR-AUC helpers
│   └── utils.py         # Reproducibility, device, and CLI helpers
├── data/                # Benchmark datasets and precomputed folds/features
├── output/              # Experiment artifacts
├── logs/                # Training logs
├── baseline/            # Classical machine-learning baselines
└── report/              # Experiment results and project report
```

The root-level `kge_nfm.py`, `kge_rf.py`, and `deepdti.py` files are preserved
legacy implementations. Use the actively maintained PyTorch pipeline in `src/`
for new experiments.

## Installation

Use Python 3.10 or newer in a clean virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

The supplied requirements use PyTorch's CUDA 12.1 wheel index and include
PyKEEN, NumPy, pandas, scikit-learn, and tqdm.

## Quick start

Run a one-fold, one-epoch CompGCN-NFM smoke test:

```bash
python src/kge_nfm.py \
  --dataset yamanishi_08 \
  --split warm_start_1_10 \
  --folds 1 \
  --kge-model compgcn \
  --kge-epochs 1 \
  --nfm-epochs 1 \
  --device auto \
  --no-tqdm
```

## Reproduce the CompGCN-NFM experiment

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
  --kge-batch-size 1024 \
  --batch-size 20000 \
  --nfm-sparse-embedding-dim 50 \
  --nfm-lr 0.001 \
  --nfm-weight-decay 0.00001 \
  --nfm-hidden-units 128,128 \
  --nfm-patience 10 \
  --output-dir output/compgcn_nfm
```

`--compgcn-composition` supports the three entity-relation composition
operators from CompGCN: `sub`, `mult`, and `corr`.

To run the earlier DistMult-NFM model for comparison, change the encoder:

```bash
python src/kge_nfm.py --kge-model distmult
```

## Supported data

The current PyTorch runner supports:

- `yamanishi_08`
- `BioKG`
- `hetionet`

`luo's_dataset` is present in `data/`, but is not enabled because this checkout
does not contain its knowledge graph triples file.

## Outputs

Each run produces fold-specific checkpoints, curves, predictions, and summary
metrics under the selected `--output-dir`:

```text
output/compgcn_nfm/
├── model/kge_nfm_fold_0.pt
├── curve/roc/0.csv
├── curve/pr/0.csv
├── curve/roc_nfm/0.csv
├── curve/pr_nfm/0.csv
├── predictions/fold_0.csv
└── auc/kge_nfm_torch_auc.csv
```

Per fold, the summary includes standalone KGE metrics (`roc_auc`, `pr_auc`) and
final fused NFM metrics (`roc_auc_nfm`, `pr_auc_nfm`).

## Legacy environment

The original root-level scripts depend on Python 3.6, TensorFlow 1.15,
Ampligraph 1.3.2, DeepCTR 0.8.4, pandas 1.1.5, NumPy 1.18.4, and
scikit-learn 0.24.1. They are retained for reference and are not required by the
current PyTorch implementation.

## License

This project is released under the [MIT License](LICENSE).
