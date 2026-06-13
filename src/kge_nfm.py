"""CLI runner for the PyTorch KGE_NFM experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from constants import TRIPLE_COLUMNS
from data_utils import (
    DatasetSpec,
    encode_labels,
    get_dataset_spec,
    load_all_dti,
    load_feature_tables,
    load_fold,
    load_kg,
    merge_feature_blocks,
)
from kge import load_kge_checkpoint, pair_embeddings, require_entities, score_triples, train_compgcn, train_distmult
from metrics_utils import pr_auc, roc_auc
from nfm import train_nfm
from utils import ensure_output_dirs, parse_hidden_units, resolve_device, set_seed


ABLATION_VARIANTS = {
    "full": (True, True),
    "without_protein": (True, False),
    "without_morgan": (False, True),
    "without_descriptors": (False, False),
}


def resolve_ablation_variants(args: argparse.Namespace) -> list[str]:
    if args.drug_features_only:
        if args.descriptor_ablation != "none":
            raise ValueError("--drug-features-only cannot be combined with --descriptor-ablation.")
        return ["without_protein"]

    selected = args.descriptor_ablation
    if selected == "none":
        return ["full"]
    if selected == "all":
        return list(ABLATION_VARIANTS)
    return ["full", selected.replace("-", "_")]


def combine_feature_blocks(
    kge_features: np.ndarray,
    drug_features: np.ndarray,
    protein_features: np.ndarray,
    variant: str,
) -> np.ndarray:
    use_morgan, use_protein = ABLATION_VARIANTS[variant]
    blocks = []
    if kge_features.shape[1] > 0:
        blocks.append(kge_features)
    if use_morgan:
        blocks.append(drug_features)
    if use_protein:
        blocks.append(protein_features)
    if not blocks:
        return np.empty((kge_features.shape[0], 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1)


def print_args(args: argparse.Namespace, data_root: Path, device: str) -> None:
    config = vars(args).copy()
    config["resolved_data_root"] = str(data_root)
    config["resolved_device"] = device

    print("Run arguments:")
    for key in sorted(config):
        print(f"  {key}: {config[key]}")


def run_fold(
    fold: int,
    args: argparse.Namespace,
    spec: DatasetSpec,
    kg: pd.DataFrame,
    drug_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    head_encoder: dict[str, int],
    tail_encoder: dict[str, int],
    device: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    fold_number = fold + 1
    print(f"Fold {fold_number} ({args.split}, {args.kge_model})")
    train, test = load_fold(spec, fold)
    output_root = Path(args.output_dir)
    kge_model = None
    triples_factory = None
    losses = []
    if args.nfm_only:
        print("NFM-only mode: KGE training and KGE embedding features are disabled.")
    else:
        train_pos = train.loc[train["label"] == 1, TRIPLE_COLUMNS]
        kge_train = pd.concat([train_pos, kg], ignore_index=True)[TRIPLE_COLUMNS].astype(str)

    if args.reuse_kge_checkpoints:
        checkpoint_dir = Path(args.kge_checkpoint_dir) if args.kge_checkpoint_dir else output_root / "model"
        checkpoint_path = checkpoint_dir / f"kge_nfm_fold_{fold}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"KGE checkpoint not found: {checkpoint_path}. "
                "Pass --kge-checkpoint-dir with the model directory from a completed KGE_NFM run."
            )
        print(f"Loading KGE checkpoint: {checkpoint_path}")
        kge_model, triples_factory, losses, checkpoint_args = load_kge_checkpoint(
            checkpoint_path,
            kge_train,
            device,
        )
        for key in ["dataset", "split"]:
            saved_value = checkpoint_args.get(key)
            current_value = getattr(args, key)
            if saved_value is not None and saved_value != current_value:
                raise ValueError(
                    f"Checkpoint {key} is '{saved_value}', but this run requested '{current_value}'."
                )
        print(f"Loaded KGE model: {checkpoint_args.get('kge_model', 'distmult')}")
    elif not args.nfm_only:
        kge_kwargs = dict(
            train_triples=kge_train,
            device=device,
            embedding_dim=args.embedding_dim,
            epochs=args.kge_epochs,
            batch_size=args.kge_batch_size,
            seed=args.seed + fold,
            lr=args.kge_lr,
            num_negs_per_pos=args.kge_num_negs,
            use_tqdm=not args.no_tqdm,
        )
        if args.kge_model == "compgcn":
            kge_model, triples_factory, losses = train_compgcn(
                **kge_kwargs,
                num_layers=args.compgcn_layers,
                layer_dropout=args.compgcn_dropout,
                composition=args.compgcn_composition,
            )
        else:
            kge_model, triples_factory, losses = train_distmult(**kge_kwargs)

    if triples_factory is not None:
        require_entities(triples_factory, train["head"].tolist() + train["tail"].tolist(), "Training fold")
        require_entities(triples_factory, test["head"].tolist() + test["tail"].tolist(), "Test fold")

    if not args.nfm_only and not args.reuse_kge_checkpoints:
        torch.save(
            {
                "model_state_dict": kge_model.state_dict(),
                "entity_to_id": triples_factory.entity_to_id,
                "relation_to_id": triples_factory.relation_to_id,
                "losses": losses,
                "args": vars(args),
            },
            output_root / "model" / f"kge_nfm_fold_{fold}.pt",
        )

    test_labels = test["label"].to_numpy(dtype=np.float32)
    train_pairs = train[TRIPLE_COLUMNS]
    test_pairs = test[TRIPLE_COLUMNS]
    if args.nfm_only:
        train_kge_features = np.empty((len(train_pairs), 0), dtype=np.float32)
        test_kge_features = np.empty((len(test_pairs), 0), dtype=np.float32)
        kge_scores = None
        roc_value = np.nan
        pr_value = np.nan
    else:
        kge_scores = score_triples(kge_model, triples_factory, test_pairs, device, args.kge_batch_size)
        roc_curve, roc_value = roc_auc(test_labels, kge_scores)
        pr_curve, pr_value = pr_auc(test_labels, kge_scores)
        train_kge_features = pair_embeddings(kge_model, triples_factory, train_pairs, device)
        test_kge_features = pair_embeddings(kge_model, triples_factory, test_pairs, device)
        roc_curve.to_csv(output_root / "curve" / "roc" / f"{fold}.csv", index=False)
        pr_curve.to_csv(output_root / "curve" / "pr" / f"{fold}.csv", index=False)
    train_drug_features, train_protein_features = merge_feature_blocks(train_pairs, drug_df, protein_df, spec)
    test_drug_features, test_protein_features = merge_feature_blocks(test_pairs, drug_df, protein_df, spec)

    if not args.nfm_only:
        print(f"roc_auc: {roc_value:.6f}")
        print(f"pr_auc: {pr_value:.6f}")
    nfm_metrics = []
    for variant in resolve_ablation_variants(args):
        print(f"Training NFM variant: {variant}")
        train_features = combine_feature_blocks(
            train_kge_features,
            train_drug_features,
            train_protein_features,
            variant,
        )
        test_features = combine_feature_blocks(
            test_kge_features,
            test_drug_features,
            test_protein_features,
            variant,
        )
        if train_features.shape[1] > 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_features = scaler.fit_transform(train_features).astype(np.float32)
            test_features = scaler.transform(test_features).astype(np.float32)

        roc_nfm, roc_nfm_value, pr_nfm, pr_nfm_value, nfm_pred = train_nfm(
            train_pairs=train_pairs,
            train_labels=train["label"].to_numpy(dtype=np.float32),
            test_pairs=test_pairs,
            test_labels=test_labels,
            train_features=train_features,
            test_features=test_features,
            head_encoder=head_encoder,
            tail_encoder=tail_encoder,
            sparse_embedding_dim=args.nfm_sparse_embedding_dim,
            epochs=args.nfm_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + fold,
            lr=args.nfm_lr,
            weight_decay=args.nfm_weight_decay,
            dropout=args.nfm_dropout,
            hidden_units=parse_hidden_units(args.nfm_hidden_units),
            patience=args.nfm_patience,
        )

        variant_roc_dir = output_root / "ablation" / "curve" / "roc" / variant
        variant_pr_dir = output_root / "ablation" / "curve" / "pr" / variant
        variant_pred_dir = output_root / "ablation" / "predictions" / variant
        for path in [variant_roc_dir, variant_pr_dir, variant_pred_dir]:
            path.mkdir(parents=True, exist_ok=True)
        roc_nfm.to_csv(variant_roc_dir / f"{fold}.csv", index=False)
        pr_nfm.to_csv(variant_pr_dir / f"{fold}.csv", index=False)

        predictions = test[TRIPLE_COLUMNS + ["label"]].copy()
        if kge_scores is not None:
            predictions["kge_score"] = kge_scores
        predictions["nfm_pred"] = nfm_pred
        predictions.to_csv(variant_pred_dir / f"fold_{fold}.csv", index=False)

        if variant == "full":
            roc_nfm.to_csv(output_root / "curve" / "roc_nfm" / f"{fold}.csv", index=False)
            pr_nfm.to_csv(output_root / "curve" / "pr_nfm" / f"{fold}.csv", index=False)
            predictions.to_csv(output_root / "predictions" / f"fold_{fold}.csv", index=False)

        print(f"roc_auc_nfm ({variant}): {roc_nfm_value:.6f}")
        print(f"pr_auc_nfm ({variant}): {pr_nfm_value:.6f}")
        nfm_metrics.append(
            {
                "fold": fold_number,
                "variant": variant,
                "feature_dim": train_features.shape[1],
                "roc_auc_nfm": roc_nfm_value,
                "pr_auc_nfm": pr_nfm_value,
            }
        )

    kge_metrics = {
        "fold": fold_number,
        "train_size": len(train),
        "test_size": len(test),
        "train_pos": int((train["label"] == 1).sum()),
        "test_pos": int((test["label"] == 1).sum()),
        "roc_auc": roc_value,
        "pr_auc": pr_value,
    }
    return kge_metrics, nfm_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KGE_NFM with a PyKEEN KGE model and a local PyTorch NFM.")
    parser.add_argument("--dataset", default="yamanishi_08", help="Dataset name: yamanishi_08, BioKG, hetionet.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing dataset folders. Defaults to <repo>/data.",
    )
    parser.add_argument(
        "--split",
        default="warm_start_1_10",
        help="Fold split directory under data_folds, e.g. warm_start_1_10, protein_coldstart, drug_coldstart.",
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of folds to run from fold 0.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--kge-epochs", type=int, default=50)
    parser.add_argument("--nfm-epochs", type=int, default=2000)
    parser.add_argument("--kge-batch-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=20000, help="NFM batch size.")
    parser.add_argument("--kge-model", choices=["distmult", "compgcn"], default="distmult")
    parser.add_argument("--embedding-dim", type=int, default=400, help="PyKEEN entity embedding dimension.")
    parser.add_argument("--nfm-sparse-embedding-dim", type=int, default=50)
    parser.add_argument("--protein-pca-components", type=int, default=100)
    parser.add_argument("--kge-lr", type=float, default=1e-3)
    parser.add_argument("--kge-num-negs", type=int, default=1)
    parser.add_argument("--compgcn-layers", type=int, default=1, help="Number of CompGCN message-passing layers.")
    parser.add_argument("--compgcn-dropout", type=float, default=0.0, help="Dropout inside each CompGCN layer.")
    parser.add_argument(
        "--compgcn-composition",
        choices=["sub", "mult", "corr"],
        default="mult",
        help="CompGCN entity-relation composition: sub, mult, or corr.",
    )
    parser.add_argument("--nfm-lr", type=float, default=1e-3)
    parser.add_argument("--nfm-weight-decay", type=float, default=1e-5)
    parser.add_argument("--nfm-dropout", type=float, default=0.0)
    parser.add_argument("--nfm-hidden-units", default="128,128")
    parser.add_argument("--nfm-patience", type=int, default=10)
    parser.add_argument("--output-dir", default="output/kge_nfm_torch")
    parser.add_argument(
        "--descriptor-ablation",
        choices=["none", "all", "without-protein", "without-morgan", "without-descriptors"],
        default="none",
        help="Train the full NFM only, or compare it with selected descriptor-removal variants.",
    )
    parser.add_argument(
        "--nfm-only",
        action="store_true",
        help="Train NFM without KGE dense embedding features; use descriptor features only.",
    )
    parser.add_argument(
        "--reuse-kge-checkpoints",
        action="store_true",
        help="Load saved KGE checkpoints instead of retraining KGE.",
    )
    parser.add_argument(
        "--kge-checkpoint-dir",
        default=None,
        help="Directory containing kge_nfm_fold_<index>.pt files for --reuse-kge-checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drug-features-only", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nfm_only and args.reuse_kge_checkpoints:
        raise ValueError("--nfm-only cannot be combined with --reuse-kge-checkpoints.")
    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root) if args.data_root is not None else repo_root / "data"
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_dir)
    ensure_output_dirs(output_root)
    print_args(args, data_root, device)

    spec = get_dataset_spec(data_root, args.dataset, args.split)
    all_dti = load_all_dti(spec)
    kg = load_kg(spec)
    drug_df, protein_df = load_feature_tables(spec, pca_components=args.protein_pca_components)

    head_encoder = encode_labels(all_dti["head"].astype(str))
    tail_encoder = encode_labels(all_dti["tail"].astype(str))

    kge_metrics_by_fold = []
    nfm_metrics_by_fold = []
    for fold in range(args.folds):
        kge_metrics, nfm_metrics = run_fold(
            fold=fold,
            args=args,
            spec=spec,
            kg=kg,
            drug_df=drug_df,
            protein_df=protein_df,
            head_encoder=head_encoder,
            tail_encoder=tail_encoder,
            device=device,
        )
        kge_metrics_by_fold.append(kge_metrics)
        nfm_metrics_by_fold.extend(nfm_metrics)

    kge_metrics = pd.DataFrame(kge_metrics_by_fold)
    nfm_metrics = pd.DataFrame(nfm_metrics_by_fold)
    legacy_nfm_metrics = nfm_metrics.loc[nfm_metrics["variant"] == "full"]
    if legacy_nfm_metrics.empty and nfm_metrics["variant"].nunique() == 1:
        legacy_nfm_metrics = nfm_metrics
    stable_metrics = kge_metrics.merge(
        legacy_nfm_metrics[["fold", "roc_auc_nfm", "pr_auc_nfm"]],
        on="fold",
        how="left",
    )
    print(stable_metrics)
    print("Average metrics:")
    print(stable_metrics[["roc_auc", "pr_auc", "roc_auc_nfm", "pr_auc_nfm"]].mean())
    print(stable_metrics.describe())
    stable_metrics.to_csv(output_root / "auc" / "kge_nfm_torch_auc.csv", index=False)

    ablation_auc_dir = output_root / "ablation" / "auc"
    nfm_metrics.to_csv(ablation_auc_dir / "nfm_ablation_by_fold.csv", index=False)
    summary = nfm_metrics.groupby("variant", sort=False).agg(
        mean_roc_auc=("roc_auc_nfm", "mean"),
        std_roc_auc=("roc_auc_nfm", "std"),
        mean_pr_auc=("pr_auc_nfm", "mean"),
        std_pr_auc=("pr_auc_nfm", "std"),
    ).reset_index()
    full_summary = summary.loc[summary["variant"] == "full"]
    if not full_summary.empty:
        summary["delta_roc_vs_full"] = summary["mean_roc_auc"] - full_summary.iloc[0]["mean_roc_auc"]
        summary["delta_pr_vs_full"] = summary["mean_pr_auc"] - full_summary.iloc[0]["mean_pr_auc"]
    else:
        summary["delta_roc_vs_full"] = np.nan
        summary["delta_pr_vs_full"] = np.nan
    summary.to_csv(ablation_auc_dir / "nfm_ablation_summary.csv", index=False)
    print("NFM descriptor ablation summary:")
    print(summary)


if __name__ == "__main__":
    main()
