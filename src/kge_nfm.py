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
    get_dataset_spec,
    load_feature_tables,
    load_fold,
    load_kg,
    merge_feature_blocks,
)
from kge import pair_embedding_blocks, require_entities, score_triples, train_compgcn, train_distmult
from metrics_utils import pr_auc, roc_auc
from nfm import train_nfm
from utils import ensure_output_dirs, parse_hidden_units, resolve_device, set_seed


def print_args(args: argparse.Namespace, data_root: Path, device: str) -> None:
    config = vars(args).copy()
    config["resolved_data_root"] = str(data_root)
    config["resolved_device"] = device

    print("Run arguments:")
    for key in sorted(config):
        print(f"  {key}: {config[key]}")


def scale_feature_fields(
    train_fields: list[np.ndarray],
    test_fields: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    field_dims = [field.shape[1] for field in train_fields]
    train_all = np.concatenate(train_fields, axis=1)
    test_all = np.concatenate(test_fields, axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_all = scaler.fit_transform(train_all).astype(np.float32)
    test_all = scaler.transform(test_all).astype(np.float32)

    offsets = np.cumsum([0, *field_dims])
    train_scaled = [train_all[:, offsets[index] : offsets[index + 1]] for index in range(len(field_dims))]
    test_scaled = [test_all[:, offsets[index] : offsets[index + 1]] for index in range(len(field_dims))]
    return train_scaled, test_scaled


def run_fold(
    fold: int,
    args: argparse.Namespace,
    spec: DatasetSpec,
    kg: pd.DataFrame,
    drug_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    device: str,
) -> dict[str, float]:
    fold_number = fold + 1
    print(f"Fold {fold_number} ({args.split}, {args.kge_model})")
    train, test = load_fold(spec, fold)
    duplicate_test_triples = int(test.duplicated(TRIPLE_COLUMNS).sum())
    if duplicate_test_triples:
        print(f"Fold {fold_number} has {duplicate_test_triples} duplicate test triples; scores will preserve all rows.")
    train_pos = train.loc[train["label"] == 1, TRIPLE_COLUMNS]
    kge_train = pd.concat([train_pos, kg], ignore_index=True)[TRIPLE_COLUMNS].astype(str)

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

    require_entities(triples_factory, train["head"].tolist() + train["tail"].tolist(), "Training fold")
    require_entities(triples_factory, test["head"].tolist() + test["tail"].tolist(), "Test fold")

    output_root = Path(args.output_dir)
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
    kge_scores = score_triples(kge_model, triples_factory, test[TRIPLE_COLUMNS], device, args.kge_batch_size)
    roc_curve, roc_value = roc_auc(test_labels, kge_scores)
    pr_curve, pr_value = pr_auc(test_labels, kge_scores)

    train_pairs = train[TRIPLE_COLUMNS]
    test_pairs = test[TRIPLE_COLUMNS]
    train_drug_kge, train_protein_kge = pair_embedding_blocks(kge_model, triples_factory, train_pairs, device)
    test_drug_kge, test_protein_kge = pair_embedding_blocks(kge_model, triples_factory, test_pairs, device)
    train_descriptor_fields = merge_feature_blocks(
        train_pairs,
        drug_df,
        protein_df,
        spec,
        use_protein_features=not args.drug_features_only,
    )
    test_descriptor_fields = merge_feature_blocks(
        test_pairs,
        drug_df,
        protein_df,
        spec,
        use_protein_features=not args.drug_features_only,
    )

    train_feature_fields = [train_drug_kge, train_protein_kge, *train_descriptor_fields]
    test_feature_fields = [test_drug_kge, test_protein_kge, *test_descriptor_fields]
    field_names = ["drug_kg_embedding", "protein_kg_embedding", "drug_descriptor"]
    if not args.drug_features_only:
        field_names.append("protein_descriptor")
    if args.include_kge_score_feature:
        train_kge_scores = score_triples(
            kge_model,
            triples_factory,
            train_pairs,
            device,
            args.kge_batch_size,
        ).reshape(-1, 1)
        train_feature_fields.append(train_kge_scores)
        test_feature_fields.append(kge_scores.reshape(-1, 1))
        field_names.append("kge_score")

    train_feature_fields, test_feature_fields = scale_feature_fields(train_feature_fields, test_feature_fields)

    roc_nfm, roc_nfm_value, pr_nfm, pr_nfm_value, nfm_pred = train_nfm(
        train_feature_fields=train_feature_fields,
        train_labels=train["label"].to_numpy(dtype=np.float32),
        test_labels=test_labels,
        test_feature_fields=test_feature_fields,
        field_embedding_dim=args.nfm_field_embedding_dim,
        epochs=args.nfm_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + fold,
        lr=args.nfm_lr,
        weight_decay=args.nfm_weight_decay,
        dropout=args.nfm_dropout,
        hidden_units=parse_hidden_units(args.nfm_hidden_units),
        patience=args.nfm_patience,
        field_names=field_names,
    )

    roc_curve.to_csv(output_root / "curve" / "roc" / f"{fold}.csv", index=False)
    pr_curve.to_csv(output_root / "curve" / "pr" / f"{fold}.csv", index=False)
    roc_nfm.to_csv(output_root / "curve" / "roc_nfm" / f"{fold}.csv", index=False)
    pr_nfm.to_csv(output_root / "curve" / "pr_nfm" / f"{fold}.csv", index=False)

    predictions = test[TRIPLE_COLUMNS + ["label"]].copy()
    predictions["kge_score"] = kge_scores
    predictions["nfm_pred"] = nfm_pred
    predictions.to_csv(output_root / "predictions" / f"fold_{fold}.csv", index=False)

    print(f"roc_auc: {roc_value:.6f}")
    print(f"pr_auc: {pr_value:.6f}")
    print(f"roc_auc_nfm: {roc_nfm_value:.6f}")
    print(f"pr_auc_nfm: {pr_nfm_value:.6f}")
    return {
        "fold": fold_number,
        "train_size": len(train),
        "test_size": len(test),
        "train_pos": int((train["label"] == 1).sum()),
        "test_pos": int((test["label"] == 1).sum()),
        "roc_auc": roc_value,
        "pr_auc": pr_value,
        "roc_auc_nfm": roc_nfm_value,
        "pr_auc_nfm": pr_nfm_value,
    }


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
    parser.add_argument(
        "--nfm-field-embedding-dim",
        "--nfm-sparse-embedding-dim",
        dest="nfm_field_embedding_dim",
        type=int,
        default=50,
        help="Latent dimension used to project each KG/descriptor field before NFM bi-interaction.",
    )
    parser.add_argument(
        "--include-kge-score-feature",
        action="store_true",
        help="Append the KGE triple score as an extra dense feature for the NFM predictor.",
    )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drug-features-only", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root) if args.data_root is not None else repo_root / "data"
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_dir)
    ensure_output_dirs(output_root)
    print_args(args, data_root, device)

    spec = get_dataset_spec(data_root, args.dataset, args.split)
    kg = load_kg(spec)
    drug_df, protein_df = load_feature_tables(spec, pca_components=args.protein_pca_components)

    metrics_by_fold = []
    for fold in range(args.folds):
        metrics_by_fold.append(
            run_fold(
                fold=fold,
                args=args,
                spec=spec,
                kg=kg,
                drug_df=drug_df,
                protein_df=protein_df,
                device=device,
            )
        )

    stable_metrics = pd.DataFrame(metrics_by_fold)
    print(stable_metrics)
    print("Average metrics:")
    print(stable_metrics[["roc_auc", "pr_auc", "roc_auc_nfm", "pr_auc_nfm"]].mean())
    print(stable_metrics.describe())
    stable_metrics.to_csv(output_root / "auc" / "kge_nfm_torch_auc.csv", index=False)


if __name__ == "__main__":
    main()
