from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


DTI_COLUMNS = ["head", "relation", "tail"]
WARM_START_SPLITS = {
    "warm_start_1_1": "Warm-start 1:1",
    "warm_start_1_10": "Warm-start 1:10",
}
COLORS = {
    "drug": "#6C5CE7",
    "target": "#D94F70",
    "positive": "#168AAD",
    "negative": "#F4A261",
    "kg": "#2A9D8F",
    "neutral": "#64748B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready analysis for the Yamanishi08 DTI dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/kaggle/input/datasets/ngcaovn/kge-dti/data/yamanishi_08"),
        help="Path to the Yamanishi08 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/yamanishi_08_analysis"),
        help="Directory for generated tables, figures, and the Markdown report.",
    )
    parser.add_argument("--dpi", type=int, default=240, help="PNG resolution.")
    return parser.parse_args()


def prepare_output(output_dir: Path) -> tuple[Path, Path]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    for path in (figures_dir, tables_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def read_triples(path: Path, deduplicate: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+" if path.suffix.lower() == ".txt" else ",",
        header=None,
        names=DTI_COLUMNS,
        usecols=[0, 1, 2],
        dtype=str,
    )
    frame = frame.dropna().reset_index(drop=True)
    return frame.drop_duplicates().reset_index(drop=True) if deduplicate else frame


def save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#94A3B8",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "font.size": 10,
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def add_bar_labels(axis: plt.Axes, bars, percent: bool = False) -> None:
    labels = [f"{bar.get_height():.1f}%" if percent else f"{bar.get_height():,.0f}" for bar in bars]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=9, color="#334155")


def gini(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0 or np.all(array == 0):
        return 0.0
    array = np.sort(array)
    index = np.arange(1, array.size + 1)
    return float(
        (2.0 * np.sum(index * array) / (array.size * np.sum(array)))
        - (array.size + 1.0) / array.size
    )


def concentration(values: pd.Series | np.ndarray, fraction: float = 0.10) -> float:
    array = np.sort(np.asarray(values, dtype=float))[::-1]
    if array.size == 0 or array.sum() == 0:
        return 0.0
    top_n = max(1, math.ceil(array.size * fraction))
    return float(array[:top_n].sum() / array.sum())


def entropy(counts: pd.Series | np.ndarray) -> tuple[float, float]:
    array = np.asarray(counts, dtype=float)
    probabilities = array[array > 0] / array.sum()
    value = float(-(probabilities * np.log2(probabilities)).sum())
    maximum = math.log2(len(probabilities)) if len(probabilities) > 1 else 0.0
    return value, value / maximum if maximum else 0.0


class UnionFind:
    def __init__(self, nodes: set[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def connected_component_stats(dti: pd.DataFrame) -> dict[str, float | int]:
    drug_nodes = {f"drug::{value}" for value in dti["head"]}
    target_nodes = {f"target::{value}" for value in dti["tail"]}
    union_find = UnionFind(drug_nodes | target_nodes)
    for row in dti.itertuples(index=False):
        union_find.union(f"drug::{row.head}", f"target::{row.tail}")

    component_sizes = Counter(union_find.find(node) for node in union_find.parent)
    sizes = np.asarray(sorted(component_sizes.values(), reverse=True), dtype=int)
    total_nodes = int(sizes.sum())
    return {
        "connected_components": int(len(sizes)),
        "largest_component_nodes": int(sizes[0]),
        "largest_component_share": float(sizes[0] / total_nodes),
        "singleton_components": int((sizes == 1).sum()),
        "median_component_size": float(np.median(sizes)),
    }


def degree_row(name: str, values: pd.Series) -> dict[str, float | int | str]:
    return {
        "entity_type": name,
        "count": int(values.size),
        "minimum": int(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q75": float(values.quantile(0.75)),
        "maximum": int(values.max()),
        "std": float(values.std(ddof=0)),
        "gini": gini(values),
        "top_10_percent_interaction_share": concentration(values),
    }


def analyze_dti(data_dir: Path, tables_dir: Path) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    dti = read_triples(data_dir / "dt_all_08.txt")
    drug_degree = dti.groupby("head").size().sort_values(ascending=False)
    target_degree = dti.groupby("tail").size().sort_values(ascending=False)
    possible_pairs = int(drug_degree.size * target_degree.size)
    density = float(len(dti) / possible_pairs)

    overview = {
        "positive_interactions": int(len(dti)),
        "unique_drugs": int(drug_degree.size),
        "unique_targets": int(target_degree.size),
        "possible_drug_target_pairs": possible_pairs,
        "observed_positive_density": density,
        "unobserved_pair_sparsity": 1.0 - density,
        "unobserved_pairs": possible_pairs - int(len(dti)),
        "average_targets_per_drug": float(drug_degree.mean()),
        "average_drugs_per_target": float(target_degree.mean()),
        "drug_degree_le_5_count": int((drug_degree <= 5).sum()),
        "drug_degree_le_5_share": float((drug_degree <= 5).mean()),
        "target_degree_le_2_count": int((target_degree <= 2).sum()),
        "target_degree_le_2_share": float((target_degree <= 2).mean()),
        **connected_component_stats(dti),
    }

    degree_summary = pd.DataFrame(
        [degree_row("Drug", drug_degree), degree_row("Target", target_degree)]
    )
    save_table(pd.DataFrame([overview]), tables_dir / "dataset_overview.csv")
    save_json(overview, tables_dir / "dataset_overview.json")
    save_table(degree_summary, tables_dir / "degree_summary.csv")
    save_table(
        drug_degree.rename_axis("drug_id").reset_index(name="degree"),
        tables_dir / "drug_degrees.csv",
    )
    save_table(
        target_degree.rename_axis("target_id").reset_index(name="degree"),
        tables_dir / "target_degrees.csv",
    )
    save_table(
        drug_degree.head(20).rename_axis("drug_id").reset_index(name="degree"),
        tables_dir / "top_drug_hubs.csv",
    )
    save_table(
        target_degree.head(20).rename_axis("target_id").reset_index(name="degree"),
        tables_dir / "top_target_hubs.csv",
    )
    return dti, overview, drug_degree, target_degree


def analyze_knowledge_graph(
    data_dir: Path, dti: pd.DataFrame, tables_dir: Path
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    for path in [data_dir / "kg_data/kegg_kg.txt", data_dir / "kg_data/yamanishi_uniprot_kg.txt"]:
        raw_frame = read_triples(path, deduplicate=False)
        frame = raw_frame.drop_duplicates().reset_index(drop=True)
        frames.append(frame)
        entities = set(frame["head"]) | set(frame["tail"])
        rows.append(
            {
                "graph": path.stem,
                "raw_triples": int(len(raw_frame)),
                "unique_triples": int(len(frame)),
                "duplicate_triples": int(raw_frame.duplicated().sum()),
                "entities": int(len(entities)),
                "relations": int(frame["relation"].nunique()),
                "unique_entity_pairs": int(frame[["head", "tail"]].drop_duplicates().shape[0]),
            }
        )

    kg = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    entities = set(kg["head"]) | set(kg["tail"])
    relation_counts = kg["relation"].value_counts()
    relation_entropy, normalized_entropy = entropy(relation_counts)
    unique_pairs = int(kg[["head", "tail"]].drop_duplicates().shape[0])
    combined = {
        "graph": "combined",
        "raw_triples": int(sum(row["raw_triples"] for row in rows)),
        "unique_triples": int(len(kg)),
        "duplicate_triples": int(sum(row["duplicate_triples"] for row in rows)),
        "entities": int(len(entities)),
        "relations": int(relation_counts.size),
        "unique_entity_pairs": unique_pairs,
        "directed_pair_density": float(unique_pairs / (len(entities) ** 2)),
        "relation_aware_density": float(len(kg) / (len(entities) ** 2 * relation_counts.size)),
        "relation_entropy_bits": relation_entropy,
        "normalized_relation_entropy": normalized_entropy,
        "top_5_relation_share": float(relation_counts.head(5).sum() / len(kg)),
        "dti_drug_coverage": float(dti["head"].isin(entities).mean()),
        "dti_target_coverage": float(dti["tail"].isin(entities).mean()),
        "unique_dti_drugs_covered": int(dti.loc[dti["head"].isin(entities), "head"].nunique()),
        "unique_dti_targets_covered": int(dti.loc[dti["tail"].isin(entities), "tail"].nunique()),
    }
    rows.append(combined)

    relation_table = relation_counts.rename_axis("relation").reset_index(name="triple_count")
    relation_table["share"] = relation_table["triple_count"] / len(kg)
    save_table(pd.DataFrame(rows), tables_dir / "knowledge_graph_summary.csv")
    save_table(relation_table, tables_dir / "knowledge_graph_relations.csv")
    return relation_table, combined


def numeric_quality(values: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(values.shape[0]),
        "dimensions": int(values.shape[1]),
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "zero_share": float(np.mean(values == 0)),
        "missing_values": int(np.isnan(values).sum()),
        "infinite_values": int(np.isinf(values).sum()),
        "constant_dimensions": int(np.sum(np.nanstd(values, axis=0) == 0)),
    }


def analyze_features(data_dir: Path, tables_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    morgan = np.loadtxt(data_dir / "morganfp.txt", delimiter=",")
    ctd = np.loadtxt(data_dir / "pro_ctd.txt", delimiter=",")
    structures = pd.read_csv(data_dir / "791drug_struc.csv")
    sequences = pd.read_csv(data_dir / "989proseq.csv")

    active_bits = np.sum(morgan != 0, axis=1)
    sequence_lengths = sequences["seq"].astype(str).str.len().to_numpy()
    smiles_lengths = structures["smiles"].astype(str).str.len().to_numpy()

    morgan_stats = numeric_quality(morgan)
    morgan_stats.update(
        {
            "feature_block": "Drug Morgan fingerprint",
            "average_nonzero_values_per_row": float(active_bits.mean()),
            "median_nonzero_values_per_row": float(np.median(active_bits)),
        }
    )
    ctd_stats = numeric_quality(ctd)
    ctd_stats.update(
        {
            "feature_block": "Protein CTD descriptor",
            "average_nonzero_values_per_row": float(np.count_nonzero(ctd, axis=1).mean()),
            "median_nonzero_values_per_row": float(np.median(np.count_nonzero(ctd, axis=1))),
        }
    )
    feature_summary = pd.DataFrame([morgan_stats, ctd_stats])
    save_table(feature_summary, tables_dir / "feature_summary.csv")

    length_summary = pd.DataFrame(
        [
            {
                "representation": "SMILES",
                "count": int(smiles_lengths.size),
                "minimum": int(smiles_lengths.min()),
                "median": float(np.median(smiles_lengths)),
                "mean": float(smiles_lengths.mean()),
                "maximum": int(smiles_lengths.max()),
            },
            {
                "representation": "Protein sequence",
                "count": int(sequence_lengths.size),
                "minimum": int(sequence_lengths.min()),
                "median": float(np.median(sequence_lengths)),
                "mean": float(sequence_lengths.mean()),
                "maximum": int(sequence_lengths.max()),
            },
        ]
    )
    save_table(length_summary, tables_dir / "sequence_structure_lengths.csv")
    return feature_summary, {
        "active_bits": active_bits,
        "sequence_lengths": sequence_lengths,
        "smiles_lengths": smiles_lengths,
    }


def pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["head"].astype(str), frame["tail"].astype(str)))


def percentage_subset(test_values: set[str], train_values: set[str]) -> float:
    return len(test_values & train_values) / len(test_values) if test_values else 1.0


def analyze_warm_start_folds(
    data_dir: Path, dti: pd.DataFrame, tables_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gold_positive_pairs = pair_set(dti)
    fold_rows: list[dict] = []
    quality_rows: list[dict] = []
    partition_rows: list[dict] = []

    for split, display_name in WARM_START_SPLITS.items():
        split_dir = data_dir / "data_folds" / split
        test_positive_counter: Counter[tuple[str, str]] = Counter()

        for fold in range(1, 11):
            train = pd.read_csv(split_dir / f"train_fold_{fold}.csv")
            test = pd.read_csv(split_dir / f"test_fold_{fold}.csv")
            train["label"] = train["label"].astype(int)
            test["label"] = test["label"].astype(int)
            train_positive = train[train["label"] == 1]
            test_positive = test[test["label"] == 1]
            train_negative = train[train["label"] == 0]
            test_negative = test[test["label"] == 0]

            train_pairs = pair_set(train)
            test_pairs = pair_set(test)
            train_positive_pairs = pair_set(train_positive)
            test_positive_pairs = pair_set(test_positive)
            test_positive_counter.update(test_positive_pairs)

            test_drugs = set(test["head"].astype(str))
            test_targets = set(test["tail"].astype(str))
            train_drugs = set(train["head"].astype(str))
            train_targets = set(train["tail"].astype(str))
            test_positive_drugs = set(test_positive["head"].astype(str))
            test_positive_targets = set(test_positive["tail"].astype(str))
            train_positive_drugs = set(train_positive["head"].astype(str))
            train_positive_targets = set(train_positive["tail"].astype(str))

            fold_rows.append(
                {
                    "setting": display_name,
                    "fold": fold,
                    "train_samples": int(len(train)),
                    "test_samples": int(len(test)),
                    "train_positive": int(len(train_positive)),
                    "train_negative": int(len(train_negative)),
                    "test_positive": int(len(test_positive)),
                    "test_negative": int(len(test_negative)),
                    "train_negative_positive_ratio": float(len(train_negative) / len(train_positive)),
                    "test_negative_positive_ratio": float(len(test_negative) / len(test_positive)),
                    "test_drug_train_coverage": percentage_subset(test_drugs, train_drugs),
                    "test_target_train_coverage": percentage_subset(test_targets, train_targets),
                    "positive_test_drug_positive_train_coverage": percentage_subset(
                        test_positive_drugs, train_positive_drugs
                    ),
                    "positive_test_target_positive_train_coverage": percentage_subset(
                        test_positive_targets, train_positive_targets
                    ),
                    "train_test_pair_overlap": int(len(train_pairs & test_pairs)),
                    "train_test_positive_pair_overlap": int(
                        len(train_positive_pairs & test_positive_pairs)
                    ),
                    "train_duplicate_rows": int(train.duplicated().sum()),
                    "test_duplicate_rows": int(test.duplicated().sum()),
                    "train_negative_gold_positive_collisions": int(
                        len(pair_set(train_negative) & gold_positive_pairs)
                    ),
                    "test_negative_gold_positive_collisions": int(
                        len(pair_set(test_negative) & gold_positive_pairs)
                    ),
                }
            )

        unique_test_positives = set(test_positive_counter)
        partition_rows.append(
            {
                "setting": display_name,
                "gold_positive_pairs": len(gold_positive_pairs),
                "unique_test_positive_pairs_across_folds": len(unique_test_positives),
                "gold_positive_coverage": len(unique_test_positives & gold_positive_pairs)
                / len(gold_positive_pairs),
                "test_positive_pairs_repeated_across_folds": sum(
                    count > 1 for count in test_positive_counter.values()
                ),
                "maximum_test_positive_repetitions": max(test_positive_counter.values()),
                "test_positive_pairs_not_in_gold_file": len(
                    unique_test_positives - gold_positive_pairs
                ),
            }
        )

    fold_stats = pd.DataFrame(fold_rows)
    metrics = [
        "train_samples",
        "test_samples",
        "train_positive",
        "train_negative",
        "test_positive",
        "test_negative",
        "train_negative_positive_ratio",
        "test_negative_positive_ratio",
        "test_drug_train_coverage",
        "test_target_train_coverage",
        "positive_test_drug_positive_train_coverage",
        "positive_test_target_positive_train_coverage",
    ]
    summary_rows: list[dict] = []
    for setting, group in fold_stats.groupby("setting", sort=False):
        row: dict[str, str | float] = {"setting": setting}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        summary_rows.append(row)
        quality_rows.append(
            {
                "setting": setting,
                "train_test_pair_overlap_total": int(group["train_test_pair_overlap"].sum()),
                "train_test_positive_pair_overlap_total": int(
                    group["train_test_positive_pair_overlap"].sum()
                ),
                "duplicate_rows_total": int(
                    group["train_duplicate_rows"].sum() + group["test_duplicate_rows"].sum()
                ),
                "negative_gold_positive_collisions_total": int(
                    group["train_negative_gold_positive_collisions"].sum()
                    + group["test_negative_gold_positive_collisions"].sum()
                ),
                "minimum_test_drug_train_coverage": float(
                    group["test_drug_train_coverage"].min()
                ),
                "minimum_test_target_train_coverage": float(
                    group["test_target_train_coverage"].min()
                ),
            }
        )

    split_summary = pd.DataFrame(summary_rows)
    quality_checks = pd.DataFrame(quality_rows).merge(
        pd.DataFrame(partition_rows), on="setting", how="left"
    )
    save_table(fold_stats, tables_dir / "warm_start_fold_statistics.csv")
    save_table(split_summary, tables_dir / "warm_start_summary.csv")
    save_table(quality_checks, tables_dir / "warm_start_quality_checks.csv")
    return fold_stats, split_summary, quality_checks


def plot_degree_distributions(
    drug_degree: pd.Series, target_degree: pd.Series, figures_dir: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    figure.suptitle("Yamanishi08 DTI Degree Distributions", fontsize=15, fontweight="bold")
    panels = [
        (axes[0], drug_degree, "Drug Degree Distribution", "Known targets per drug", COLORS["drug"]),
        (axes[1], target_degree, "Target Degree Distribution", "Known drugs per target", COLORS["target"]),
    ]
    for axis, values, title, xlabel, color in panels:
        axis.hist(values.to_numpy(), bins=30, color=color, edgecolor="white", linewidth=0.6)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Entity count (log scale)")
        axis.grid(axis="y")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, figures_dir / "dti_degree_distribution.png", dpi)


def plot_degree_rank(
    drug_degree: pd.Series, target_degree: pd.Series, figures_dir: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for values, label, color in [
        (drug_degree, "Drugs", COLORS["drug"]),
        (target_degree, "Targets", COLORS["target"]),
    ]:
        ranked = np.sort(values.to_numpy())[::-1]
        axis.plot(np.arange(1, len(ranked) + 1), ranked, linewidth=2.2, color=color, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Rank-Degree Profile of the DTI Graph")
    axis.set_xlabel("Entity rank (log scale)")
    axis.set_ylabel("Degree (log scale)")
    axis.grid(which="both")
    axis.legend()
    figure.tight_layout()
    save_figure(figure, figures_dir / "dti_degree_rank.png", dpi)


def plot_network_summary(overview: dict, degree_summary: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    figure.suptitle("Yamanishi08 Network Sparsity and Imbalance", fontsize=15, fontweight="bold")
    panels = [
        (
            ["Observed", "Unobserved"],
            [overview["observed_positive_density"] * 100, overview["unobserved_pair_sparsity"] * 100],
            "Interaction Matrix",
            "Pair share (%)",
            [COLORS["positive"], "#C9D5E1"],
        ),
        (
            ["Drug <= 5", "Target <= 2"],
            [overview["drug_degree_le_5_share"] * 100, overview["target_degree_le_2_share"] * 100],
            "Low-Degree Entities",
            "Entity share (%)",
            [COLORS["drug"], COLORS["target"]],
        ),
        (
            degree_summary["entity_type"].tolist(),
            (degree_summary["top_10_percent_interaction_share"] * 100).tolist(),
            "Hub Concentration",
            "Top 10% interaction share (%)",
            [COLORS["drug"], COLORS["target"]],
        ),
    ]
    for axis, (labels, values, title, ylabel, colors) in zip(axes, panels):
        bars = axis.bar(labels, values, color=colors, width=0.6)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 105)
        axis.grid(axis="y")
        add_bar_labels(axis, bars, percent=True)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, figures_dir / "network_sparsity_summary.png", dpi)


def plot_kg_relations(relation_table: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    top = relation_table.head(15).sort_values("triple_count")
    figure, axis = plt.subplots(figsize=(10, 6.5))
    bars = axis.barh(top["relation"], top["triple_count"], color=COLORS["kg"])
    axis.set_title("Most Frequent Relations in the Combined Knowledge Graph")
    axis.set_xlabel("Triple count")
    axis.grid(axis="x")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.bar_label(bars, labels=[f"{value:,}" for value in top["triple_count"]], padding=4, fontsize=8)
    axis.margins(x=0.12)
    figure.tight_layout()
    save_figure(figure, figures_dir / "knowledge_graph_relation_distribution.png", dpi)


def plot_feature_characteristics(
    feature_arrays: dict[str, np.ndarray], figures_dir: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    figure.suptitle("Yamanishi08 Descriptor Inputs", fontsize=15, fontweight="bold")
    panels = [
        ("active_bits", "Morgan Fingerprint Activity", "Nonzero bits per drug", COLORS["drug"], 30),
        ("smiles_lengths", "SMILES Length", "Characters", COLORS["positive"], 30),
        ("sequence_lengths", "Protein Sequence Length", "Residues", COLORS["negative"], 35),
    ]
    for axis, (key, title, xlabel, color, bins) in zip(axes, panels):
        axis.hist(feature_arrays[key], bins=bins, color=color, edgecolor="white", linewidth=0.6)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        axis.grid(axis="y")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, figures_dir / "feature_characteristics.png", dpi)


def plot_warm_start_balance(fold_stats: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    summary = fold_stats.groupby("setting", sort=False)[
        ["train_positive", "train_negative", "test_positive", "test_negative"]
    ].mean()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    figure.suptitle("Warm-Start Class Balance Across Ten Folds", fontsize=15, fontweight="bold")
    labels = [setting.replace("Warm-start ", "") for setting in summary.index]
    positions = np.arange(len(summary))
    for axis, (prefix, title) in zip(
        axes,
        [("train", "Mean Training Composition"), ("test", "Mean Test Composition")],
    ):
        positives = summary[f"{prefix}_positive"].to_numpy(dtype=float)
        negatives = summary[f"{prefix}_negative"].to_numpy(dtype=float)
        axis.bar(positions, positives, color=COLORS["positive"], label="Positive")
        axis.bar(positions, negatives, bottom=positives, color=COLORS["negative"], label="Sampled negative")
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Pairs per fold")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"))
        axis.grid(axis="y")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.98, 0.95))
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(figure, figures_dir / "warm_start_class_balance.png", dpi)


def plot_fold_stability(fold_stats: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    figure.suptitle("Fold Stability of the Warm-Start Protocol", fontsize=15, fontweight="bold")
    panels = [
        ("test_samples", "Total Test Pairs per Fold", "Test pairs"),
        ("test_negative_positive_ratio", "Realized Test Negative-to-Positive Ratio", "Negative / positive"),
    ]
    line_colors = [COLORS["drug"], COLORS["target"]]
    grouped = list(fold_stats.groupby("setting", sort=False))
    for axis, (metric, title, ylabel) in zip(axes, panels):
        for color, (setting, group) in zip(line_colors, grouped):
            axis.plot(group["fold"], group[metric], marker="o", linewidth=2, markersize=4, color=color, label=setting)
        axis.set_title(title)
        axis.set_xlabel("Fold")
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid()
        if metric.endswith("ratio"):
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.98, 0.95))
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(figure, figures_dir / "warm_start_fold_stability.png", dpi)


def format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_report(
    output_dir: Path,
    overview: dict,
    degree_summary: pd.DataFrame,
    kg_summary: dict,
    feature_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    quality_checks: pd.DataFrame,
) -> None:
    drug_degree = degree_summary.loc[degree_summary["entity_type"] == "Drug"].iloc[0]
    target_degree = degree_summary.loc[degree_summary["entity_type"] == "Target"].iloc[0]
    split_lines: list[str] = []
    for row in split_summary.itertuples(index=False):
        split_lines.extend(
            [
                f"### {row.setting}",
                "",
                f"- Mean training pairs: {row.train_samples_mean:,.1f} "
                f"({row.train_positive_mean:,.1f} positive, {row.train_negative_mean:,.1f} sampled negative)",
                f"- Mean test pairs: {row.test_samples_mean:,.1f} "
                f"({row.test_positive_mean:,.1f} positive, {row.test_negative_mean:,.1f} sampled negative)",
                f"- Realized test negative-to-positive ratio: {row.test_negative_positive_ratio_mean:.3f}:1",
                f"- Minimum test-drug training coverage: {format_percent(row.test_drug_train_coverage_min)}",
                f"- Minimum test-target training coverage: {format_percent(row.test_target_train_coverage_min)}",
                "",
            ]
        )

    quality_lines: list[str] = []
    for row in quality_checks.itertuples(index=False):
        quality_lines.append(
            f"- **{row.setting}:** pair overlap={row.train_test_pair_overlap_total}, "
            f"duplicate rows={row.duplicate_rows_total}, "
            f"negative/known-positive collisions={row.negative_gold_positive_collisions_total}, "
            f"gold positives covered across test folds={format_percent(row.gold_positive_coverage)}."
        )

    morgan = feature_summary.loc[
        feature_summary["feature_block"] == "Drug Morgan fingerprint"
    ].iloc[0]
    ctd = feature_summary.loc[
        feature_summary["feature_block"] == "Protein CTD descriptor"
    ].iloc[0]
    lines = [
        "# Yamanishi08 Dataset Analysis",
        "",
        "This analysis is restricted to Yamanishi08 and the two warm-start settings used by the project. "
        "Unobserved drug-target pairs are described as *sampled negatives* rather than confirmed biological negatives.",
        "",
        "## Dataset Overview",
        "",
        f"- Known positive interactions: {overview['positive_interactions']:,}",
        f"- Drugs: {overview['unique_drugs']:,}",
        f"- Target proteins: {overview['unique_targets']:,}",
        f"- Possible drug-target pairs: {overview['possible_drug_target_pairs']:,}",
        f"- Observed positive density: {format_percent(overview['observed_positive_density'])}",
        f"- Unobserved-pair sparsity: {format_percent(overview['unobserved_pair_sparsity'])}",
        f"- Connected components: {overview['connected_components']:,}; largest component contains "
        f"{format_percent(overview['largest_component_share'])} of DTI nodes",
        "",
        "## Degree Imbalance",
        "",
        f"- Median drug degree: {drug_degree['median']:.1f}; maximum: {int(drug_degree['maximum'])}; "
        f"Gini coefficient: {drug_degree['gini']:.3f}",
        f"- Median target degree: {target_degree['median']:.1f}; maximum: {int(target_degree['maximum'])}; "
        f"Gini coefficient: {target_degree['gini']:.3f}",
        f"- Drugs with at most five interactions: {overview['drug_degree_le_5_count']:,} "
        f"({format_percent(overview['drug_degree_le_5_share'])})",
        f"- Targets with at most two interactions: {overview['target_degree_le_2_count']:,} "
        f"({format_percent(overview['target_degree_le_2_share'])})",
        f"- Top 10% of drugs account for {format_percent(drug_degree['top_10_percent_interaction_share'])} of interactions",
        f"- Top 10% of targets account for {format_percent(target_degree['top_10_percent_interaction_share'])} of interactions",
        "",
        "**Interpretation.** The graph combines extreme pair sparsity with concentrated hubs. Aggregate metrics can "
        "therefore be dominated by well-connected entities, while low-degree drugs and targets remain the harder cases.",
        "",
        "## Knowledge Graph Context",
        "",
        f"- Raw KG rows: {kg_summary['raw_triples']:,}",
        f"- Unique KG triples: {kg_summary['unique_triples']:,}",
        f"- Duplicate KG triples removed: {kg_summary['duplicate_triples']:,}",
        f"- Entities: {kg_summary['entities']:,}",
        f"- Relations: {kg_summary['relations']:,}",
        f"- Directed entity-pair density: {format_percent(kg_summary['directed_pair_density'])}",
        f"- Relation-aware density: {kg_summary['relation_aware_density']:.3e}",
        f"- DTI drug coverage in the KG: {format_percent(kg_summary['dti_drug_coverage'])}",
        f"- DTI target coverage in the KG: {format_percent(kg_summary['dti_target_coverage'])}",
        "",
        "## Descriptor Quality",
        "",
        f"- Morgan fingerprints: {int(morgan['rows'])} x {int(morgan['dimensions'])}; "
        f"zero share={format_percent(morgan['zero_share'])}; "
        f"mean active bits={morgan['average_nonzero_values_per_row']:.2f}",
        f"- Protein CTD descriptors: {int(ctd['rows'])} x {int(ctd['dimensions'])}; "
        f"zero share={format_percent(ctd['zero_share'])}; "
        f"constant dimensions={int(ctd['constant_dimensions'])}",
        f"- Missing or infinite values: {int(morgan['missing_values'] + ctd['missing_values'])} missing, "
        f"{int(morgan['infinite_values'] + ctd['infinite_values'])} infinite",
        "",
        "## Warm-Start Settings",
        "",
        *split_lines,
        "## Data Quality Checks",
        "",
        *quality_lines,
        "",
        "The ten files are repeated warm-start holdouts rather than a disjoint ten-fold partition: "
        "test sets contain 3,120 unique positives in total, and some positive pairs occur in multiple test folds. "
        "Accordingly, results should be reported as the mean and standard deviation across repeated splits.",
        "",
        "These checks are important because leakage, mislabeled sampled negatives, or incomplete warm-start entity coverage "
        "would make performance estimates difficult to interpret.",
        "",
        "## Generated Figures",
        "",
        "- `figures/dti_degree_distribution.png`",
        "- `figures/dti_degree_rank.png`",
        "- `figures/network_sparsity_summary.png`",
        "- `figures/knowledge_graph_relation_distribution.png`",
        "- `figures/feature_characteristics.png`",
        "- `figures/warm_start_class_balance.png`",
        "- `figures/warm_start_fold_stability.png`",
        "",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_plot_style()
    figures_dir, tables_dir = prepare_output(args.output_dir)

    print("[1/4] Analyzing the Yamanishi08 DTI graph...")
    dti, overview, drug_degree, target_degree = analyze_dti(args.data_dir, tables_dir)
    degree_summary = pd.read_csv(tables_dir / "degree_summary.csv")

    print("[2/4] Analyzing knowledge graph coverage and relation imbalance...")
    relation_table, kg_summary = analyze_knowledge_graph(args.data_dir, dti, tables_dir)

    print("[3/4] Checking molecular and protein feature blocks...")
    feature_summary, feature_arrays = analyze_features(args.data_dir, tables_dir)

    print("[4/4] Auditing warm-start 1:1 and 1:10 folds...")
    fold_stats, split_summary, quality_checks = analyze_warm_start_folds(
        args.data_dir, dti, tables_dir
    )

    plot_degree_distributions(drug_degree, target_degree, figures_dir, args.dpi)
    plot_degree_rank(drug_degree, target_degree, figures_dir, args.dpi)
    plot_network_summary(overview, degree_summary, figures_dir, args.dpi)
    plot_kg_relations(relation_table, figures_dir, args.dpi)
    plot_feature_characteristics(feature_arrays, figures_dir, args.dpi)
    plot_warm_start_balance(fold_stats, figures_dir, args.dpi)
    plot_fold_stability(fold_stats, figures_dir, args.dpi)
    write_report(
        args.output_dir,
        overview,
        degree_summary,
        kg_summary,
        feature_summary,
        split_summary,
        quality_checks,
    )

    print(f"Completed. Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
