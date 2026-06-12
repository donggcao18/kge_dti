from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


TRIPLE_COLUMNS = ["head", "relation", "tail"]
DEFAULT_SPLITS = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    dti_path: Path
    kg_paths: tuple[Path, ...]
    drug_feature_path: Path | None
    protein_feature_path: Path | None
    drug_structure_path: Path | None = None
    protein_sequence_path: Path | None = None


def build_config(data_root: Path, dataset: str) -> DatasetConfig:
    if dataset == "yamanishi_08":
        root = data_root / dataset
        return DatasetConfig(
            name=dataset,
            root=root,
            dti_path=root / "dt_all_08.txt",
            kg_paths=(
                root / "kg_data" / "kegg_kg.txt",
                root / "kg_data" / "yamanishi_uniprot_kg.txt",
            ),
            drug_feature_path=root / "morganfp.txt",
            protein_feature_path=root / "pro_ctd.txt",
            drug_structure_path=root / "791drug_struc.csv",
            protein_sequence_path=root / "989proseq.csv",
        )
    if dataset == "BioKG":
        root = data_root / dataset
        return DatasetConfig(
            name=dataset,
            root=root,
            dti_path=root / "dti.csv",
            kg_paths=(root / "kg.csv",),
            drug_feature_path=root / "fp_df.csv",
            protein_feature_path=root / "prodes_df.csv",
            drug_structure_path=root / "comp_struc.csv",
            protein_sequence_path=root / "pro_seq.csv",
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def ensure_dirs(dataset_out: Path) -> dict[str, Path]:
    dirs = {
        "root": dataset_out,
        "figures": dataset_out / "figures",
        "tables": dataset_out / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_triples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    sep = r"\s+" if path.suffix == ".txt" else ","
    frame = pd.read_csv(path, sep=sep)
    if set(TRIPLE_COLUMNS).issubset(frame.columns):
        return frame[TRIPLE_COLUMNS].astype(str)

    frame = pd.read_csv(path, sep=sep, header=None, usecols=[0, 1, 2])
    frame.columns = TRIPLE_COLUMNS
    return frame.astype(str)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_plot_style() -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#718096",
            "axes.labelcolor": "#4A5568",
            "xtick.color": "#718096",
            "ytick.color": "#718096",
            "grid.color": "#CBD5E0",
            "font.size": 11,
            "axes.titleweight": "bold",
        }
    )


def svg_text(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path: Path, body: str, width: int = 1200, height: int = 650) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def svg_axes(x: int, y: int, w: int, h: int, title: str, xlabel: str, ylabel: str) -> str:
    return f"""
<text x="{x + w / 2:.1f}" y="{y - 25}" text-anchor="middle" font-size="22" font-weight="700" fill="#334155">{svg_text(title)}</text>
<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#64748B" stroke-width="2"/>
<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#64748B" stroke-width="2"/>
<text x="{x + w / 2:.1f}" y="{y + h + 52}" text-anchor="middle" font-size="16" fill="#475569">{svg_text(xlabel)}</text>
<text x="{x - 52}" y="{y + h / 2:.1f}" transform="rotate(-90 {x - 52} {y + h / 2:.1f})" text-anchor="middle" font-size="16" fill="#475569">{svg_text(ylabel)}</text>
"""


def svg_hist_panel(values: Iterable[float], x: int, y: int, w: int, h: int, title: str, xlabel: str, ylabel: str, color: str) -> str:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return ""
    counts, edges = np.histogram(arr, bins=min(30, max(5, int(np.sqrt(arr.size)))))
    max_count = max(int(counts.max()), 1)
    body = [svg_axes(x, y, w, h, title, xlabel, ylabel)]
    for i, count in enumerate(counts):
        bar_w = w / len(counts) * 0.82
        bx = x + i * (w / len(counts)) + (w / len(counts) - bar_w) / 2
        bh = h * (count / max_count)
        by = y + h - bh
        body.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.82" stroke="#334155" stroke-width="0.8"/>')
    for tick in np.linspace(0, max_count, 5):
        ty = y + h - h * (tick / max_count)
        body.append(f'<line x1="{x}" y1="{ty:.1f}" x2="{x + w}" y2="{ty:.1f}" stroke="#CBD5E1" stroke-dasharray="5,5"/>')
        body.append(f'<text x="{x - 8}" y="{ty + 4:.1f}" text-anchor="end" font-size="12" fill="#64748B">{int(tick)}</text>')
    body.append(f'<text x="{x}" y="{y + h + 24}" text-anchor="start" font-size="12" fill="#64748B">{edges[0]:.0f}</text>')
    body.append(f'<text x="{x + w}" y="{y + h + 24}" text-anchor="end" font-size="12" fill="#64748B">{edges[-1]:.0f}</text>')
    return "\n".join(body)


def svg_horizontal_bars(path: Path, labels: list[str], values: list[float], title: str, color: str) -> None:
    width, height = 1100, max(520, 70 + 28 * len(labels))
    x, y, w, h = 260, 70, 820, height - 120
    max_value = max(values) if values else 1
    body = [f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="24" font-weight="700" fill="#334155">{svg_text(title)}</text>']
    for idx, (label, value) in enumerate(zip(labels, values)):
        row_h = h / max(len(labels), 1)
        by = y + idx * row_h + row_h * 0.18
        bw = w * (value / max_value)
        body.append(f'<text x="{x - 10}" y="{by + row_h * 0.45:.1f}" text-anchor="end" font-size="13" fill="#475569">{svg_text(label)}</text>')
        body.append(f'<rect x="{x}" y="{by:.1f}" width="{bw:.1f}" height="{row_h * 0.64:.1f}" fill="{color}" opacity="0.85"/>')
        body.append(f'<text x="{x + bw + 6:.1f}" y="{by + row_h * 0.45:.1f}" font-size="12" fill="#475569">{value:.0f}</text>')
    write_svg(path, "\n".join(body), width=width, height=height)


def svg_grouped_bars(path: Path, data: pd.DataFrame, title: str) -> None:
    width, height = 1100, 620
    x, y, w, h = 90, 80, 980, 430
    labels = list(data.index)
    columns = list(data.columns)
    colors = ["#22C55E", "#94A3B8", "#F97316", "#3B82F6"]
    max_value = float(data.to_numpy().max()) if data.size else 1.0
    group_w = w / max(len(labels), 1)
    bar_w = group_w / (len(columns) + 1)
    body = [f'<text x="{width / 2}" y="40" text-anchor="middle" font-size="24" font-weight="700" fill="#334155">{svg_text(title)}</text>']
    body.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#64748B" stroke-width="2"/>')
    body.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#64748B" stroke-width="2"/>')
    for i, label in enumerate(labels):
        gx = x + i * group_w
        for j, col in enumerate(columns):
            value = float(data.loc[label, col])
            bh = h * (value / max_value)
            bx = gx + (j + 0.5) * bar_w
            by = y + h - bh
            body.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w * 0.82:.1f}" height="{bh:.1f}" fill="{colors[j % len(colors)]}" opacity="0.85"/>')
        body.append(f'<text x="{gx + group_w / 2:.1f}" y="{y + h + 35}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(18 {gx + group_w / 2:.1f} {y + h + 35})">{svg_text(label)}</text>')
    for j, col in enumerate(columns):
        lx = x + j * 190
        body.append(f'<rect x="{lx}" y="{height - 35}" width="16" height="16" fill="{colors[j % len(colors)]}" opacity="0.85"/>')
        body.append(f'<text x="{lx + 22}" y="{height - 22}" font-size="13" fill="#475569">{svg_text(col)}</text>')
    write_svg(path, "\n".join(body), width=width, height=height)


def series_stats(values: pd.Series | np.ndarray, label: str) -> dict[str, float | int | str]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"metric": label, "count": 0}
    return {
        "metric": label,
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def dti_network_analysis(dti: pd.DataFrame, dirs: dict[str, Path], dataset: str) -> dict:
    drug_degrees = dti.groupby("head").size().sort_values(ascending=False)
    target_degrees = dti.groupby("tail").size().sort_values(ascending=False)

    n_interactions = len(dti)
    n_drugs = dti["head"].nunique()
    n_targets = dti["tail"].nunique()
    matrix_size = n_drugs * n_targets
    density = n_interactions / matrix_size if matrix_size else 0.0

    summary = {
        "dataset": dataset,
        "positive_interactions": int(n_interactions),
        "unique_drugs": int(n_drugs),
        "unique_targets": int(n_targets),
        "possible_pairs": int(matrix_size),
        "matrix_density": float(density),
        "matrix_sparsity": float(1.0 - density),
        "avg_targets_per_drug": float(n_interactions / n_drugs) if n_drugs else 0.0,
        "avg_drugs_per_target": float(n_interactions / n_targets) if n_targets else 0.0,
        "drugs_degree_le_5": int((drug_degrees <= 5).sum()),
        "drugs_degree_le_5_pct": float((drug_degrees <= 5).mean()) if len(drug_degrees) else 0.0,
        "targets_degree_le_2": int((target_degrees <= 2).sum()),
        "targets_degree_le_2_pct": float((target_degrees <= 2).mean()) if len(target_degrees) else 0.0,
    }

    save_json(summary, dirs["tables"] / "dti_network_summary.json")
    save_csv(pd.DataFrame([summary]), dirs["tables"] / "dti_network_summary.csv")

    save_csv(
        drug_degrees.reset_index(name="degree").rename(columns={"head": "drug_id"}),
        dirs["tables"] / "drug_degrees.csv",
    )
    save_csv(
        target_degrees.reset_index(name="degree").rename(columns={"tail": "target_id"}),
        dirs["tables"] / "target_degrees.csv",
    )
    save_csv(
        drug_degrees.head(25).reset_index(name="degree").rename(columns={"head": "drug_id"}),
        dirs["tables"] / "top_25_drug_hubs.csv",
    )
    save_csv(
        target_degrees.head(25).reset_index(name="degree").rename(columns={"tail": "target_id"}),
        dirs["tables"] / "top_25_target_hubs.csv",
    )

    plot_degree_distribution(drug_degrees, target_degrees, dirs["figures"], dataset)
    plot_ccdf(drug_degrees, target_degrees, dirs["figures"], dataset)
    plot_top_hubs(drug_degrees, target_degrees, dirs["figures"], dataset)

    return summary


def plot_degree_distribution(
    drug_degrees: pd.Series,
    target_degrees: pd.Series,
    figures_dir: Path,
    dataset: str,
) -> None:
    if not HAS_MATPLOTLIB:
        body = "\n".join(
            [
                svg_hist_panel(
                    drug_degrees.values,
                    90,
                    95,
                    470,
                    390,
                    f"{dataset}: Drug Degree Distribution",
                    "Number of Associated Targets",
                    "Count of Drugs",
                    "#8B5CF6",
                ),
                svg_hist_panel(
                    target_degrees.values,
                    690,
                    95,
                    470,
                    390,
                    f"{dataset}: Protein Target Degree Distribution",
                    "Number of Associated Drugs",
                    "Count of Targets",
                    "#D946EF",
                ),
            ]
        )
        write_svg(figures_dir / "degree_distribution.svg", body, width=1240, height=620)
        return

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    axes[0].hist(drug_degrees.values, bins=30, color="#8B5CF6", alpha=0.82, edgecolor="#4C1D95")
    axes[0].set_title(f"{dataset}: Drug Degree Distribution")
    axes[0].set_xlabel("Number of Associated Targets")
    axes[0].set_ylabel("Count of Drugs")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    axes[1].hist(target_degrees.values, bins=30, color="#D946EF", alpha=0.82, edgecolor="#86198F")
    axes[1].set_title(f"{dataset}: Protein Target Degree Distribution")
    axes[1].set_xlabel("Number of Associated Drugs")
    axes[1].set_ylabel("Count of Targets")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(figures_dir / "degree_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ccdf(
    drug_degrees: pd.Series,
    target_degrees: pd.Series,
    figures_dir: Path,
    dataset: str,
) -> None:
    if not HAS_MATPLOTLIB:
        # SVG fallback: save the same heavy-tail information as a ranked-degree line chart.
        body = []
        for panel, (degrees, title, color) in enumerate(
            [
                (drug_degrees, "Drug Degree Rank Plot", "#7C3AED"),
                (target_degrees, "Target Degree Rank Plot", "#C026D3"),
            ]
        ):
            x0, y0, w, h = (90 + panel * 600), 95, 470, 390
            values = np.sort(degrees.values)[::-1]
            max_x = max(len(values) - 1, 1)
            max_y = max(float(values.max()), 1.0)
            points = []
            for i, value in enumerate(values):
                px = x0 + w * (i / max_x)
                py = y0 + h - h * (float(value) / max_y)
                points.append(f"{px:.1f},{py:.1f}")
            body.append(svg_axes(x0, y0, w, h, f"{dataset}: {title}", "Rank", "Degree"))
            body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        write_svg(figures_dir / "degree_rank_plot.svg", "\n".join(body), width=1240, height=620)
        return

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, degrees, title, color in [
        (axes[0], drug_degrees, "Drug Degree CCDF", "#7C3AED"),
        (axes[1], target_degrees, "Target Degree CCDF", "#C026D3"),
    ]:
        values = np.sort(degrees.values)
        ccdf = 1.0 - np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, ccdf, marker="o", markersize=3, linewidth=1.6, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{dataset}: {title}")
        ax.set_xlabel("Degree")
        ax.set_ylabel("P(Degree >= x)")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(figures_dir / "degree_ccdf_loglog.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_hubs(
    drug_degrees: pd.Series,
    target_degrees: pd.Series,
    figures_dir: Path,
    dataset: str,
    top_n: int = 15,
) -> None:
    if not HAS_MATPLOTLIB:
        top_drugs = drug_degrees.head(top_n).sort_values(ascending=False)
        top_targets = target_degrees.head(top_n).sort_values(ascending=False)
        svg_horizontal_bars(
            figures_dir / "top_drug_hubs.svg",
            [str(x) for x in top_drugs.index],
            [float(x) for x in top_drugs.values],
            f"{dataset}: Top {top_n} Drug Hubs",
            "#7C3AED",
        )
        svg_horizontal_bars(
            figures_dir / "top_target_hubs.svg",
            [str(x) for x in top_targets.index],
            [float(x) for x in top_targets.values],
            f"{dataset}: Top {top_n} Target Hubs",
            "#C026D3",
        )
        return

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_drugs = drug_degrees.head(top_n).sort_values()
    axes[0].barh(top_drugs.index, top_drugs.values, color="#7C3AED", alpha=0.85)
    axes[0].set_title(f"{dataset}: Top {top_n} Drug Hubs")
    axes[0].set_xlabel("Degree")
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.65)

    top_targets = target_degrees.head(top_n).sort_values()
    axes[1].barh(top_targets.index, top_targets.values, color="#C026D3", alpha=0.85)
    axes[1].set_title(f"{dataset}: Top {top_n} Target Hubs")
    axes[1].set_xlabel("Degree")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.65)

    fig.tight_layout()
    fig.savefig(figures_dir / "top_hubs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def kg_analysis(config: DatasetConfig, dirs: dict[str, Path]) -> dict:
    rows = []
    relation_frames = []
    all_kg_frames = []

    for path in config.kg_paths:
        kg = read_triples(path)
        all_kg_frames.append(kg)
        entities = pd.concat([kg["head"], kg["tail"]], ignore_index=True).nunique()
        rows.append(
            {
                "kg_file": path.name,
                "triples": len(kg),
                "entities": int(entities),
                "relations": int(kg["relation"].nunique()),
                "unique_heads": int(kg["head"].nunique()),
                "unique_tails": int(kg["tail"].nunique()),
            }
        )
        rel = kg["relation"].value_counts().rename_axis("relation").reset_index(name="count")
        rel.insert(0, "kg_file", path.name)
        relation_frames.append(rel)

    kg_all = pd.concat(all_kg_frames, ignore_index=True) if all_kg_frames else pd.DataFrame(columns=TRIPLE_COLUMNS)
    rows.append(
        {
            "kg_file": "combined",
            "triples": len(kg_all),
            "entities": int(pd.concat([kg_all["head"], kg_all["tail"]], ignore_index=True).nunique()) if len(kg_all) else 0,
            "relations": int(kg_all["relation"].nunique()) if len(kg_all) else 0,
            "unique_heads": int(kg_all["head"].nunique()) if len(kg_all) else 0,
            "unique_tails": int(kg_all["tail"].nunique()) if len(kg_all) else 0,
        }
    )

    kg_summary = pd.DataFrame(rows)
    relation_counts = pd.concat(relation_frames, ignore_index=True) if relation_frames else pd.DataFrame()
    combined_relations = kg_all["relation"].value_counts().rename_axis("relation").reset_index(name="count")

    save_csv(kg_summary, dirs["tables"] / "kg_summary.csv")
    save_csv(relation_counts, dirs["tables"] / "kg_relation_counts_by_file.csv")
    save_csv(combined_relations, dirs["tables"] / "kg_relation_counts_combined.csv")
    plot_relation_distribution(combined_relations, dirs["figures"], config.name)

    return kg_summary.iloc[-1].to_dict()


def plot_relation_distribution(relation_counts: pd.DataFrame, figures_dir: Path, dataset: str, top_n: int = 20) -> None:
    if relation_counts.empty:
        return
    if not HAS_MATPLOTLIB:
        top = relation_counts.head(top_n)
        svg_horizontal_bars(
            figures_dir / "kg_top_relations.svg",
            [str(x) for x in top["relation"]],
            [float(x) for x in top["count"]],
            f"{dataset}: Top {top_n} Knowledge Graph Relations",
            "#2563EB",
        )
        return

    set_plot_style()
    top = relation_counts.head(top_n).sort_values("count")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top["relation"], top["count"], color="#2563EB", alpha=0.85)
    ax.set_title(f"{dataset}: Top {top_n} Knowledge Graph Relations")
    ax.set_xlabel("Triple Count")
    ax.grid(True, axis="x", linestyle="--", alpha=0.65)
    fig.tight_layout()
    fig.savefig(figures_dir / "kg_top_relations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def feature_analysis(config: DatasetConfig, dirs: dict[str, Path]) -> None:
    rows = []

    if config.name == "yamanishi_08":
        if config.drug_feature_path and config.drug_feature_path.exists():
            drug = np.loadtxt(config.drug_feature_path, delimiter=",")
            active_bits = (drug == 1).sum(axis=1)
            rows.append(
                {
                    "feature_block": "drug_morgan_fingerprint",
                    "rows": int(drug.shape[0]),
                    "columns": int(drug.shape[1]),
                    "min": float(np.min(drug)),
                    "max": float(np.max(drug)),
                    "mean": float(np.mean(drug)),
                    "sparsity_zero": float((drug == 0).mean()),
                    "avg_active_bits": float(active_bits.mean()),
                }
            )
            plot_feature_density(active_bits, dirs["figures"], config.name, "drug_active_bits", "Active Morgan Bits per Drug")

        if config.protein_feature_path and config.protein_feature_path.exists():
            protein = np.loadtxt(config.protein_feature_path, delimiter=",")
            rows.append(
                {
                    "feature_block": "protein_ctd_descriptor",
                    "rows": int(protein.shape[0]),
                    "columns": int(protein.shape[1]),
                    "min": float(np.min(protein)),
                    "max": float(np.max(protein)),
                    "mean": float(np.mean(protein)),
                    "sparsity_zero": float((protein == 0).mean()),
                    "avg_active_bits": np.nan,
                }
            )
            plot_feature_value_distribution(protein, dirs["figures"], config.name, "protein_ctd_values")
    else:
        for label, path, id_col in [
            ("drug_fingerprint", config.drug_feature_path, "comp_id"),
            ("protein_descriptor", config.protein_feature_path, "pro_ids"),
        ]:
            if path and path.exists():
                frame = pd.read_csv(path)
                numeric = frame.drop(columns=[id_col], errors="ignore").select_dtypes(include=[np.number])
                values = numeric.to_numpy(dtype=float)
                rows.append(
                    {
                        "feature_block": label,
                        "rows": int(frame.shape[0]),
                        "columns": int(numeric.shape[1]),
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                        "mean": float(np.nanmean(values)),
                        "sparsity_zero": float(np.nanmean(values == 0)),
                        "avg_active_bits": float(np.nanmean((values == 1).sum(axis=1))) if values.size else np.nan,
                    }
                )
                if label == "drug_fingerprint":
                    plot_feature_density((values == 1).sum(axis=1), dirs["figures"], config.name, "drug_active_bits", "Active Fingerprint Bits per Drug")
                else:
                    plot_feature_value_distribution(values, dirs["figures"], config.name, "protein_descriptor_values")

    if config.drug_structure_path and config.drug_structure_path.exists():
        structure = pd.read_csv(config.drug_structure_path)
        smiles_col = next((c for c in structure.columns if c.lower() in {"smiles", "canonical_smiles"}), None)
        if smiles_col:
            lengths = structure[smiles_col].astype(str).str.len()
            rows.append(series_stats(lengths, "drug_smiles_length"))
            plot_feature_density(lengths, dirs["figures"], config.name, "smiles_lengths", "SMILES Length")

    if config.protein_sequence_path and config.protein_sequence_path.exists():
        seq = pd.read_csv(config.protein_sequence_path)
        seq_col = next((c for c in seq.columns if c.lower() in {"seq", "sequence"}), None)
        if seq_col:
            lengths = seq[seq_col].astype(str).str.len()
            rows.append(series_stats(lengths, "protein_sequence_length"))
            plot_feature_density(lengths, dirs["figures"], config.name, "protein_sequence_lengths", "Protein Sequence Length")

    if rows:
        save_csv(pd.DataFrame(rows), dirs["tables"] / "feature_summary.csv")


def plot_feature_density(values: Iterable[float], figures_dir: Path, dataset: str, slug: str, xlabel: str) -> None:
    if not HAS_MATPLOTLIB:
        body = svg_hist_panel(values, 90, 95, 820, 390, f"{dataset}: {xlabel} Distribution", xlabel, "Count", "#059669")
        write_svg(figures_dir / f"{slug}.svg", body, width=1000, height=620)
        return

    set_plot_style()
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=35, color="#059669", alpha=0.82, edgecolor="#065F46")
    ax.set_title(f"{dataset}: {xlabel} Distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.65)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{slug}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_value_distribution(values: np.ndarray, figures_dir: Path, dataset: str, slug: str) -> None:
    if not HAS_MATPLOTLIB:
        arr = values.reshape(-1)
        if arr.size > 250_000:
            rng = np.random.default_rng(42)
            arr = rng.choice(arr, size=250_000, replace=False)
        body = svg_hist_panel(arr, 90, 95, 820, 390, f"{dataset}: Descriptor Value Distribution", "Feature Value", "Frequency", "#0EA5E9")
        write_svg(figures_dir / f"{slug}.svg", body, width=1000, height=620)
        return

    set_plot_style()
    arr = values.reshape(-1)
    if arr.size > 250_000:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=250_000, replace=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=50, color="#0EA5E9", alpha=0.82, edgecolor="#075985")
    ax.set_title(f"{dataset}: Descriptor Value Distribution")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.65)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{slug}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def split_analysis(config: DatasetConfig, dirs: dict[str, Path], splits: list[str]) -> None:
    rows = []
    for split in splits:
        split_root = config.root / "data_folds" / split
        if not split_root.exists():
            continue
        for fold in range(1, 11):
            train_path = split_root / f"train_fold_{fold}.csv"
            test_path = split_root / f"test_fold_{fold}.csv"
            if not train_path.exists() or not test_path.exists():
                continue
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            train.columns = [str(col).strip() for col in train.columns]
            test.columns = [str(col).strip() for col in test.columns]
            train_pos = train[train["label"] == 1]
            test_pos = test[test["label"] == 1]
            rows.append(
                {
                    "split": split,
                    "fold": fold,
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "train_pos": int((train["label"] == 1).sum()),
                    "train_neg": int((train["label"] == 0).sum()),
                    "test_pos": int((test["label"] == 1).sum()),
                    "test_neg": int((test["label"] == 0).sum()),
                    "train_drugs": int(train["head"].nunique()),
                    "test_drugs": int(test["head"].nunique()),
                    "train_targets": int(train["tail"].nunique()),
                    "test_targets": int(test["tail"].nunique()),
                    "all_drug_overlap": int(len(set(train["head"]) & set(test["head"]))),
                    "all_target_overlap": int(len(set(train["tail"]) & set(test["tail"]))),
                    "positive_drug_overlap": int(len(set(train_pos["head"]) & set(test_pos["head"]))),
                    "positive_target_overlap": int(len(set(train_pos["tail"]) & set(test_pos["tail"]))),
                }
            )

    if not rows:
        return

    fold_stats = pd.DataFrame(rows)
    save_csv(fold_stats, dirs["tables"] / "fold_level_split_stats.csv")

    agg = (
        fold_stats.groupby("split")
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
    save_csv(agg, dirs["tables"] / "split_summary.csv")

    plot_split_class_balance(fold_stats, dirs["figures"], config.name)
    plot_split_entity_overlap(fold_stats, dirs["figures"], config.name)


def plot_split_class_balance(fold_stats: pd.DataFrame, figures_dir: Path, dataset: str) -> None:
    if not HAS_MATPLOTLIB:
        summary = fold_stats.groupby("split")[["train_pos", "train_neg", "test_pos", "test_neg"]].mean()
        svg_grouped_bars(figures_dir / "split_class_balance.svg", summary, f"{dataset}: Mean Class Balance")
        return

    set_plot_style()
    summary = fold_stats.groupby("split")[["train_pos", "train_neg", "test_pos", "test_neg"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary[["train_pos", "train_neg"]].plot(kind="bar", stacked=True, ax=axes[0], color=["#22C55E", "#94A3B8"])
    axes[0].set_title(f"{dataset}: Mean Train Class Balance")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Samples")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.65)

    summary[["test_pos", "test_neg"]].plot(kind="bar", stacked=True, ax=axes[1], color=["#22C55E", "#94A3B8"])
    axes[1].set_title(f"{dataset}: Mean Test Class Balance")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Samples")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.65)

    fig.tight_layout()
    fig.savefig(figures_dir / "split_class_balance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_split_entity_overlap(fold_stats: pd.DataFrame, figures_dir: Path, dataset: str) -> None:
    if not HAS_MATPLOTLIB:
        summary = fold_stats.groupby("split")[["positive_drug_overlap", "positive_target_overlap"]].mean()
        svg_grouped_bars(figures_dir / "positive_entity_overlap.svg", summary, f"{dataset}: Mean Positive Entity Overlap")
        return

    set_plot_style()
    summary = fold_stats.groupby("split")[["positive_drug_overlap", "positive_target_overlap"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=ax, color=["#F97316", "#3B82F6"])
    ax.set_title(f"{dataset}: Mean Positive Entity Overlap")
    ax.set_xlabel("Split")
    ax.set_ylabel("Overlap Count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.65)
    fig.tight_layout()
    fig.savefig(figures_dir / "positive_entity_overlap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(dataset: str, dataset_out: Path) -> None:
    tables = dataset_out / "tables"
    figures = dataset_out / "figures"
    network = pd.read_csv(tables / "dti_network_summary.csv").iloc[0].to_dict()
    kg_path = tables / "kg_summary.csv"
    kg = pd.read_csv(kg_path).tail(1).iloc[0].to_dict() if kg_path.exists() else {}

    lines = [
        f"# {dataset} Dataset Analysis",
        "",
        "## DTI Network",
        "",
        f"- Positive interactions: {int(network['positive_interactions']):,}",
        f"- Unique drugs: {int(network['unique_drugs']):,}",
        f"- Unique targets: {int(network['unique_targets']):,}",
        f"- Matrix density: {network['matrix_density']:.4%}",
        f"- Matrix sparsity: {network['matrix_sparsity']:.4%}",
        f"- Drugs with degree <= 5: {int(network['drugs_degree_le_5']):,} ({network['drugs_degree_le_5_pct']:.2%})",
        f"- Targets with degree <= 2: {int(network['targets_degree_le_2']):,} ({network['targets_degree_le_2_pct']:.2%})",
        "",
    ]
    if kg:
        lines += [
            "## Knowledge Graph",
            "",
            f"- Triples: {int(kg['triples']):,}",
            f"- Entities: {int(kg['entities']):,}",
            f"- Relations: {int(kg['relations']):,}",
            "",
        ]
    lines += [
        "## Generated Figures",
        "",
    ]
    figure_paths = sorted([*figures.glob("*.png"), *figures.glob("*.svg")])
    for fig in figure_paths:
        lines.append(f"- `{fig.relative_to(dataset_out)}`")
    lines.append("")
    (dataset_out / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def comparative_analysis(result_root: Path, datasets: list[str]) -> None:
    rows = []
    for dataset in datasets:
        summary_path = result_root / dataset / "tables" / "dti_network_summary.csv"
        kg_path = result_root / dataset / "tables" / "kg_summary.csv"
        if not summary_path.exists():
            continue
        row = pd.read_csv(summary_path).iloc[0].to_dict()
        if kg_path.exists():
            kg = pd.read_csv(kg_path).tail(1).iloc[0].to_dict()
            row["kg_triples"] = kg.get("triples", np.nan)
            row["kg_entities"] = kg.get("entities", np.nan)
            row["kg_relations"] = kg.get("relations", np.nan)
        rows.append(row)
    if not rows:
        return
    comparison = pd.DataFrame(rows)
    comparison.to_csv(result_root / "dataset_comparison.csv", index=False)

    if not HAS_MATPLOTLIB:
        plot_data = comparison.set_index("dataset")
        columns = [c for c in ["positive_interactions", "matrix_density", "kg_triples"] if c in plot_data.columns]
        svg_grouped_bars(result_root / "dataset_comparison.svg", plot_data[columns], "Dataset Comparison")
        return

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    comparison.plot(x="dataset", y="positive_interactions", kind="bar", ax=axes[0], legend=False, color="#7C3AED")
    axes[0].set_title("Positive Interactions")
    axes[0].set_ylabel("Count")
    comparison.plot(x="dataset", y="matrix_density", kind="bar", ax=axes[1], legend=False, color="#059669")
    axes[1].set_title("DTI Matrix Density")
    axes[1].set_ylabel("Density")
    if "kg_triples" in comparison.columns:
        comparison.plot(x="dataset", y="kg_triples", kind="bar", ax=axes[2], legend=False, color="#2563EB")
        axes[2].set_title("KG Triples")
        axes[2].set_ylabel("Count")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.65)
    fig.tight_layout()
    fig.savefig(result_root / "dataset_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(config: DatasetConfig, output_root: Path, splits: list[str]) -> None:
    dataset_out = output_root / config.name
    dirs = ensure_dirs(dataset_out)
    dti = read_triples(config.dti_path)

    print(f"[{config.name}] DTI network analysis")
    dti_network_analysis(dti, dirs, config.name)

    print(f"[{config.name}] Knowledge graph analysis")
    kg_analysis(config, dirs)

    print(f"[{config.name}] Feature analysis")
    feature_analysis(config, dirs)

    print(f"[{config.name}] Split analysis")
    split_analysis(config, dirs, splits)

    write_markdown_report(config.name, dataset_out)
    print(f"[{config.name}] Wrote outputs to {dataset_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DTI datasets and generate report-ready figures.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Dataset root directory.")
    parser.add_argument("--output-root", type=Path, default=Path("analysis") / "results", help="Output directory.")
    parser.add_argument("--datasets", nargs="+", default=["yamanishi_08", "BioKG"], help="Datasets to analyze.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, help="Fold splits to summarize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for dataset in args.datasets:
        config = build_config(args.data_root, dataset)
        analyze_dataset(config, args.output_root, args.splits)
        completed.append(dataset)
    comparative_analysis(args.output_root, completed)
    print(f"Done. Results are under {args.output_root}")


if __name__ == "__main__":
    main()
