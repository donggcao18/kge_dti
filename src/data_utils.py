from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from constants import TRIPLE_COLUMNS


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    dti_path: Path | None
    kg_paths: tuple[Path, ...]
    fold_root: Path
    drug_feature_path: Path
    protein_feature_path: Path
    drug_id_col: str
    protein_id_col: str
    yamanishi_features: bool = False


def get_dataset_spec(data_root: Path, dataset: str, split: str) -> DatasetSpec:
    if dataset == "yamanishi_08":
        root = data_root / dataset
        return DatasetSpec(
            root=root,
            dti_path=root / "dt_all_08.txt",
            kg_paths=(
                root / "kg_data" / "kegg_kg.txt",
                root / "kg_data" / "yamanishi_uniprot_kg.txt",
            ),
            fold_root=root / "data_folds" / split,
            drug_feature_path=root / "morganfp.txt",
            protein_feature_path=root / "pro_ctd.txt",
            drug_id_col="drug_id",
            protein_id_col="pro_id",
            yamanishi_features=True,
        )
    if dataset == "BioKG":
        root = data_root / dataset
        return DatasetSpec(
            root=root,
            dti_path=root / "dti.csv",
            kg_paths=(root / "kg.csv",),
            fold_root=root / "data_folds" / split,
            drug_feature_path=root / "fp_df.csv",
            protein_feature_path=root / "prodes_df.csv",
            drug_id_col="comp_id",
            protein_id_col="pro_ids",
        )
    if dataset == "hetionet":
        root = data_root / dataset
        return DatasetSpec(
            root=root,
            dti_path=root / "dti.csv",
            kg_paths=(root / "kg.csv",),
            fold_root=root / "data_folds" / split,
            drug_feature_path=root / "fp_df.csv",
            protein_feature_path=root / "prodes_df.csv",
            drug_id_col="comp_id",
            protein_id_col="gene_id",
        )
    if dataset in {"luo", "luo's_dataset", "luo.s_dataset"}:
        raise ValueError(
            "luo's_dataset does not include a KG triples file in this checkout. "
            "Add a KG file or adapt get_dataset_spec() before using it here."
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def read_triples(path: Path, sep: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if sep is None:
        sep = "\t" if path.suffix == ".txt" else ","
    frame = pd.read_csv(path, sep=sep)
    if not set(TRIPLE_COLUMNS).issubset(frame.columns):
        frame = pd.read_csv(path, sep=sep, header=None, usecols=[0, 1, 2])
        frame.columns = TRIPLE_COLUMNS
    return frame[TRIPLE_COLUMNS].astype(str)


def load_all_dti(spec: DatasetSpec) -> pd.DataFrame:
    if spec.dti_path is None:
        raise ValueError(f"No all-DTI file configured for {spec.root.name}")
    sep = "\t" if spec.dti_path.suffix == ".txt" else ","
    return read_triples(spec.dti_path, sep=sep)


def load_kg(spec: DatasetSpec) -> pd.DataFrame:
    frames = [read_triples(path) for path in spec.kg_paths]
    kg = pd.concat(frames, ignore_index=True)
    return kg[TRIPLE_COLUMNS].astype(str)


def load_fold(spec: DatasetSpec, fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = spec.fold_root / f"train_fold_{fold_index + 1}.csv"
    test_path = spec.fold_root / f"test_fold_{fold_index + 1}.csv"
    train = pd.read_csv(train_path)[TRIPLE_COLUMNS + ["label"]]
    test = pd.read_csv(test_path)[TRIPLE_COLUMNS + ["label"]]
    train[TRIPLE_COLUMNS] = train[TRIPLE_COLUMNS].astype(str)
    test[TRIPLE_COLUMNS] = test[TRIPLE_COLUMNS].astype(str)
    train["label"] = train["label"].astype(float)
    test["label"] = test["label"].astype(float)
    return train, test


def load_feature_tables(spec: DatasetSpec, pca_components: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.yamanishi_features:
        drug_ids = pd.read_csv(spec.root / "791drug_struc.csv")[["drug_id"]]
        protein_ids = pd.read_csv(spec.root / "989proseq.csv").iloc[:, [0]].copy()
        protein_ids.columns = ["pro_id"]

        drug_feats = np.loadtxt(spec.drug_feature_path, delimiter=",")
        protein_feats = np.loadtxt(spec.protein_feature_path, delimiter=",")

        scaler = MinMaxScaler(feature_range=(0, 1))
        protein_scaled = scaler.fit_transform(protein_feats)
        n_components = min(pca_components, protein_scaled.shape[0], protein_scaled.shape[1])
        protein_pca = PCA(n_components=n_components).fit_transform(protein_scaled)
        protein_pca = scaler.fit_transform(protein_pca)

        drug_df = pd.concat([drug_ids, pd.DataFrame(drug_feats)], axis=1)
        protein_df = pd.concat([protein_ids, pd.DataFrame(protein_pca)], axis=1)
        return drug_df, protein_df

    drug_df = pd.read_csv(spec.drug_feature_path)
    protein_df = pd.read_csv(spec.protein_feature_path)
    return drug_df, protein_df


def merge_features(
    pairs: pd.DataFrame,
    drug_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    spec: DatasetSpec,
    use_protein_features: bool,
) -> np.ndarray:
    drug_features, protein_features = merge_feature_blocks(pairs, drug_df, protein_df, spec)
    return np.concatenate([drug_features, protein_features], axis=1) if use_protein_features else drug_features


def merge_feature_blocks(
    pairs: pd.DataFrame,
    drug_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    spec: DatasetSpec,
) -> tuple[np.ndarray, np.ndarray]:
    drug_merged = pairs.merge(drug_df, how="left", left_on="head", right_on=spec.drug_id_col)
    protein_merged = pairs.merge(protein_df, how="left", left_on="tail", right_on=spec.protein_id_col)

    drug_features = drug_merged.drop(columns=[*TRIPLE_COLUMNS, spec.drug_id_col], errors="ignore")
    protein_features = protein_merged.drop(columns=[*TRIPLE_COLUMNS, spec.protein_id_col], errors="ignore")
    drug_features = drug_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    protein_features = protein_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

    for name, features in [("drug", drug_features), ("protein", protein_features)]:
        if np.isnan(features).any():
            missing = pairs.loc[np.isnan(features).any(axis=1), ["head", "tail"]].head()
            raise ValueError(f"Missing {name} descriptor features for some pairs, examples:\n{missing}")
    return drug_features, protein_features


def encode_labels(values: Iterable[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}
