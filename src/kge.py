from typing import Iterable

import numpy as np
import pandas as pd
import torch

from constants import TRIPLE_COLUMNS


def _make_triples_factory(train_triples: pd.DataFrame, create_inverse_triples: bool):
    from pykeen.triples import TriplesFactory

    return TriplesFactory.from_labeled_triples(
        train_triples[TRIPLE_COLUMNS].to_numpy(dtype=str),
        create_inverse_triples=create_inverse_triples,
    )


def _make_distmult(triples_factory, embedding_dim: int, seed: int, device: str):
    from pykeen.models import DistMult

    return DistMult(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        loss="MarginRankingLoss",
        loss_kwargs={"margin": 1.0},
        random_seed=seed,
    ).to(device)


def _make_compgcn(
    triples_factory,
    embedding_dim: int,
    seed: int,
    device: str,
    num_layers: int,
    layer_dropout: float,
    composition: str,
):
    from pykeen.models import CompGCN
    from pykeen.nn.compositions import (
        CircularCorrelationCompositionModule,
        MultiplicationCompositionModule,
        SubtractionCompositionModule,
    )

    if num_layers < 1:
        raise ValueError(f"CompGCN requires at least one layer, got {num_layers}.")
    if not 0.0 <= layer_dropout < 1.0:
        raise ValueError(f"CompGCN dropout must be in [0.0, 1.0), got {layer_dropout}.")

    composition_modules = {
        "sub": SubtractionCompositionModule,
        "mult": MultiplicationCompositionModule,
        "corr": CircularCorrelationCompositionModule,
    }
    if composition not in composition_modules:
        raise ValueError(
            f"Unsupported CompGCN composition '{composition}'. "
            f"Expected one of: {', '.join(composition_modules)}."
        )

    return CompGCN(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        encoder_kwargs={
            "num_layers": num_layers,
            "dims": embedding_dim,
            "layer_kwargs": {
                "dropout": layer_dropout,
                "composition": composition_modules[composition],
            },
        },
        loss="MarginRankingLoss",
        loss_kwargs={"margin": 1.0},
        random_seed=seed,
    ).to(device)


def require_entities(factory, entities: Iterable[str], context: str) -> None:
    missing = sorted(set(entities).difference(factory.entity_to_id))
    if missing:
        examples = ", ".join(missing[:10])
        raise ValueError(
            f"{context} has {len(missing)} entities missing from the PyKEEN training graph. "
            f"Examples: {examples}"
        )


def train_distmult(
    train_triples: pd.DataFrame,
    device: str,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
    num_negs_per_pos: int,
    use_tqdm: bool,
):
    from pykeen.training import SLCWATrainingLoop

    triples_factory = _make_triples_factory(train_triples, create_inverse_triples=False)
    model = _make_distmult(triples_factory, embedding_dim, seed, device)

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer="adam",
        optimizer_kwargs={"lr": lr},
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": num_negs_per_pos},
    )
    losses = training_loop.train(
        triples_factory=triples_factory,
        num_epochs=epochs,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        use_tqdm_batch=use_tqdm,
    )
    return model, triples_factory, losses


def train_compgcn(
    train_triples: pd.DataFrame,
    device: str,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
    num_negs_per_pos: int,
    use_tqdm: bool,
    num_layers: int = 1,
    layer_dropout: float = 0.0,
    composition: str = "mult",
):
    from pykeen.training import SLCWATrainingLoop

    triples_factory = _make_triples_factory(train_triples, create_inverse_triples=True)
    model = _make_compgcn(
        triples_factory,
        embedding_dim,
        seed,
        device,
        num_layers,
        layer_dropout,
        composition,
    )

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer="adam",
        optimizer_kwargs={"lr": lr},
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": num_negs_per_pos},
    )
    losses = training_loop.train(
        triples_factory=triples_factory,
        num_epochs=epochs,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        use_tqdm_batch=use_tqdm,
    )
    return model, triples_factory, losses


def load_kge_checkpoint(checkpoint_path, train_triples: pd.DataFrame, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model_name = saved_args.get("kge_model", "distmult")
    seed = int(saved_args.get("seed", 0))
    embedding_dim = int(saved_args.get("embedding_dim", 400))
    triples_factory = _make_triples_factory(
        train_triples,
        create_inverse_triples=model_name == "compgcn",
    )

    if checkpoint.get("entity_to_id") != triples_factory.entity_to_id:
        raise ValueError(f"Entity mapping in {checkpoint_path} does not match the current fold data.")
    if checkpoint.get("relation_to_id") != triples_factory.relation_to_id:
        raise ValueError(f"Relation mapping in {checkpoint_path} does not match the current fold data.")

    if model_name == "compgcn":
        model = _make_compgcn(
            triples_factory,
            embedding_dim,
            seed,
            device,
            int(saved_args.get("compgcn_layers", 1)),
            float(saved_args.get("compgcn_dropout", 0.0)),
            saved_args.get("compgcn_composition", "mult"),
        )
    else:
        model = _make_distmult(triples_factory, embedding_dim, seed, device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, triples_factory, checkpoint.get("losses", []), saved_args


def _prepare_model_for_inference(model) -> None:
    was_training = model.training
    model.eval()

    # CompGCN caches enriched full-graph embeddings. If we switch from training
    # to inference, clear the cache so BatchNorm/dropout use eval behavior.
    if was_training:
        for module in model.modules():
            if hasattr(module, "enriched_representations"):
                module.enriched_representations = None


def entity_embeddings(model, triples_factory, labels: Iterable[str], device: str) -> np.ndarray:
    _prepare_model_for_inference(model)

    ids = triples_factory.entities_to_ids(list(labels))
    indices = torch.as_tensor(ids, dtype=torch.long, device=device)

    with torch.no_grad():
        representation = model.entity_representations[0](indices=indices)

    return representation.detach().cpu().numpy()


def pair_embeddings(model, triples_factory, pairs: pd.DataFrame, device: str) -> np.ndarray:
    heads, tails = pair_embedding_blocks(model, triples_factory, pairs, device)
    return np.concatenate([heads, tails], axis=1)


def pair_embedding_blocks(model, triples_factory, pairs: pd.DataFrame, device: str) -> tuple[np.ndarray, np.ndarray]:
    _prepare_model_for_inference(model)

    heads = entity_embeddings(model, triples_factory, pairs["head"].astype(str).tolist(), device)
    tails = entity_embeddings(model, triples_factory, pairs["tail"].astype(str).tolist(), device)
    return heads, tails


def score_triples(model, triples_factory, triples: pd.DataFrame, device: str, batch_size: int) -> np.ndarray:
    _prepare_model_for_inference(model)

    mapped = triples_factory.map_triples(triples[TRIPLE_COLUMNS].to_numpy(dtype=str))
    scores = []

    with torch.no_grad():
        for start in range(0, mapped.shape[0], batch_size):
            batch = mapped[start : start + batch_size].to(device)
            score = model.predict_hrt(batch).detach().cpu().reshape(-1).numpy()
            scores.append(score)

    return np.concatenate(scores)
