import copy
from collections.abc import Sequence
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from metrics_utils import pr_auc, roc_auc


def _module_key(name: str, index: int) -> str:
    key = re.sub(r"[^0-9a-zA-Z_]", "_", name).strip("_").lower()
    return key or f"field_{index}"


class FieldFeatureDataset(Dataset):
    def __init__(
        self,
        feature_fields: Sequence[np.ndarray],
        labels: np.ndarray | None = None,
    ) -> None:
        if not feature_fields:
            raise ValueError("NFM requires at least one feature field.")

        row_count = feature_fields[0].shape[0]
        self.feature_fields = []
        for index, field in enumerate(feature_fields):
            if field.ndim != 2:
                raise ValueError(f"NFM field {index} must be a 2D array, got shape {field.shape}.")
            if field.shape[0] != row_count:
                raise ValueError(
                    f"NFM field {index} has {field.shape[0]} rows, expected {row_count}."
                )
            self.feature_fields.append(torch.as_tensor(field, dtype=torch.float32))

        if labels is not None and labels.shape[0] != row_count:
            raise ValueError(f"NFM labels have {labels.shape[0]} rows, expected {row_count}.")
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feature_fields[0].shape[0]

    def __getitem__(self, index: int):
        item = {
            "fields": tuple(field[index] for field in self.feature_fields),
        }
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item


class TorchNFM(nn.Module):
    def __init__(
        self,
        field_dims: Sequence[int],
        field_embedding_dim: int,
        field_names: Sequence[str] | None = None,
        hidden_units: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if field_embedding_dim <= 0:
            raise ValueError(f"field_embedding_dim must be positive, got {field_embedding_dim}.")
        if len(field_dims) < 2:
            raise ValueError(f"NFM requires at least two feature fields, got {len(field_dims)}.")
        if any(dim <= 0 for dim in field_dims):
            raise ValueError(f"All NFM field dimensions must be positive, got {tuple(field_dims)}.")
        if field_names is None:
            field_names = [f"field_{index}" for index in range(len(field_dims))]
        if len(field_names) != len(field_dims):
            raise ValueError(f"Got {len(field_names)} field names for {len(field_dims)} field dimensions.")

        self.field_names = tuple(field_names)
        self.field_keys = tuple(_module_key(name, index) for index, name in enumerate(self.field_names))
        if len(set(self.field_keys)) != len(self.field_keys):
            raise ValueError(f"NFM field names must be unique after normalization, got {self.field_names}.")

        self.field_projections = nn.ModuleDict(
            {
                key: nn.Linear(field_dim, field_embedding_dim)
                for key, field_dim in zip(self.field_keys, field_dims, strict=False)
            }
        )
        self.linear_terms = nn.ModuleDict(
            {
                key: nn.Linear(field_dim, 1)
                for key, field_dim in zip(self.field_keys, field_dims, strict=False)
            }
        )

        layers: list[nn.Module] = []
        input_dim = field_embedding_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [*self.field_projections.values(), *self.linear_terms.values(), *self.mlp]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feature_fields: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_fields) != len(self.field_projections):
            raise ValueError(
                f"Expected {len(self.field_projections)} NFM fields, got {len(feature_fields)}."
            )

        projected_fields = torch.stack(
            [
                self.field_projections[key](field)
                for key, field in zip(self.field_keys, feature_fields, strict=False)
            ],
            dim=1,
        )
        summed = projected_fields.sum(dim=1)
        squared_sum = summed * summed
        sum_squared = (projected_fields * projected_fields).sum(dim=1)
        bi_interaction = 0.5 * (squared_sum - sum_squared)

        deep_logit = self.mlp(bi_interaction).squeeze(1)
        linear_logit = torch.stack(
            [
                self.linear_terms[key](field).squeeze(1)
                for key, field in zip(self.field_keys, feature_fields, strict=False)
            ],
            dim=0,
        ).sum(dim=0)
        return deep_logit + linear_logit


def train_nfm(
    train_feature_fields: Sequence[np.ndarray],
    train_labels: np.ndarray,
    test_feature_fields: Sequence[np.ndarray],
    test_labels: np.ndarray,
    field_embedding_dim: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden_units: tuple[int, ...],
    patience: int,
    field_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, float, pd.DataFrame, float, np.ndarray]:
    torch.manual_seed(seed)
    field_dims = tuple(field.shape[1] for field in train_feature_fields)
    if field_names is None:
        field_names = [f"field_{index}" for index in range(len(field_dims))]

    print("NFM fields:")
    for name, dim in zip(field_names, field_dims, strict=False):
        print(f"  {name}: input_dim={dim} -> projected_dim={field_embedding_dim}")

    train_dataset = FieldFeatureDataset(train_feature_fields, train_labels)
    test_dataset = FieldFeatureDataset(test_feature_fields)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TorchNFM(
        field_dims=field_dims,
        field_embedding_dim=field_embedding_dim,
        field_names=field_names,
        hidden_units=hidden_units,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    min_delta = 0.0001

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            fields = [field.to(device) for field in batch["fields"]]
            label = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(fields)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * label.shape[0]
            total_count += label.shape[0]

        epoch_loss = total_loss / max(total_count, 1)
        print(f"NFM epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"NFM early stopping at epoch {epoch + 1}; best loss: {best_loss:.6f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model([field.to(device) for field in batch["fields"]])
            predictions.append(torch.sigmoid(logits).detach().cpu().numpy())
    pred = np.concatenate(predictions).reshape(-1)
    roc_curve, roc_value = roc_auc(test_labels, pred)
    pr_curve, pr_value = pr_auc(test_labels, pred)
    return roc_curve, roc_value, pr_curve, pr_value, pred
