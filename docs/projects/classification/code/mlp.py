"""MLP binária em PyTorch, com parada antecipada — reutilizada em toda a Etapa 4/5.

Saída de 1 logit (BCEWithLogitsLoss, com pos_weight para o desbalanceamento).
Regularização opcional: weight decay (no otimizador) e dropout (na rede).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float = 0.0,
                 batchnorm: bool = False):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logit


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_with_early_stopping(
    model: nn.Module, X_train, y_train, X_val, y_val, *, lr: float, weight_decay: float = 0.0,
    batch_size: int = 64, max_epochs: int = 300, patience: int = 20, pos_weight: float | None = None,
    checkpoint_path: Path | None = None, seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val)

    pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = Xt.shape[0]
    checkpoint_path = checkpoint_path or Path("best.pt")

    best, wait = float("inf"), 0
    history = {"train_loss": [], "val_loss": []}
    rng = np.random.default_rng(seed)

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best:
            best, wait = val_loss, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(torch.load(checkpoint_path))
    history["stopped_epoch"] = epoch + 1
    return history


def predict_proba(model: nn.Module, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        return torch.sigmoid(logits).numpy()
