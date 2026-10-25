"""Exercise 2 — Classificação binária que uma reta não resolve.

Gera o dataset de 3 blobs (Classe 1 dividida em dois lados opostos da Classe
0), mede a linha de base linear e treina o MLP do zero (``mlp.py``). Salva as
Figuras 1-3 em ``figures/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/mlp/code/exercise2_binary.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlp import MLP  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
RNG = np.random.default_rng(42)

ETA = 0.3
EPOCHS = 2000
HIDDEN = [8]


def generate_data():
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers, cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)  # classe 0: 1 grupo | classe 1: 2 grupos
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def plot_figure1(X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for c in (0, 1):
        ax.scatter(*X[y == c].T, s=14, alpha=0.7, label=f"Classe {c}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 1 — Dataset do Exercise 2 (1000 pontos)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-binary-data.png", dpi=150)
    plt.close(fig)


def plot_figure2(losses: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (BCE) — treino")
    ax.set_title("Figura 2 — Perda de treino por época")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-binary-loss.png", dpi=150)
    plt.close(fig)


def plot_figure3(model: MLP, X_test: np.ndarray, y_test: np.ndarray) -> None:
    pad = 1.0
    x1 = np.linspace(X_test[:, 0].min() - pad, X_test[:, 0].max() + pad, 200)
    x2 = np.linspace(X_test[:, 1].min() - pad, X_test[:, 1].max() + pad, 200)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    pred = model.predict(grid).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx1, xx2, pred, levels=[-0.5, 0.5, 1.5], colors=["#cfe3fb", "#fbdcc6"], alpha=0.8)
    for c in (0, 1):
        ax.scatter(*X_test[y_test == c].T, s=18, edgecolor="k", linewidth=0.3, label=f"Classe {c}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 3 — Fronteira de decisão do MLP (teste)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-binary-boundary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = generate_data()
    plot_figure1(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))

    baseline = LogisticRegression(random_state=42).fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    print(f"Linha de base (regressão logística): acurácia no teste = {baseline_acc:.4f} "
          f"({baseline_acc * 100:.2f}%)")

    model = MLP(layer_sizes=[2, *HIDDEN, 1], task="binary", rng=RNG, eta=ETA)
    losses = model.fit(X_train, y_train, epochs=EPOCHS)
    plot_figure2(losses)
    plot_figure3(model, X_test, y_test)

    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"\nMLP — arquitetura: {model.layer_sizes}, eta={ETA}, épocas={EPOCHS}")
    print(f"Perda final de treino: {losses[-1]:.4f}")
    print(f"Acurácia no teste: {test_acc:.4f} ({test_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
