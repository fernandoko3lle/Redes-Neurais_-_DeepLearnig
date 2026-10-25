"""Exercise 3 — Multiclasse, a mesma rede.

Estende o Exercise 2 para 3 classes: reutiliza ``mlp.py`` sem alterações
(só muda o tamanho da saída e o ``task``, que já eram argumentos). Salva as
Figuras 4-5 em ``figures/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/mlp/code/exercise3_multiclass.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlp import MLP  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
RNG = np.random.default_rng(42)

ETA = 0.3
EPOCHS = 3000
HIDDEN = [16]


def generate_data():
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
        n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def plot_figure4(losses: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (entropia cruzada) — treino")
    ax.set_title("Figura 4 — Perda de treino por época (Exercise 3)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-multiclass-loss.png", dpi=150)
    plt.close(fig)


def plot_figure5(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    ax.set_title("Figura 5 — Matriz de confusão (teste)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-multiclass-confusion.png", dpi=150)
    plt.close(fig)
    return cm


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = generate_data()

    baseline = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    print(f"Linha de base (regressão logística): acurácia no teste = {baseline_acc:.4f} "
          f"({baseline_acc * 100:.2f}%)")

    model = MLP(layer_sizes=[4, *HIDDEN, 3], task="multiclass", rng=RNG, eta=ETA)
    losses = model.fit(X_train, y_train, epochs=EPOCHS)
    plot_figure4(losses)

    y_pred = model.predict(X_test)
    test_acc = (y_pred == y_test).mean()
    cm = plot_figure5(y_test, y_pred)

    print(f"\nMLP — arquitetura: {model.layer_sizes}, eta={ETA}, épocas={EPOCHS}")
    print(f"Perda final de treino: {losses[-1]:.4f}")
    print(f"Acurácia no teste: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Matriz de confusão (linhas=verdadeiro, colunas=predito):\n{cm}")

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    i, j = np.unravel_index(off_diag.argmax(), off_diag.shape)
    print(f"Par mais confundido: verdadeiro={i} predito={j} ({off_diag[i, j]} casos)")


if __name__ == "__main__":
    main()
