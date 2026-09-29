"""Exercise 2 — Perceptron em dados sobrepostos (algoritmo pocket).

Reutiliza a classe ``Perceptron`` do Exercise 1 sem alterações. Como os dados
não são separáveis, o laço nunca para de atualizar — este script acompanha,
além dos pesos finais, os melhores pesos vistos durante o treino (pocket).

Uso (a partir da raiz do repositório):

    python docs/exercises/perceptron/code/exercise2_pocket.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exercise1_separable import ETA, MAX_EPOCHS, Perceptron  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
RNG = np.random.default_rng(42)

N_PER_CLASS = 1000
MEAN_0 = np.array([3.0, 3.0])
COV_0 = np.array([[1.5, 0.0], [0.0, 1.5]])
MEAN_1 = np.array([4.0, 4.0])
COV_1 = np.array([[1.5, 0.0], [0.0, 1.5]])


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    class0 = RNG.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
    class1 = RNG.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])
    return X, y


def train_with_pocket(model: Perceptron, X: np.ndarray, y: np.ndarray, max_epochs: int = MAX_EPOCHS):
    """Mesmo laço de atualização do Exercise 1; só acrescenta a cópia do 'bolso'."""
    current_accuracies, pocket_accuracies = [], []
    best_acc = -1.0
    pocket_w, pocket_b = model.w.copy(), model.b
    pocket_epoch = 0

    for epoch in range(1, max_epochs + 1):
        for xi, yi in zip(X, y):
            changed = model.update(xi, yi)
            if changed:
                acc = float((model.predict(X) == y).mean())
                if acc > best_acc:
                    best_acc = acc
                    pocket_w, pocket_b = model.w.copy(), model.b
                    pocket_epoch = epoch
        current_accuracies.append(float((model.predict(X) == y).mean()))
        pocket_accuracies.append(best_acc)

    return pocket_w, pocket_b, pocket_epoch, current_accuracies, pocket_accuracies


def plot_figure4(X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[y == 0].T, s=10, alpha=0.6, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=10, alpha=0.6, label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 4 — Dados sobrepostos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-overlapping-data.png", dpi=150)
    plt.close(fig)


def plot_figure5(model_final: Perceptron, model_pocket: Perceptron, X: np.ndarray, y: np.ndarray) -> None:
    wrong_pocket = model_pocket.predict(X) != y

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[y == 0].T, s=10, alpha=0.5, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=10, alpha=0.5, label="Classe 1")
    ax.scatter(*X[wrong_pocket].T, s=40, facecolors="none", edgecolors="red", linewidths=1.2,
               label="Mal classificado (pocket)")

    x1 = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    x2_final = -(model_final.w[0] * x1 + model_final.b) / model_final.w[1]
    x2_pocket = -(model_pocket.w[0] * x1 + model_pocket.b) / model_pocket.w[1]
    ax.plot(x1, x2_final, "k:", linewidth=1.5, label="Fronteira final")
    ax.plot(x1, x2_pocket, "k--", linewidth=1.5, label="Fronteira pocket")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 5 — Fronteiras final vs. pocket")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-overlapping-boundaries.png", dpi=150)
    plt.close(fig)


def plot_figure6(current_accuracies: list[float], pocket_accuracies: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(current_accuracies) + 1)
    ax.plot(epochs, current_accuracies, label="Pesos atuais")
    ax.plot(epochs, pocket_accuracies, label="Melhor até agora (pocket)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Figura 6 — Acurácia por época (dados sobrepostos)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-overlapping-accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    X, y = generate_data()
    plot_figure4(X, y)

    model = Perceptron(n_features=2, rng=RNG, eta=ETA)
    pocket_w, pocket_b, pocket_epoch, current_acc, pocket_acc = train_with_pocket(model, X, y)

    model_pocket = Perceptron(n_features=2, eta=ETA, w_init=pocket_w)
    model_pocket.b = pocket_b

    plot_figure5(model, model_pocket, X, y)
    plot_figure6(current_acc, pocket_acc)

    print(f"Pesos finais:  w = {model.w}, b = {model.b:.4f}, acurácia = {current_acc[-1]:.4f} "
          f"({current_acc[-1] * 100:.2f}%)")
    print(f"Pesos pocket:  w = {pocket_w}, b = {pocket_b:.4f}, acurácia = {pocket_acc[-1]:.4f} "
          f"({pocket_acc[-1] * 100:.2f}%)")
    print(f"Época em que o pocket foi atualizado pela última vez: {pocket_epoch}")


if __name__ == "__main__":
    main()
