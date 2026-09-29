"""Exercise 1 — Perceptron em dados separáveis.

Gera duas classes gaussianas 2D bem separadas, treina um perceptron escrito do
zero e mede a convergência. A classe ``Perceptron`` daqui é reutilizada, sem
alterações, pelo Exercise 2.

Uso (a partir da raiz do repositório):

    python docs/exercises/perceptron/code/exercise1_separable.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)

N_PER_CLASS = 1000
ETA = 0.01
MAX_EPOCHS = 100

MEAN_0 = np.array([1.5, 1.5])
COV_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
MEAN_1 = np.array([5.0, 5.0])
COV_1 = np.array([[0.5, 0.0], [0.0, 0.5]])


class Perceptron:
    """Perceptron de camada única com regra de atualização dirigida pelo erro."""

    def __init__(
        self,
        n_features: int,
        rng: np.random.Generator | None = None,
        eta: float = ETA,
        w_init: np.ndarray | None = None,
    ):
        # w_init permite repetir o treino com os MESMOS pesos iniciais e só outro eta
        # (item D do Exercise 1) — sem ele, cada instância sortearia pesos diferentes.
        self.w = np.array(w_init, dtype=float) if w_init is not None else rng.normal(0.0, 0.01, size=n_features)  # (1)!
        self.b = 0.0
        self.eta = eta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w + self.b >= 0).astype(int)

    def update(self, xi: np.ndarray, yi: int) -> bool:
        """Atualiza com uma amostra; devolve True se os pesos mudaram."""
        y_hat = int(xi @ self.w + self.b >= 0)
        error = yi - y_hat
        if error == 0:
            return False
        self.w = self.w + self.eta * error * xi
        self.b = self.b + self.eta * error
        return True

    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = MAX_EPOCHS) -> list[float]:
        """Treina até uma época sem atualização, ou max_epochs. Devolve acurácia/época."""
        accuracies = []
        self.updates_per_epoch = []  # guardado para a Análise (item D)
        for _ in range(max_epochs):
            updates = sum(self.update(xi, yi) for xi, yi in zip(X, y))
            self.updates_per_epoch.append(updates)
            accuracies.append(float((self.predict(X) == y).mean()))
            if updates == 0:
                break
        return accuracies


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    class0 = RNG.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
    class1 = RNG.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])
    return X, y


def plot_figure1(X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[y == 0].T, s=10, alpha=0.6, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=10, alpha=0.6, label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 1 — Dados separáveis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-separable-data.png", dpi=150)
    plt.close(fig)


def plot_decision_boundary(ax, model: Perceptron, X: np.ndarray) -> None:
    x1 = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    if abs(model.w[1]) > 1e-12:
        x2 = -(model.w[0] * x1 + model.b) / model.w[1]
        ax.plot(x1, x2, "k--", linewidth=1.5, label="Fronteira")
    else:
        x1_line = -model.b / model.w[0]
        ax.axvline(x1_line, color="k", linestyle="--", linewidth=1.5, label="Fronteira")


def plot_figure2(model: Perceptron, X: np.ndarray, y: np.ndarray) -> None:
    pred = model.predict(X)
    wrong = pred != y

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[y == 0].T, s=10, alpha=0.6, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=10, alpha=0.6, label="Classe 1")
    ax.scatter(*X[wrong].T, s=60, facecolors="none", edgecolors="red", linewidths=1.5,
               label="Mal classificado")
    plot_decision_boundary(ax, model, X)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 2 — Fronteira de decisão (dados separáveis)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-separable-boundary.png", dpi=150)
    plt.close(fig)


def plot_figure3(accuracies: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(accuracies) + 1), accuracies, marker="o", markersize=3)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Figura 3 — Acurácia por época (dados separáveis)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-separable-accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    X, y = generate_data()
    plot_figure1(X, y)

    model = Perceptron(n_features=2, rng=RNG, eta=ETA)
    w_init = model.w.copy()  # guardado para repetir o treino com o mesmo início (item D)
    accuracies = model.fit(X, y)
    plot_figure2(model, X, y)
    plot_figure3(accuracies)

    print(f"eta={ETA}: w final = {model.w}, b final = {model.b:.4f}")
    print(f"épocas até convergir: {len(accuracies)}")
    print(f"acurácia final: {accuracies[-1]:.4f} ({accuracies[-1] * 100:.2f}%)")
    print(f"atualizações por época: {model.updates_per_epoch}")
    direction_001 = model.w / np.linalg.norm(model.w)
    print(f"direção de w (eta=0.01): {direction_001}")

    # Item D: repetir com eta = 1.0 e os MESMOS pesos iniciais — "sem mudar mais nada".
    model_eta1 = Perceptron(n_features=2, eta=1.0, w_init=w_init)
    accuracies_eta1 = model_eta1.fit(X, y)
    direction_eta1 = model_eta1.w / np.linalg.norm(model_eta1.w)
    print(f"\neta=1.0: w final = {model_eta1.w}, b final = {model_eta1.b:.4f}")
    print(f"épocas até convergir: {len(accuracies_eta1)}")
    print(f"acurácia final: {accuracies_eta1[-1]:.4f} ({accuracies_eta1[-1] * 100:.2f}%)")
    print(f"direção de w (eta=1.0): {direction_eta1}")
    print(f"produto escalar entre as duas direções: {direction_001 @ direction_eta1:.6f}")


if __name__ == "__main__":
    main()
