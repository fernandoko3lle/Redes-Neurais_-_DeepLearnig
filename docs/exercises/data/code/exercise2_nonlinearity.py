"""Exercise 2 — Não-linearidade em dimensões maiores.

Gera o Dataset I (gaussianas 5D deslocadas, correlacionadas) e o Dataset II
(cascas concêntricas 5D), projeta os dois em 2D via PCA e mede a distância
entre centros e o raio de cada ponto. Salva as Figuras 4-5 em ``figures/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise2_nonlinearity.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)

N_PER_CLASS = 500

# --- Dataset I: gaussianas multivariadas deslocadas ------------------------
MU_A = np.zeros(5)
COV_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)
MU_B = np.full(5, 1.5)
COV_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)

# --- Dataset II: cascas concêntricas ----------------------------------------
# O enunciado dá o raio como rho ~ N(mu, 0.4); interpretamos 0.4 como o
# desvio padrão (não a variância) do raio de cada casca.
RADIUS_STD = 0.4
RADIUS_CORE = 2.0
RADIUS_SHELL = 5.0


def make_dataset_i() -> tuple[np.ndarray, np.ndarray]:
    a = RNG.multivariate_normal(MU_A, COV_A, size=N_PER_CLASS)
    b = RNG.multivariate_normal(MU_B, COV_B, size=N_PER_CLASS)
    X = np.vstack([a, b])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


def random_unit_vectors(n: int, dim: int = 5) -> np.ndarray:
    v = RNG.normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_dataset_ii() -> tuple[np.ndarray, np.ndarray]:
    u_core = random_unit_vectors(N_PER_CLASS)
    u_shell = random_unit_vectors(N_PER_CLASS)
    r_core = RNG.normal(RADIUS_CORE, RADIUS_STD, size=N_PER_CLASS)
    r_shell = RNG.normal(RADIUS_SHELL, RADIUS_STD, size=N_PER_CLASS)
    core = r_core[:, None] * u_core
    shell = r_shell[:, None] * u_shell
    X = np.vstack([core, shell])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


def center_distance(X: np.ndarray, y: np.ndarray) -> float:
    mean0 = X[y == 0].mean(axis=0)
    mean1 = X[y == 1].mean(axis=0)
    return float(np.linalg.norm(mean0 - mean1))


def plot_figure4(datasets: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    explained = {}
    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(X)
        for c, label in ((0, "Classe 0"), (1, "Classe 1")):
            ax.scatter(*proj[y == c].T, s=12, alpha=0.7, label=label)
        ax.set_title(name)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        explained[name] = pca.explained_variance_ratio_
    fig.suptitle("Figura 4 — Projeção PCA (2D) dos dois datasets 5D")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-pca-projection.png", dpi=150)
    plt.close(fig)
    return explained


def plot_figure5(datasets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        radius = np.linalg.norm(X, axis=1)
        for c, label in ((0, "Classe 0"), (1, "Classe 1")):
            ax.hist(radius[y == c], bins=25, alpha=0.6, label=label)
        ax.set_title(name)
        ax.set_xlabel(r"$\|x\|$")
        ax.set_ylabel("Contagem")
        ax.legend()
    fig.suptitle("Figura 5 — Histograma do raio por classe")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-radius-histogram.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Dataset I (gaussianas)": make_dataset_i(),
        "Dataset II (cascas)": make_dataset_ii(),
    }

    explained = plot_figure4(datasets)
    plot_figure5(datasets)

    for name, (X, y) in datasets.items():
        dist = center_distance(X, y)
        var2 = explained[name][:2].sum()
        print(f"{name}:")
        print(f"  distância entre centros (5D) = {dist:.4f}")
        print(f"  variância explicada PC1+PC2  = {var2:.4f} ({var2 * 100:.2f}%)")


if __name__ == "__main__":
    main()
