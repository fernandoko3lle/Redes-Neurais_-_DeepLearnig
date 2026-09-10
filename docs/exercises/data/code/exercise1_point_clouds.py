"""Exercise 1 — Nuvens de pontos: geometria e espalhamento em 2D.

Gera as 4 classes gaussianas do enunciado em 4 escalas de espalhamento, salva
as Figuras 1-3 em ``figures/`` e imprime os números que alimentam a tabela
*Results summary* do relatório (razão de separação e taxa de mistura).

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

CLASSES = {
    0: {"mean": np.array([2.0, 3.0]), "std": np.array([0.8, 2.5])},
    1: {"mean": np.array([5.0, 6.0]), "std": np.array([1.2, 1.9])},
    2: {"mean": np.array([8.0, 1.0]), "std": np.array([0.9, 0.9])},
    3: {"mean": np.array([15.0, 4.0]), "std": np.array([0.5, 2.0])},
}
N_PER_CLASS = 100
SCALES = (0.5, 1.0, 2.0, 4.0)


def generate(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Amostra 100 pontos por classe; as médias não mudam, só o espalhamento."""
    xs, ys = [], []
    for label, params in CLASSES.items():
        xs.append(RNG.normal(params["mean"], params["std"] * scale, size=(N_PER_CLASS, 2)))
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def separation_ratios(scale: float) -> dict[tuple[int, int], float]:
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), por par de classes."""
    sigma_bar = {c: (p["std"] * scale).mean() for c, p in CLASSES.items()}
    ratios = {}
    for i, j in combinations(CLASSES, 2):
        dist = np.linalg.norm(CLASSES[i]["mean"] - CLASSES[j]["mean"])
        ratios[(i, j)] = dist / (sigma_bar[i] + sigma_bar[j])
    return ratios


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Fração de pontos cujo centro de classe mais próximo não é o verdadeiro."""
    means = np.stack([CLASSES[c]["mean"] for c in sorted(CLASSES)])
    dists = np.linalg.norm(X[:, None, :] - means[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    return float((nearest != y).mean())


def bisector_segment(p1: np.ndarray, p2: np.ndarray, length: float = 7.0) -> np.ndarray:
    """Segmento da bissetriz perpendicular entre dois centros — esboço de fronteira."""
    midpoint = (p1 + p2) / 2
    direction = p2 - p1
    perp = np.array([-direction[1], direction[0]])
    perp = perp / np.linalg.norm(perp)
    return np.stack([midpoint - perp * length / 2, midpoint + perp * length / 2])


def plot_figure1(X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in CLASSES:
        ax.scatter(*X[y == c].T, s=14, alpha=0.75, label=f"Classe {c}")
    centers = np.stack([CLASSES[c]["mean"] for c in CLASSES])
    ax.scatter(*centers.T, marker="x", s=120, c="black", linewidths=2, label="Centro")

    # Esboço (item C): fronteira aproximada entre pares de classes vizinhas,
    # a bissetriz perpendicular do segmento que une os dois centros.
    for i, j in ((0, 1), (1, 2), (2, 3)):
        seg = bisector_segment(CLASSES[i]["mean"], CLASSES[j]["mean"])
        ax.plot(*seg.T, "k--", linewidth=1.3, alpha=0.8)
    ax.plot([], [], "k--", linewidth=1.3, label="Fronteira aproximada (esboço)")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 1 — Nuvens gaussianas (scale = 1.0)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
    plt.close(fig)  # (2)!


def plot_figure2(datasets: dict[float, tuple[np.ndarray, np.ndarray]]) -> None:
    all_x = np.vstack([X for X, _ in datasets.values()])
    margin = 1.0
    xlim = (all_x[:, 0].min() - margin, all_x[:, 0].max() + margin)
    ylim = (all_x[:, 1].min() - margin, all_x[:, 1].max() + margin)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    for ax, scale in zip(axes, SCALES):
        X, y = datasets[scale]
        for c in CLASSES:
            ax.scatter(*X[y == c].T, s=10, alpha=0.7)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"scale = {scale}")
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    fig.suptitle("Figura 2 — Espalhamento das 4 classes por fator de escala")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-scale-comparison.png", dpi=150)
    plt.close(fig)


def plot_figure3(rates: dict[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(rates.keys()), list(rates.values()), marker="o")
    ax.set_xlabel("Fator de escala $s$")
    ax.set_ylabel("Taxa de mistura")
    ax.set_title("Figura 3 — Taxa de mistura vs. fator de escala")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-mixing-rate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    datasets = {scale: generate(scale) for scale in SCALES}

    plot_figure1(*datasets[1.0])
    plot_figure2(datasets)

    rates = {scale: mixing_rate(*datasets[scale]) for scale in SCALES}
    plot_figure3(rates)

    ratios = separation_ratios(1.0)
    print("Razão de separação r_ij (s = 1.0):")
    for (i, j), r in sorted(ratios.items(), key=lambda item: item[1]):
        print(f"  par ({i}, {j}): r_ij = {r:.3f}")
    min_pair, min_r = min(ratios.items(), key=lambda item: item[1])
    print(f"Menor r_ij: par {min_pair} = {min_r:.3f}")
    print(f"Menor r_ij projetado em s=2.0 (r_ij(1)/2): {min_r / 2:.3f}")

    print("\nTaxa de mistura por escala:")
    for scale, rate in rates.items():
        print(f"  scale={scale:>4}: {rate:.4f} ({rate * 100:.2f}%)")


if __name__ == "__main__":
    main()
