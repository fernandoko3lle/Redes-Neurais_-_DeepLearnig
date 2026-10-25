"""Exercise 4 — Uma segunda camada oculta ajuda?

Mesmos dados do Exercise 3, mesmo orçamento de treino, mesma semente de
dados. Único eixo que muda: profundidade (1 camada oculta vs. 2). Compara as
duas arquiteturas em >= 3 sementes de inicialização e salva a Figura 6.

Uso (a partir da raiz do repositório):

    python docs/exercises/mlp/code/exercise4_depth.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlp import MLP  # noqa: E402
from exercise3_multiclass import ETA, EPOCHS, HIDDEN as HIDDEN_EX3, generate_data  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
HIDDEN_EX4 = [16, 16]
SEEDS = [42, 43, 44]


def n_params(layer_sizes: list[int]) -> int:
    return sum(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))


def run(layer_sizes: list[int], X_train, y_train, X_test, y_test, seed: int):
    rng = np.random.default_rng(seed)
    model = MLP(layer_sizes=layer_sizes, task="multiclass", rng=rng, eta=ETA)
    losses = model.fit(X_train, y_train, epochs=EPOCHS)
    acc = (model.predict(X_test) == y_test).mean()
    return losses, acc


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = generate_data()

    layers_ex3 = [4, *HIDDEN_EX3, 3]
    layers_ex4 = [4, *HIDDEN_EX4, 3]

    results = {"Exercise 3 (1 camada)": {"layers": layers_ex3, "losses": [], "accs": []},
               "Exercise 4 (2 camadas)": {"layers": layers_ex4, "losses": [], "accs": []}}

    for name, info in results.items():
        for seed in SEEDS:
            losses, acc = run(info["layers"], X_train, y_train, X_test, y_test, seed)
            info["losses"].append(losses)
            info["accs"].append(acc)
            print(f"{name} | seed={seed} | perda final={losses[-1]:.4f} | acurácia teste={acc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, info in results.items():
        ax.plot(range(1, EPOCHS + 1), info["losses"][0], label=f"{name} (seed {SEEDS[0]})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (entropia cruzada) — treino")
    ax.set_title("Figura 6 — Perda de treino: Exercise 3 vs. Exercise 4")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-depth-comparison.png", dpi=150)
    plt.close(fig)

    print("\n=== Resumo (média de", len(SEEDS), "sementes) ===")
    for name, info in results.items():
        mean_loss = np.mean([losses[-1] for losses in info["losses"]])
        mean_acc = np.mean(info["accs"])
        params = n_params(info["layers"])
        print(f"{name}: camadas={info['layers']}, parâmetros={params}, "
              f"perda final média={mean_loss:.4f}, acurácia média={mean_acc:.4f} ({mean_acc*100:.2f}%)")

    acc3 = np.mean(results["Exercise 3 (1 camada)"]["accs"])
    acc4 = np.mean(results["Exercise 4 (2 camadas)"]["accs"])
    print(f"\nDiferença de acurácia (Ex4 - Ex3): {acc4 - acc3:+.4f} ({(acc4 - acc3) * 100:+.2f} p.p.)")


if __name__ == "__main__":
    main()
