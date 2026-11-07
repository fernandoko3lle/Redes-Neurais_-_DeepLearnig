"""Exercise 2 — Um espaço latente que dá para ver (d = 2).

Mesmo modelo e orçamento de treino do Exercise 1, só mudando d=2. Produz as
Figuras 4-6 (dispersão do latente, plano decodificado via Phi^-1, e a
interpolação em pixels vs. em espaço latente usando o modelo d=16 salvo pelo
Exercise 1).

Uso (a partir da raiz do repositório, DEPOIS de rodar exercise1_mnist.py):

    python docs/exercises/vae/code/exercise2_latent2d.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vae import VAE, vae_loss  # noqa: E402
from exercise1_mnist import BATCH_SIZE, D_LATENT as D_EX1, DATA_DIR, EPOCHS, LR, load_data, run_epoch  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
D_LATENT = 2


def train_2d_model():
    train, val, test = load_data()
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(42)
    model = VAE(d=D_LATENT)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        tr_recon, tr_kl = run_epoch(model, train_loader, optimizer)
        va_recon, va_kl = run_epoch(model, val_loader, optimizer=None)
        print(f"época {epoch:2d} | treino recon={tr_recon:7.3f} kl={tr_kl:6.3f} | "
              f"val recon={va_recon:7.3f} kl={va_kl:6.3f}")

    test_recon, test_kl = run_epoch(model, test_loader, optimizer=None)
    print(f"\nTeste (d={D_LATENT}): reconstrução={test_recon:.4f} nats, KL={test_kl:.4f} nats")
    return model, test


def plot_figure4(model: VAE, test) -> np.ndarray:
    model.eval()
    loader = DataLoader(test, batch_size=512, shuffle=False)
    all_mu, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            mu, _ = model.encode(x.view(x.size(0), -1))
            all_mu.append(mu)
            all_y.append(y)
    mu = torch.cat(all_mu).numpy()
    y = torch.cat(all_y).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(mu[:, 0], mu[:, 1], c=y, cmap="tab10", s=4, alpha=0.6)
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("Figura 4 — Espaço latente 2D do conjunto de teste (10.000 imagens)")
    legend = ax.legend(*scatter.legend_elements(), title="Dígito", loc="center left",
                        bbox_to_anchor=(1.0, 0.5))
    fig.add_artist(legend)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-latent-scatter.png", dpi=150)
    plt.close(fig)
    return mu


def plot_figure5(model: VAE) -> None:
    model.eval()
    q = np.linspace(0.05, 0.95, 15)
    grid_1d = norm.ppf(q)  # z = Phi^-1(q): mesma massa do prior em cada célula

    fig, axes = plt.subplots(15, 15, figsize=(9, 9))
    with torch.no_grad():
        for i, zi in enumerate(grid_1d):           # linha = eixo z2 (de cima para baixo)
            for j, zj in enumerate(grid_1d):        # coluna = eixo z1
                z = torch.tensor([[zj, zi]], dtype=torch.float32)
                img = model.decode(z).view(28, 28)
                axes[i, j].imshow(img, cmap="gray")
                axes[i, j].axis("off")
    fig.suptitle("Figura 5 — Plano latente decodificado (grade 15×15, espaçamento Φ⁻¹)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-latent-plane.png", dpi=150)
    plt.close(fig)


def plot_figure6(model_d16: VAE, test, idx_a: int, idx_b: int) -> None:
    model_d16.eval()
    x_a = test[idx_a][0].view(-1)
    x_b = test[idx_b][0].view(-1)

    with torch.no_grad():
        mu_a, _ = model_d16.encode(x_a.unsqueeze(0))
        mu_b, _ = model_d16.encode(x_b.unsqueeze(0))

    ts = np.linspace(0, 1, 10)
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    with torch.no_grad():
        for i, t in enumerate(ts):
            pixel_interp = (1 - t) * x_a + t * x_b
            axes[0, i].imshow(pixel_interp.view(28, 28), cmap="gray")
            axes[0, i].axis("off")

            z_interp = (1 - t) * mu_a + t * mu_b
            latent_interp = model_d16.decode(z_interp).squeeze(0)
            axes[1, i].imshow(latent_interp.view(28, 28), cmap="gray")
            axes[1, i].axis("off")

    fig.suptitle(f"Figura 6 — Interpolação em pixels (cima) vs. em espaço latente d=16 (baixo), "
                 f"teste[{idx_a}] → teste[{idx_b}]")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-interpolation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    model, test = train_2d_model()
    plot_figure4(model, test)
    plot_figure5(model)

    model_d16 = VAE(d=D_EX1)
    model_d16.load_state_dict(torch.load(DATA_DIR / "vae_d16.pt"))

    labels = np.array([test[i][1] for i in range(len(test))])
    idx_a = int(np.where(labels == 4)[0][0])
    idx_b = int(np.where(labels == 9)[0][0])
    print(f"\nÍndices de teste interpolados: idx_a={idx_a} (dígito {labels[idx_a]}), "
          f"idx_b={idx_b} (dígito {labels[idx_b]})")
    plot_figure6(model_d16, test, idx_a, idx_b)


if __name__ == "__main__":
    main()
