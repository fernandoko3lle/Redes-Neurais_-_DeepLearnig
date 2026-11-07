"""Exercise 1 — Um VAE no MNIST (d = 16).

Carrega e divide o MNIST (50k treino / 10k validação / 10k teste), treina o
VAE de ``vae.py`` por 20 épocas registrando reconstrução e KL separadamente,
e produz as Figuras 1-3.

Uso (a partir da raiz do repositório):

    python docs/exercises/vae/code/exercise1_mnist.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vae import VAE, vae_loss  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
DATA_DIR = HERE / "data"

D_LATENT = 16
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3


def load_data():
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)
    train, val = random_split(train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(42))
    return train, val, test


def run_epoch(model: VAE, loader: DataLoader, optimizer=None) -> tuple[float, float]:
    """Uma passada pelo loader; treina se ``optimizer`` for dado, senão só avalia."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_recon, total_kl, n_batches = 0.0, 0.0, 0
    for x, _ in loader:
        x = x.view(x.size(0), -1)
        if train_mode:
            optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        recon, kl = vae_loss(x, x_hat, mu, logvar)
        if train_mode:
            (recon + kl).backward()
            optimizer.step()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1
    return total_recon / n_batches, total_kl / n_batches


def plot_figure1(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = range(1, EPOCHS + 1)
    axes[0].plot(epochs, history["train_recon"], label="Treino")
    axes[0].plot(epochs, history["val_recon"], label="Validação")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Reconstrução (nats/imagem)")
    axes[0].set_title("Termo de reconstrução")
    axes[0].legend()

    axes[1].plot(epochs, history["train_kl"], label="Treino")
    axes[1].plot(epochs, history["val_kl"], label="Validação")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("KL (nats/imagem)")
    axes[1].set_title("Termo KL")
    axes[1].legend()

    fig.suptitle("Figura 1 — Reconstrução e KL por época (d = 16)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-loss-terms.png", dpi=150)
    plt.close(fig)


def plot_figure2(model: VAE, test) -> None:
    model.eval()
    digit_images = {}
    for x, label in test:
        if label not in digit_images:
            digit_images[label] = x.view(-1)
        if len(digit_images) == 10:
            break
    digit_images = [digit_images[d] for d in range(10)]
    X = torch.stack(digit_images)

    with torch.no_grad():
        mu, _ = model.encode(X)
        recon = model.decode(mu)

    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(X[i].view(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(str(i), fontsize=9)
        axes[1, i].imshow(recon[i].view(28, 28), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstrução", fontsize=9)
    fig.suptitle("Figura 2 — Original (cima) vs. reconstrução a partir de μ (baixo)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-reconstructions.png", dpi=150)
    plt.close(fig)


def plot_figure3(model: VAE) -> None:
    model.eval()
    torch.manual_seed(0)
    z = torch.randn(64, D_LATENT)
    with torch.no_grad():
        samples = model.decode(z)

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(samples[i].view(28, 28), cmap="gray")
        ax.axis("off")
    fig.suptitle("Figura 3 — 64 amostras de z ~ N(0, I) (prior), decodificadas")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-prior-samples.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    train, val, test = load_data()
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(42)
    model = VAE(d=D_LATENT)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}
    for epoch in range(1, EPOCHS + 1):
        tr_recon, tr_kl = run_epoch(model, train_loader, optimizer)
        va_recon, va_kl = run_epoch(model, val_loader, optimizer=None)
        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)
        history["val_recon"].append(va_recon)
        history["val_kl"].append(va_kl)
        print(f"época {epoch:2d} | treino recon={tr_recon:7.3f} kl={tr_kl:6.3f} | "
              f"val recon={va_recon:7.3f} kl={va_kl:6.3f}")

    test_recon, test_kl = run_epoch(model, test_loader, optimizer=None)
    print(f"\nTeste (d={D_LATENT}): reconstrução={test_recon:.4f} nats, KL={test_kl:.4f} nats, "
          f"ELBO negativo={test_recon + test_kl:.4f} nats")
    print(f"KL de validação — primeira época: {history['val_kl'][0]:.4f}, "
          f"última época: {history['val_kl'][-1]:.4f}")

    plot_figure1(history)
    plot_figure2(model, test)
    plot_figure3(model)

    torch.save(model.state_dict(), HERE / "data" / "vae_d16.pt")


if __name__ == "__main__":
    main()
