"""VAE — arquitetura e perda do enunciado, reutilizado sem alterações nos dois exercícios.

Encoder 784->512->256 (ReLU), duas cabeças lineares 256->d (mu, log-variância),
decoder d->256->512->784 (ReLU nas ocultas, sigmoide na saída). Reparametrização,
BCE de reconstrução e KL em forma fechada escritas à mão (sem
torch.distributions.kl_divergence e sem classe de VAE de biblioteca).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.enc1 = nn.Linear(784, 512)
        self.enc2 = nn.Linear(512, 256)
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)
        self.dec1 = nn.Linear(d, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, 784)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I) — o truque que deixa a amostragem diferenciável."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.dec3(h))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstrução (BCE, nats/imagem) e KL (forma fechada, nats/imagem), separados."""
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / x.size(0)
    return recon, kl
