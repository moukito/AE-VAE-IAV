from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, 128 * 8 * 8)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(self.projection(z).view(-1, 128, 8, 8))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim * 2)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.encoder(x)
        return stats.chunk(2, dim=1)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decoder(self.sample(mu, logvar)), mu, logvar
