from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from data import make_loaders, seed_everything
from models import ConvAutoencoder, ConvVAE


@dataclass
class TrainConfig:
    data_root: str = "datasets/camus_64"
    output_dir: str = "outputs"
    model: str = "ae"
    latent_dim: int = 32
    kl_weight: float = 1e-4
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 12345
    num_workers: int = 0
    device: str = "auto"


def _device(value: str) -> torch.device:
    return torch.device(
        "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def _loss(model: nn.Module, batch: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, ConvVAE):
        reconstruction, mu, logvar = model(batch)
        reconstruction_loss = nn.functional.mse_loss(reconstruction, batch)
        kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
        return reconstruction_loss + kl_weight * kl, reconstruction_loss
    reconstruction = model(batch)
    return nn.functional.mse_loss(reconstruction, batch), nn.functional.mse_loss(reconstruction, batch)


def train(config: TrainConfig) -> dict[str, list[float]]:
    seed_everything(config.seed)
    device = _device(config.device)
    from data import discover_samples, split_by_patient

    splits = split_by_patient(discover_samples(config.data_root), config.seed)
    loaders = make_loaders(splits, config.batch_size, config.num_workers)
    model: nn.Module = ConvVAE(config.latent_dim) if config.model == "vae" else ConvAutoencoder(config.latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {"train_loss": [], "val_loss": [], "val_reconstruction": [], "test_loss": [], "test_reconstruction": []}
    best = float("inf")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    for _ in range(config.epochs):
        model.train()
        train_values = []
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss(model, batch, config.kl_weight)
            loss.backward()
            optimizer.step()
            train_values.append(loss.item())
        model.eval()
        val_values, reconstruction_values = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                loss, reconstruction = _loss(model, batch.to(device), config.kl_weight)
                val_values.append(loss.item())
                reconstruction_values.append(reconstruction.item())
        train_loss, val_loss = sum(train_values) / len(train_values), sum(val_values) / len(val_values)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_reconstruction"].append(sum(reconstruction_values) / len(reconstruction_values))
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "config": asdict(config), "history": history}, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_values, test_reconstruction_values = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            loss, reconstruction = _loss(model, batch.to(device), config.kl_weight)
            test_values.append(loss.item())
            test_reconstruction_values.append(reconstruction.item())
    history["test_loss"] = [sum(test_values) / len(test_values)]
    history["test_reconstruction"] = [sum(test_reconstruction_values) / len(test_reconstruction_values)]
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
