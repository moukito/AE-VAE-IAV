from __future__ import annotations

import argparse

from training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an autoencoder or VAE on CAMUS images.")
    for field, default in TrainConfig().__dict__.items():
        argument = "--" + field.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(argument, action="store_true", default=default)
        else:
            parser.add_argument(argument, type=type(default), default=default)
    args = parser.parse_args()
    train(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
