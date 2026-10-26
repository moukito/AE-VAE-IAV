from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VIEWS = ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES")


@dataclass(frozen=True)
class Sample:
    image: Path
    patient: str
    view: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_samples(data_root: str | Path) -> list[Sample]:
    root = Path(data_root)
    samples = []
    for patient_dir in sorted(root.glob("patient*")):
        for view in VIEWS:
            path = patient_dir / f"{patient_dir.name}_{view}.nii.gz"
            if path.exists():
                samples.append(Sample(path, patient_dir.name, view))
    if not samples:
        raise FileNotFoundError(f"No CAMUS images found under {root}")
    return samples


def split_by_patient(
        samples: list[Sample], seed: int = 12345, train_fraction: float = 0.8, val_fraction: float = 0.1
) -> dict[str, list[Sample]]:
    patients = sorted({sample.patient for sample in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_train = max(1, int(len(patients) * train_fraction))
    n_val = max(1, int(len(patients) * val_fraction))
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train: n_train + n_val])
    test_patients = set(patients[n_train + n_val:])
    if not test_patients:
        raise ValueError("The split leaves no test patients; reduce train/validation fractions.")
    return {
        "train": [s for s in samples if s.patient in train_patients],
        "val": [s for s in samples if s.patient in val_patients],
        "test": [s for s in samples if s.patient in test_patients],
    }


class CamusDataset(Dataset[torch.Tensor]):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        array = np.asarray(nib.load(self.samples[index].image).get_fdata(), dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D image, got {array.shape} from {self.samples[index].image}")
        peak = float(array.max())
        if peak > 0:
            array = array / peak
        return torch.from_numpy(array).unsqueeze(0)


def make_loaders(
        splits: dict[str, list[Sample]], batch_size: int, num_workers: int = 0
) -> dict[str, DataLoader[torch.Tensor]]:
    return {
        name: DataLoader(
            CamusDataset(items),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, items in splits.items()
    }
