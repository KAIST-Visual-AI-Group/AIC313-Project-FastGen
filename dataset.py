"""Dataset and dataloader API for Pokemon Generation One (22k)."""

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from download_dataset import download_dataset
DATASET_NAME = "pokemon-generation-one-22k"
DEFAULT_DATA_ROOT = Path("./data/pokemon-generation-one-22k")
DEFAULT_SPLIT_DIR = Path("./data/pokemon-generation-one-22k")
CATEGORY_MAPPING_FILE = "category_to_id.json"


def category_from_filename(path: str | Path) -> str:
    """Return the category directory from an original Kaggle image path."""
    path = Path(path)
    if path.parent.name not in {"", "PokemonData"}:
        return path.parent.name
    return path.stem.rsplit("-", 1)[0]


def read_split(split_file: str | Path) -> list[str]:
    """Read a newline-separated split manifest."""
    return [
        line.strip()
        for line in Path(split_file).read_text().splitlines()
        if line.strip()
    ]


def build_category_mapping(*split_files: str | Path) -> dict[str, int]:
    categories = {
        category_from_filename(path)
        for split_file in split_files
        for path in read_split(split_file)
    }
    return {name: index for index, name in enumerate(sorted(categories))}


def load_or_save_category_mapping(split_dir, train_split, val_split):
    """Load a fixed category mapping or create it from the split manifests."""
    split_dir = Path(split_dir)
    mapping_path = split_dir / CATEGORY_MAPPING_FILE
    if mapping_path.exists():
        with mapping_path.open("r") as file:
            category_to_id = json.load(file)
        return {str(name): int(index) for name, index in category_to_id.items()}

    category_to_id = build_category_mapping(
        split_dir / train_split,
        split_dir / val_split,
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w") as file:
        json.dump(category_to_id, file, indent=2, sort_keys=True)
    return category_to_id


class PokemonDataset(Dataset):
    """Pokemon image dataset backed by an explicit split manifest."""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        split_file: str | Path = DEFAULT_SPLIT_DIR / "train_split.txt",
        transform: Optional[Callable] = None,
        return_category: bool = False,
        category_to_id: Optional[dict[str, int]] = None,
    ):
        self.data_root = Path(data_root)
        download_dataset(self.data_root)
        self.paths = read_split(split_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.return_category = return_category
        self.category_to_id = category_to_id or {
            name: index for index, name in enumerate(
                sorted({category_from_filename(path) for path in self.paths})
            )
        }
        self.categories = [category_from_filename(path) for path in self.paths]
        if return_category:
            unknown = set(self.categories) - set(self.category_to_id)
            if unknown:
                raise ValueError(f"Unknown categories: {sorted(unknown)}")
            self.category_ids = [self.category_to_id[name] for name in self.categories]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.data_root / self.paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        if self.return_category:
            return image, torch.tensor(self.category_ids[index], dtype=torch.long)
        return image


class PokemonDataModule:
    """Small explicit API for train/validation datasets and dataloaders."""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        split_dir: str | Path = DEFAULT_SPLIT_DIR,
        batch_size: int = 32,
        num_workers: int = 4,
        return_category: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.split_dir = Path(split_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_to_id = load_or_save_category_mapping(
            self.split_dir,
            "train_split.txt",
            "val_split.txt",
        )
        self.train_dataset = PokemonDataset(
            self.data_root,
            self.split_dir / "train_split.txt",
            transform=transform,
            return_category=return_category,
            category_to_id=self.category_to_id,
        )
        self.val_dataset = PokemonDataset(
            self.data_root,
            self.split_dir / "val_split.txt",
            transform=transform,
            return_category=return_category,
            category_to_id=self.category_to_id,
        )

    def train_dataloader(
        self, sampler: Optional[Sampler] = None, shuffle: Optional[bool] = None
    ) -> DataLoader:
        """Return the training loader; pass a sampler for distributed training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None) if shuffle is None else shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
