"""Download the original Pokemon Generation One dataset from Kaggle."""

import argparse
import shutil
from pathlib import Path


KAGGLE_DATASET = "bhawks/pokemon-generation-one-22k"


def download_dataset(data_root="./data/pokemon-generation-one-22k", dataset=KAGGLE_DATASET):
    """Download and materialize the dataset under ``data_root``."""
    data_root = Path(data_root)
    image_root = data_root / "PokemonData"
    if image_root.is_dir() and any(image_root.iterdir()):
        print(f"Dataset already exists: {image_root}")
        return data_root

    try:
        import kagglehub
    except ImportError as error:
        raise ImportError(
            "kagglehub is required to download the dataset. "
            "Install it with: pip install kagglehub"
        ) from error

    print(f"Downloading Kaggle dataset: {dataset}")
    downloaded_path = Path(kagglehub.dataset_download(dataset))
    source_root = downloaded_path / "PokemonData"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"PokemonData directory was not found under {downloaded_path}"
        )

    data_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, image_root, dirs_exist_ok=True)
    print(f"Dataset ready: {image_root}")
    return data_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Pokemon dataset")
    parser.add_argument(
        "--data_root",
        default="./data/pokemon-generation-one-22k",
        help="Directory containing the PokemonData directory",
    )
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    args = parser.parse_args()
    download_dataset(args.data_root, args.dataset)
