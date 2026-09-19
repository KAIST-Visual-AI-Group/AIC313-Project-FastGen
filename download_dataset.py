"""Download the original Pokemon Generation One dataset from Kaggle."""

import argparse
import shutil
from pathlib import Path

from PIL import Image


KAGGLE_DATASET = "bhawks/pokemon-generation-one-22k"
NORMALIZED_MARKER = ".transparency_composited_on_white"


def composite_transparency_on_white(image_path):
    """Rewrite one image in place as RGB with transparency composited on white.

    Returns ``True`` when the file was rewritten.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        if "A" not in source.getbands() and "transparency" not in source.info:
            return False
        rgba = source.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white, rgba).convert("RGB")
        image_format = source.format
    composited.save(image_path, format=image_format)
    return True


def composite_transparency_in_tree(image_root):
    """Composite every transparent image under ``image_root`` onto white."""
    rewritten = 0
    for image_path in sorted(Path(image_root).rglob("*")):
        if not image_path.is_file():
            continue
        try:
            rewritten += composite_transparency_on_white(image_path)
        except OSError:
            continue
    return rewritten


def download_dataset(data_root="./data/pokemon-generation-one-22k", dataset=KAGGLE_DATASET):
    """Download and materialize the dataset under ``data_root``.

    Images keep their original filenames, but transparency is composited onto
    white once, at materialization time, so training, FID references, and
    generated samples all share the same opaque-white background convention.
    """
    data_root = Path(data_root)
    image_root = data_root / "PokemonData"
    marker = data_root / NORMALIZED_MARKER
    if image_root.is_dir() and any(image_root.iterdir()):
        print(f"Dataset already exists: {image_root}")
        if not marker.exists():
            print("Compositing transparency onto white for the existing dataset")
            rewritten = composite_transparency_in_tree(image_root)
            marker.touch()
            print(f"Composited {rewritten} transparent images onto white")
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
    rewritten = composite_transparency_in_tree(image_root)
    marker.touch()
    print(f"Composited {rewritten} transparent images onto white")
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
