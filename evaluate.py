"""Generate samples and measure FID for one selected NFE-specific model."""

import argparse
from pathlib import Path

import torch
from cleanfid import fid
from tqdm import tqdm

from dataset import PokemonDataModule
from model import Model

MAX_MODEL_PARAMETERS = 100_000_000
SAMPLES_PER_CATEGORY = 20


def prepare_reference_set(data_module, reference_dir, max_images=None):
    """Export validation images as RGB PNGs for FID evaluation."""
    from torchvision.transforms.functional import to_pil_image

    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in data_module.val_dataloader():
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        for image in images:
            if max_images is not None and count >= max_images:
                return reference_dir
            image = ((image.float().cpu() + 1) / 2).clamp(0, 1)
            to_pil_image(image).convert("RGB").save(
                reference_dir / f"{count:04d}.png"
            )
            count += 1
    return reference_dir


def compute_fid(generated_dir, reference_dir, device="cuda", batch_size=64):
    """Compute clean-fid for two image directories."""
    return float(
        fid.compute_fid(
            str(generated_dir),
            str(reference_dir),
            device=device,
            batch_size=batch_size,
            num_workers=4,
        )
    )


def generate_samples(
    model,
    output_dir,
    categories,
    batch_size=32,
    device="cuda",
):
    """Generate normalized 64x64 RGB PNG images with category-aware names."""
    from torchvision.transforms.functional import to_pil_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = torch.device(device)
    num_samples = len(categories)
    image_counts = {}

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size), desc=str(output_dir)):
            count = min(batch_size, num_samples - start)
            batch_categories = categories[start : start + count].to(device)
            samples = model.sample(
                (count, 3, 64, 64),
                device=device,
                category=batch_categories,
            )
            for sample, category in zip(samples, batch_categories.tolist()):
                image_idx = image_counts.get(category, 0)
                image_counts[category] = image_idx + 1
                image = ((sample.float().cpu() + 1) / 2).clamp(0, 1)
                to_pil_image(image).convert("RGB").save(
                    output_dir / f"{category:0>4}_{image_idx:0>2}.png"
                )
    return output_dir


def evaluate(model, reference_dir, output_dir, categories, batch_size=32, device="cuda"):
    """Generate samples for one model and return its FID score."""
    generated_dir = generate_samples(
        model,
        output_dir,
        categories,
        batch_size=batch_size,
        device=device,
    )
    return compute_fid(
        generated_dir,
        reference_dir,
        device=device,
        batch_size=batch_size,
    )


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"Loading {args.evaluate_mode} model checkpoint "
        f"from: {args.model_checkpoint}"
    )
    model = Model.load_checkpoint(
        args.model_checkpoint,
        evaluate_mode=args.evaluate_mode,
        device=device,
    )
    parameter_count = model.count_parameters()
    print(f"Model parameters: {parameter_count:,}")
    if parameter_count > MAX_MODEL_PARAMETERS:
        print(
            f"WARNING: model has more than {MAX_MODEL_PARAMETERS:,} parameters. "
            "Evaluation stopped."
        )
        raise SystemExit(1)

    data_module = PokemonDataModule(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    reference_dir = Path(args.reference_dir)
    if not any(reference_dir.glob("*.png")):
        prepare_reference_set(data_module, reference_dir)

    categories = torch.arange(
        len(data_module.category_to_id), dtype=torch.long
    ).repeat_interleave(SAMPLES_PER_CATEGORY)
    print(
        f"Generating {SAMPLES_PER_CATEGORY} samples per category "
        f"for {len(data_module.category_to_id)} categories "
        f"({len(categories)} total samples)"
    )
    output_dir = Path(args.output_dir) / args.evaluate_mode
    score = evaluate(
        model,
        reference_dir,
        output_dir,
        categories=categories,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"{args.evaluate_mode} FID: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one student model")
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument(
        "--evaluate_mode",
        required=True,
        choices=("one_nfe", "few_nfe"),
    )
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--reference_dir", default="./results/reference")
    parser.add_argument("--output_dir", default="./results/evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
