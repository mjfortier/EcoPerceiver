import argparse
import csv
from pathlib import Path
from utils import resolve_checkpoint_path, resolve_config_path, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run ERA5 inference and save predictions to CSV.")
    parser.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help="Run directory.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to config YAML (default: <run_path>/../config.yml).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to checkpoint file (default: run_path/last.pth, else latest checkpoint-*.pth).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <run_path>/eval/era5_predictions.csv).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: use entire dataset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override (default: value from config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of dataloader workers for inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from ecoperceiver.dataset import EcoPerceiverLoaderConfig
    from ecoperceiver.era5_dataset import ERA5Dataset
    from ecoperceiver.components import EcoPerceiverConfig
    from ecoperceiver.model import EcoPerceiver

    repo_root = Path(__file__).resolve().parent.parent
    run_path = args.run_path.resolve()
    config_path = resolve_config_path(run_path, args.config_path)
    checkpoint_path = resolve_checkpoint_path(run_path, args.checkpoint_path)
    data_path = (repo_root / "experiments/data").resolve()
    output_csv_path = (
        (run_path / "eval" / "era5_predictions.csv")
        if args.output_csv is None
        else args.output_csv.resolve()
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(f"Model targets: {config['model']['targets']}")
    print(f"Context length: {config['model']['context_length']}")
    print(f"Latent space dim: {config['model']['latent_space_dim']}")
    print(f"Run path: {run_path}")
    print(f"Config path: {config_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Data path: {data_path}")

    model_config = EcoPerceiverConfig(**config["model"])
    relative_pretrained_path = repo_root / "ecoperceiver" / "resnet18_weights.pth"
    model = EcoPerceiver(model_config, relative_pretrained_path)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    device = resolve_device("auto")
    model = model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to evaluation mode")

    dataset_config = EcoPerceiverLoaderConfig(**config["dataset"])
    dataset = ERA5Dataset(data_path, config=dataset_config)
    if args.max_samples is None:
        max_samples = len(dataset)
    else:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer when provided.")
        max_samples = min(args.max_samples, len(dataset))
    dataset_for_loader = Subset(dataset, range(max_samples)) if max_samples < len(dataset) else dataset

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer when provided.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    dataloader_batch_size = args.batch_size if args.batch_size is not None else config["dataloader"]["batch_size"]
    dataloader = DataLoader(
        dataset_for_loader,
        batch_size=dataloader_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=config["dataloader"]["pin_memory"],
        collate_fn=dataset.collate_fn,
    )

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Using {max_samples} samples for inference")
    print(f"Batch size: {dataloader_batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Output CSV: {output_csv_path}")

    print("Testing model inference...")
    rows_written = 0
    batches_processed = 0
    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = None
        include_ground_truth = False
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference", unit="batch"):
                batches_processed += 1
                if hasattr(batch, "to"):
                    batch = batch.to(device)

                res = model(batch)
                yhat = res.predictions

                if writer is None:
                    fieldnames = ["lat", "lon", "igbp", "timestamp"] + [f"pred_{flux}" for flux in res.flux_labels]
                    include_ground_truth = res.ground_truth is not None
                    if include_ground_truth:
                        fieldnames += [f"gt_{flux}" for flux in res.flux_labels]
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    print(f"Pred shape per batch: {yhat.shape}")

                preds_cpu = yhat.detach().cpu()
                gt_cpu = res.ground_truth.detach().cpu() if include_ground_truth else None
                lat_idx = batch.aux_columns.index("lat") if "lat" in batch.aux_columns else None
                lon_idx = batch.aux_columns.index("lon") if "lon" in batch.aux_columns else None
                aux_cpu = batch.aux_values.detach().cpu()

                for i in range(preds_cpu.shape[0]):
                    ts = batch.timestamps[i][-1] if len(batch.timestamps[i]) > 0 else ""
                    lat_val = float(aux_cpu[i, lat_idx].item() * 180.0) if lat_idx is not None else float("nan")
                    lon_val = float(aux_cpu[i, lon_idx].item() * 180.0) if lon_idx is not None else float("nan")
                    row = {
                        "igbp": batch.igbp[i],
                        "timestamp": ts,
                        "lat": f"{lat_val:.2f}",
                        "lon": f"{lon_val:.2f}",
                    }
                    for j, flux in enumerate(res.flux_labels):
                        row[f"pred_{flux}"] = f"{preds_cpu[i, j].item():.4f}"
                        if include_ground_truth:
                            row[f"gt_{flux}"] = f"{gt_cpu[i, j].item():.4f}"
                    writer.writerow(row)
                    rows_written += 1

    print(f"Saved predictions for {rows_written} samples across {batches_processed} batches to: {output_csv_path}")


if __name__ == "__main__":
    main()
