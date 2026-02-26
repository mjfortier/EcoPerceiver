from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader
from utils import (
    resolve_checkpoint_path,
    resolve_config_path,
    resolve_data_dir,
    resolve_device,
)

mp.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EcoPerceiver inference on test sites.")
    parser.add_argument(
        "--run-path",
        required=True,
        type=Path,
        help="Run folder path (typically .../seed_<n>) where checkpoints are stored.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments/data/carbonsense_v2"),
        help="Path to extracted dataset directory (must contain carbonsense_v2.sql).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional explicit config.yml path. If omitted, resolves from run folder.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint filename inside run folder (e.g. checkpoint-9.pth). If omitted, auto-selects best checkpoint.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="test_sites_metrics.csv",
        help="Output CSV filename inside <run_folder>/eval.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional eval batch size override. Defaults to config dataloader batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of dataloader workers for inference.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin_memory in inference dataloader.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. 'auto' picks cuda if available.",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Optional explicit list of sites to evaluate. Defaults to config test_sites.",
    )
    return parser.parse_args()


def compute_rmse_and_nse(
    sum_sq_error: np.ndarray,
    count_valid: np.ndarray,
    sum_obs: np.ndarray,
    sum_obs_sq: np.ndarray,
) -> tuple[list[float], list[float]]:
    rmse = []
    nse = []
    for j in range(len(count_valid)):
        if count_valid[j] > 0:
            rmse_j = np.sqrt(sum_sq_error[j] / count_valid[j])
        else:
            rmse_j = float("nan")
        rmse.append(rmse_j)

        if count_valid[j] > 1:
            denom = sum_obs_sq[j] - (sum_obs[j] ** 2) / count_valid[j]
            if denom > 0:
                nse_j = 1.0 - (sum_sq_error[j] / denom)
            else:
                nse_j = float("nan")
        else:
            nse_j = float("nan")
        nse.append(nse_j)

    return rmse, nse


def evaluate_site(
    model: EcoPerceiver,
    device: torch.device,
    dataset: EcoPerceiverDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    num_targets: int,
) -> dict:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
    )

    site_total_loss = 0.0
    site_num_batches = 0
    site_num_samples = 0
    site_sum_sq_error = np.zeros(num_targets, dtype=np.float64)
    site_count_valid = np.zeros(num_targets, dtype=np.int64)
    site_sum_obs = np.zeros(num_targets, dtype=np.float64)
    site_sum_obs_sq = np.zeros(num_targets, dtype=np.float64)
    site_igbp = None

    with torch.no_grad():
        for batch in dataloader:
            if hasattr(batch, "to"):
                batch = batch.to(device)

            if site_igbp is None:
                if hasattr(batch, "igbp") and len(batch.igbp) > 0:
                    unique_igbp = set(batch.igbp)
                    site_igbp = next(iter(unique_igbp))
                    if len(unique_igbp) > 1:
                        print(f"  Warning: multiple IGBP values found: {unique_igbp}")
                else:
                    site_igbp = "unknown"

            res = model(batch)
            yhat = res.predictions
            y = res.ground_truth
            loss = res.loss

            batch_samples = yhat.shape[0]
            site_num_samples += batch_samples
            site_num_batches += 1

            batch_loss = loss.mean().item() if isinstance(loss, torch.Tensor) else float(loss)
            # Keep loss sample-weighted to avoid bias from smaller last batches.
            site_total_loss += batch_loss * batch_samples

            yhat_np = yhat.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            mask = np.isfinite(y_np) & np.isfinite(yhat_np)
            err_sq = (yhat_np - y_np) ** 2

            site_sum_sq_error += np.where(mask, err_sq, 0.0).sum(axis=0)
            site_count_valid += mask.sum(axis=0)
            y_obs_valid = np.where(mask, y_np, 0.0)
            site_sum_obs += y_obs_valid.sum(axis=0)
            site_sum_obs_sq += (y_obs_valid**2).sum(axis=0)

    return {
        "igbp": site_igbp or "unknown",
        "total_loss": site_total_loss,
        "num_batches": site_num_batches,
        "num_samples": site_num_samples,
        "sum_sq_error": site_sum_sq_error,
        "count_valid": site_count_valid,
        "sum_obs": site_sum_obs,
        "sum_obs_sq": site_sum_obs_sq,
    }


def main() -> None:
    args = parse_args()
    from ecoperceiver.components import EcoPerceiverConfig
    from ecoperceiver.dataset import EcoPerceiverDataset, EcoPerceiverLoaderConfig
    from ecoperceiver.model import EcoPerceiver

    run_folder = args.run_path.expanduser().resolve()
    if not run_folder.exists():
        raise FileNotFoundError(f"Run folder does not exist: {run_folder}")

    config_path = resolve_config_path(run_folder, args.config_path)
    explicit_checkpoint_path = None
    if args.checkpoint_path is not None:
        checkpoint_name = Path(args.checkpoint_path)
        if checkpoint_name.is_absolute() or checkpoint_name.name != args.checkpoint_path:
            raise ValueError(
                "checkpoint-path must be a filename only (e.g. checkpoint-9.pth), not a full path."
            )
        explicit_checkpoint_path = run_folder / checkpoint_name
    checkpoint_path = resolve_checkpoint_path(run_folder, explicit_checkpoint_path)
    results_dir = run_folder / "eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = results_dir / args.output_name

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolved_data_dir = resolve_data_dir(args.data_dir)

    test_sites = args.sites if args.sites is not None else config.get("test_sites", [])
    if not test_sites:
        raise ValueError("No test sites provided (config test_sites is empty and --sites not set).")

    model_config = EcoPerceiverConfig(**config["model"])
    model = EcoPerceiver(model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    device = resolve_device(args.device)
    model = model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to eval mode")

    dataset_config = EcoPerceiverLoaderConfig(**config["dataset"])
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer when provided.")
    eval_batch_size = (
        args.batch_size if args.batch_size is not None else config.get("dataloader", {}).get("batch_size", 512)
    )
    if eval_batch_size <= 0:
        raise ValueError(
            "Resolved batch size must be positive. "
            "Provide --batch-size or fix dataloader.batch_size in config."
        )
    eval_num_workers = args.num_workers
    eval_pin_memory = args.pin_memory

    flux_labels = list(config["model"]["targets"])
    num_targets = len(flux_labels)
    fieldnames = (
        ["site", "igbp", "num_batches", "num_samples", "avg_loss"]
        + [f"rmse_{name}" for name in flux_labels]
        + [f"nse_{name}" for name in flux_labels]
    )

    print("Configuration:")
    print(f"  Run folder: {run_folder}")
    print(f"  Config path: {config_path}")
    print(f"  Checkpoint path: {checkpoint_path}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Resolved data dir: {resolved_data_dir}")
    print(f"  Test sites: {len(test_sites)}")
    print(f"  Batch size: {eval_batch_size}")
    print(f"  Num workers: {eval_num_workers}")
    print(f"  Pin memory: {eval_pin_memory}")
    print(f"  Output CSV: {output_csv_path}")

    global_total_loss = 0.0
    global_total_batches = 0
    global_total_samples = 0
    global_sum_sq_error = np.zeros(num_targets, dtype=np.float64)
    global_count_valid = np.zeros(num_targets, dtype=np.int64)
    global_sum_obs = np.zeros(num_targets, dtype=np.float64)
    global_sum_obs_sq = np.zeros(num_targets, dtype=np.float64)

    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for site in test_sites:
            print(f"\nEvaluating site {site}...")
            dataset = EcoPerceiverDataset(resolved_data_dir, config=dataset_config, sites=[site])

            if len(dataset) == 0:
                print(f"  Warning: site {site} has no samples, skipping.")
                continue

            site_metrics = evaluate_site(
                model=model,
                device=device,
                dataset=dataset,
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
                pin_memory=eval_pin_memory,
                num_targets=num_targets,
            )

            if site_metrics["num_batches"] == 0:
                print(f"  No batches processed for site {site}, skipping.")
                continue

            site_avg_loss = site_metrics["total_loss"] / site_metrics["num_samples"]
            site_rmse, site_nse = compute_rmse_and_nse(
                sum_sq_error=site_metrics["sum_sq_error"],
                count_valid=site_metrics["count_valid"],
                sum_obs=site_metrics["sum_obs"],
                sum_obs_sq=site_metrics["sum_obs_sq"],
            )

            print(f"  Finished site {site}")
            print(f"    Batches: {site_metrics['num_batches']}")
            print(f"    Samples: {site_metrics['num_samples']}")
            print(f"    Avg loss: {site_avg_loss:.4f}")
            for name, val in zip(flux_labels, site_rmse):
                print(f"    RMSE {name}: {val:.4f}" if np.isfinite(val) else f"    RMSE {name}: nan")
            for name, val in zip(flux_labels, site_nse):
                print(f"    NSE {name}: {val:.4f}" if np.isfinite(val) else f"    NSE {name}: nan")

            row = {
                "site": site,
                "igbp": site_metrics["igbp"],
                "num_batches": site_metrics["num_batches"],
                "num_samples": site_metrics["num_samples"],
                "avg_loss": round(site_avg_loss, 4),
            }
            for name, val in zip(flux_labels, site_rmse):
                row[f"rmse_{name}"] = round(val, 4) if np.isfinite(val) else float("nan")
            for name, val in zip(flux_labels, site_nse):
                row[f"nse_{name}"] = round(val, 4) if np.isfinite(val) else float("nan")
            writer.writerow(row)

            global_total_loss += site_metrics["total_loss"]
            global_total_batches += site_metrics["num_batches"]
            global_total_samples += site_metrics["num_samples"]
            global_sum_sq_error += site_metrics["sum_sq_error"]
            global_count_valid += site_metrics["count_valid"]
            global_sum_obs += site_metrics["sum_obs"]
            global_sum_obs_sq += site_metrics["sum_obs_sq"]

            del dataset

        global_avg_loss = (
            global_total_loss / global_total_samples if global_total_samples > 0 else float("nan")
        )
        global_rmse, global_nse = compute_rmse_and_nse(
            sum_sq_error=global_sum_sq_error,
            count_valid=global_count_valid,
            sum_obs=global_sum_obs,
            sum_obs_sq=global_sum_obs_sq,
        )

        total_row = {
            "site": "TOTAL",
            "igbp": "",
            "num_batches": global_total_batches,
            "num_samples": global_total_samples,
            "avg_loss": round(global_avg_loss, 4) if np.isfinite(global_avg_loss) else float("nan"),
        }
        for name, val in zip(flux_labels, global_rmse):
            total_row[f"rmse_{name}"] = round(val, 4) if np.isfinite(val) else float("nan")
        for name, val in zip(flux_labels, global_nse):
            total_row[f"nse_{name}"] = round(val, 4) if np.isfinite(val) else float("nan")
        writer.writerow(total_row)

    print(f"\nSaved metrics to {output_csv_path}")
    print("TOTAL:")
    print(f"  Batches: {global_total_batches}")
    print(f"  Samples: {global_total_samples}")
    print(f"  Avg loss: {global_avg_loss:.4f}")
    for name, val in zip(flux_labels, global_rmse):
        print(f"  RMSE {name}: {val:.4f}" if np.isfinite(val) else f"  RMSE {name}: nan")
    for name, val in zip(flux_labels, global_nse):
        print(f"  NSE {name}: {val:.4f}" if np.isfinite(val) else f"  NSE {name}: nan")


if __name__ == "__main__":
    main()
