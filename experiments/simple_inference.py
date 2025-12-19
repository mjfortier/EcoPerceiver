import csv
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import yaml
import numpy as np

from ecoperceiver.dataset import EcoPerceiverLoaderConfig, EcoPerceiverDataset
from ecoperceiver.components import EcoPerceiverConfig
from ecoperceiver.model import EcoPerceiver

mp.set_sharing_strategy("file_system")

def main():
    # Paths
    config_path = Path(
        "runs/multi_gpu4_newdataset_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/config.yml"
    )
    checkpoint_path = Path(
        "runs/multi_gpu4_newdataset_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0/checkpoint-9.pth"
    )
    data_path = Path("data/carbonsense_v2")

    output_csv_path = Path("eval_test_sites_metrics.csv")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Read test sites from config
    test_sites = config.get("test_sites", [])
    if not test_sites:
        raise ValueError("No test_sites found in config.yml")

    print("Configuration loaded:")
    print(f"  Model targets: {config['model']['targets']}")
    print(f"  Context length: {config['model']['context_length']}")
    print(f"  Latent space dim: {config['model']['latent_space_dim']}")
    print(f"  Test sites: {len(test_sites)} sites")
    print(f"  Checkpoint path: {checkpoint_path}")
    print(f"  Checkpoint exists: {checkpoint_path.exists()}")

    # Build model config
    model_config = EcoPerceiverConfig(
        targets=tuple(config["model"]["targets"]),
        latent_space_dim=config["model"]["latent_space_dim"],
        num_frequencies=config["model"]["num_frequencies"],
        input_embedding_dim=config["model"]["input_embedding_dim"],
        context_length=config["model"]["context_length"],
        obs_dropout=config["model"]["obs_dropout"],
        weight_sharing=config["model"]["weight_sharing"],
        layers=config["model"]["layers"],
        pretrained_path=config["model"]["pretrained_path"],
    )

    model = EcoPerceiver(model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Device (you are on CPU in your logs, so this will stay "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint if available
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Continuing with randomly initialized weights")

    model = model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to evaluation mode")

    # Dataset loader config
    dataset_config = EcoPerceiverLoaderConfig(**config["dataset"])

    # Hard override dataloader params to avoid multiprocessing issues
    dl_cfg = config.get("dataloader", {})
    eval_batch_size = dl_cfg.get("batch_size", 512)
    eval_num_workers = 0      # critical: no worker processes
    eval_pin_memory = False   # not needed on CPU and frees some resources

    flux_labels = list(config["model"]["targets"])
    num_targets = len(flux_labels)

    # CSV setup: one file, one file descriptor for entire run
    fieldnames = [
        "site",
        "num_batches",
        "num_samples",
        "avg_loss",
    ] + [f"rmse_{name}" for name in flux_labels]

    # Global accumulators for TOTAL row
    global_total_loss = 0.0
    global_total_batches = 0
    global_total_samples = 0
    global_sum_sq_error = np.zeros(num_targets, dtype=np.float64)
    global_count_valid = np.zeros(num_targets, dtype=np.int64)

    with output_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for site in test_sites:
            print(f"\nEvaluating site {site}...")
            dataset = EcoPerceiverDataset(
                data_path,
                config=dataset_config,
                sites=[site],
            )

            if len(dataset) == 0:
                print(f"  Warning: site {site} has no samples, skipping.")
                continue

            dataloader = DataLoader(
                dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=eval_num_workers,
                pin_memory=eval_pin_memory,
                collate_fn=dataset.collate_fn,
            )

            site_total_loss = 0.0
            site_num_batches = 0
            site_num_samples = 0
            site_sum_sq_error = np.zeros(num_targets, dtype=np.float64)
            site_count_valid = np.zeros(num_targets, dtype=np.int64)

            with torch.no_grad():
                for batch in dataloader:
                    if hasattr(batch, "to"):
                        batch = batch.to(device)

                    res = model(batch)  # EcoPerceiverOutput
                    yhat = res.predictions  # [B, num_targets]
                    y = res.ground_truth    # [B, num_targets]
                    loss = res.loss

                    B = yhat.shape[0]
                    site_num_samples += B

                    # Loss averaged over loss tensor
                    if isinstance(loss, torch.Tensor):
                        batch_loss = loss.mean().item()
                    else:
                        batch_loss = float(loss)

                    site_total_loss += batch_loss
                    site_num_batches += 1

                    # RMSE parts
                    yhat_np = yhat.detach().cpu().numpy()
                    y_np = y.detach().cpu().numpy()

                    mask = np.isfinite(y_np)
                    err_sq = (yhat_np - y_np) ** 2

                    site_sum_sq_error += np.where(mask, err_sq, 0.0).sum(axis=0)
                    site_count_valid += mask.sum(axis=0)

            if site_num_batches == 0:
                print(f"  No batches processed for site {site}, skipping.")
                continue

            site_avg_loss = site_total_loss / site_num_batches

            # Per flux RMSE
            site_rmse = []
            for j in range(num_targets):
                if site_count_valid[j] > 0:
                    rmse_j = np.sqrt(site_sum_sq_error[j] / site_count_valid[j])
                else:
                    rmse_j = float("nan")
                site_rmse.append(rmse_j)

            # Log to console
            print(f"  Finished site {site}")
            print(f"    Batches: {site_num_batches}")
            print(f"    Samples: {site_num_samples}")
            print(f"    Avg loss: {site_avg_loss:.2f}")
            for name, val in zip(flux_labels, site_rmse):
                if np.isfinite(val):
                    print(f"    RMSE {name}: {val:.2f}")
                else:
                    print(f"    RMSE {name}: nan")

            # Row for this site (round to 2 decimal places)
            row = {
                "site": site,
                "num_batches": site_num_batches,
                "num_samples": site_num_samples,
                "avg_loss": round(site_avg_loss, 2),
            }
            for name, val in zip(flux_labels, site_rmse):
                row[f"rmse_{name}"] = round(val, 2) if np.isfinite(val) else float("nan")
            writer.writerow(row)

            # Update global accumulators
            global_total_loss += site_total_loss
            global_total_batches += site_num_batches
            global_total_samples += site_num_samples
            global_sum_sq_error += site_sum_sq_error
            global_count_valid += site_count_valid

            # Help GC a bit between sites
            del dataloader, dataset

        # Global TOTAL row
        if global_total_batches > 0:
            global_avg_loss = global_total_loss / global_total_batches
        else:
            global_avg_loss = float("nan")

        global_rmse = []
        for j in range(num_targets):
            if global_count_valid[j] > 0:
                rmse_j = np.sqrt(global_sum_sq_error[j] / global_count_valid[j])
            else:
                rmse_j = float("nan")
            global_rmse.append(rmse_j)

        total_row = {
            "site": "TOTAL",
            "num_batches": global_total_batches,
            "num_samples": global_total_samples,
            "avg_loss": round(global_avg_loss, 2) if np.isfinite(global_avg_loss) else float("nan"),
        }
        for name, val in zip(flux_labels, global_rmse):
            total_row[f"rmse_{name}"] = round(val, 2) if np.isfinite(val) else float("nan")

        writer.writerow(total_row)

        print(f"\nSaved metrics for all test sites to {output_csv_path.resolve()}")
        print("TOTAL row:")
        print(f"  Batches: {global_total_batches}")
        print(f"  Samples: {global_total_samples}")
        print(f"  Avg loss: {global_avg_loss:.2f}")
        for name, val in zip(flux_labels, global_rmse):
            if np.isfinite(val):
                print(f"  RMSE {name}: {val:.2f}")
            else:
                print(f"  RMSE {name}: nan")


if __name__ == "__main__":
    main()
