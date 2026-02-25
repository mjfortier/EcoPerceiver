import torch
import yaml
import csv
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
from ecoperceiver.dataset import EcoPerceiverLoaderConfig
from ecoperceiver.era5_dataset import ERA5Dataset
from ecoperceiver.components import EcoPerceiverConfig
from ecoperceiver.model import EcoPerceiver

base_dir = Path(__file__).resolve().parent
run_path = base_dir.parent / "experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"
config_path = run_path.parent / "config.yml"
checkpoint_path = run_path / "checkpoint-11.pth"
data_path = base_dir.parent / "experiments/data"

with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("Configuration loaded:")
print(f"Model targets: {config['model']['targets']}")
print(f"Context length: {config['model']['context_length']}")
print(f"Latent space dim: {config['model']['latent_space_dim']}")
print(f"Checkpoint path: {checkpoint_path}")
print(f"Checkpoint exists: {checkpoint_path.exists()}")

model_config = EcoPerceiverConfig(**config["model"])

relative_pretrained_path = base_dir.parent / "ecoperceiver" / "resnet18_weights.pth"
model = EcoPerceiver(model_config, relative_pretrained_path)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

if checkpoint_path.exists():
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to evaluation mode")
else:
    print(f"Checkpoint not found at {checkpoint_path}")

dataset_config = EcoPerceiverLoaderConfig(**config['dataset'])
dataset = ERA5Dataset(data_path, config=dataset_config)
max_samples = min(1_000_000, len(dataset))
dataset_subset = Subset(dataset, range(max_samples))

dataloader = DataLoader(
    dataset_subset, 
    batch_size=config["dataloader"]["batch_size"],  
    shuffle=False,  
    num_workers=8, 
    pin_memory=True, 
    collate_fn=dataset.collate_fn
)

print(f"Dataset created with {len(dataset)} samples")
print(f"Running inference on first {max_samples} samples")

print("Testing model inference...")
output_csv_path = base_dir / "era5_predictions.csv"
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
