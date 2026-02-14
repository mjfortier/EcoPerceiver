import torch
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
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

model = EcoPerceiver(model_config)
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

dataloader = DataLoader(
    dataset, 
    batch_size=config["dataloader"]["batch_size"],  
    shuffle=False,  
    num_workers=8, 
    pin_memory=True, 
    collate_fn=dataset.collate_fn
)

print(f"Dataset created with {len(dataset)} samples")

print("Testing model inference...")
with torch.no_grad():
    batch = next(iter(dataloader))
    if hasattr(batch, "to"):
        batch = batch.to(device)

    res = model(batch)

    yhat = res.predictions
    print(f"Pred shape: {yhat.shape}")
    if yhat.shape[0] > 0:
        first_pred = yhat[0].detach().cpu()
        print("First inferenced sample:")
        for flux, value in zip(res.flux_labels, first_pred.tolist()):
            print(f"  {flux}: {value:.6f}")
    if res.loss is not None:
        print(f"Loss: {float(res.loss.mean().item()):.4f}")
    else:
        print("Loss: N/A (no target_values in inference batch)")
