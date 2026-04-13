import torch
from einops import rearrange

from ecoperceiver.components import ModisLinearInputModule
from ecoperceiver.era5_dataset import ERA5Batch
from ecoperceiver.model import EcoPerceiver


class ERA5ModisLinearInputModule(ModisLinearInputModule):
    def forward(self, batch: ERA5Batch):
        modis_values = batch.modis_values
        modis_present = batch.modis_present
        device = self.spectral_embeddings.weight.device
        batch_size, context_length, _ = batch.predictor_values.shape

        if modis_values is None or modis_present is None or not modis_present.any().item():
            return None, None

        masks = []
        for i in range(batch_size):
            if modis_present[i].item():
                masks.append(self._generate_mask((modis_values[i],)))
            else:
                masks.append(torch.ones(self.channels))

        mask = torch.stack(masks).to(dtype=torch.bool, device=device)
        if (~mask).sum() == 0:
            return None, None
        mask = mask.unsqueeze(1).repeat(1, context_length, 1)

        images = rearrange(modis_values.to(device), "B C H W -> B C (H W)")
        projected = []
        for i, proj in enumerate(self.spectral_projection):
            projected.append(proj(images[:, i, :]))
        projected_images = torch.stack(projected, dim=1)
        embeddings = self.spectral_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        output = torch.cat([projected_images, embeddings], dim=-1)
        return output, mask


class ERA5EcoPerceiver(EcoPerceiver):
    def __init__(self, config, relative_pretrained_path=None):
        super().__init__(config, relative_pretrained_path)
        era5_modis_module = ERA5ModisLinearInputModule(config)
        era5_modis_module.apply(self._initialize_weights)
        self.auxiliary_modules[0] = era5_modis_module
