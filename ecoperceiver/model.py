import torch
from typing import Tuple
from torch import nn
from .components import AttentionLayer
from dataclasses import dataclass
from einops import rearrange
torch.manual_seed(0)


@dataclass
class EcoPerceiverConfig():
    latent_hidden_dim: int = 256
    input_embedding_dim: int = 128
    tabular_inputs: Tuple = ()
    spectral_data_channels: int = 7
    spectral_data_resolution: Tuple = (8,8)
    weight_sharing: bool = False
    mlp_ratio: int = 3
    num_frequencies: int = 12
    context_length: int = 64
    num_heads: int = 8
    obs_dropout: float = 0.0
    layers: str = 'cscscsss' # c = cross-attention (with input), s = self-attention
    targets: Tuple = ('NEE_VUT_REF')
    causal: bool = True


class FourierFeatureMapping(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, values):
        embeddings = torch.arange(self.num_frequencies).to(values.device)
        embeddings = torch.pi * 2 ** embeddings
        embeddings = embeddings * values.unsqueeze(-1)
        sin_emb = torch.sin(embeddings)
        cos_emb = torch.cos(embeddings)
        return torch.cat([sin_emb, cos_emb], dim=-1)


class EcoPerceiverModel(nn.Module):
    def __init__(self, config: EcoPerceiverConfig):
        super().__init__()
        self.config = config
        self.input_embeddings = nn.Embedding(len(self.config.tabular_inputs), self.config.input_embedding_dim)

        self.fourier = FourierFeatureMapping(self.config.num_frequencies)
        self.input_hidden_dim = self.config.input_embedding_dim + self.config.num_frequencies * 2
        self.obs_dropout = nn.Dropout(p=self.config.obs_dropout)

        latent_hidden_dim = self.config.latent_hidden_dim
        context_length = self.config.context_length

        self.latent_embeddings = nn.Embedding(context_length, latent_hidden_dim)

        num_pixels = self.config.spectral_data_resolution[0] * self.config.spectral_data_resolution[1]
        self.channels = self.config.spectral_data_channels # for brevity
        self.spectral_projections = nn.ModuleList(
            [nn.Linear(num_pixels, self.config.num_frequencies * 2) for _ in range(self.channels)]
        )
        self.spectral_embeddings = nn.Embedding(self.channels, self.config.input_embedding_dim)
        self.layer_norm_ec = nn.LayerNorm(self.input_hidden_dim, eps=1e-12)
        self.layer_norm_eo = nn.LayerNorm(self.input_hidden_dim, eps=1e-12)

        self.layer_types = self.config.layers
        layers = []
        if self.config.weight_sharing:
            cross_attention_block = [
                AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio, kv_hidden_size=self.input_hidden_dim),
                AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio)
            ]
            for i in range(len(self.layer_types)//2):
                block_type = self.layer_types[i*2:(i+1)*2]
                if block_type == 'cs':
                    layers.extend(cross_attention_block)
                else:
                    layers.extend([
                        AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio),
                        AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio)
                    ])
            
        else:
            for l in self.layer_types:
                if l == 'c':
                    layers.append(
                        AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio, kv_hidden_size=self.input_hidden_dim)
                    )
                elif l == 's':
                    layers.append(
                        AttentionLayer(config.latent_hidden_dim, config.num_heads, config.mlp_ratio)
                    )

        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(latent_hidden_dim, 1)
        self.causal_mask = nn.Parameter(torch.zeros((1, context_length, context_length), dtype=torch.bool), requires_grad=False)
        if self.config.causal:
            for y in range(context_length):
                for x in range(context_length):
                    self.causal_mask[:,y,x] = y < x
        
        self.apply(self.initialize_weights)
    
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def process_spectral_inputs(self, spectral_data, B, L):
        device = self.input_embeddings.weight.device
        
        imgs = []
        indices = []
        for b, t, img in spectral_data:
            imgs.append(img.flatten(1).to(device))
            indices.append(b*L + t)
        img_data = torch.stack(imgs) # (M, C, num_pixels)
        img_map = torch.zeros(B*L, device=device, dtype=torch.bool)
        img_map[indices] = True
        
        img_data = img_data.transpose(0,1) # (C, M, num_pixels)
        img_data_proj = torch.zeros((self.channels, img_data.shape[1], self.config.num_frequencies * 2), device=device)
        for i, proj in enumerate(self.spectral_projections):
            img_data_proj[i] = proj(img_data[i])
        
        # add embeddings
        img_data = img_data_proj.transpose(0,1) # (M, C, 2*F)
        spec_embeddings = self.spectral_embeddings.weight.unsqueeze(0).repeat(len(spectral_data), 1, 1) # (M, C, I)
        img_data = torch.cat([img_data, spec_embeddings], dim=-1)  # (M, C, IH)
        
        return img_data, img_map

    def forward(self, batch):
        '''
        B - batch size
        L - sequence length
        P - # of observations (input variables)
        M - # of observations with images
        F - # of frequencies
        C - # of spectral channels
        I - input embedding dim
        IH - total input dim (I + 2*F)
        H - latent hidden dim
        '''
        predictors = batch['predictors']
        labels = batch['predictor_labels']
        mask = batch['predictor_mask']
        spectral_data = batch['modis_imgs']
        fluxes = batch['targets']

        device = self.input_embeddings.weight.device

        # Marshall data
        mask = mask.to(device)
        if self.training:
            dropout_mask = ~self.obs_dropout(torch.ones(mask.shape, device=device)).to(torch.bool)
            mask = mask | dropout_mask
        # Don't drop DoY or ToD
        doy_index = labels.index('DOY')
        tod_index = labels.index('TOD')
        mask[:,:,doy_index] = False
        mask[:,:,tod_index] = False
        
        observations = predictors.to(device)
        fluxes = fluxes.to(device)
        if len(spectral_data) == 0:
            return self.forward_no_images(observations, mask, fluxes)

        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = self.input_embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
        combined_obs = torch.cat([fourier_obs, embedding_obs], dim=-1) # (B, L, P, IH)
        combined_obs = rearrange(combined_obs, 'B L P IH -> (B L) P IH') # (B*L, P, IH)
        mask = rearrange(mask, 'B L P -> (B L) P').unsqueeze(1) # (B*L, 1, P)

        # images
        img_data, img_map = self.process_spectral_inputs(spectral_data, B, L)

        # divide obs
        mask_with_image = torch.cat([mask[img_map], torch.zeros((len(spectral_data), 1, self.channels), dtype=bool, device=device)], dim=-1) # (M, 1, P+C)
        obs_with_image = torch.cat([combined_obs[img_map], img_data], dim=1)
        mask_without_image = mask[~img_map]
        obs_without_image = combined_obs[~img_map]

        obs_with_image = self.layer_norm_eo(obs_with_image)
        obs_without_image = self.layer_norm_ec(obs_without_image)

        hidden = self.latent_embeddings.weight.unsqueeze(0).repeat(B,1,1) # (B, L, H)

        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'c':
                hidden = rearrange(hidden, 'B L H -> (B L) H').unsqueeze(1) # (B*L, 1, H)
                hidden_with_image = hidden[img_map]
                hidden_without_image = hidden[~img_map]

                hidden_with_image, _ = self.layers[i](hidden_with_image, obs_with_image, mask=mask_with_image)
                hidden_without_image, _ = self.layers[i](hidden_without_image, obs_without_image, mask=mask_without_image)

                hidden[img_map] = hidden_with_image
                hidden[~img_map] = hidden_without_image
                hidden = rearrange(hidden.squeeze(), '(B L) H -> B L H', B=B, L=L)
            elif layer_type == 's':
                hidden, _ = self.layers[i](hidden, hidden, mask=self.causal_mask)
        
        op = self.output_proj(hidden[:,-1,:]).squeeze() # B
        loss = self.loss(fluxes.squeeze(), op)
        
        return {
            'loss': loss,
            'logits': op,
        }
    
    def forward_no_images(self, observations, mask, fluxes):
        '''
        B - batch size
        L - sequence length
        P - # of observations (input variables)
        M - # of observations with images
        F - # of frequencies
        C - # of spectral channels
        I - input embedding dim
        IH - total input dim (I + 2*F)
        H - latent hidden dim
        '''
        device = self.input_embeddings.weight.device
        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = self.input_embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
        combined_obs = torch.cat([fourier_obs, embedding_obs], dim=-1) # (B, L, P, IH)
        combined_obs = rearrange(combined_obs, 'B L P IH -> (B L) P IH') # (B*L, P, IH)
        mask = rearrange(mask, 'B L P -> (B L) P').unsqueeze(1) # (B*L, 1, P)

        combined_obs = self.layer_norm_ec(combined_obs)
        hidden = self.latent_embeddings.weight.unsqueeze(0).repeat(B,1,1) # (B, L, H)

        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'c':
                hidden = rearrange(hidden, 'B L H -> (B L) H').unsqueeze(1) # (B*L, 1, H)

                hidden, _ = self.layers[i](hidden, combined_obs, mask=mask)
                hidden = rearrange(hidden.squeeze(), '(B L) H -> B L H', B=B, L=L)
            elif layer_type == 's':
                hidden, _ = self.layers[i](hidden, hidden, mask=self.causal_mask)
        
        op = self.output_proj(hidden[:,-1,:]).squeeze() # B
        loss = self.loss(fluxes.squeeze(), op)
        return {
            'loss': loss,
            'logits': op,
        }
    
    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean()


class FauxVanillaModel(nn.Module):
    def __init__(self, config: EcoPerceiverConfig):
        super().__init__()
        self.config = config
        self.input_embeddings = nn.Embedding(len(self.config.tabular_inputs), self.config.input_embedding_dim)

        self.fourier = FourierFeatureMapping(self.config.num_frequencies)
        self.input_hidden_dim = self.config.input_embedding_dim + self.config.num_frequencies * 2
        #self.obs_dropout = nn.Dropout(p=self.config.obs_dropout)

        self.latent_hidden_dim = self.config.latent_hidden_dim
        context_length = self.config.context_length

        num_pixels = self.config.spectral_data_resolution[0] * self.config.spectral_data_resolution[1]
        self.channels = self.config.spectral_data_channels # for brevity
        self.spectral_projections = nn.ModuleList(
            [nn.Linear(num_pixels, self.config.num_frequencies * 2) for _ in range(self.channels)]
        )
        self.spectral_embeddings = nn.Embedding(self.channels, self.config.input_embedding_dim)
        self.hidden_proj = nn.Linear(self.input_hidden_dim, self.latent_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.latent_hidden_dim, eps=1e-12)

        self.layer_types = self.config.layers
        layers = []
        for l in self.layer_types:
            if l == 's':
                layers.append(
                    AttentionLayer(config.latent_hidden_dim, config.num_heads, config.mlp_ratio)
                )

        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(self.latent_hidden_dim, 1)
        self.apply(self.initialize_weights)
    
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    

    def forward(self, batch):
        '''
        B - batch size
        L - sequence length (num_observations) ((1))
        P - # of observations (input variables)
        M - # of observations with images
        F - # of frequencies
        C - # of spectral channels
        I - input embedding dim
        IH - total input dim (I + 2*F)
        H - latent hidden dim
        '''
        predictors = batch['predictors']
        labels = batch['predictor_labels']
        mask = batch['predictor_mask']
        spectral_data = batch['modis_imgs']
        fluxes = batch['targets']

        device = self.input_embeddings.weight.device
        # Marshall data
        mask = mask.to(device)
        doy_index = labels.index('DOY')
        tod_index = labels.index('TOD')
        mask[:,:,doy_index] = False
        mask[:,:,tod_index] = False
        
        observations = predictors.to(device)
        fluxes = fluxes.to(device)
        # if len(spectral_data) == 0:
        #     faux_spectral = torch.zeros(self.num_channels, 8, 8)
        #     return self.forward_no_images(observations, mask, fluxes)

        B, L, P = observations.shape
        # spectral mask
        spectral_mask = torch.ones(B, 1, 9).to(torch.bool)
        spec_dict = {}
        for i, s in enumerate(spectral_data):
            spectral_mask[s[0],:,:] = False
            spec_dict[i] = s[0]

        spectral_mask = spectral_mask.to(device)
        mask = torch.cat([mask, spectral_mask], dim=-1).unsqueeze(-2).repeat(1,1,30,1)
        
        spectral_values = torch.zeros(B, 9, self.config.num_frequencies * 2).to(device) # (B, L(1), C, 2*F)

        if len(spectral_data) > 0:
            imgs = []
            for _, _, img in spectral_data:
                imgs.append(img.flatten(1).to(device))
            img_data = torch.stack(imgs).to(device) # (M, C, num_pixels)
            img_data = img_data.transpose(0,1) # (C, M, num_pixels)
            img_data_proj = torch.zeros((self.channels, img_data.shape[1], self.config.num_frequencies * 2), device=device)
            for i, proj in enumerate(self.spectral_projections):
                img_data_proj[i] = proj(img_data[i])
            img_data = img_data_proj.transpose(0,1) # (M, C, 2*F)
            for k, v in spec_dict.items():
                spectral_values[v,:,:] = img_data[k,:,:]
        
        spectral_values = torch.cat([spectral_values, self.spectral_embeddings.weight.unsqueeze(0).repeat(B, 1, 1)], dim=-1) # (B, L(1), C, 2*F + I)

        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)

        embedding_obs = self.input_embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
        combined_obs = torch.cat([fourier_obs, embedding_obs], dim=-1) # (B, L, P, IH)
        mask = mask.squeeze()
        hidden = torch.cat([combined_obs.squeeze(), spectral_values], dim=-2)
        hidden = self.hidden_proj(hidden)
        hidden = self.layer_norm(hidden)

        for layer in self.layers:
            hidden, _ = layer(hidden, hidden, mask=mask)
        
        op = self.output_proj(hidden[:,-1,:]).squeeze()
        loss = self.loss(fluxes.squeeze(), op)
        
        return {
            'loss': loss,
            'logits': op,
        }
    

    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean()
