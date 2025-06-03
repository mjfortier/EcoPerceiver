import torch
from torch import nn
from einops import rearrange
from ecoperceiver.constants import *
from ecoperceiver.components import EcoSageConfig, ECInputModule, ModisLinearInputModule, AttentionLayer, \
                       GeoInputModule, IGBPInputModule, PhenocamRGBInputModule, FluxOutputModule
torch.manual_seed(0)


class EcoSage(nn.Module):
    def __init__(self, config: EcoSageConfig):
        super().__init__()
        self.config = config
        self.windowed_modules = nn.ModuleList([ECInputModule(config)])
        self.auxiliary_modules = nn.ModuleList([ModisLinearInputModule(config), GeoInputModule(config), IGBPInputModule(config), PhenocamRGBInputModule(config)])

        self.input_hidden_dim = 2 * self.config.num_frequencies + self.config.input_embedding_dim
        self.latent_embeddings = nn.Embedding(self.config.context_length, self.config.latent_space_dim)
        layers = []
        for l in self.config.layers:
            if l in ['w', 'c']:
                layers.append(AttentionLayer(self.config.latent_space_dim, self.config.num_heads, self.config.mlp_ratio, kv_hidden_size=self.input_hidden_dim))
            else:
                layers.append(AttentionLayer(self.config.latent_space_dim, self.config.num_heads, self.config.mlp_ratio))
        self.layers = nn.ModuleList(layers)
        self.output_module = FluxOutputModule(config)
        self.apply(self.initialize_weights)


    def initialize_weights(self, module):
        if hasattr(module, '_skip_init') and module._skip_init:
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


    def forward(self, batch):
        device = self.latent_embeddings.weight.device
        windowed_inputs = []
        windowed_masks = []
        for m in self.windowed_modules:
            ip, mask = m(batch)
            windowed_inputs.append(ip)
            windowed_masks.append(mask)
        windowed_input = torch.cat(windowed_inputs, dim=-2)
        windowed_mask = torch.cat(windowed_masks, dim=-1)
        aux_inputs = []
        aux_masks = []
        for m in self.auxiliary_modules:
            ip, mask = m(batch)
            if ip == None or mask == None:
                continue
            aux_inputs.append(ip)
            aux_masks.append(mask)

        aux_input = torch.cat(aux_inputs, dim=-2)
        aux_mask = torch.cat(aux_masks, dim=-1)

        B, L, _ = batch.predictor_values.shape
        hidden = self.latent_embeddings.weight.unsqueeze(0).repeat(B,1,1)

        for i, layer_type in enumerate(self.config.layers):
            if layer_type == 'w':
                hidden = rearrange(hidden, 'B L H -> (B L) H').unsqueeze(1)
                hidden, _ = self.layers[i](hidden, windowed_input, mask=windowed_mask)
                hidden = rearrange(hidden.squeeze(), '(B L) H -> B L H', B=B, L=L)
            elif layer_type == 'c':
                hidden, _ = self.layers[i](hidden, aux_input, mask=aux_mask)
            else:
                hidden, _ = self.layers[i](hidden)
        
        output_dict = self.output_module(hidden, batch)
        
        loss = torch.Tensor([0.0]).to(device)
        # for now, naively just add up the losses
        for op in output_dict.values():
            loss += self.loss(op)
            
        return output_dict, loss


    def loss(self, output: torch.Tensor) -> torch.Tensor:
        filtered = output[~torch.any(output.isnan(),dim=1)]
        loss = (filtered[:,0] - filtered[:,1]) ** 2
        return loss.mean().unsqueeze(0)
