import math
import torch
from torch import nn
from typing import Optional, Tuple, List
from einops import rearrange
import numpy as np
from collections import OrderedDict
from ecoperceiver.dataset import EcoSageBatch
from dataclasses import dataclass
from ecoperceiver.constants import *
from torchvision.models import resnet18


@dataclass
class EcoSageConfig:
    # General config
    latent_space_dim: int = 64
    num_frequencies: int = 12
    input_embedding_dim: int = 22
    context_length: int = 64
    num_heads: int = 8
    mlp_ratio: int = 3
    obs_dropout: float = 0.0
    layers: str = 'wcswcswcswcsss' # w = windowed cross-attention, c = cross-attention (with input), s = self-attention
    #causal: bool = True
    targets: Tuple = ('NEE',)

    # ECInputModule config
    allowable_ec_predictors: List[str] = EC_PREDICTORS

    # ModisLinearInputModule config
    modis_channels: int = 9
    modis_resolution: Tuple = (8,8)
    mask_ratio: float = 0.3

    # GeoInputModule
    allowable_geo_predictors: List[str] = GEO_PREDICTORS

    # PhenocamRGBInputModule
    num_tokens_per_image: int = 4
    cnn_model: str = 'resnet18' # test before changing this...
    pretrained_path: str = None


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(input)

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


class MultiheadAttentionBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            kv_hidden_size=None,
            qkv_bias=True,
            attention_dropout_prob=0.0,
            hidden_dropout_prob=0.0
            )-> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"The hidden size {hidden_size} is not a multiple of the number of attention heads ({num_heads}).")

        self.hidden_size = hidden_size
        self.kv_hidden_size = hidden_size if kv_hidden_size is None else kv_hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.key = nn.Linear(self.kv_hidden_size, hidden_size, bias=qkv_bias)
        self.value = nn.Linear(self.kv_hidden_size, hidden_size, bias=qkv_bias)

        self.attention_dropout = nn.Dropout(attention_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            q_states: torch.Tensor,
            kv_states: Optional[torch.Tensor],
            mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        '''
        hidden_states - (B, L, H)
        cross_attention_states - (B, Lc, Hc)
        mask: (B, L, Lc)
            - where mask is True, attention score will be set to '-inf'
        '''
        
        query_layer = self.transpose_for_scores(self.query(q_states)) # (B, num_heads, L, head_size)
        key_layer   = self.transpose_for_scores(self.key(kv_states)) # (B, num_heads, Lc, head_size)
        value_layer = self.transpose_for_scores(self.value(kv_states)) # (B, num_heads, Lc, head_size)
    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (B, num_heads, L, Lc)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.where(~mask, float('-inf'))
        attention_probs = self.attention_dropout(attention_scores) # NOTE: This was in the original paper, but it drops entire tokens so we could try removing it
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)


        context_layer = torch.matmul(attention_probs, value_layer) # (B, num_heads, L, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape) # (B, L, H)

        output = self.dense(context_layer)
        output = self.hidden_dropout(output)
        
        return (output, attention_probs)


class AttentionLayer(nn.Module):
    '''
    Basic (pre-LN) attention layer:
        - Layernorm & MHA
        - residual
        - Layernorm & FFN
        - residual
    '''
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio,
            kv_hidden_size=None,
            qkv_bias=True,
            activation='gelu', # can also pass a nn.Module
            attention_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
            eps=1e-12,
            )-> None:
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = MultiheadAttentionBlock(hidden_size, num_heads, kv_hidden_size=kv_hidden_size, qkv_bias=qkv_bias,
                                                 attention_dropout_prob=attention_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)
        
        self.layernorm_2 = nn.LayerNorm(hidden_size, eps=eps)
    
        if isinstance(activation, str):
            activation = ACT2FN[activation]
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            activation,
            nn.Linear(mlp_ratio * hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        q_states: torch.Tensor,
        kv_states: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        if kv_states is None:
            kv_states = q_states
        
        residual = q_states
        h = self.layernorm_1(q_states)
        h, att = self.attention(h, kv_states, mask=mask)
        h = h + residual

        residual = h
        h = self.layernorm_2(h)
        h = self.ffn(h)
        h = self.dropout(h)
        h = h + residual
        
        if output_attentions:
            return (h, att)
        else:
            return (h, None)


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


#---------------#
# Input Modules #
#---------------#


class ECInputModule(nn.Module):
    def __init__(self, config: EcoSageConfig) -> None:
        super().__init__()
        self.config = config
        self.fourier = FourierFeatureMapping(self.config.num_frequencies)
        self.allowable_vars = self.config.allowable_ec_predictors
        self.embeddings = nn.Embedding(len(self.allowable_vars), self.config.input_embedding_dim)


    def _filter_nan_vars(self, input_vars, ec_data):
        keep_vars = []
        keep_indices = []
        for i, v in enumerate(input_vars):
            if (~ec_data[:,:,i].isnan()).sum() > 0: # there's at least one valid entry
                keep_vars.append(v)
                keep_indices.append(i)
        keep_indices = torch.IntTensor(keep_indices).to(ec_data.device)
        keep_data = torch.index_select(ec_data, dim=-1, index=keep_indices)
        return keep_vars, keep_data


    def forward(self, batch: EcoSageBatch) -> Tuple:
        device = self.embeddings.weight.device
        input_vars = batch.predictor_columns
        ec_data = batch.predictor_values.to(device)
        embedding_map = {v: self.embeddings.weight[i] for i, v in enumerate(self.allowable_vars)}
        assert set(input_vars) <= set(self.allowable_vars), f'ERROR: found unseen variables in EC Input Module: f{[c for c in input_vars if c not in self.allowable_vars]}'
        
        keep_vars, ec_data = self._filter_nan_vars(input_vars, ec_data)
        B, L, _ = ec_data.shape
        mask = ec_data.isnan().to(device)
        fourier_encoded = self.fourier(ec_data)
        embeddings = torch.stack([embedding_map[i] for i in keep_vars]).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)
        output = torch.cat([fourier_encoded, embeddings], dim=-1)
        output = rearrange(output, 'B L P IH -> (B L) P IH')
        mask = rearrange(mask, 'B L P -> (B L) P').unsqueeze(1)
        output = output.nan_to_num(0.0) # the mask takes care of this
        return output, mask


class ModisLinearInputModule(nn.Module):
    def __init__(self, config: EcoSageConfig) -> None:
        super().__init__()
        self.config = config
        h, w = self.config.modis_resolution
        self.num_pixels = h * w
        self.channels = self.config.modis_channels
        self.hidden_size = self.config.num_frequencies * 2
        self.spectral_projection = nn.ModuleList(
            [nn.Linear(self.num_pixels, self.hidden_size) for _ in range(self.channels)]
        )
        self.spectral_embeddings = nn.Embedding(self.channels, self.config.input_embedding_dim)
        self.mask_ratio = self.config.mask_ratio
        self.image_placeholder = nn.Parameter(torch.zeros(self.channels, self.num_pixels), requires_grad=False)

    def _generate_mask(self, sample: Tuple) -> torch.Tensor:
        if len(sample) == 0:
            return torch.Tensor([True] * self.channels)
        mask = []
        img = sample[0]
        for i in range(self.channels):
            ratio_nan = torch.where(img[i,:,:] < 0.0, 1.0, 0.0).sum() / self.num_pixels
            mask.append(ratio_nan > self.mask_ratio)
        return torch.Tensor(mask)


    def forward(self, batch: EcoSageBatch) -> Tuple:
        modis_data = batch.modis
        device = self.spectral_embeddings.weight.device
        B, L, _ = batch.predictor_values.shape

        images = []
        masks = []
        for sample in modis_data:
            masks.append(self._generate_mask(sample))
            if len(sample) == 0:
                images.append(self.image_placeholder)
            else:
                images.append(rearrange(sample[0], 'C H W -> C (H W)').to(device))
        
        mask = torch.stack(masks).to(bool).to(device)
        if (~mask).sum() == 0: # no valid MODIS imagery
            return None, None
        mask = mask.unsqueeze(1).repeat(1, L, 1)
        
        images = torch.stack(images)
        projected = []
        for i, proj in enumerate(self.spectral_projection):
            projected.append(proj(images[:,i,:]))
        projected_images = torch.stack(projected, dim=1) # (B, C, Enc)
        embeddings = self.spectral_embeddings.weight.unsqueeze(0).repeat(B,1,1)
        output = torch.cat([projected_images, embeddings], dim=-1) # (B, C, IH)
        return output, mask


class GeoInputModule(nn.Module):
    def __init__(self, config: EcoSageConfig) -> None:
        super().__init__()
        self.config = config
        self.fourier = FourierFeatureMapping(self.config.num_frequencies)
        self.allowable_vars = self.config.allowable_geo_predictors
        self.embeddings = nn.Embedding(len(self.allowable_vars), self.config.input_embedding_dim)


    def _filter_nan_vars(self, input_vars, aux_data):
        keep_vars = []
        keep_indices = []
        for i, v in enumerate(input_vars):
            if (~aux_data[:,i].isnan()).sum() > 0: # there's at least one valid entry
                keep_vars.append(v)
                keep_indices.append(i)
        keep_indices = torch.IntTensor(keep_indices).to(aux_data.device)
        keep_data = torch.index_select(aux_data, dim=-1, index=keep_indices)
        return keep_vars, keep_data


    def forward(self, batch: EcoSageBatch) -> Tuple:
        device = self.embeddings.weight.device
        embedding_map = {v: self.embeddings.weight[i] for i, v in enumerate(self.allowable_vars)}
        input_vars = batch.aux_columns
        aux_data = batch.aux_values.to(device)
        B, L, _ = batch.predictor_values.shape
        
        keep_vars, aux_data = self._filter_nan_vars(input_vars, aux_data)
        mask = aux_data.isnan()
        fourier_encoded = self.fourier(aux_data)

        embeddings = torch.stack([embedding_map[i] for i in keep_vars]).unsqueeze(0).repeat(B, 1, 1)
        output = torch.cat([fourier_encoded, embeddings], dim=-1)
        mask = mask.unsqueeze(1).repeat(1, L, 1)
        output = output.nan_to_num(0.0) # the mask takes care of this
        return output, mask


class IGBPInputModule(nn.Module):
    def __init__(self, config: EcoSageConfig) -> None:
        super().__init__()
        self.config = config
        self.allowable_vars = IGBP_CODES
        self.embedding_length = 2 * self.config.num_frequencies + self.config.input_embedding_dim
        self.embeddings = nn.Embedding(len(self.allowable_vars), self.embedding_length)


    def forward(self, batch: EcoSageBatch) -> Tuple:
        device = self.embeddings.weight.device
        embedding_map = {v: self.embeddings.weight[i] for i, v in enumerate(self.allowable_vars)}
        codes = batch.igbp
        B, L, _ = batch.predictor_values.shape

        embeddings = torch.zeros((B, 1, self.embedding_length), device=device)
        mask = torch.zeros((len(codes), L, 1), device=device).to(bool)
        for i, code in enumerate(codes):
            if code in self.allowable_vars:
                embeddings[i,:,:] = embedding_map[code]
            else:
                mask[i,:,:] = True
        
        return embeddings, mask



class PhenocamRGBInputModule(nn.Module):
    def __init__(self, config: EcoSageConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.pretrained_path:
            resnet_model = resnet18(weights=None)
            state_dict = torch.load(self.config.pretrained_path, map_location="cpu")
            resnet_model.load_state_dict(state_dict)
        else:
            from torchvision.models import ResNet18_Weights
            resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*(list(resnet_model.children())[:-1])) # cut off last layer
        self.cnn._skip_init = True # don't override initial weights
        self.embeddings = nn.Embedding(self.config.num_tokens_per_image, self.config.input_embedding_dim)
        self.encoding_length = self.config.num_frequencies * 2

        resnet_output_size = 512 # this may need to be changed if the model changes significantly
        self.fc = nn.Linear(resnet_output_size, self.encoding_length * self.config.num_tokens_per_image)
    

    def forward(self, batch: EcoSageBatch) -> Tuple:
        device = self.embeddings.weight.device
        phenocam_data = batch.phenocam_rgb
        B, L, _ = batch.predictor_values.shape
        
        valid_samples = [i for i, s in enumerate(phenocam_data) if len(s) > 0]
        if len(valid_samples) == 0:
            return None, None
        
        cnn_input = torch.stack([s[0] for s in phenocam_data if len(s) > 0], dim=0).to(device)
        hidden = self.cnn(cnn_input).squeeze(2,3)
        hidden = self.fc(hidden)
        
        hidden = rearrange(hidden, 'B (T H) -> B T H', T=self.config.num_tokens_per_image, H=self.encoding_length)
        embeddings = self.embeddings.weight.unsqueeze(0).repeat(len(valid_samples),1,1)
        output = torch.cat([hidden, embeddings], dim=-1)
        
        full_output = torch.zeros((B, self.config.num_tokens_per_image, self.encoding_length + self.config.input_embedding_dim), device=device)
        mask = torch.ones((B, L, self.config.num_tokens_per_image), device=device).to(bool)
        for op_ind, full_ind in enumerate(valid_samples):
            full_output[full_ind,:,:] = output[op_ind,:,:]
            mask[full_ind,:,:] = False

        return full_output, mask


class FluxOutputModule(nn.Module):
    def __init__(self, config: EcoSageConfig) -> None:
        super().__init__()
        self.config = config
        self.allowable_vars = self.config.targets

        self.fc = nn.ModuleDict({
            v: nn.Linear(self.config.latent_space_dim, 1) for v in self.allowable_vars
        })
    
    def forward(self, hidden: torch.Tensor, batch: EcoSageBatch):
        final_tokens = hidden[:,-1,:].squeeze()
        
        op = {}
        for i, var in enumerate(batch.target_columns):
            if var not in self.allowable_vars:
                print(f'WARNING: skipping unseen target {var}')
                continue
            pred = self.fc[var](final_tokens).squeeze()
            target = batch.target_values[:,i].squeeze().to(pred.device)
            op[var] = torch.stack([target, pred], dim=1)
        return op
