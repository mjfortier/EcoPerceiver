import math
import torch
from torch import nn
from typing import Optional
from collections import OrderedDict

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