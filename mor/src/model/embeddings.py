import math

from typing import Optional, Tuple



import torch

import torch.nn as nn



from .config import MoRConfig





class RMSNorm(nn.Module):

    def __init__(self, hidden_dim: int, eps: float = 1e-6):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_dim))

        self.eps = eps



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        variance = x.pow(2).mean(-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)

        return self.weight * x





class RotaryEmbedding(nn.Module):

    def __init__(

        self,

        dim: int,

        max_seq_len: int = 2048,

        base: float = 10000.0,

    ):

        super().__init__()

        self.dim = dim

        self.max_seq_len = max_seq_len

        self.base = base



        inv_freq = 1.0 / (

            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)

        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)



        self._set_cos_sin_cache(max_seq_len)



    def _set_cos_sin_cache(self, seq_len: int):

        self.max_seq_len_cached = seq_len



        t = torch.arange(seq_len, dtype=torch.float32)



        freqs = torch.outer(t, self.inv_freq)



        emb = torch.cat((freqs, freqs), dim=-1)



        self.register_buffer("cos_cached", emb.cos(), persistent=False)

        self.register_buffer("sin_cached", emb.sin(), persistent=False)



    def forward(

        self,

        x: torch.Tensor,

        position_ids: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        seq_len = x.shape[-2]



        if seq_len > self.max_seq_len_cached:

            self._set_cos_sin_cache(seq_len)



        if position_ids is not None:

            cos = self.cos_cached[position_ids]

            sin = self.sin_cached[position_ids]

        else:

            cos = self.cos_cached[:seq_len].unsqueeze(0)

            sin = self.sin_cached[:seq_len].unsqueeze(0)



        return cos.to(x.dtype), sin.to(x.dtype)





def rotate_half(x: torch.Tensor) -> torch.Tensor:

    x1 = x[..., : x.shape[-1] // 2]

    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)





def apply_rotary_pos_emb(

    q: torch.Tensor,

    k: torch.Tensor,

    cos: torch.Tensor,

    sin: torch.Tensor,

    unsqueeze_dim: int = 1,

) -> Tuple[torch.Tensor, torch.Tensor]:

    cos = cos.unsqueeze(unsqueeze_dim)

    sin = sin.unsqueeze(unsqueeze_dim)



    q_embed = (q * cos) + (rotate_half(q) * sin)

    k_embed = (k * cos) + (rotate_half(k) * sin)



    return q_embed, k_embed





class TokenEmbedding(nn.Module):

    def __init__(

        self,

        vocab_size: int,

        hidden_dim: int,

        initializer_range: float = 0.02,

    ):

        super().__init__()

        self.vocab_size = vocab_size

        self.hidden_dim = hidden_dim



        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_dim))

        nn.init.normal_(self.weight, std=initializer_range)



    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        return nn.functional.embedding(input_ids, self.weight)





class MoREmbeddings(nn.Module):

    def __init__(self, config: MoRConfig):

        super().__init__()

        self.config = config



        self.token_embedding = TokenEmbedding(

            vocab_size=config.vocab_size,

            hidden_dim=config.hidden_dim,

            initializer_range=config.initializer_range,

        )



        self.rotary_emb = RotaryEmbedding(

            dim=config.head_dim,

            max_seq_len=config.max_seq_len,

            base=config.rope_theta,

        )



        self.dropout = nn.Dropout(config.dropout)



    def forward(

        self,

        input_ids: torch.Tensor,

        position_ids: Optional[torch.Tensor] = None,

    ) -> torch.Tensor:

        embeddings = self.token_embedding(input_ids)

        embeddings = self.dropout(embeddings)

        return embeddings



    def get_rotary_emb(

        self,

        hidden_states: torch.Tensor,

        position_ids: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.rotary_emb(hidden_states, position_ids)

