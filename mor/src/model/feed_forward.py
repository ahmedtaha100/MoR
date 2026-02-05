import torch

import torch.nn as nn

import torch.nn.functional as F



from .config import MoRConfig





class SwiGLUFFN(nn.Module):

    def __init__(self, config: MoRConfig):

        super().__init__()

        self.config = config



        self.hidden_dim = config.hidden_dim

        self.ffn_dim = config.ffn_dim



        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)



        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)



        self.dropout = nn.Dropout(config.dropout)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = F.silu(self.gate_proj(x))



        up = self.up_proj(x)



        hidden = gate * up



        output = self.down_proj(hidden)



        output = self.dropout(output)



        return output





class StandardFFN(nn.Module):

    def __init__(self, config: MoRConfig):

        super().__init__()



        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)

        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.up_proj(x)

        x = F.gelu(x)

        x = self.down_proj(x)

        x = self.dropout(x)

        return x





def create_ffn(config: MoRConfig) -> nn.Module:

    if config.ffn_hidden_act == "silu":

        return SwiGLUFFN(config)

    else:

        return StandardFFN(config)

