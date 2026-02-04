from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from .config import MoRConfig
from .embeddings import RMSNorm
from .attention import MoRAttention, KVCache
from .feed_forward import SwiGLUFFN


class TransformerBlock(nn.Module):
    def __init__(self, config: MoRConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.self_attn = MoRAttention(config, layer_idx=layer_idx)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_dim, eps=config.rms_norm_eps
        )

        self.mlp = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
        active_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        reuse_kv: bool = False,
        apply_active_mask_to_kv: bool = True,
        use_selective_cache: bool = False,
        shared_kv: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache], Optional[Tuple]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights, present_key_value, computed_kv = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
            active_mask=active_mask,
            output_attentions=output_attentions,
            reuse_kv=reuse_kv,
            apply_active_mask_to_kv=apply_active_mask_to_kv,
            use_selective_cache=use_selective_cache,
            shared_kv=shared_kv,
        )

        hidden_states = residual + attn_output

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, present_key_value, computed_kv


class SharedTransformerBlock(nn.Module):
    def __init__(self, config: MoRConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=layer_idx + i)
            for i in range(config.num_shared_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = False,
        active_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        router_weight: Optional[torch.Tensor] = None,
        reuse_kv: bool = False,
        apply_active_mask_to_kv: bool = True,
        use_selective_cache: bool = False,
        create_cache_fn=None,
        shared_kvs: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, Optional[list], Optional[List[KVCache]], Optional[List[Tuple]]]:
        from .attention import KVCache, SelectiveKVCache

        all_attentions = [] if output_attentions else None
        all_key_values = [] if use_cache else None
        all_computed_kvs = []

        input_hidden_states = hidden_states

        for i, layer in enumerate(self.layers):
            if past_key_values is not None and i < len(past_key_values):
                past_key_value = past_key_values[i]
            elif use_cache:
                if create_cache_fn is not None:
                    past_key_value = create_cache_fn()
                elif use_selective_cache:
                    past_key_value = SelectiveKVCache()
                else:
                    past_key_value = KVCache()
            else:
                past_key_value = None

            layer_shared_kv = None
            if shared_kvs is not None and i < len(shared_kvs):
                layer_shared_kv = shared_kvs[i]

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                past_key_value=past_key_value,
                use_cache=use_cache,
                active_mask=active_mask,
                output_attentions=output_attentions,
                reuse_kv=reuse_kv,
                apply_active_mask_to_kv=apply_active_mask_to_kv,
                use_selective_cache=use_selective_cache,
                shared_kv=layer_shared_kv,
            )

            hidden_states = layer_outputs[0]
            computed_kv = layer_outputs[3]
            all_computed_kvs.append(computed_kv)

            if output_attentions:
                all_attentions.append(layer_outputs[1])

            if use_cache:
                all_key_values.append(layer_outputs[2])

        if router_weight is not None:
            if router_weight.dim() == 2:
                router_weight = router_weight.unsqueeze(-1)

            hidden_states = router_weight * hidden_states + (1 - router_weight) * input_hidden_states

        return hidden_states, all_attentions, all_key_values, all_computed_kvs

    def get_num_layers(self) -> int:
        return len(self.layers)
