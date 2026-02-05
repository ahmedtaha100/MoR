import math

from typing import Optional, Tuple



import torch

import torch.nn as nn

import torch.nn.functional as F



from .config import MoRConfig

from .embeddings import apply_rotary_pos_emb





class KVCache:

                                                   



    def __init__(self):

        self.key_cache: Optional[torch.Tensor] = None

        self.value_cache: Optional[torch.Tensor] = None

        self.seq_len: int = 0



    def update(

        self,

        key: torch.Tensor,

        value: torch.Tensor,

        layer_idx: int,

        active_mask: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.key_cache is None:

            self.key_cache = key

            self.value_cache = value

        else:

            self.key_cache = torch.cat([self.key_cache, key], dim=2)

            self.value_cache = torch.cat([self.value_cache, value], dim=2)



        self.seq_len = self.key_cache.shape[2]

        return self.key_cache, self.value_cache



    def get(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        return self.key_cache, self.value_cache



    def get_seq_len(self) -> int:

        return self.seq_len



    def has_cache(self) -> bool:

        return self.key_cache is not None



    def reset(self):

        self.key_cache = None

        self.value_cache = None

        self.seq_len = 0





class SelectiveKVCache:

    


       



    def __init__(self):

        self.key_cache: Optional[torch.Tensor] = None

        self.value_cache: Optional[torch.Tensor] = None

        self.position_ids: Optional[torch.Tensor] = None

        self.cached_len: int = 0

        self.total_seq_len: int = 0



    def update(

        self,

        key: torch.Tensor,

        value: torch.Tensor,

        layer_idx: int,

        active_mask: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor]:

                                                                    

        seq_len = key.shape[2]

        if self.key_cache is None:

            self.key_cache = key

            self.value_cache = value

        else:

            self.key_cache = torch.cat([self.key_cache, key], dim=2)

            self.value_cache = torch.cat([self.value_cache, value], dim=2)



        self.cached_len = self.key_cache.shape[2]

        self.total_seq_len += seq_len

        return self.key_cache, self.value_cache



    def update_selective(

        self,

        key: torch.Tensor,

        value: torch.Tensor,

        position_ids: torch.Tensor,

        active_mask: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        










           

        batch_size, num_heads, seq_len, head_dim = key.shape

        device = key.device



        self.total_seq_len += seq_len



        if not active_mask.any():

            if self.key_cache is not None:

                return self.key_cache, self.value_cache, self.position_ids

            else:

                empty_k = key.new_zeros(batch_size, num_heads, 0, head_dim)

                empty_v = value.new_zeros(batch_size, num_heads, 0, head_dim)

                empty_pos = position_ids.new_zeros(batch_size, 0)

                return empty_k, empty_v, empty_pos



        max_active = active_mask.sum(dim=1).max().item()



        new_keys_list = []

        new_values_list = []

        new_positions_list = []



        for b in range(batch_size):

            active_indices = active_mask[b].nonzero(as_tuple=True)[0]

            num_active = active_indices.shape[0]



            batch_keys = key[b, :, active_indices, :]

            batch_values = value[b, :, active_indices, :]

            batch_positions = position_ids[b, active_indices]



            if num_active < max_active:

                pad_len = max_active - num_active

                batch_keys = F.pad(batch_keys, (0, 0, 0, pad_len), value=0)

                batch_values = F.pad(batch_values, (0, 0, 0, pad_len), value=0)

                batch_positions = F.pad(batch_positions, (0, pad_len), value=-1)



            new_keys_list.append(batch_keys)

            new_values_list.append(batch_values)

            new_positions_list.append(batch_positions)



        new_keys = torch.stack(new_keys_list, dim=0)

        new_values = torch.stack(new_values_list, dim=0)

        new_positions = torch.stack(new_positions_list, dim=0)



        if self.key_cache is None:

            self.key_cache = new_keys

            self.value_cache = new_values

            self.position_ids = new_positions

        else:

            self.key_cache = torch.cat([self.key_cache, new_keys], dim=2)

            self.value_cache = torch.cat([self.value_cache, new_values], dim=2)

            self.position_ids = torch.cat([self.position_ids, new_positions], dim=1)



        self.cached_len = self.key_cache.shape[2]



        return self.key_cache, self.value_cache, self.position_ids



    def get(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        return self.key_cache, self.value_cache



    def get_positions(self) -> Optional[torch.Tensor]:

        return self.position_ids



    def get_seq_len(self) -> int:

        return self.total_seq_len



    def get_cached_len(self) -> int:

        return self.cached_len



    def has_cache(self) -> bool:

        return self.key_cache is not None



    def reset(self):

        self.key_cache = None

        self.value_cache = None

        self.position_ids = None

        self.cached_len = 0

        self.total_seq_len = 0





class MoRAttention(nn.Module):

    def __init__(self, config: MoRConfig, layer_idx: int = 0):

        super().__init__()

        self.config = config

        self.layer_idx = layer_idx



        self.hidden_dim = config.hidden_dim

        self.num_heads = config.num_attention_heads

        self.num_kv_heads = config.num_kv_heads

        self.head_dim = config.head_dim

        self.num_kv_groups = self.num_heads // self.num_kv_heads



        assert self.hidden_dim == self.num_heads * self.head_dim



        self.q_proj = nn.Linear(

            self.hidden_dim, self.num_heads * self.head_dim, bias=False

        )

        self.k_proj = nn.Linear(

            self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False

        )

        self.v_proj = nn.Linear(

            self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False

        )

        self.o_proj = nn.Linear(

            self.num_heads * self.head_dim, self.hidden_dim, bias=False

        )



        self.attention_dropout = nn.Dropout(config.attention_dropout)



        self.scale = self.head_dim ** -0.5



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

        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,

        merge_shared_kv: bool = False,

    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        







           

        batch_size, seq_len, _ = hidden_states.shape



        query_states = self.q_proj(hidden_states)

        key_states = self.k_proj(hidden_states)

        value_states = self.v_proj(hidden_states)



        query_states = query_states.view(

            batch_size, seq_len, self.num_heads, self.head_dim

        ).transpose(1, 2)



        key_states = key_states.view(

            batch_size, seq_len, self.num_kv_heads, self.head_dim

        ).transpose(1, 2)



        value_states = value_states.view(

            batch_size, seq_len, self.num_kv_heads, self.head_dim

        ).transpose(1, 2)



        if cos is not None and sin is not None:

            query_states, key_states = apply_rotary_pos_emb(

                query_states, key_states, cos, sin

            )



        computed_kv = (key_states.clone(), value_states.clone())



        selective_cache_positions = None



        if shared_kv is not None:

            if merge_shared_kv and active_mask is not None:

                shared_key, shared_value = shared_kv

                mask = active_mask.unsqueeze(1).unsqueeze(-1)

                key_states = torch.where(mask, key_states, shared_key)

                value_states = torch.where(mask, value_states, shared_value)

            else:

                key_states, value_states = shared_kv

        elif past_key_value is not None:

            if reuse_kv and past_key_value.has_cache():

                key_states, value_states = past_key_value.get()

                if isinstance(past_key_value, SelectiveKVCache):

                    selective_cache_positions = past_key_value.get_positions()

            elif use_selective_cache and isinstance(past_key_value, SelectiveKVCache):

                if active_mask is not None and position_ids is not None:

                    key_states, value_states, selective_cache_positions = past_key_value.update_selective(

                        key_states, value_states, position_ids, active_mask

                    )

                else:

                    key_states, value_states = past_key_value.update(

                        key_states, value_states, self.layer_idx

                    )

            else:

                key_states, value_states = past_key_value.update(

                    key_states, value_states, self.layer_idx

                )



        if self.num_kv_groups > 1:

            key_states = self._repeat_kv(key_states, self.num_kv_groups)

            value_states = self._repeat_kv(value_states, self.num_kv_groups)



        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale



        if selective_cache_positions is not None:

            kv_len = key_states.shape[2]

            valid_mask = selective_cache_positions >= 0



            if attention_mask is not None:

                mask_floor = torch.finfo(attention_mask.dtype).min / 2

                kv_allowed = attention_mask.max(dim=-2).values.squeeze(1)

                kv_allowed = kv_allowed > mask_floor

                kv_len_total = kv_allowed.shape[1]

                pos_clamped = selective_cache_positions.clamp(min=0, max=kv_len_total - 1)

                kv_keep = kv_allowed.gather(1, pos_clamped)

                valid_mask = valid_mask & kv_keep



            valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)

            attn_weights = attn_weights.masked_fill(~valid_mask, float("-inf"))



            q_positions = position_ids.unsqueeze(-1)

            kv_positions = selective_cache_positions.unsqueeze(1)

            causal_mask = q_positions < kv_positions

            causal_mask = causal_mask.unsqueeze(1)

            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))



        elif attention_mask is not None:

            if attn_weights.shape[-1] == attention_mask.shape[-1]:

                attn_weights = attn_weights + attention_mask

            elif attention_mask.shape[-1] > attn_weights.shape[-1]:

                attn_weights = attn_weights + attention_mask[..., :attn_weights.shape[-1]]



        if active_mask is not None and apply_active_mask_to_kv and selective_cache_positions is None:

            kv_len = key_states.shape[2]

            q_len = query_states.shape[2]



            if q_len == kv_len and active_mask.shape[1] == kv_len:

                active_mask_expanded = active_mask.unsqueeze(1).unsqueeze(2)



                diag_mask = torch.eye(q_len, dtype=torch.bool, device=attn_weights.device)

                diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)

                combined_mask = active_mask_expanded | diag_mask



                mask_value = -1e9



                attn_weights = attn_weights.masked_fill(

                    ~combined_mask, mask_value

                )



        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(

            query_states.dtype

        )



        attn_weights = self.attention_dropout(attn_weights)



        attn_output = torch.matmul(attn_weights, value_states)



        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)



        attn_output = self.o_proj(attn_output)



        attn_weights_out = attn_weights if output_attentions else None

        kv_cache_out = past_key_value if use_cache else None



        return attn_output, attn_weights_out, kv_cache_out, computed_kv



    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

        if n_rep == 1:

            return hidden_states



        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

        hidden_states = hidden_states.unsqueeze(2).expand(

            batch, num_kv_heads, n_rep, seq_len, head_dim

        )

        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)





def create_causal_mask(

    seq_len: int,

    device: torch.device,

    dtype: torch.dtype = torch.float32,

) -> torch.Tensor:

    mask = torch.triu(

        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),

        diagonal=1,

    )

    return mask.unsqueeze(0).unsqueeze(0)





def create_attention_mask(

    attention_mask: torch.Tensor,

    seq_len: int,

    is_causal: bool = True,

) -> torch.Tensor:

    batch_size = attention_mask.shape[0]

    device = attention_mask.device

    dtype = attention_mask.dtype



    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)



    expanded_mask = (1.0 - expanded_mask.float()) * torch.finfo(torch.float32).min



    if is_causal:

        causal_mask = create_causal_mask(seq_len, device, dtype=torch.float32)

        expanded_mask = expanded_mask + causal_mask



    return expanded_mask
