from typing import Optional, Tuple, List, Dict



import torch

import torch.nn as nn



from .config import MoRConfig

from .transformer_block import TransformerBlock, SharedTransformerBlock

from .router import MoRRouter, ExpertChoiceRouter, TokenChoiceRouter

from .attention import KVCache, SelectiveKVCache





class RecursiveBlock(nn.Module):

    def __init__(self, config: MoRConfig, layer_idx_start: int = 0):

        super().__init__()

        self.config = config

        self.num_recursions = config.num_recursion_steps

        self.router_type = config.router_type

        self.kv_cache_strategy = config.kv_cache_strategy



        self.shared_block = SharedTransformerBlock(config, layer_idx=layer_idx_start)



        self.router = MoRRouter(config)



    def _derive_base_active_mask(

        self,

        attention_mask: Optional[torch.Tensor],

        position_ids: Optional[torch.Tensor],

        batch_size: int,

        seq_len: int,

        device: torch.device,

    ) -> torch.Tensor:

        if attention_mask is None or position_ids is None:

            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)



        if attention_mask.dim() == 4:

            attn = attention_mask[:, 0]

        elif attention_mask.dim() == 3:

            attn = attention_mask

        else:

            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)



        if attn.shape[0] == 1 and batch_size > 1:

            attn = attn.expand(batch_size, -1, -1).contiguous()

        kv_len = attn.shape[-1]

        pos_clamped = position_ids.clamp(min=0, max=kv_len - 1)

        diag_vals = attn.gather(2, pos_clamped.unsqueeze(-1)).squeeze(-1)

        mask_floor = torch.finfo(attention_mask.dtype).min / 2

        return diag_vals > mask_floor



    def _get_layer_indices(self, recursion_step: int) -> List[int]:

        d = self.config.num_shared_layers

        n = self.config.num_recursion_steps

        strategy = self.config.sharing_strategy

        if strategy == "sequence" or strategy == "middle_sequence":

            base = recursion_step * d

            return [(base + i) // n for i in range(d)]

        return list(range(d))



    def forward(

        self,

        hidden_states: torch.Tensor,

        attention_mask: Optional[torch.Tensor] = None,

        position_ids: Optional[torch.Tensor] = None,

        cos: Optional[torch.Tensor] = None,

        sin: Optional[torch.Tensor] = None,

        past_key_values: Optional[Dict[int, List[KVCache]]] = None,

        use_cache: bool = False,

        output_attentions: bool = False,

    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor]]:

        if self.router_type == "expert_choice":

            return self._forward_expert_choice(

                hidden_states, attention_mask, position_ids,

                cos, sin, past_key_values, use_cache, output_attentions

            )

        else:

            return self._forward_token_choice(

                hidden_states, attention_mask, position_ids,

                cos, sin, past_key_values, use_cache, output_attentions

            )



    def _forward_expert_choice(

        self,

        hidden_states: torch.Tensor,

        attention_mask: Optional[torch.Tensor] = None,

        position_ids: Optional[torch.Tensor] = None,

        cos: Optional[torch.Tensor] = None,

        sin: Optional[torch.Tensor] = None,

        past_key_values: Optional[Dict] = None,

        use_cache: bool = False,

        output_attentions: bool = False,

    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor]]:

        batch_size, seq_len, _ = hidden_states.shape

        device = hidden_states.device



        all_attentions = [] if output_attentions else None

        all_key_values = {} if use_cache else None

        aux_losses = []

        routing_info = {

            "selected_masks": [],

            "router_weights": [],

        }



        output_states = hidden_states.clone()



        base_active_mask = self._derive_base_active_mask(

            attention_mask, position_ids, batch_size, seq_len, device

        )

        active_mask = base_active_mask.clone()



        exit_states = hidden_states.clone()



        first_recursion_kv = None

        first_recursion_computed_kvs = None



        is_selective_mode = self.kv_cache_strategy == "selective"

        is_shared_mode = self.kv_cache_strategy == "shared"



        for r in range(self.num_recursions):

            router_weights, selected_mask, aux_loss = self.router.route_expert_choice(

                hidden_states,

                recursion_step=r,

                active_mask=active_mask,

                is_training=self.training,

            )



            if aux_loss is not None:

                aux_losses.append(aux_loss)



            routing_info["selected_masks"].append(selected_mask)

            routing_info["router_weights"].append(router_weights)



            is_shared_first = is_shared_mode and r == 0

            is_shared_reuse = is_shared_mode and r > 0



            if is_shared_reuse and first_recursion_kv is not None:

                kv_cache = first_recursion_kv

            else:

                kv_cache = past_key_values.get(r) if past_key_values else None



            shared_kvs_for_block = None

            if is_shared_reuse and first_recursion_computed_kvs is not None:

                shared_kvs_for_block = first_recursion_computed_kvs



            if is_shared_first:

                block_active_mask = None

                apply_kv_mask = False

                use_selective = False

            elif is_shared_reuse:

                block_active_mask = None

                apply_kv_mask = False

                use_selective = False

            elif is_selective_mode:

                block_active_mask = selected_mask

                apply_kv_mask = True

                use_selective = True

            else:

                block_active_mask = selected_mask

                apply_kv_mask = True

                use_selective = False



            def create_selective_cache():

                return SelectiveKVCache()



            layer_indices = self._get_layer_indices(r)

            block_output, attentions, key_values, computed_kvs = self.shared_block(

                hidden_states=hidden_states,

                attention_mask=attention_mask,

                position_ids=position_ids,

                cos=cos,

                sin=sin,

                past_key_values=kv_cache,

                use_cache=use_cache,

                active_mask=block_active_mask,

                output_attentions=output_attentions,

                reuse_kv=is_shared_reuse,

                apply_active_mask_to_kv=apply_kv_mask,

                use_selective_cache=use_selective,

                create_cache_fn=create_selective_cache if use_selective and kv_cache is None else None,

                shared_kvs=shared_kvs_for_block,
                layer_indices=layer_indices,

            )



            if is_shared_first:

                first_recursion_computed_kvs = computed_kvs



            if output_attentions:

                all_attentions.append(attentions)

            if use_cache:

                if is_shared_mode:

                    if r == 0:

                        first_recursion_kv = key_values

                        all_key_values[0] = key_values

                else:

                    all_key_values[r] = key_values



            router_weights_expanded = router_weights.unsqueeze(-1)

            selected_expanded = selected_mask.unsqueeze(-1)



            weighted_output = router_weights_expanded * block_output



            hidden_states = torch.where(

                selected_expanded,

                weighted_output + hidden_states,

                hidden_states,

            )



            exiting_mask = active_mask & ~selected_mask

            exit_states = torch.where(

                exiting_mask.unsqueeze(-1),

                hidden_states,

                exit_states,

            )



            active_mask = selected_mask & base_active_mask



        output_states = torch.where(

            active_mask.unsqueeze(-1),

            hidden_states,

            exit_states,

        )



        total_aux_loss = self.router.get_total_aux_loss(aux_losses)



        aux_outputs = {

            "attentions": all_attentions,

            "key_values": all_key_values,

            "routing_info": routing_info,

        }



        return output_states, aux_outputs, total_aux_loss



    def _forward_token_choice(

        self,

        hidden_states: torch.Tensor,

        attention_mask: Optional[torch.Tensor] = None,

        position_ids: Optional[torch.Tensor] = None,

        cos: Optional[torch.Tensor] = None,

        sin: Optional[torch.Tensor] = None,

        past_key_values: Optional[Dict] = None,

        use_cache: bool = False,

        output_attentions: bool = False,

    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor]]:

        batch_size, seq_len, _ = hidden_states.shape

        device = hidden_states.device



        h1 = hidden_states



        router_weights, assigned_depth, depth_mask, aux_loss = self.router.route_token_choice(

            hidden_states

        )



        aux_losses = [aux_loss] if aux_loss is not None else []



        all_attentions = [] if output_attentions else None

        all_key_values = {} if use_cache else None

        routing_info = {

            "assigned_depth": assigned_depth,

            "depth_mask": depth_mask,

            "router_weights": router_weights,

        }



        first_recursion_kv = None

        first_recursion_computed_kvs = None



        is_selective_mode = self.kv_cache_strategy == "selective"

        is_shared_mode = self.kv_cache_strategy == "shared"



        base_active_mask = self._derive_base_active_mask(

            attention_mask, position_ids, batch_size, seq_len, device

        )



        for r in range(self.num_recursions):

            active_mask = self.router.router.get_active_mask_for_depth(assigned_depth, r)

            active_mask = active_mask & base_active_mask



            r_weight = router_weights[:, :, r]



            is_shared_first = is_shared_mode and r == 0

            is_shared_reuse = is_shared_mode and r > 0



            if is_shared_reuse and first_recursion_kv is not None:

                kv_cache = first_recursion_kv

            else:

                kv_cache = past_key_values.get(r) if past_key_values else None



            shared_kvs_for_block = None

            if is_shared_reuse and first_recursion_computed_kvs is not None:

                shared_kvs_for_block = first_recursion_computed_kvs



            if is_shared_first:

                block_active_mask = None

                apply_kv_mask = False

                use_selective = False

            elif is_shared_reuse:

                block_active_mask = None

                apply_kv_mask = False

                use_selective = False

            elif is_selective_mode:

                block_active_mask = active_mask

                apply_kv_mask = True

                use_selective = True

            else:

                block_active_mask = active_mask

                apply_kv_mask = True

                use_selective = False



            def create_selective_cache():

                return SelectiveKVCache()



            layer_indices = self._get_layer_indices(r)

            block_output, attentions, key_values, computed_kvs = self.shared_block(

                hidden_states=hidden_states,

                attention_mask=attention_mask,

                position_ids=position_ids,

                cos=cos,

                sin=sin,

                past_key_values=kv_cache,

                use_cache=use_cache,

                active_mask=block_active_mask,

                output_attentions=output_attentions,

                reuse_kv=is_shared_reuse,

                apply_active_mask_to_kv=apply_kv_mask,

                use_selective_cache=use_selective,

                create_cache_fn=create_selective_cache if use_selective and kv_cache is None else None,

                shared_kvs=shared_kvs_for_block,
                layer_indices=layer_indices,

            )



            if is_shared_first:

                first_recursion_computed_kvs = computed_kvs



            if output_attentions:

                all_attentions.append(attentions)

            if use_cache:

                if is_shared_mode:

                    if r == 0:

                        first_recursion_kv = key_values

                        all_key_values[0] = key_values

                else:

                    all_key_values[r] = key_values



            r_weight_expanded = r_weight.unsqueeze(-1)

            active_expanded = active_mask.unsqueeze(-1)



            is_final = (assigned_depth == r)

            is_final_expanded = is_final.unsqueeze(-1)



            weighted_output = r_weight_expanded * block_output



            final_output = weighted_output + h1

            intermediate_output = weighted_output



            hidden_states = torch.where(

                active_expanded,

                torch.where(is_final_expanded, final_output, intermediate_output),

                hidden_states,

            )



        total_aux_loss = self.router.get_total_aux_loss(aux_losses)



        aux_outputs = {

            "attentions": all_attentions,

            "key_values": all_key_values,

            "routing_info": routing_info,

        }



        return hidden_states, aux_outputs, total_aux_loss



    def get_recursion_depth_per_token(

        self,

        routing_info: Dict,

    ) -> torch.Tensor:

        if self.router_type == "token_choice":

            return routing_info["assigned_depth"] + 1

        else:

            selected_masks = routing_info["selected_masks"]

            batch_size = selected_masks[0].shape[0]

            seq_len = selected_masks[0].shape[1]

            device = selected_masks[0].device



            depths = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)



            for r, mask in enumerate(selected_masks):

                depths = torch.where(mask, torch.tensor(r + 1, device=device), depths)



            return depths
