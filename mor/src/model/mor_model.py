from typing import Optional, Tuple, Dict, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass

from .config import MoRConfig
from .embeddings import MoREmbeddings, RMSNorm
from .transformer_block import TransformerBlock
from .recursive_block import RecursiveBlock
from .attention import create_causal_mask, KVCache


@dataclass
class MoRModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    routing_info: Optional[Dict] = None
    attentions: Optional[List] = None
    past_key_values: Optional[Dict] = None


class MoRModel(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config

        self.embeddings = MoREmbeddings(config)

        self.layers = nn.ModuleDict()

        if config.has_unique_first_layer:
            self.layers["first"] = TransformerBlock(config, layer_idx=0)
            recursive_start_idx = 1
        else:
            recursive_start_idx = 0

        self.recursive_block = RecursiveBlock(
            config,
            layer_idx_start=recursive_start_idx,
        )

        if config.has_unique_last_layer:
            last_idx = recursive_start_idx + config.num_shared_layers * config.num_recursion_steps
            self.layers["last"] = TransformerBlock(config, layer_idx=last_idx)

        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _recursive_block_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        past_key_values: Optional[Dict],
        use_cache: bool,
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor]]:
        return self.recursive_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        past_seq_len = 0
        if past_key_values is not None:
            if "first" in past_key_values and past_key_values["first"] is not None:
                first_kv = past_key_values["first"]
                if first_kv.key_cache is not None:
                    past_seq_len = first_kv.get_seq_len()
            elif "recursive" in past_key_values and past_key_values["recursive"]:
                recursive_kv = past_key_values["recursive"]
                for r_idx in recursive_kv:
                    r_cache = recursive_kv[r_idx]
                    if r_cache is not None and isinstance(r_cache, list) and len(r_cache) > 0:
                        layer_cache = r_cache[0]
                        if layer_cache is not None and hasattr(layer_cache, 'get_seq_len'):
                            past_seq_len = layer_cache.get_seq_len()
                            break

        if position_ids is None:
            position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embeddings(input_ids, position_ids)

        cos, sin = self.embeddings.get_rotary_emb(hidden_states, position_ids)

        total_seq_len = past_seq_len + seq_len
        if past_seq_len > 0:
            if seq_len == 1:
                causal_mask = None
            else:
                causal_mask = torch.zeros(
                    (1, 1, seq_len, total_seq_len),
                    device=device,
                    dtype=hidden_states.dtype
                )
                future_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
                    diagonal=1,
                )
                causal_mask[:, :, :, past_seq_len:] = future_mask

            if attention_mask is not None:
                padding_mask = (1.0 - attention_mask[:, :total_seq_len].float()).unsqueeze(1).unsqueeze(2) * torch.finfo(hidden_states.dtype).min
                if causal_mask is None:
                    causal_mask = padding_mask.expand(-1, -1, seq_len, -1)
                else:
                    causal_mask = causal_mask + padding_mask
        else:
            causal_mask = create_causal_mask(seq_len, device, hidden_states.dtype)
            if attention_mask is not None:
                padding_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * torch.finfo(hidden_states.dtype).min
                causal_mask = causal_mask + padding_mask

        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        total_aux_loss = torch.tensor(0.0, device=device)

        first_kv_cache = None
        last_kv_cache = None
        recursive_kv = None

        if past_key_values is not None:
            first_kv_cache = past_key_values.get("first")
            last_kv_cache = past_key_values.get("last")
            recursive_kv = past_key_values.get("recursive")

        new_first_kv = None
        new_last_kv = None

        if "first" in self.layers:
            if first_kv_cache is None and use_cache:
                first_kv_cache = KVCache()

            hidden_states, attn_weights, new_first_kv, _ = self.layers["first"](
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                past_key_value=first_kv_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        if self.gradient_checkpointing and self.training:
            hidden_states, aux_outputs, aux_loss = checkpoint(
                self._recursive_block_forward,
                hidden_states,
                causal_mask,
                position_ids,
                cos,
                sin,
                None,
                False,
                output_attentions,
                use_reentrant=False,
            )
        else:
            hidden_states, aux_outputs, aux_loss = self.recursive_block(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                past_key_values=recursive_kv,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        if aux_loss is not None:
            total_aux_loss = total_aux_loss + aux_loss

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        if output_attentions and aux_outputs.get("attentions"):
            all_attentions.extend(aux_outputs["attentions"])

        if "last" in self.layers:
            if last_kv_cache is None and use_cache:
                last_kv_cache = KVCache()

            hidden_states, attn_weights, new_last_kv, _ = self.layers["last"](
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                past_key_value=last_kv_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        hidden_states = self.norm(hidden_states)

        all_key_values = None
        if use_cache:
            all_key_values = {
                "first": new_first_kv,
                "last": new_last_kv,
                "recursive": aux_outputs.get("key_values"),
            }

        outputs = {
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "routing_info": aux_outputs.get("routing_info"),
            "key_values": all_key_values,
        }

        return hidden_states, outputs, total_aux_loss

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.token_embedding

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.token_embedding = value


class MoRForCausalLM(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config

        self.model = MoRModel(config)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[MoRModelOutput, Tuple]:
        hidden_states, aux_outputs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if aux_loss is not None and aux_loss.item() > 0:
                loss = loss + aux_loss

        if return_dict:
            return MoRModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                aux_loss=aux_loss,
                routing_info=aux_outputs.get("routing_info"),
                attentions=aux_outputs.get("attentions"),
                past_key_values=aux_outputs.get("key_values"),
            )
        else:
            return (loss, logits, hidden_states, aux_loss, aux_outputs.get("key_values"))

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated = input_ids.clone()
        past_key_values = None

        for step in range(max_new_tokens):
            if past_key_values is None or not use_cache:
                context = generated
                position_ids = None
            else:
                context = generated[:, -1:]
                position_ids = None

            if generated.shape[1] > self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
                past_key_values = None
                position_ids = None

            with torch.no_grad():
                outputs = self.forward(
                    context,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :]

                if use_cache and outputs.past_key_values is not None:
                    past_key_values = outputs.past_key_values

            if temperature != 1.0:
                logits = logits / temperature

            if do_sample:
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_recursion_depths(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        routing_info = outputs.routing_info
        if routing_info is None:
            return None

        return self.model.recursive_block.get_recursion_depth_per_token(routing_info)
