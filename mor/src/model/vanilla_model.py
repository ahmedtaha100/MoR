from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .config import MoRConfig
from .embeddings import MoREmbeddings, RMSNorm
from .transformer_block import TransformerBlock
from .attention import create_causal_mask


@dataclass
class VanillaModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[list] = None


class VanillaTransformer(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config

        self.embeddings = MoREmbeddings(config)

        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embeddings(input_ids, position_ids)

        cos, sin = self.embeddings.get_rotary_emb(hidden_states, position_ids)

        if attention_mask is None:
            causal_mask = create_causal_mask(seq_len, device, hidden_states.dtype)
        else:
            causal_mask = create_causal_mask(seq_len, device, hidden_states.dtype)
            padding_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * torch.finfo(hidden_states.dtype).min
            causal_mask = causal_mask + padding_mask

        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                past_key_value=past_kv,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_attentions and layer_outputs[1] is not None:
                all_attentions.append(layer_outputs[1])
            if use_cache:
                all_key_values.append(layer_outputs[2])

        hidden_states = self.norm(hidden_states)

        outputs = {
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "key_values": all_key_values,
        }

        return hidden_states, outputs


class VanillaForCausalLM(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config

        self.model = VanillaTransformer(config)

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
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[VanillaModelOutput, Tuple]:
        hidden_states, aux_outputs = self.model(
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

        if return_dict:
            return VanillaModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                attentions=aux_outputs.get("attentions"),
            )
        else:
            return (loss, logits, hidden_states)

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
