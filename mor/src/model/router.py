import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoRConfig


class ExpertChoiceRouter(nn.Module):
    def __init__(self, config: MoRConfig, recursion_step: int = 0):
        super().__init__()
        self.config = config
        self.recursion_step = recursion_step

        capacity_schedule = config.get_capacity_schedule()
        self.base_capacity = capacity_schedule[recursion_step]
        self.capacity = self.base_capacity
        self.capacity_warmup_steps = config.capacity_warmup_steps
        self.router_alpha = config.router_alpha
        self.z_loss_coeff = config.router_z_loss_coeff

        if config.router_architecture == "linear":
            self.router = nn.Linear(config.hidden_dim, 1, bias=False)
        elif config.router_architecture == "mlp":
            self.router = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, 1),
            )
        elif config.router_architecture == "wide_mlp":
            self.router = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )

        self.activation = config.router_activation
        self._current_step = 0

    def set_training_step(self, step: int):
        self._current_step = step
        if self.capacity_warmup_steps > 0 and step < self.capacity_warmup_steps:
            warmup_factor = step / self.capacity_warmup_steps
            self.capacity = self.base_capacity * warmup_factor + (1.0 - warmup_factor) * 1.0
        else:
            self.capacity = self.base_capacity

    def forward(
        self,
        hidden_states: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        router_logits = self.router(hidden_states).squeeze(-1)

        scaled_logits = router_logits * self.router_alpha

        if self.activation == "sigmoid":
            router_scores = torch.sigmoid(scaled_logits)
        elif self.activation == "tanh":
            router_scores = (torch.tanh(scaled_logits) + 1) / 2
        else:
            router_scores = torch.sigmoid(scaled_logits)

        if active_mask is not None:
            masked_scores = router_scores.clone()
            masked_scores[~active_mask] = float("-inf")
        else:
            masked_scores = router_scores
            active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)

        num_active = active_mask.sum(dim=-1)

        k_per_batch = (self.capacity * num_active).long().clamp(min=1)

        selected_mask = torch.zeros_like(router_scores, dtype=torch.bool)

        for b in range(batch_size):
            k = k_per_batch[b].item()
            if k > 0:
                batch_scores = masked_scores[b]
                _, top_indices = torch.topk(batch_scores, k=min(k, num_active[b].item()))
                selected_mask[b, top_indices] = True

        router_weights = router_scores * selected_mask.float()

        aux_loss = None
        if return_aux_loss and self.training:
            targets = selected_mask.float()

            if active_mask is not None:
                active_logits = router_logits[active_mask]
                active_targets = targets[active_mask]
            else:
                active_logits = router_logits.view(-1)
                active_targets = targets.view(-1)

            if active_logits.numel() > 0:
                bce_loss = F.binary_cross_entropy_with_logits(
                    active_logits, active_targets, reduction="mean"
                )
                z_loss = compute_router_z_loss(router_logits)
                aux_loss = bce_loss + self.z_loss_coeff * z_loss

        return router_weights, selected_mask, aux_loss

    def forward_inference(
        self,
        hidden_states: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        router_logits = self.router(hidden_states).squeeze(-1)

        scaled_logits = router_logits * self.router_alpha

        if self.activation == "sigmoid":
            router_scores = torch.sigmoid(scaled_logits)
        elif self.activation == "tanh":
            router_scores = (torch.tanh(scaled_logits) + 1) / 2
        else:
            router_scores = torch.sigmoid(scaled_logits)

        selected_mask = router_scores > threshold

        if active_mask is not None:
            selected_mask = selected_mask & active_mask

        router_weights = router_scores * selected_mask.float()

        return router_weights, selected_mask


class TokenChoiceRouter(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.num_recursions = config.num_recursion_steps
        self.router_alpha = config.router_alpha
        self.z_loss_coeff = config.router_z_loss_coeff

        if config.router_architecture == "linear":
            self.router = nn.Linear(config.hidden_dim, self.num_recursions, bias=False)
        elif config.router_architecture == "mlp":
            self.router = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, self.num_recursions),
            )
        elif config.router_architecture == "wide_mlp":
            self.router = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, self.num_recursions),
            )

        self.expert_bias = nn.Parameter(torch.zeros(self.num_recursions))
        self.use_bias_balancing = False

        self.activation = config.router_activation

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        router_logits = self.router(hidden_states)

        scaled_logits = router_logits * self.router_alpha

        if self.use_bias_balancing:
            router_logits_biased = scaled_logits + self.expert_bias
        else:
            router_logits_biased = scaled_logits

        if self.activation == "softmax":
            router_probs = F.softmax(scaled_logits, dim=-1)
        elif self.activation == "sigmoid":
            router_probs = torch.sigmoid(scaled_logits)
            router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)
        else:
            router_probs = F.softmax(scaled_logits, dim=-1)

        assigned_depth = router_logits_biased.argmax(dim=-1)

        depth_mask = torch.zeros_like(router_logits, dtype=torch.bool)
        depth_mask.scatter_(-1, assigned_depth.unsqueeze(-1), True)

        router_weights = router_probs

        aux_loss = None
        if return_aux_loss and self.training:
            balance_loss = self._compute_balancing_loss(router_probs, assigned_depth)
            z_loss = compute_router_z_loss(router_logits)
            aux_loss = balance_loss + self.z_loss_coeff * z_loss

        return router_weights, assigned_depth, depth_mask, aux_loss

    def _compute_balancing_loss(
        self,
        router_probs: torch.Tensor,
        assigned_depth: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_experts = router_probs.shape

        flat_probs = router_probs.view(-1, num_experts)
        flat_assigned = assigned_depth.view(-1)

        num_tokens = flat_probs.shape[0]

        one_hot = F.one_hot(flat_assigned, num_classes=num_experts).float()
        tokens_per_expert = one_hot.sum(dim=0)
        f = tokens_per_expert / num_tokens

        P = flat_probs.mean(dim=0)

        balance_loss = num_experts * (f * P).sum()

        return balance_loss

    def get_active_mask_for_depth(
        self,
        assigned_depth: torch.Tensor,
        current_depth: int,
    ) -> torch.Tensor:
        return assigned_depth >= current_depth


def compute_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    if router_logits.dim() == 2:
        return (router_logits ** 2).mean()
    else:
        log_z = torch.logsumexp(router_logits, dim=-1)
        return (log_z ** 2).mean()


class MoRRouter(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.router_type = config.router_type

        if config.router_type == "expert_choice":
            self.routers = nn.ModuleList([
                ExpertChoiceRouter(config, recursion_step=i)
                for i in range(config.num_recursion_steps)
            ])
        else:
            self.router = TokenChoiceRouter(config)

        self.aux_loss_coeff = config.router_aux_loss_coeff
        self.z_loss_coeff = config.router_z_loss_coeff

    def set_training_step(self, step: int):
        if self.router_type == "expert_choice":
            for router in self.routers:
                router.set_training_step(step)

    def route_expert_choice(
        self,
        hidden_states: torch.Tensor,
        recursion_step: int,
        active_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        router = self.routers[recursion_step]

        if is_training:
            return router(hidden_states, active_mask, return_aux_loss=True)
        else:
            weights, mask = router.forward_inference(hidden_states, active_mask)
            return weights, mask, None

    def route_token_choice(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.router(hidden_states, return_aux_loss=True)

    def get_total_aux_loss(self, aux_losses: list) -> torch.Tensor:
        valid_losses = [l for l in aux_losses if l is not None]
        if not valid_losses:
            if self.router_type == "expert_choice":
                device = next(self.routers[0].parameters()).device
            else:
                device = next(self.router.parameters()).device
            return torch.tensor(0.0, device=device)

        total_loss = sum(valid_losses) / len(valid_losses)
        return self.aux_loss_coeff * total_loss
