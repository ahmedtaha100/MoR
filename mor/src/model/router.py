import math

from typing import Optional, Tuple, Literal, Dict, List



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



        self.aux_mode = config.expert_choice_aux_mode

        self.use_aux_router = self.aux_mode == "aux_router"

        if self.use_aux_router:

            if config.router_architecture == "linear":

                self.aux_router = nn.Linear(config.hidden_dim, 1, bias=False)

            elif config.router_architecture == "mlp":

                self.aux_router = nn.Sequential(

                    nn.Linear(config.hidden_dim, config.hidden_dim // 4),

                    nn.GELU(),

                    nn.Linear(config.hidden_dim // 4, 1),

                )

            elif config.router_architecture == "wide_mlp":

                self.aux_router = nn.Sequential(

                    nn.Linear(config.hidden_dim, config.hidden_dim),

                    nn.GELU(),

                    nn.Linear(config.hidden_dim, 1),

                )

        else:

            self.aux_router = None



        self.activation = config.router_activation

        self._current_step = 0

        self.last_aux_scores = None

        self.last_aux_selected_mask = None



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



        aux_logits = None

        aux_scores = None

        if self.use_aux_router:

            aux_input = hidden_states.detach() if self.training else hidden_states

            aux_logits = self.aux_router(aux_input).squeeze(-1)

            aux_scaled = aux_logits * self.router_alpha

            if self.activation == "sigmoid":

                aux_scores = torch.sigmoid(aux_scaled)

            elif self.activation == "tanh":

                aux_scores = (torch.tanh(aux_scaled) + 1) / 2

            else:

                aux_scores = torch.sigmoid(aux_scaled)



        scores_for_selection = aux_scores if (self.use_aux_router and not self.training) else router_scores



        if active_mask is not None:

            masked_scores = scores_for_selection.clone()

            masked_scores[~active_mask] = float("-inf")

        else:

            masked_scores = scores_for_selection

            active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)



        num_active = active_mask.sum(dim=-1)



        k_per_batch = self.capacity * num_active



        selected_mask = torch.zeros_like(router_scores, dtype=torch.bool)



        for b in range(batch_size):

            num_active_b = int(num_active[b].item())

            if num_active_b == 0:

                continue

            k = int(k_per_batch[b].item())

            if k < 1:

                k = 1

            if k > num_active_b:

                k = num_active_b

            batch_scores = masked_scores[b]

            _, top_indices = torch.topk(batch_scores, k=k)

            selected_mask[b, top_indices] = True



        aux_selected_mask = None

        if aux_scores is not None:

            if active_mask is not None:

                aux_masked_scores = aux_scores.clone()

                aux_masked_scores[~active_mask] = float("-inf")

            else:

                aux_masked_scores = aux_scores

            aux_selected_mask = torch.zeros_like(router_scores, dtype=torch.bool)

            for b in range(batch_size):

                num_active_b = int(num_active[b].item())

                if num_active_b == 0:

                    continue

                k = int(k_per_batch[b].item())

                if k < 1:

                    k = 1

                if k > num_active_b:

                    k = num_active_b

                batch_scores = aux_masked_scores[b]

                _, top_indices = torch.topk(batch_scores, k=k)

                aux_selected_mask[b, top_indices] = True



        self.last_aux_scores = aux_scores

        self.last_aux_selected_mask = aux_selected_mask



        router_weights = scores_for_selection * selected_mask.float()



        aux_loss = None

        if return_aux_loss and self.training:

            targets = selected_mask.float()

            aux_logits_target = None

            if self.aux_mode == "loss":

                aux_logits_target = router_logits

            elif self.aux_mode == "aux_router":

                aux_logits_target = aux_logits

            if aux_logits_target is not None:

                if active_mask is not None:

                    active_logits = aux_logits_target[active_mask]

                    active_targets = targets[active_mask]

                else:

                    active_logits = aux_logits_target.view(-1)

                    active_targets = targets.view(-1)

                if active_logits.numel() > 0:

                    bce_loss = F.binary_cross_entropy_with_logits(

                        active_logits, active_targets, reduction="mean"

                    )

                    z_loss = compute_router_z_loss(aux_logits_target)

                    aux_loss = bce_loss + self.z_loss_coeff * z_loss



        return router_weights, selected_mask, aux_loss



    def forward_inference(

        self,

        hidden_states: torch.Tensor,

        active_mask: Optional[torch.Tensor] = None,

        threshold: float = 0.5,

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, _ = hidden_states.shape

        if self.use_aux_router:

            router_logits = self.aux_router(hidden_states).squeeze(-1)

        else:

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

        k_per_batch = self.capacity * num_active

        selected_mask = torch.zeros_like(router_scores, dtype=torch.bool)

        for b in range(batch_size):

            num_active_b = int(num_active[b].item())

            if num_active_b == 0:

                continue

            k = int(k_per_batch[b].item())

            if k < 1:

                k = 1

            if k > num_active_b:

                k = num_active_b

            batch_scores = masked_scores[b]

            _, top_indices = torch.topk(batch_scores, k=k)

            selected_mask[b, top_indices] = True

        router_weights = router_scores * selected_mask.float()

        return router_weights, selected_mask





class TokenChoiceRouter(nn.Module):

    def __init__(self, config: MoRConfig):

        super().__init__()

        self.config = config

        self.num_recursions = config.num_recursion_steps

        self.router_alpha = config.router_alpha

        self.z_loss_coeff = config.router_z_loss_coeff

        self.balance_mode = config.token_choice_balance_mode

        self.bias_update_rate = config.token_choice_bias_update_rate



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



        self.expert_bias = nn.Parameter(torch.zeros(self.num_recursions), requires_grad=False)

        self.use_bias_balancing = self.balance_mode == "loss_free"



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



        if self.training and self.use_bias_balancing and self.bias_update_rate > 0:

            with torch.no_grad():

                flat_assigned = assigned_depth.view(-1)

                one_hot = F.one_hot(flat_assigned, num_classes=self.num_recursions).float()

                counts = one_hot.sum(dim=0)

                avg = counts.mean()

                error = avg - counts

                update = self.bias_update_rate * torch.sign(error)

                self.expert_bias.add_(update)



        depth_mask = torch.zeros_like(router_logits, dtype=torch.bool)

        depth_mask.scatter_(-1, assigned_depth.unsqueeze(-1), True)



        router_weights = router_probs



        aux_loss = None

        if return_aux_loss and self.training and self.balance_mode == "loss":

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



        return router(hidden_states, active_mask, return_aux_loss=is_training)



    def route_token_choice(

        self,

        hidden_states: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        return self.router(hidden_states, return_aux_loss=True)



    def compute_metrics(self, routing_info: Dict) -> Dict[str, torch.Tensor]:

        if routing_info is None:

            return {}

        if self.router_type == "expert_choice":

            selected_masks = routing_info.get("selected_masks")

            if not selected_masks:

                return {}

            total = selected_masks[0].numel()

            if total > 0:

                dead_ratio = 1.0 - selected_masks[-1].float().sum() / total

            else:

                dead_ratio = torch.tensor(0.0, device=selected_masks[0].device)

            metrics = {"router/dead_token_ratio": dead_ratio}

            accs = []

            for i, mask in enumerate(selected_masks):

                if i < len(self.routers):

                    aux_mask = self.routers[i].last_aux_selected_mask

                    if aux_mask is not None and aux_mask.shape == mask.shape:

                        accs.append((aux_mask == mask).float().mean())

            if accs:

                metrics["router/sampling_accuracy"] = torch.stack(accs).mean()

            return metrics

        else:

            depth_mask = routing_info.get("depth_mask")

            if depth_mask is None:

                return {}

            counts = depth_mask.float().sum(dim=(0, 1))

            total = counts.sum()

            if total > 0:

                frac = counts / total

                target = 1.0 / self.config.num_recursion_steps

                max_vio = (frac - target).abs().max()

            else:

                max_vio = torch.tensor(0.0, device=depth_mask.device)

            return {"router/max_vio": max_vio}



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
