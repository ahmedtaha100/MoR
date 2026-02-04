import pytest
import torch

import sys
sys.path.insert(0, '/Users/ahmedtaeha/Desktop/General/GitHub/MoR/mor/src')

from model.config import MoRConfig, get_mor_135m_config
from model.router import (
    ExpertChoiceRouter,
    TokenChoiceRouter,
    MoRRouter,
    compute_router_z_loss,
)


class TestExpertChoiceRouter:
    def test_output_shape(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=0)

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        router_weights, selected_mask, aux_loss = router(
            hidden_states, return_aux_loss=True
        )

        assert router_weights.shape == (batch, seq)
        assert selected_mask.shape == (batch, seq)
        assert selected_mask.dtype == torch.bool

    def test_capacity_filtering(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=0)
        router.eval()

        batch, seq = 1, 100
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        router_weights, selected_mask, _ = router(
            hidden_states, return_aux_loss=False
        )

        capacity = router.capacity
        expected_selected = int(capacity * seq)
        actual_selected = selected_mask.sum().item()

        assert abs(actual_selected - expected_selected) <= 2

    def test_hierarchical_filtering(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=1)
        router.eval()

        batch, seq = 1, 100
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        active_mask = torch.zeros(batch, seq, dtype=torch.bool)
        active_mask[:, :50] = True

        router_weights, selected_mask, _ = router(
            hidden_states, active_mask=active_mask, return_aux_loss=False
        )

        assert (selected_mask & ~active_mask).sum() == 0

    def test_aux_loss_training(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=0)
        router.train()

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        _, _, aux_loss = router(hidden_states, return_aux_loss=True)

        assert aux_loss is not None
        assert aux_loss.shape == ()

    def test_gradient_flow(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=0)
        router.train()

        batch, seq = 2, 32
        hidden_states = torch.randn(batch, seq, config.hidden_dim, requires_grad=True)

        router_weights, selected_mask, aux_loss = router(
            hidden_states, return_aux_loss=True
        )

        loss = router_weights.sum()
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        assert hidden_states.grad is not None

    def test_inference_mode(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=0)
        router.eval()

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        router_weights, selected_mask = router.forward_inference(
            hidden_states, threshold=0.5
        )

        assert router_weights.shape == (batch, seq)
        assert selected_mask.shape == (batch, seq)


class TestTokenChoiceRouter:
    def test_output_shape(self):
        config = get_mor_135m_config()
        router = TokenChoiceRouter(config)

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        router_weights, assigned_depth, depth_mask, aux_loss = router(
            hidden_states, return_aux_loss=True
        )

        assert router_weights.shape == (batch, seq, config.num_recursion_steps)
        assert assigned_depth.shape == (batch, seq)
        assert depth_mask.shape == (batch, seq, config.num_recursion_steps)

    def test_depth_assignment(self):
        config = get_mor_135m_config()
        router = TokenChoiceRouter(config)

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        _, assigned_depth, depth_mask, _ = router(hidden_states)

        assert depth_mask.sum(dim=-1).eq(1).all()

        assert (assigned_depth >= 0).all()
        assert (assigned_depth < config.num_recursion_steps).all()

    def test_active_mask_for_depth(self):
        config = get_mor_135m_config()
        router = TokenChoiceRouter(config)

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        _, assigned_depth, _, _ = router(hidden_states)

        active_0 = router.get_active_mask_for_depth(assigned_depth, 0)
        assert active_0.all()

        max_depth = config.num_recursion_steps - 1
        active_max = router.get_active_mask_for_depth(assigned_depth, max_depth)
        expected_active = assigned_depth >= max_depth
        assert torch.equal(active_max, expected_active)

    def test_balancing_loss(self):
        config = get_mor_135m_config()
        router = TokenChoiceRouter(config)
        router.train()

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        _, _, _, aux_loss = router(hidden_states, return_aux_loss=True)

        assert aux_loss is not None
        assert aux_loss.shape == ()

    def test_gradient_flow(self):
        config = get_mor_135m_config()
        router = TokenChoiceRouter(config)
        router.train()

        batch, seq = 2, 32
        hidden_states = torch.randn(batch, seq, config.hidden_dim, requires_grad=True)

        router_weights, _, _, aux_loss = router(hidden_states, return_aux_loss=True)

        loss = router_weights.sum()
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        assert hidden_states.grad is not None


class TestMoRRouter:
    def test_expert_choice_routing(self):
        config = get_mor_135m_config()
        config.router_type = "expert_choice"
        router = MoRRouter(config)
        router.train()

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        for r in range(config.num_recursion_steps):
            active_mask = torch.ones(batch, seq, dtype=torch.bool)
            weights, mask, aux_loss = router.route_expert_choice(
                hidden_states, recursion_step=r, active_mask=active_mask
            )

            assert weights.shape == (batch, seq)
            assert mask.shape == (batch, seq)

    def test_token_choice_routing(self):
        config = get_mor_135m_config()
        config.router_type = "token_choice"
        router = MoRRouter(config)

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        weights, assigned_depth, depth_mask, aux_loss = router.route_token_choice(
            hidden_states
        )

        assert weights.shape == (batch, seq, config.num_recursion_steps)
        assert assigned_depth.shape == (batch, seq)

    def test_total_aux_loss(self):
        config = get_mor_135m_config()
        router = MoRRouter(config)

        aux_losses = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)]
        total = router.get_total_aux_loss(aux_losses)

        expected = config.router_aux_loss_coeff * sum(aux_losses) / len(aux_losses)
        assert torch.allclose(total, expected)

        aux_losses_with_none = [torch.tensor(0.1), None, torch.tensor(0.3)]
        total = router.get_total_aux_loss(aux_losses_with_none)
        valid_losses = [l for l in aux_losses_with_none if l is not None]
        expected = config.router_aux_loss_coeff * sum(valid_losses) / len(valid_losses)
        assert torch.allclose(total, expected)


class TestRouterZLoss:
    def test_2d_logits(self):
        logits = torch.randn(8, 64)
        z_loss = compute_router_z_loss(logits)

        assert z_loss.shape == ()
        assert z_loss >= 0

    def test_3d_logits(self):
        logits = torch.randn(8, 64, 4)
        z_loss = compute_router_z_loss(logits)

        assert z_loss.shape == ()
        assert z_loss >= 0


class TestRouterEdgeCases:
    def test_single_token(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=0)

        hidden_states = torch.randn(1, 1, config.hidden_dim)
        weights, mask, _ = router(hidden_states, return_aux_loss=False)

        assert weights.shape == (1, 1)
        assert mask.shape == (1, 1)

    def test_all_inactive_tokens(self):
        config = get_mor_135m_config()
        router = ExpertChoiceRouter(config, recursion_step=0)

        batch, seq = 1, 16
        hidden_states = torch.randn(batch, seq, config.hidden_dim)
        active_mask = torch.zeros(batch, seq, dtype=torch.bool)

        weights, mask, _ = router(
            hidden_states, active_mask=active_mask, return_aux_loss=False
        )

        assert mask.sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
