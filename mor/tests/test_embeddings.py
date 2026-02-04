import pytest
import torch
import math

import sys
sys.path.insert(0, '/Users/ahmedtaeha/Desktop/General/GitHub/MoR/mor/src')

from model.config import MoRConfig, get_mor_135m_config
from model.embeddings import (
    RMSNorm,
    RotaryEmbedding,
    TokenEmbedding,
    MoREmbeddings,
    rotate_half,
    apply_rotary_pos_emb,
)


class TestRMSNorm:
    def test_shape_preservation(self):
        hidden_dim = 768
        norm = RMSNorm(hidden_dim)

        for batch_size in [1, 4]:
            for seq_len in [8, 128]:
                x = torch.randn(batch_size, seq_len, hidden_dim)
                out = norm(x)
                assert out.shape == x.shape

    def test_normalization(self):
        hidden_dim = 768
        norm = RMSNorm(hidden_dim)

        x = torch.randn(2, 16, hidden_dim)
        out = norm(x)

        assert not torch.allclose(out, x)

        rms = out.pow(2).mean(dim=-1).sqrt()
        assert rms.shape == (2, 16)

    def test_gradient_flow(self):
        hidden_dim = 768
        norm = RMSNorm(hidden_dim)

        x = torch.randn(2, 16, hidden_dim, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None


class TestRotaryEmbedding:
    def test_output_shape(self):
        hidden_dim = 768
        num_heads = 12
        head_dim = hidden_dim // num_heads
        max_seq_len = 2048

        rope = RotaryEmbedding(head_dim, max_seq_len)

        x = torch.randn(2, 128, hidden_dim)
        position_ids = torch.arange(128).unsqueeze(0).expand(2, -1)

        cos, sin = rope(x, position_ids)

        assert cos.shape[1:] == (128, head_dim)
        assert sin.shape[1:] == (128, head_dim)

    def test_position_dependent(self):
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 2048)

        x = torch.randn(1, 4, 768)
        pos1 = torch.tensor([[0, 1, 2, 3]])
        pos2 = torch.tensor([[10, 11, 12, 13]])

        cos1, sin1 = rope(x, pos1)
        cos2, sin2 = rope(x, pos2)

        assert not torch.allclose(cos1, cos2)
        assert not torch.allclose(sin1, sin2)


class TestRotateHalf:
    def test_rotation(self):
        x = torch.tensor([[[[1., 2., 3., 4.]]]])

        rotated = rotate_half(x)

        expected = torch.tensor([[[[-3., -4., 1., 2.]]]])
        assert torch.allclose(rotated, expected)


class TestApplyRotaryPosEmb:
    def test_output_shape(self):
        batch, heads, seq, head_dim = 2, 12, 128, 64

        q = torch.randn(batch, heads, seq, head_dim)
        k = torch.randn(batch, heads, seq, head_dim)
        cos = torch.randn(1, seq, head_dim)
        sin = torch.randn(1, seq, head_dim)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_gradient_flow(self):
        batch, heads, seq, head_dim = 2, 8, 32, 64

        q = torch.randn(batch, heads, seq, head_dim, requires_grad=True)
        k = torch.randn(batch, heads, seq, head_dim, requires_grad=True)
        cos = torch.randn(1, seq, head_dim)
        sin = torch.randn(1, seq, head_dim)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        loss = (q_rot.sum() + k_rot.sum())
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None


class TestMoREmbeddings:
    def test_output_shape(self):
        config = get_mor_135m_config()
        embeddings = MoREmbeddings(config)

        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        out = embeddings(input_ids, position_ids)

        assert out.shape == (batch_size, seq_len, config.hidden_dim)

    def test_rotary_emb(self):
        config = get_mor_135m_config()
        embeddings = MoREmbeddings(config)

        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        hidden = embeddings(input_ids, position_ids)
        cos, sin = embeddings.get_rotary_emb(hidden, position_ids)

        head_dim = config.hidden_dim // config.num_attention_heads
        assert cos.shape[1:] == (seq_len, head_dim)
        assert sin.shape[1:] == (seq_len, head_dim)

    def test_gradient_flow(self):
        config = get_mor_135m_config()
        embeddings = MoREmbeddings(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        out = embeddings(input_ids, position_ids)
        loss = out.sum()
        loss.backward()

        assert embeddings.token_embedding.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
