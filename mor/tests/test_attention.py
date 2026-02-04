import pytest
import torch
import math

import sys
sys.path.insert(0, '/Users/ahmedtaeha/Desktop/General/GitHub/MoR/mor/src')

from model.config import MoRConfig, get_mor_135m_config
from model.attention import (
    KVCache,
    MoRAttention,
    create_causal_mask,
    create_attention_mask,
)
from model.embeddings import RotaryEmbedding


class TestCausalMask:
    def test_shape(self):
        seq_len = 128
        mask = create_causal_mask(seq_len, torch.device("cpu"), torch.float32)

        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_causality(self):
        seq_len = 4
        mask = create_causal_mask(seq_len, torch.device("cpu"), torch.float32)

        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[0, 0, i, j] == float("-inf")
                else:
                    assert mask[0, 0, i, j] == 0.0


class TestKVCache:
    def test_initialization(self):
        cache = KVCache()

        assert cache.key_cache is None
        assert cache.value_cache is None
        assert cache.seq_len == 0

    def test_update(self):
        cache = KVCache()

        batch, heads, seq, head_dim = 2, 4, 16, 64
        k1 = torch.randn(batch, heads, seq, head_dim)
        v1 = torch.randn(batch, heads, seq, head_dim)

        k_out, v_out = cache.update(k1, v1, layer_idx=0)

        assert cache.seq_len == seq
        assert torch.equal(k_out, k1)
        assert torch.equal(v_out, v1)

        k2 = torch.randn(batch, heads, 1, head_dim)
        v2 = torch.randn(batch, heads, 1, head_dim)

        k_out, v_out = cache.update(k2, v2, layer_idx=0)

        assert cache.seq_len == seq + 1
        assert k_out.shape == (batch, heads, seq + 1, head_dim)


class TestMoRAttention:
    def test_output_shape(self):
        config = get_mor_135m_config()
        attn = MoRAttention(config, layer_idx=0)

        batch, seq = 2, 128
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)

        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        outputs = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
        )

        out = outputs[0]
        assert out.shape == (batch, seq, config.hidden_dim)

    def test_with_attention_weights(self):
        config = get_mor_135m_config()
        attn = MoRAttention(config, layer_idx=0)

        batch, seq = 2, 32
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        outputs = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            output_attentions=True,
        )

        out, attn_weights = outputs[0], outputs[1]
        assert attn_weights is not None
        assert attn_weights.shape == (batch, config.num_attention_heads, seq, seq)

    def test_gqa(self):
        config = get_mor_135m_config()
        assert config.num_attention_heads > config.num_kv_heads

        attn = MoRAttention(config, layer_idx=0)

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        outputs = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
        )

        out = outputs[0]
        assert out.shape == (batch, seq, config.hidden_dim)

    def test_kv_cache(self):
        config = get_mor_135m_config()
        attn = MoRAttention(config, layer_idx=0)

        batch = 2
        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)

        seq1 = 32
        hidden1 = torch.randn(batch, seq1, config.hidden_dim)
        pos1 = torch.arange(seq1).unsqueeze(0).expand(batch, -1)
        cos1, sin1 = rope(hidden1, pos1)
        mask1 = create_causal_mask(seq1, hidden1.device, hidden1.dtype)

        kv = KVCache()

        outputs1 = attn(
            hidden1,
            attention_mask=mask1,
            position_ids=pos1,
            cos=cos1,
            sin=sin1,
            past_key_value=kv,
            use_cache=True,
        )

        out1 = outputs1[0]
        kv_out = outputs1[-1]
        assert kv.seq_len == seq1

        hidden2 = torch.randn(batch, 1, config.hidden_dim)
        pos2 = torch.tensor([[seq1]]).expand(batch, -1)
        cos2, sin2 = rope(hidden2, pos2)

        outputs2 = attn(
            hidden2,
            position_ids=pos2,
            cos=cos2,
            sin=sin2,
            past_key_value=kv,
            use_cache=True,
        )

        out2 = outputs2[0]
        assert out2.shape == (batch, 1, config.hidden_dim)
        assert kv.seq_len == seq1 + 1

    def test_gradient_flow(self):
        config = get_mor_135m_config()
        attn = MoRAttention(config, layer_idx=0)

        batch, seq = 2, 32
        hidden_states = torch.randn(batch, seq, config.hidden_dim, requires_grad=True)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        outputs = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
        )

        out = outputs[0]
        loss = out.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert attn.q_proj.weight.grad is not None

    def test_active_mask(self):
        config = get_mor_135m_config()
        attn = MoRAttention(config, layer_idx=0)

        batch, seq = 2, 16
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        active_mask = torch.zeros(batch, seq, dtype=torch.bool)
        active_mask[:, ::2] = True

        outputs = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            active_mask=active_mask,
        )

        out = outputs[0]
        assert out.shape == (batch, seq, config.hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
