import pytest
import torch

import sys
sys.path.insert(0, '/Users/ahmedtaeha/Desktop/General/GitHub/MoR/mor/src')

from model.config import MoRConfig, get_mor_135m_config
from model.transformer_block import TransformerBlock, SharedTransformerBlock
from model.embeddings import RotaryEmbedding, RMSNorm
from model.attention import create_causal_mask, KVCache


class TestTransformerBlock:
    def test_output_shape(self):
        config = get_mor_135m_config()
        block = TransformerBlock(config, layer_idx=0)

        batch, seq = 2, 128
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)

        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
        )

        out = outputs[0]
        assert out.shape == (batch, seq, config.hidden_dim)

    def test_residual_connection(self):
        config = get_mor_135m_config()
        block = TransformerBlock(config, layer_idx=0)

        batch, seq = 2, 32
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
        )

        out = outputs[0]

        assert not torch.equal(out, hidden_states)

    def test_pre_norm(self):
        config = get_mor_135m_config()
        block = TransformerBlock(config, layer_idx=0)

        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')

    def test_gradient_flow(self):
        config = get_mor_135m_config()
        block = TransformerBlock(config, layer_idx=0)

        batch, seq = 2, 32
        hidden_states = torch.randn(batch, seq, config.hidden_dim, requires_grad=True)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        outputs = block(
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
        assert block.self_attn.q_proj.weight.grad is not None
        assert block.mlp.gate_proj.weight.grad is not None

    def test_kv_cache(self):
        config = get_mor_135m_config()
        block = TransformerBlock(config, layer_idx=0)

        batch = 2
        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)

        seq1 = 32
        hidden1 = torch.randn(batch, seq1, config.hidden_dim)
        pos1 = torch.arange(seq1).unsqueeze(0).expand(batch, -1)
        cos1, sin1 = rope(hidden1, pos1)
        mask1 = create_causal_mask(seq1, hidden1.device, hidden1.dtype)

        kv = KVCache()

        out1, attn_weights, kv_out = block(
            hidden1,
            attention_mask=mask1,
            position_ids=pos1,
            cos=cos1,
            sin=sin1,
            past_key_value=kv,
            use_cache=True,
        )

        assert kv.seq_len == seq1

        hidden2 = torch.randn(batch, 1, config.hidden_dim)
        pos2 = torch.tensor([[seq1]]).expand(batch, -1)
        cos2, sin2 = rope(hidden2, pos2)

        out2, _, kv2 = block(
            hidden2,
            position_ids=pos2,
            cos=cos2,
            sin=sin2,
            past_key_value=kv,
            use_cache=True,
        )

        assert out2.shape == (batch, 1, config.hidden_dim)


class TestSharedTransformerBlock:
    def test_output_shape(self):
        config = get_mor_135m_config()
        block = SharedTransformerBlock(config, layer_idx=0)

        batch, seq = 2, 128
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        out, attentions, key_values, _ = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
        )

        assert out.shape == (batch, seq, config.hidden_dim)

    def test_multiple_layers(self):
        config = get_mor_135m_config()
        block = SharedTransformerBlock(config, layer_idx=0)

        assert hasattr(block, 'layers')
        assert len(block.layers) == config.num_shared_layers

    def test_with_active_mask(self):
        config = get_mor_135m_config()
        block = SharedTransformerBlock(config, layer_idx=0)

        batch, seq = 2, 64
        hidden_states = torch.randn(batch, seq, config.hidden_dim)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        active_mask = torch.ones(batch, seq, dtype=torch.bool)
        active_mask[:, seq//2:] = False

        out, _, _, _ = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            active_mask=active_mask,
        )

        assert out.shape == (batch, seq, config.hidden_dim)

    def test_gradient_flow(self):
        config = get_mor_135m_config()
        block = SharedTransformerBlock(config, layer_idx=0)

        batch, seq = 2, 32
        hidden_states = torch.randn(batch, seq, config.hidden_dim, requires_grad=True)

        head_dim = config.head_dim
        rope = RotaryEmbedding(head_dim, config.max_seq_len)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(hidden_states, position_ids)
        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)

        out, _, _, _ = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
        )

        loss = out.sum()
        loss.backward()

        assert hidden_states.grad is not None


class TestBlockComponents:
    def test_attention_ffn_order(self):
        config = get_mor_135m_config()
        block = TransformerBlock(config, layer_idx=0)

        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'mlp')

    def test_normalization_layers(self):
        config = get_mor_135m_config()
        block = TransformerBlock(config, layer_idx=0)

        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')

        assert isinstance(block.input_layernorm, RMSNorm)
        assert isinstance(block.post_attention_layernorm, RMSNorm)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
