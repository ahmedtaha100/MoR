import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from model.config import MoRConfig, get_mor_135m_config
from model.feed_forward import SwiGLUFFN, StandardFFN, create_ffn


class TestSwiGLUFFN:
    def test_output_shape(self):
        config = get_mor_135m_config()
        ffn = SwiGLUFFN(config)

        batch, seq = 2, 128
        x = torch.randn(batch, seq, config.hidden_dim)

        out = ffn(x)

        assert out.shape == x.shape

    def test_swiglu_components(self):
        config = get_mor_135m_config()
        ffn = SwiGLUFFN(config)

        assert hasattr(ffn, 'gate_proj')
        assert hasattr(ffn, 'up_proj')
        assert hasattr(ffn, 'down_proj')

        assert ffn.gate_proj.out_features == config.ffn_dim
        assert ffn.up_proj.out_features == config.ffn_dim

        assert ffn.down_proj.out_features == config.hidden_dim

    def test_gradient_flow(self):
        config = get_mor_135m_config()
        ffn = SwiGLUFFN(config)

        x = torch.randn(2, 32, config.hidden_dim, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert ffn.gate_proj.weight.grad is not None
        assert ffn.up_proj.weight.grad is not None
        assert ffn.down_proj.weight.grad is not None

    def test_no_bias(self):
        config = get_mor_135m_config()
        ffn = SwiGLUFFN(config)

        assert ffn.gate_proj.bias is None
        assert ffn.up_proj.bias is None
        assert ffn.down_proj.bias is None


class TestStandardFFN:
    def test_output_shape(self):
        config = get_mor_135m_config()
        ffn = StandardFFN(config)

        batch, seq = 2, 128
        x = torch.randn(batch, seq, config.hidden_dim)

        out = ffn(x)

        assert out.shape == x.shape

    def test_two_layer_structure(self):
        config = get_mor_135m_config()
        ffn = StandardFFN(config)

        assert hasattr(ffn, 'up_proj')
        assert hasattr(ffn, 'down_proj')


class TestCreateFFN:
    def test_swiglu_creation(self):
        config = get_mor_135m_config()
        ffn = create_ffn(config)

        assert isinstance(ffn, SwiGLUFFN)

    def test_standard_creation(self):
        config = get_mor_135m_config()
        config.ffn_hidden_act = "gelu"
        ffn = create_ffn(config)

        assert isinstance(ffn, StandardFFN)


class TestFFNNumerics:
    def test_no_nan_output(self):
        config = get_mor_135m_config()
        ffn = SwiGLUFFN(config)

        for scale in [0.01, 1.0, 100.0]:
            x = torch.randn(2, 32, config.hidden_dim) * scale
            out = ffn(x)

            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_deterministic(self):
        config = get_mor_135m_config()
        config.dropout = 0.0
        ffn = SwiGLUFFN(config)
        ffn.eval()

        x = torch.randn(2, 32, config.hidden_dim)

        out1 = ffn(x)
        out2 = ffn(x)

        assert torch.equal(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
