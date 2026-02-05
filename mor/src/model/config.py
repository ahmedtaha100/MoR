from dataclasses import dataclass
from typing import Literal, Optional
import yaml


@dataclass
class MoRConfig:
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_kv_heads: Optional[int] = None
    head_dim: int = 64
    num_layers: int = 12
    num_shared_layers: int = 4
    num_recursion_steps: int = 3
    ffn_dim: int = 2048
    ffn_hidden_act: str = "silu"
    vocab_size: int = 49152
    max_seq_len: int = 2048
    router_type: Literal["expert_choice", "token_choice"] = "expert_choice"
    capacity_ratio: float = 0.5
    router_aux_loss_coeff: float = 0.001
    router_z_loss_coeff: float = 0.001
    router_activation: Literal["sigmoid", "tanh", "softmax"] = "sigmoid"
    router_architecture: Literal["linear", "mlp", "wide_mlp"] = "linear"
    router_alpha: float = 1.0
    expert_choice_aux_mode: Literal["loss", "aux_router", "none"] = "loss"
    token_choice_balance_mode: Literal["loss", "loss_free", "none"] = "loss"
    token_choice_bias_update_rate: float = 0.01
    capacity_warmup_steps: int = 0
    kv_cache_strategy: Literal["selective", "shared", "hybrid"] = "selective"
    sharing_strategy: Literal["middle_cycle", "cycle", "sequence", "middle_sequence"] = "middle_cycle"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    gradient_checkpointing: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        assert self.num_attention_heads % self.num_kv_heads == 0
        assert self.num_recursion_steps >= 1
        if self.sharing_strategy.startswith("middle"):
            expected_layers = 2 + self.num_shared_layers * self.num_recursion_steps
            if self.num_layers != expected_layers:
                self.num_layers = expected_layers
        else:
            expected_layers = self.num_shared_layers * self.num_recursion_steps
            if self.num_layers != expected_layers:
                self.num_layers = expected_layers
        assert 0.0 < self.capacity_ratio <= 1.0

    @property
    def num_unique_layers(self) -> int:
        if self.sharing_strategy.startswith("middle"):
            return 2 + self.num_shared_layers
        return self.num_shared_layers

    @property
    def has_unique_first_layer(self) -> bool:
        return self.sharing_strategy.startswith("middle")

    @property
    def has_unique_last_layer(self) -> bool:
        return self.sharing_strategy.startswith("middle")

    def get_capacity_schedule(self) -> list[float]:
        return [
            self.capacity_ratio * (self.num_recursion_steps - i) / self.num_recursion_steps
            for i in range(self.num_recursion_steps)
        ]

    @classmethod
    def from_yaml(cls, path: str) -> "MoRConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MoRConfig":
        return cls(**config_dict)


def get_mor_135m_config() -> MoRConfig:
    return MoRConfig(
        hidden_dim=576,
        num_attention_heads=9,
        num_kv_heads=3,
        head_dim=64,
        num_shared_layers=10,
        num_recursion_steps=3,
        ffn_dim=1536,
        vocab_size=49152,
        max_seq_len=2048,
        router_type="expert_choice",
        kv_cache_strategy="selective",
        sharing_strategy="middle_cycle",
    )


def get_mor_360m_config() -> MoRConfig:
    return MoRConfig(
        hidden_dim=1024,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=64,
        num_shared_layers=12,
        num_recursion_steps=3,
        ffn_dim=2816,
        vocab_size=49152,
        max_seq_len=2048,
        router_type="expert_choice",
        kv_cache_strategy="selective",
        sharing_strategy="middle_cycle",
    )


def get_vanilla_360m_config() -> MoRConfig:
    return MoRConfig(
        hidden_dim=1024,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=64,
        num_shared_layers=24,
        num_recursion_steps=1,
        ffn_dim=2816,
        vocab_size=49152,
        max_seq_len=2048,
        router_type="expert_choice",
        kv_cache_strategy="selective",
        sharing_strategy="cycle",
    )
