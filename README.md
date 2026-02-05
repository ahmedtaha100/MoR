# Mixture-of-Recursions (MoR)

A PyTorch implementation of the Mixture-of-Recursions architecture from the NeurIPS 2025 paper:

**[Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](./2507.10524v3.pdf)**

## Overview

MoR combines the benefits of parameter sharing in recursive transformers with adaptive computation through dynamic routing. This enables:

- **Parameter Efficiency**: Shared layers reduce model size while maintaining expressiveness
- **Adaptive Computation**: Different tokens receive different amounts of computation based on their complexity
- **Memory Efficiency**: Recursion-wise KV caching reduces memory and compute during inference

## Architecture

```
                    ┌─────────────────────────────────┐
                    │     Token Embeddings + RoPE     │
                    └─────────────────────────────────┘
                                    │
                    ┌─────────────────────────────────┐
                    │    First Unique Layer (L_0)     │  ← Middle-Cycle
                    └─────────────────────────────────┘
                                    │
              ┌─────────────────────────────────────────────┐
              │           Recursive Block (N_r times)       │
              │  ┌─────────────────────────────────────┐    │
              │  │         Shared Transformer          │    │
              │  │     (Attention + SwiGLU FFN)        │    │
              │  └─────────────────────────────────────┘    │
              │                     │                       │
              │  ┌─────────────────────────────────────┐    │
              │  │    Router: Select tokens to continue │   │
              │  │    (Expert-Choice or Token-Choice)   │   │
              │  └─────────────────────────────────────┘    │
              └─────────────────────────────────────────────┘
                                    │
                    ┌─────────────────────────────────┐
                    │   Last Unique Layer (L_{L-1})   │  ← Middle-Cycle
                    └─────────────────────────────────┘
                                    │
                    ┌─────────────────────────────────┐
                    │      RMSNorm + LM Head          │
                    └─────────────────────────────────┘
```

## Features

- **Routing Strategies**:
  - **Expert-Choice**: Router selects top-k tokens at each step (hierarchical filtering)
  - **Token-Choice**: Each token chooses its recursion depth upfront

- **KV Caching Strategies**:
  - **Selective (Recursion-wise)**: Separate cache per recursion depth
  - **Shared**: Reuse KV from first recursion

- **Architecture Components**:
  - RMSNorm (pre-normalization)
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - SwiGLU Feed-Forward Networks

## Installation

```bash
cd mor
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train MoR-135M model
python scripts/train.py --model mor_135m --max_steps 10000

# Train with custom configuration
python scripts/train.py \
    --model mor_135m \
    --learning_rate 3e-4 \
    --batch_size 32 \
    --max_steps 50000 \
    --wandb_project my_project
```

### Using the Model

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("mor/src").resolve()))
from model.config import MoRConfig, get_mor_135m_config
from model.mor_model import MoRForCausalLM

# Load configuration
config = get_mor_135m_config()

# Create model
model = MoRForCausalLM(config)

# Forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 128))
outputs = model(input_ids, labels=input_ids)

print(f"Loss: {outputs.loss.item():.4f}")
print(f"Aux Loss: {outputs.aux_loss.item():.4f}")

# Generate text
generated = model.generate(input_ids[:, :10], max_new_tokens=50)
```

### Analyzing Routing

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("mor/src").resolve()))
from evaluation import get_routing_heatmap, plot_routing_heatmap

# Get routing information
depths, routing_info = get_routing_heatmap(model, input_ids)

# Visualize
plot_routing_heatmap(depths, save_path="routing.png")
```

## Project Structure

```
mor/
├── configs/                 # YAML configuration files
│   ├── mor_135m.yaml
│   ├── mor_360m.yaml
│   └── vanilla_360m.yaml
├── src/
│   ├── model/              # Core model implementation
│   │   ├── config.py       # Model configuration
│   │   ├── embeddings.py   # RoPE, RMSNorm, embeddings
│   │   ├── attention.py    # Multi-head attention with GQA
│   │   ├── feed_forward.py # SwiGLU FFN
│   │   ├── transformer_block.py
│   │   ├── router.py       # Expert-choice and Token-choice routers
│   │   ├── recursive_block.py  # Core MoR recursion
│   │   ├── kv_cache.py     # KV caching strategies
│   │   ├── mor_model.py    # Full MoR model
│   │   └── vanilla_model.py # Baseline vanilla transformer
│   ├── data/               # Data loading utilities
│   ├── training/           # Training loop and utilities
│   └── evaluation/         # Metrics and visualization
├── scripts/                # Training and evaluation scripts
├── tests/                  # Unit tests
└── notebooks/              # Jupyter notebooks
```

## Configuration

### Model Configurations

| Config | Params | Hidden | Heads | Layers | Recursions |
|--------|--------|--------|-------|--------|------------|
| mor_135m | ~70M | 576 | 9 | 32 | 3 |
| mor_360m | ~180M | 960 | 15 | 32 | 3 |
| vanilla_360m | ~360M | 960 | 15 | 32 | 1 |

### Key Hyperparameters

- `num_recursion_steps`: Number of times to apply the shared block (N_r)
- `router_type`: "expert_choice" or "token_choice"
- `capacity_ratio`: Fraction of tokens continuing at each step
- `router_aux_loss_coeff`: Weight for auxiliary routing loss

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_router.py -v
```

## References

- Paper (PDF in this repo): [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](./2507.10524v3.pdf)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (SwiGLU)
- [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE)

## Author

Ahmed Taha | [ahmedtaha.io](https://ahmedtaha.io)

## License

MIT License
