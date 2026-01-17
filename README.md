# Parallax 🚀

A research-grade JAX Transformer implementation from scratch, demonstrating the skills required for a Research Engineering role at labs like DeepMind or Google Research.

[![Tests](https://img.shields.io/badge/tests-36%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![JAX](https://img.shields.io/badge/JAX-0.4.30+-orange)]()

## Features

### Core Architecture (Phase 1)

- **RMSNorm**: Root Mean Square Normalization (Llama/Gemma style)
- **RoPE**: Rotary Positional Embeddings for relative position encoding
- **Multi-Head Attention**: With KV-Cache for O(1) inference
- **SwiGLU FFN**: Gated linear unit activation (PaLM style)
- **Weight Tying**: Shared input/output embeddings for memory efficiency
- **Scaled Initialization**: 1/√(2N) scaling for deep residual networks

### Training Loop (Phase 2)

- **Cross-Entropy Loss**: With label smoothing support
- **Z-Loss**: Logit stability regularization (Google/PaLM technique)
- **AdamW Optimizer**: With cosine decay and linear warmup
- **Gradient Clipping**: Global norm clipping for stability
- **JIT Compilation**: Pure `@jax.jit` compiled train step

### Distributed Training (Phase 3)

- **`jax.pmap`**: Data-parallel training across devices
- **Gradient Averaging**: `jax.lax.pmean` for synchronized updates
- **RNG Folding**: Unique dropout masks per device
- **Batch Sharding**: Automatic batch distribution

### Inference (Phase 5)

- **KV-Cache**: Efficient autoregressive generation
- **Sampling Strategies**: Temperature, top-k, top-p (nucleus)
- **Compiled Generation**: `jax.lax.while_loop` for speed

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ichbingautam/parallax.git
cd parallax

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Run the Overfit Test (Prove Correctness)

The overfit test is the most important validation - if a model can't memorize 10 tokens, it won't learn 10 billion.

```bash
python scripts/train.py --mode overfit --steps 500
```

Expected output:

```
✅ OVERFIT TEST PASSED - Model is mathematically correct!
```

### Train on Tiny Shakespeare

```bash
python scripts/train.py --mode single --dataset tiny_shakespeare --steps 1000
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
parallax/
├── parallax/
│   ├── config.py           # Typed configuration classes
│   ├── layers/
│   │   ├── attention.py    # Multi-Head Attention + RoPE + KV-Cache
│   │   ├── feedforward.py  # SwiGLU FFN
│   │   ├── normalization.py # RMSNorm
│   │   └── embeddings.py   # Token embeddings with weight tying
│   ├── model/
│   │   ├── block.py        # TransformerBlock
│   │   └── transformer.py  # TransformerLM (full decoder stack)
│   ├── training/
│   │   ├── data.py         # Data pipeline and tokenization
│   │   ├── loss.py         # Cross-entropy + Z-loss
│   │   ├── optimizer.py    # AdamW + schedules
│   │   └── train_step.py   # JIT-compiled training step
│   ├── distributed/
│   │   └── pmap_trainer.py # jax.pmap distributed training
│   └── inference/
│       └── generate.py     # Autoregressive generation
├── tests/
│   ├── test_layers.py      # Unit tests (14 tests)
│   ├── test_model.py       # Model tests (8 tests)
│   └── test_training.py    # Training tests + overfit (14 tests)
└── scripts/
    ├── train.py            # Main training script
    └── generate.py         # Interactive text generation
```

## Portfolio Artifacts

| Phase | Deliverable | Skill Demonstrated |
|-------|-------------|-------------------|
| 1 | `TransformerLM` class | Architecture, Math translation |
| 2 | Converging loss curve | Optimization stability, Numerics |
| 3 | `pmap` training script | Distributed systems, Parallelism |
| 5 | Interactive demo | End-to-end ML engineering |

## Research Engineering Practices

This codebase demonstrates research engineering best practices:

1. **Parametric Typing**: Using `chex.dataclass` for typed, immutable configs
2. **Unit Testing**: 36 tests covering numerical correctness
3. **The Overfit Test**: Critical validation that proves mathematical correctness
4. **Observability**: Logging gradient norm, parameter norm, Z-loss
5. **Pure Functions**: Stateless, JIT-compiled training steps
6. **Modern Techniques**: RMSNorm, RoPE, SwiGLU, Z-loss, KV-cache

## Configuration Presets

```python
from parallax.config import TINY_CONFIG, SMALL_CONFIG

# TINY: 128d, 4L, 4H, ~400K params (for quick experiments)
# SMALL: 512d, 6L, 8H, ~25M params (for real training)
```

## License

MIT

## Acknowledgments

This implementation draws inspiration from:

- [Llama](https://github.com/facebookresearch/llama) - RMSNorm, RoPE, SwiGLU
- [PaLM](https://arxiv.org/abs/2204.02311) - Z-loss technique
- [minGPT](https://github.com/karpathy/minGPT) - Educational approach
- [Flax](https://github.com/google/flax) - JAX neural network library
