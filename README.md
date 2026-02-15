# Parallax 🚀

A research-grade JAX Transformer implementation from scratch, demonstrating the skills required for a Research Engineering role at labs like DeepMind or Google Research.

[![CI](https://github.com/ichbingautam/parallax/actions/workflows/ci.yml/badge.svg)](https://github.com/ichbingautam/parallax/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.30+-orange)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 Project Overview

This project implements a **decoder-only Transformer language model** using modern techniques from state-of-the-art models like **Llama**, **Gemma**, and **PaLM**. It serves as a portfolio piece demonstrating:

- Deep understanding of Transformer architecture mathematics
- Production-quality JAX/Flax implementation
- Distributed training with `jax.pmap`
- Research engineering best practices

### Why This Project?

Research Engineering at labs like DeepMind differs from traditional software engineering:

| Standard SWE | Research Engineering |
|--------------|---------------------|
| Build features | Build experimental testbeds |
| Code is a product | Code is a scientific instrument |
| Ship fast | Ship correctly (numerical precision matters) |
| Unit tests | Overfit tests + gradient checks |

---

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     TransformerLM                           │
├─────────────────────────────────────────────────────────────┤
│  Input IDs → Embeddings → [Block × N] → RMSNorm → Logits   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   TransformerBlock                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌───────────────────┐    ┌─────────┐       │
│  │ RMSNorm │───►│ Multi-Head Attn   │───►│    +    │       │
│  └─────────┘    │  (RoPE + KV-Cache)│    └────┬────┘       │
│       │         └───────────────────┘         │            │
│       └───────────────────────────────────────┘            │
│                           │                                 │
│  ┌─────────┐    ┌───────────────────┐    ┌─────────┐       │
│  │ RMSNorm │───►│   SwiGLU FFN      │───►│    +    │       │
│  └─────────┘    └───────────────────┘    └────┬────┘       │
│       │                                        │            │
│       └────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Key Techniques Explained

#### 1. RMSNorm (Root Mean Square Normalization)

**Used in**: Llama, Gemma, Mistral

Unlike LayerNorm which subtracts mean and divides by std, RMSNorm only divides by RMS:

```python
RMSNorm(x) = x / √(mean(x²) + ε) × γ
```

**Why?** Faster computation (no mean subtraction) with equal or better training stability.

#### 2. Rotary Positional Embeddings (RoPE)

**Used in**: Llama, GPT-NeoX, PaLM

RoPE encodes position by *rotating* query and key vectors:

```python
q_rotated = q × cos(θ) + rotate_half(q) × sin(θ)
```

**Why?**

- Relative position awareness through rotation
- Better extrapolation to longer sequences than absolute embeddings
- Position info naturally decays with distance

#### 3. SwiGLU Activation

**Used in**: Llama, PaLM, Gemma

```python
SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)  # ⊙ = element-wise multiply
```

**Why?** Outperforms ReLU and GELU in practice (empirically shown in PaLM paper).

#### 4. Weight Tying

Shares weights between input embeddings and output projection:

```python
logits = hidden_states @ embedding_weights.T
```

**Why?** Reduces parameters by `vocab_size × hidden_dim` (~16M for 32k vocab, 512 dim).

#### 5. Scaled Initialization

Output projections scaled by `1/√(2N)` where N = num_layers:

```python
attn_out_proj.weight *= 1 / sqrt(2 * num_layers)
ffn_down_proj.weight *= 1 / sqrt(2 * num_layers)
```

**Why?** Prevents signal explosion in deep residual networks.

#### 6. Z-Loss (Logit Stability)

**Used in**: PaLM, Gemini

```python
z_loss = 1e-4 × log²(Σ exp(logits))
```

**Why?** Prevents logits from drifting to extreme values, crucial for bfloat16 training on TPUs.

#### 7. KV-Cache

Caches key/value projections during autoregressive generation:

```
Step 1: Process "The" → cache K₁, V₁
Step 2: Process "cat" → cache K₂, V₂, attend to [K₁,K₂], [V₁,V₂]
Step 3: Process "sat" → cache K₃, V₃, attend to [K₁,K₂,K₃], [V₁,V₂,V₃]
```

**Why?** Reduces generation complexity from O(N²) to O(N).

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- For GPU: CUDA 12.0+
- For TPU: Google Cloud account

### Local Development

```bash
# Clone repository
git clone https://github.com/ichbingautam/parallax.git
cd parallax

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install for CPU/GPU
pip install -e ".[dev]"

# Install for TPU
pip install -e ".[dev,tpu]"
```

---

## 🚀 Quick Start

### 1. Run the Overfit Test (Validate Correctness)

The **overfit test** is the most important validation for any new model implementation:

> If a model can't memorize 10 tokens, it won't learn 10 billion.

```bash
python scripts/train.py --mode overfit --steps 500
```

Expected output:

```
Final Loss: 0.0275
Accuracy: 100.0%
Target:     'B CD EF GH AB CD EF GH AB CD EF GH...'
Prediction: 'B CD EF GH AB CD EF GH AB CD EF GH...'

✅ OVERFIT TEST PASSED - Model is mathematically correct!
```

### 2. Train on Tiny Shakespeare

```bash
python scripts/train.py --mode single --dataset tiny_shakespeare --steps 5000
```

### 3. Distributed Training (Multi-GPU/TPU)

```bash
python scripts/train.py --mode distributed --dataset tiny_shakespeare
```

### 4. Interactive Generation

```bash
python scripts/generate.py --prompt "To be or not to be" --temperature 0.8
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
parallax/
├── parallax/                    # Main package
│   ├── config.py                # Typed configuration (chex.dataclass)
│   ├── layers/
│   │   ├── attention.py         # MultiHeadAttention + RoPE + KV-Cache
│   │   ├── feedforward.py       # SwiGLU FFN
│   │   ├── normalization.py     # RMSNorm
│   │   └── embeddings.py        # Weight-tied embeddings
│   ├── model/
│   │   ├── block.py             # TransformerBlock (pre-norm residual)
│   │   └── transformer.py       # TransformerLM (full decoder stack)
│   ├── training/
│   │   ├── data.py              # Character tokenizer + dataset
│   │   ├── loss.py              # Cross-entropy + Z-loss
│   │   ├── optimizer.py         # AdamW + cosine schedule
│   │   └── train_step.py        # JIT-compiled training step
│   ├── distributed/
│   │   └── pmap_trainer.py      # jax.pmap distributed training
│   └── inference/
│       └── generate.py          # Autoregressive generation
├── tests/                       # 36 unit tests
│   ├── test_layers.py           # Layer correctness tests
│   ├── test_model.py            # Model integration tests
│   └── test_training.py         # Training + overfit tests
├── scripts/
│   ├── train.py                 # Main training script
│   └── generate.py              # Interactive generation
├── terraform/                   # GCP TPU infrastructure
│   ├── main.tf                  # Provider configuration
│   ├── gcp.tf                   # TPU VM, GCS, VPC resources
│   ├── variables.tf             # Input variables
│   └── outputs.tf               # Access information
└── .github/workflows/           # CI/CD
    └── ci.yml                   # Lint, test, type check
```

---

## 🧪 Testing Strategy

### Unit Tests (36 tests)

| Category | Tests | Purpose |
|----------|-------|---------|
| Layers | 14 | Numerical correctness of RMSNorm, RoPE, Attention, FFN |
| Model | 8 | Forward pass shapes, param counting, KV-cache consistency |
| Training | 14 | Loss functions, optimizer schedules, **overfit test** |

### The Overfit Test

This is the **critical correctness check** for any new model:

```python
def test_overfit_single_batch():
    """Train on 'AB AB AB' until loss → 0 and accuracy → 100%"""
    # If this fails, there's a bug in the implementation
```

### Run All Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_layers.py -v

# With coverage
pytest tests/ --cov=parallax
```

---

## 🌐 Distributed Training

### jax.pmap (Data Parallelism)

```python
# Batch is sharded across devices
# Each device computes local gradients
# Gradients are averaged with pmean before update

@jax.pmap
def train_step(state, batch, rng):
    grads = compute_gradients(state.params, batch)
    grads = jax.lax.pmean(grads, axis_name='devices')  # All-reduce
    return state.apply_gradients(grads=grads)
```

### RNG Folding

Each device gets unique random dropout masks:

```python
device_id = jax.lax.axis_index('devices')
rng = jax.random.fold_in(rng, device_id)  # Unique per device
```

### Scaling Efficiency

| Devices | Expected Speedup | Actual (typical) |
|---------|------------------|------------------|
| 1 | 1.0x | 1.0x |
| 4 | 4.0x | ~3.8x |
| 8 | 8.0x | ~7.6x |

Communication overhead causes ~5% efficiency loss.

---

## ☁️ Infrastructure (Terraform)

Deploy TPU training infrastructure to GCP:

### Quick Deploy

```bash
cd terraform

# Initialize
terraform init

# Configure (copy example)
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project_id

# Preview
terraform plan

# Deploy
terraform apply

# Connect
gcloud compute tpus tpu-vm ssh parallax-dev-tpu --zone=us-central1-a

# Destroy when done!
terraform destroy
```

### Resources Created

| Resource | Type | Purpose |
|----------|------|---------|
| TPU VM | v3-8 | 8 TPU v3 cores for training |
| GCS Bucket | Standard | Checkpoints and data storage |
| VPC Network | Custom | Isolated network |
| Service Account | IAM | Minimal permissions |

### ⚠️ Cost Warning

- **TPU v3-8**: ~$8/hour (on-demand) or ~$2.40/hour (preemptible)
- **Always run `terraform destroy` when not training!**

---

## 🔄 CI/CD Pipeline

GitHub Actions runs on every push:

```yaml
jobs:
  lint:     # Ruff linter + formatter
  test:     # pytest on Python 3.10, 3.11, 3.12
  typecheck: # Pyright (advisory)
```

---

## 📊 Configuration Presets

```python
from parallax.config import TINY_CONFIG, SMALL_CONFIG, BASE_CONFIG

# TINY: For quick experiments and tests
# - 128 hidden, 4 layers, 4 heads
# - ~400K parameters
# - Trains in seconds on CPU

# SMALL: For real training
# - 512 hidden, 6 layers, 8 heads
# - ~25M parameters
# - Trains in minutes on GPU

# BASE: Production-like
# - 768 hidden, 12 layers, 12 heads
# - ~125M parameters
# - Requires GPU/TPU
```

---

## 🎓 Research Engineering Practices

This codebase demonstrates key practices:

1. **Parametric Typing**: `chex.dataclass` for immutable, validated configs
2. **Pure Functions**: Stateless, JIT-compiled `train_step`
3. **Observability**: Logging gradient norm, parameter norm, Z-loss
4. **The Overfit Test**: Mathematical correctness validation
5. **Numerical Stability**: Float32 for norms, Z-loss for logits
6. **Modern Techniques**: RMSNorm, RoPE, SwiGLU, KV-cache

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864) - Rotary Positional Embeddings
- [GLU Variants](https://arxiv.org/abs/2002.05202) - SwiGLU activation
- [PaLM](https://arxiv.org/abs/2204.02311) - Z-loss technique
- [Llama](https://arxiv.org/abs/2302.13971) - RMSNorm + RoPE + SwiGLU combination

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Llama](https://github.com/facebookresearch/llama) - Architecture inspiration
- [minGPT](https://github.com/karpathy/minGPT) - Educational approach
- [Flax](https://github.com/google/flax) - JAX neural network library
- [Optax](https://github.com/deepmind/optax) - Optimization library
