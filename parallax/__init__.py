"""Parallax: A research-grade JAX Transformer implementation from scratch.

This library implements a decoder-only Transformer architecture with modern
techniques used in state-of-the-art language models:
- RMSNorm (Root Mean Square Normalization)
- Rotary Positional Embeddings (RoPE)
- SwiGLU activation function
- KV-Cache for efficient inference
- Weight tying for memory efficiency
"""

__version__ = "0.1.0"

from parallax.config import TransformerConfig

__all__ = ["TransformerConfig", "__version__"]
