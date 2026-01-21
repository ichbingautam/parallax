"""Typed configuration classes for the Transformer model.

Uses chex.dataclass for immutable, type-checked configuration objects.
This follows the Research Engineering practice of explicit, parametric typing.
"""

from typing import Literal

import chex


@chex.dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the Transformer language model.

    Attributes:
        vocab_size: Size of the vocabulary (number of unique tokens).
        hidden_dim: Dimension of the model's hidden representations (d_model).
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks in the decoder stack.
        ffn_dim: Dimension of the feed-forward network's hidden layer.
            Typically 4 * hidden_dim, but modern models use 8/3 * hidden_dim with SwiGLU.
        max_seq_len: Maximum sequence length the model can process.
        dropout_rate: Dropout probability for regularization.
        tie_word_embeddings: Whether to share weights between input embeddings
            and output projection (saves memory, common in GPT-2/3).
        use_rope: Whether to use Rotary Positional Embeddings (RoPE).
        rope_theta: Base for RoPE frequency computation (10000 is standard).
        layer_norm_epsilon: Epsilon for numerical stability in normalization.
        attention_bias: Whether to include bias in attention projections.
        ffn_bias: Whether to include bias in feed-forward projections.
        dtype: Data type for computations ('float32' or 'bfloat16').
    """

    vocab_size: int = 32000
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1376  # ~8/3 * hidden_dim for SwiGLU
    max_seq_len: int = 2048
    dropout_rate: float = 0.1
    tie_word_embeddings: bool = True
    use_rope: bool = True
    rope_theta: float = 10000.0
    layer_norm_epsilon: float = 1e-6
    attention_bias: bool = False
    ffn_bias: bool = False
    dtype: Literal["float32", "bfloat16"] = "float32"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0.0 <= self.dropout_rate < 1.0, "dropout_rate must be in [0, 1)"

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_dim // self.num_heads

    def num_params(self, include_embeddings: bool = True) -> int:
        """Estimate the number of parameters in the model.

        Args:
            include_embeddings: Whether to count embedding parameters.

        Returns:
            Estimated parameter count.
        """
        # Embedding: vocab_size * hidden_dim
        embed_params = self.vocab_size * self.hidden_dim if include_embeddings else 0

        # Per-layer parameters:
        # - Attention: Q, K, V, O projections = 4 * hidden_dim^2
        # - FFN (SwiGLU): gate, up, down = 3 * hidden_dim * ffn_dim
        # - RMSNorm: 2 * hidden_dim (attn_norm + ffn_norm)
        attn_params = 4 * self.hidden_dim * self.hidden_dim
        ffn_params = 3 * self.hidden_dim * self.ffn_dim
        norm_params = 2 * self.hidden_dim
        layer_params = attn_params + ffn_params + norm_params

        # Output head (if not tied, same as embedding)
        output_params = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_dim

        # Final norm
        final_norm_params = self.hidden_dim

        return embed_params + self.num_layers * layer_params + output_params + final_norm_params


# Preset configurations for common model sizes
TINY_CONFIG = TransformerConfig(
    vocab_size=256,  # Character-level
    hidden_dim=128,
    num_heads=4,
    num_layers=4,
    ffn_dim=344,  # ~8/3 * 128
    max_seq_len=256,
    dropout_rate=0.1,
)

SMALL_CONFIG = TransformerConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    ffn_dim=1376,
    max_seq_len=2048,
    dropout_rate=0.1,
)

BASE_CONFIG = TransformerConfig(
    vocab_size=32000,
    hidden_dim=768,
    num_heads=12,
    num_layers=12,
    ffn_dim=2048,
    max_seq_len=2048,
    dropout_rate=0.1,
)
