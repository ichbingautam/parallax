"""Transformer Block: Attention + FFN with pre-norm residual connections.

This module implements a single Transformer decoder block with:
- Pre-norm architecture (norm before attention/FFN)
- Residual connections around both sublayers
- Scaled initialization for residual paths

The pre-norm architecture provides better training stability for deep networks.
"""

import jax.numpy as jnp
from flax import linen as nn

from parallax.config import TransformerConfig
from parallax.layers.attention import KVCache, MultiHeadAttention
from parallax.layers.feedforward import FeedForward
from parallax.layers.normalization import RMSNorm

Array = jnp.ndarray


class TransformerBlock(nn.Module):
    """Single Transformer decoder block.

    Architecture (Pre-Norm):
        x -> RMSNorm -> Attention -> + -> RMSNorm -> FFN -> +
        |______________________________|__________________|

    Attributes:
        config: Transformer configuration.
        layer_idx: Index of this block in the stack (for scaled init).
    """

    config: TransformerConfig
    layer_idx: int = 0

    def setup(self) -> None:
        """Initialize sublayers."""
        self.attn_norm = RMSNorm(config=self.config, name="attn_norm")
        self.attention = MultiHeadAttention(config=self.config, name="attention")
        self.ffn_norm = RMSNorm(config=self.config, name="ffn_norm")
        self.ffn = FeedForward(config=self.config, name="ffn")

    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        cache: KVCache | None = None,
        cache_index: int = 0,
        deterministic: bool = True,
    ) -> tuple[Array, KVCache | None]:
        """Apply transformer block.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim].
            mask: Optional attention mask.
            cache: Optional KV-Cache for inference.
            cache_index: Current position in the cache.
            deterministic: If True, disable dropout.

        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_dim].
                - Updated KV-Cache (or None).
        """
        # Attention sublayer with pre-norm residual
        residual = x
        x = self.attn_norm(x)
        x, cache = self.attention(
            x,
            mask=mask,
            cache=cache,
            cache_index=cache_index,
            deterministic=deterministic,
        )
        x = residual + x

        # FFN sublayer with pre-norm residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x, deterministic=deterministic)
        x = residual + x

        return x, cache
