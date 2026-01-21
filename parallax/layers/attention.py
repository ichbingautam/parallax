"""Multi-Head Attention with Rotary Positional Embeddings (RoPE) and KV-Cache.

This module implements the core attention mechanism used in modern LLMs:
- Scaled dot-product attention with causal masking
- Rotary Positional Embeddings for relative position encoding
- KV-Cache support for efficient autoregressive inference

References:
- RoPE: https://arxiv.org/abs/2104.09864
- Attention: https://arxiv.org/abs/1706.03762
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from parallax.config import TransformerConfig

Array = jnp.ndarray


class KVCache(NamedTuple):
    """Key-Value cache for efficient autoregressive inference.

    During generation, we cache the key and value projections of previous
    tokens to avoid recomputing them at each step, reducing complexity
    from O(N^2) to O(N) for sequence length N.

    Attributes:
        key: Cached key tensor of shape [batch, num_heads, cache_len, head_dim].
        value: Cached value tensor of shape [batch, num_heads, cache_len, head_dim].
    """

    key: Array
    value: Array

    @classmethod
    def init(
        cls,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> "KVCache":
        """Initialize an empty KV cache.

        Args:
            batch_size: Batch size.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length to cache.
            head_dim: Dimension of each attention head.
            dtype: Data type for the cache tensors.

        Returns:
            Empty KVCache with zero-initialized tensors.
        """
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        return cls(
            key=jnp.zeros(shape, dtype=dtype),
            value=jnp.zeros(shape, dtype=dtype),
        )


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    RoPE encodes position information by rotating the query and key vectors
    in a way that makes their dot product depend on relative position.
    This is superior to absolute position embeddings for extrapolation.

    The rotation is applied in pairs of dimensions, with different rotation
    frequencies for different dimension pairs.

    Attributes:
        config: Transformer configuration.
    """

    config: TransformerConfig

    def setup(self) -> None:
        """Precompute rotation frequencies."""
        head_dim = self.config.head_dim
        theta = self.config.rope_theta

        # Compute inverse frequencies for each pair of dimensions
        # freqs[i] = theta^(-2i/d) for i in [0, d/2)
        dim_pairs = head_dim // 2
        inv_freq = 1.0 / (theta ** (jnp.arange(0, dim_pairs) * 2.0 / head_dim))
        self.inv_freq = inv_freq

    def __call__(self, x: Array, offset: int = 0) -> Array:
        """Apply rotary positional embedding.

        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim].
            offset: Position offset for KV-cache (start position in sequence).

        Returns:
            Tensor with rotary position encoding applied.
        """
        seq_len = x.shape[2]

        # Compute position indices
        positions = jnp.arange(offset, offset + seq_len)

        # Compute rotation angles: positions * inv_freq
        # Shape: [seq_len, head_dim // 2]
        angles = jnp.outer(positions, self.inv_freq)

        # Compute sin and cos
        sin = jnp.sin(angles)
        cos = jnp.cos(angles)

        # Apply rotation using the formula:
        # x_rot = x * cos + rotate_half(x) * sin
        # where rotate_half swaps pairs and negates the second element
        return self._apply_rotation(x, sin, cos)

    def _apply_rotation(self, x: Array, sin: Array, cos: Array) -> Array:
        """Apply rotation to input tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim].
            sin: Sine values [seq_len, head_dim // 2].
            cos: Cosine values [seq_len, head_dim // 2].

        Returns:
            Rotated tensor.
        """
        # Split x into pairs
        x1, x2 = jnp.split(x, 2, axis=-1)

        # Broadcast sin/cos to match x shape
        # [seq_len, head_dim // 2] -> [1, 1, seq_len, head_dim // 2]
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]

        # Apply rotation
        # For each pair (x1, x2), the rotated values are:
        # (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        return jnp.concatenate([x1_rot, x2_rot], axis=-1)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE and KV-Cache support.

    Implements scaled dot-product attention with:
    - Multiple parallel attention heads
    - Rotary positional embeddings
    - Causal masking for autoregressive models
    - KV-Cache for efficient inference

    Attributes:
        config: Transformer configuration.
    """

    config: TransformerConfig

    def setup(self) -> None:
        """Initialize projection layers and RoPE."""
        hidden_dim = self.config.hidden_dim
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim
        use_bias = self.config.attention_bias

        # Q, K, V, O projections
        self.q_proj = nn.Dense(
            num_heads * head_dim,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            name="q_proj",
        )
        self.k_proj = nn.Dense(
            num_heads * head_dim,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            name="k_proj",
        )
        self.v_proj = nn.Dense(
            num_heads * head_dim,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            name="v_proj",
        )
        self.o_proj = nn.Dense(
            hidden_dim,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            name="o_proj",
        )

        # RoPE
        if self.config.use_rope:
            self.rope = RotaryPositionalEmbedding(config=self.config)

        # Dropout
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        cache: KVCache | None = None,
        cache_index: int = 0,
        deterministic: bool = True,
    ) -> tuple[Array, KVCache | None]:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].
            mask: Optional attention mask. If None, causal mask is applied.
            cache: Optional KV-Cache for inference.
            cache_index: Current position in the cache (for incremental decoding).
            deterministic: If True, disable dropout.

        Returns:
            Tuple of:
                - Output tensor of shape [batch, seq_len, hidden_dim].
                - Updated KV-Cache (or None if cache was None).
        """
        batch_size, seq_len, _ = x.shape
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE to Q and K
        if self.config.use_rope:
            q = self.rope(q, offset=cache_index)
            k = self.rope(k, offset=cache_index)

        # Handle KV-Cache
        if cache is not None:
            # Update cache with new K, V
            k_cache = jax.lax.dynamic_update_slice(cache.key, k, (0, 0, cache_index, 0))
            v_cache = jax.lax.dynamic_update_slice(cache.value, v, (0, 0, cache_index, 0))
            cache = KVCache(key=k_cache, value=v_cache)

            # Use cached K, V for attention
            k = k_cache[:, :, : cache_index + seq_len, :]
            v = v_cache[:, :, : cache_index + seq_len, :]

        # Compute attention scores
        # [batch, heads, q_len, head_dim] @ [batch, heads, head_dim, kv_len]
        # -> [batch, heads, q_len, kv_len]
        scale = head_dim**-0.5
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Apply causal mask
        kv_len = k.shape[2]
        if mask is None:
            # Create causal mask: position i can attend to positions <= i
            # For cached inference, we need to account for the cache offset
            q_positions = jnp.arange(cache_index, cache_index + seq_len)
            k_positions = jnp.arange(kv_len)
            mask = q_positions[:, None] >= k_positions[None, :]
            mask = mask[None, None, :, :]  # [1, 1, q_len, kv_len]

        # Apply mask (set masked positions to -inf)
        attn_scores = jnp.where(mask, attn_scores, jnp.finfo(attn_scores.dtype).min)

        # Softmax
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Dropout
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        # Apply attention to values
        # [batch, heads, q_len, kv_len] @ [batch, heads, kv_len, head_dim]
        # -> [batch, heads, q_len, head_dim]
        output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Reshape back to [batch, seq_len, hidden_dim]
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output, cache
