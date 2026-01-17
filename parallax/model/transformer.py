"""Transformer Language Model: Full decoder stack with embeddings.

This is the complete language model combining:
- Token embeddings (with optional weight tying)
- Stack of TransformerBlocks
- Final RMSNorm
- Output projection to vocabulary logits

The model uses scaled initialization for deep residual networks.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from parallax.config import TransformerConfig
from parallax.layers.attention import KVCache
from parallax.layers.embeddings import Embeddings
from parallax.layers.normalization import RMSNorm
from parallax.model.block import TransformerBlock

Array = jnp.ndarray
PyTree = Any


class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model.

    Architecture:
        Token IDs -> Embed -> [TransformerBlock x N] -> RMSNorm -> Unembed -> Logits

    Features:
    - Weight tying between input embeddings and output projection
    - Pre-norm architecture throughout
    - RoPE for positional encoding
    - KV-Cache support for efficient inference

    Attributes:
        config: Transformer configuration.
    """

    config: TransformerConfig

    def setup(self) -> None:
        """Initialize all model components."""
        # Embeddings (with weight tying handled internally)
        self.embeddings = Embeddings(config=self.config, name="embeddings")

        # Transformer blocks
        self.blocks = [
            TransformerBlock(config=self.config, layer_idx=i, name=f"block_{i}")
            for i in range(self.config.num_layers)
        ]

        # Final normalization
        self.final_norm = RMSNorm(config=self.config, name="final_norm")

    def __call__(
        self,
        input_ids: Array,
        mask: Array | None = None,
        cache: list[KVCache] | None = None,
        cache_index: int = 0,
        deterministic: bool = True,
    ) -> tuple[Array, list[KVCache] | None]:
        """Forward pass through the language model.

        Args:
            input_ids: Token indices [batch, seq_len].
            mask: Optional attention mask.
            cache: Optional list of KV-Caches (one per layer).
            cache_index: Current position in the cache.
            deterministic: If True, disable dropout.

        Returns:
            Tuple of:
                - Logits over vocabulary [batch, seq_len, vocab_size].
                - Updated list of KV-Caches (or None).
        """
        # Embed tokens
        x = self.embeddings(input_ids)

        # Apply transformer blocks
        new_caches = [] if cache is not None else None
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache = block(
                x,
                mask=mask,
                cache=layer_cache,
                cache_index=cache_index,
                deterministic=deterministic,
            )
            if new_caches is not None:
                new_caches.append(updated_cache)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary logits
        logits = self.embeddings.unembed(x)

        return logits, new_caches

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> list[KVCache]:
        """Initialize empty KV-Caches for all layers.

        Args:
            batch_size: Batch size.
            max_seq_len: Maximum sequence length. Defaults to config.max_seq_len.
            dtype: Data type for cache tensors.

        Returns:
            List of empty KV-Caches, one per layer.
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_seq_len

        return [
            KVCache.init(
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                max_seq_len=max_seq_len,
                head_dim=self.config.head_dim,
                dtype=dtype,
            )
            for _ in range(self.config.num_layers)
        ]


def create_model(config: TransformerConfig, rng: jax.Array) -> tuple[TransformerLM, PyTree]:
    """Create and initialize a TransformerLM model.

    Uses scaled initialization for residual layers to prevent signal explosion
    in deep networks. Output projections are scaled by 1/sqrt(2*num_layers).

    Args:
        config: Transformer configuration.
        rng: Random key for initialization.

    Returns:
        Tuple of (model, params).
    """
    model = TransformerLM(config=config)

    # Create dummy inputs for initialization
    dummy_input = jnp.zeros((1, 1), dtype=jnp.int32)

    # Initialize parameters
    params = model.init(rng, dummy_input, deterministic=True)

    # Apply scaled initialization to residual layers
    # This prevents gradient explosion in deep networks
    # Scale output projections by 1/sqrt(2*num_layers)
    scale = 1.0 / jnp.sqrt(2.0 * config.num_layers)

    def scale_residual_weights(params: PyTree) -> PyTree:
        """Scale weights of residual output projections."""
        params = params.copy()
        params_dict = params["params"]

        for i in range(config.num_layers):
            block_key = f"block_{i}"
            if block_key in params_dict:
                block = params_dict[block_key]

                # Scale attention output projection
                if "attention" in block and "o_proj" in block["attention"]:
                    o_proj = block["attention"]["o_proj"]
                    if "kernel" in o_proj:
                        block["attention"]["o_proj"]["kernel"] = o_proj["kernel"] * scale

                # Scale FFN down projection
                if "ffn" in block and "down_proj" in block["ffn"]:
                    down_proj = block["ffn"]["down_proj"]
                    if "kernel" in down_proj:
                        block["ffn"]["down_proj"]["kernel"] = down_proj["kernel"] * scale

        return params

    params = scale_residual_weights(params)

    return model, params


def count_params(params: PyTree) -> int:
    """Count the total number of parameters in a pytree.

    Args:
        params: Parameter pytree.

    Returns:
        Total number of parameters.
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
