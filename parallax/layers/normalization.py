"""Normalization layers for the Transformer.

RMSNorm (Root Mean Square Normalization) is used instead of LayerNorm in modern
Transformers like Llama and Gemma. It's computationally cheaper and provides
better training stability for deep networks.

Reference: https://arxiv.org/abs/1910.07467
"""

import jax.numpy as jnp
from flax import linen as nn

from parallax.config import TransformerConfig

# Type alias for arrays
Array = jnp.ndarray


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input by its RMS (root mean square) and applies a learned
    scale parameter. Unlike LayerNorm, RMSNorm doesn't subtract the mean,
    which makes it faster and equally effective for Transformers.

    Formula:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale

    Attributes:
        config: Transformer configuration.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., hidden_dim].

        Returns:
            Normalized tensor of the same shape.
        """
        epsilon = self.config.layer_norm_epsilon

        # Compute RMS along the last dimension
        # Using float32 for numerical stability in the norm computation
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + epsilon)

        # Normalize
        x_normed = x_f32 / rms

        # Learned scale parameter (initialized to ones)
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.config.hidden_dim,),
        )

        # Apply scale and cast back to original dtype
        return (x_normed * scale).astype(x.dtype)
