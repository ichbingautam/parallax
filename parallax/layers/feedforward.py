"""Feed-Forward Network with SwiGLU activation.

The SwiGLU activation function is used in modern LLMs like Llama, PaLM, and Gemma.
It provides better performance than ReLU or GELU while maintaining computational
efficiency through the gated linear unit structure.

Reference: https://arxiv.org/abs/2002.05202
"""

import jax.numpy as jnp
from flax import linen as nn

from parallax.config import TransformerConfig

Array = jnp.ndarray


class FeedForward(nn.Module):
    """Feed-Forward Network with SwiGLU activation.

    The SwiGLU FFN computes:
        FFN(x) = down(SiLU(gate(x)) * up(x))

    Where:
    - gate, up: Linear projections from hidden_dim to ffn_dim
    - down: Linear projection from ffn_dim back to hidden_dim
    - SiLU: Sigmoid Linear Unit (swish) activation

    This gated structure allows the network to learn which features to pass
    through, similar to LSTM gates but in a simpler form.

    Attributes:
        config: Transformer configuration.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, x: Array, deterministic: bool = True) -> Array:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].
            deterministic: If True, disable dropout.

        Returns:
            Output tensor of the same shape.
        """
        hidden_dim = self.config.hidden_dim
        ffn_dim = self.config.ffn_dim
        use_bias = self.config.ffn_bias

        # Gate and up projections (compute in parallel for efficiency)
        gate = nn.Dense(
            ffn_dim,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            name="gate_proj",
        )(x)

        up = nn.Dense(
            ffn_dim,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            name="up_proj",
        )(x)

        # SwiGLU: SiLU(gate) * up
        hidden = nn.silu(gate) * up

        # Down projection back to hidden_dim
        output = nn.Dense(
            hidden_dim,
            use_bias=use_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            name="down_proj",
        )(hidden)

        # Dropout
        output = nn.Dropout(rate=self.config.dropout_rate)(output, deterministic=deterministic)

        return output
