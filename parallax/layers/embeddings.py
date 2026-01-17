"""Token embeddings with optional weight tying.

Weight tying (sharing parameters between input embeddings and output projection)
is a common technique to reduce model size. It was popularized by GPT-2 and is
used in many modern LLMs.

Reference: https://arxiv.org/abs/1608.05859
"""

import jax.numpy as jnp
from flax import linen as nn

from parallax.config import TransformerConfig

Array = jnp.ndarray


class Embeddings(nn.Module):
    """Token embeddings with optional weight tying.

    Maps token indices to dense vectors. When tie_word_embeddings is True,
    the embedding matrix is also used as the output projection weights,
    effectively sharing parameters and reducing model size.

    Attributes:
        config: Transformer configuration.
    """

    config: TransformerConfig

    def setup(self) -> None:
        """Initialize the embedding table."""
        self.embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="token_embedding",
        )

    def __call__(self, input_ids: Array) -> Array:
        """Embed token indices.

        Args:
            input_ids: Integer tensor of token indices [batch, seq_len].

        Returns:
            Embedded tokens [batch, seq_len, hidden_dim].
        """
        return self.embedding(input_ids)

    def unembed(self, hidden_states: Array) -> Array:
        """Project hidden states to vocabulary logits.

        When tie_word_embeddings is True, uses the transpose of the embedding
        matrix as the output projection. Otherwise, uses a separate learned
        projection.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_dim].

        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size].
        """
        if self.config.tie_word_embeddings:
            # Use embedding weights transposed: [hidden_dim, vocab_size]
            embedding_weights = self.embedding.embedding
            return jnp.dot(hidden_states, embedding_weights.T)
        else:
            # Separate output projection
            return nn.Dense(
                self.config.vocab_size,
                use_bias=False,
                kernel_init=nn.initializers.normal(stddev=0.02),
                name="lm_head",
            )(hidden_states)
