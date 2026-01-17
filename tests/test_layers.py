"""Unit tests for Transformer layers.

Tests numerical correctness and shape consistency of:
- RMSNorm
- RotaryPositionalEmbedding
- MultiHeadAttention
- FeedForward
- Embeddings
"""

import jax
import jax.numpy as jnp
import pytest

from parallax.config import TINY_CONFIG, TransformerConfig
from parallax.layers import (
    Embeddings,
    FeedForward,
    KVCache,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
)


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_output_shape(self) -> None:
        """Output shape should match input shape."""
        config = TINY_CONFIG
        norm = RMSNorm(config=config)

        x = jnp.ones((2, 4, config.hidden_dim))
        params = norm.init(jax.random.PRNGKey(0), x)
        output = norm.apply(params, x)

        assert output.shape == x.shape

    def test_normalization_property(self) -> None:
        """Normalized output should have RMS close to 1."""
        config = TINY_CONFIG
        norm = RMSNorm(config=config)

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, config.hidden_dim))
        params = norm.init(jax.random.PRNGKey(1), x)
        output = norm.apply(params, x)

        # After normalization (before scale), RMS should be ~1
        rms = jnp.sqrt(jnp.mean(output**2, axis=-1))
        # With learned scale=1 initially, RMS should be close to 1
        assert jnp.allclose(rms, 1.0, atol=0.1)

    def test_numerical_stability(self) -> None:
        """Should handle very small inputs without NaN."""
        config = TINY_CONFIG
        norm = RMSNorm(config=config)

        x = jnp.ones((1, 1, config.hidden_dim)) * 1e-10
        params = norm.init(jax.random.PRNGKey(0), x)
        output = norm.apply(params, x)

        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))


class TestRotaryPositionalEmbedding:
    """Tests for RoPE layer."""

    def test_output_shape(self) -> None:
        """Output shape should match input shape."""
        config = TINY_CONFIG
        rope = RotaryPositionalEmbedding(config=config)

        batch, heads, seq_len, head_dim = 2, 4, 8, config.head_dim
        x = jnp.ones((batch, heads, seq_len, head_dim))
        params = rope.init(jax.random.PRNGKey(0), x)
        output = rope.apply(params, x)

        assert output.shape == x.shape

    def test_rotation_is_norm_preserving(self) -> None:
        """Rotation should preserve vector norms."""
        config = TINY_CONFIG
        rope = RotaryPositionalEmbedding(config=config)

        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 8, config.head_dim))
        params = rope.init(jax.random.PRNGKey(1), x)
        output = rope.apply(params, x)

        # Norms should be preserved (rotation is orthogonal)
        input_norms = jnp.linalg.norm(x, axis=-1)
        output_norms = jnp.linalg.norm(output, axis=-1)
        assert jnp.allclose(input_norms, output_norms, atol=1e-5)

    def test_position_offset(self) -> None:
        """Different offsets should produce different outputs."""
        config = TINY_CONFIG
        rope = RotaryPositionalEmbedding(config=config)

        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 4, config.head_dim))
        params = rope.init(jax.random.PRNGKey(1), x)

        output_offset_0 = rope.apply(params, x, offset=0)
        output_offset_4 = rope.apply(params, x, offset=4)

        # Different offsets should produce different outputs
        assert not jnp.allclose(output_offset_0, output_offset_4)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention layer."""

    def test_output_shape(self) -> None:
        """Output shape should be [batch, seq_len, hidden_dim]."""
        config = TINY_CONFIG
        attn = MultiHeadAttention(config=config)

        batch, seq_len = 2, 8
        x = jnp.ones((batch, seq_len, config.hidden_dim))
        params = attn.init(jax.random.PRNGKey(0), x, deterministic=True)
        output, _ = attn.apply(params, x, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_causal_masking(self) -> None:
        """Attention should be causal (no attending to future)."""
        config = TransformerConfig(
            vocab_size=256,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            max_seq_len=16,
            dropout_rate=0.0,  # Disable dropout for determinism
            use_rope=False,  # Simpler test without RoPE
        )
        attn = MultiHeadAttention(config=config)

        batch, seq_len = 1, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, config.hidden_dim))
        params = attn.init(jax.random.PRNGKey(1), x, deterministic=True)

        # Compute outputs for full sequence
        full_output, _ = attn.apply(params, x, deterministic=True)

        # Output at position i should only depend on positions 0..i
        # Test: changing position j>i should not affect output at i
        x_modified = x.at[:, -1, :].set(x[:, -1, :] * 2)  # Modify last position
        modified_output, _ = attn.apply(params, x_modified, deterministic=True)

        # All positions except the last should be unchanged
        assert jnp.allclose(full_output[:, :-1, :], modified_output[:, :-1, :], atol=1e-5)

    def test_kv_cache(self) -> None:
        """KV-Cache should produce same results as full computation."""
        config = TINY_CONFIG
        attn = MultiHeadAttention(config=config)

        batch = 1
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, 4, config.hidden_dim))
        params = attn.init(jax.random.PRNGKey(1), x, deterministic=True)

        # Full computation
        full_output, _ = attn.apply(params, x, deterministic=True)

        # Incremental computation with cache
        cache = KVCache.init(
            batch_size=batch,
            num_heads=config.num_heads,
            max_seq_len=16,
            head_dim=config.head_dim,
        )

        outputs = []
        for i in range(4):
            token = x[:, i : i + 1, :]
            output, cache = attn.apply(
                params,
                token,
                cache=cache,
                cache_index=i,
                deterministic=True,
            )
            outputs.append(output)

        cached_output = jnp.concatenate(outputs, axis=1)

        # Should produce same results (within numerical tolerance)
        assert jnp.allclose(full_output, cached_output, atol=1e-4)


class TestFeedForward:
    """Tests for FeedForward layer."""

    def test_output_shape(self) -> None:
        """Output shape should match input shape."""
        config = TINY_CONFIG
        ffn = FeedForward(config=config)

        batch, seq_len = 2, 8
        x = jnp.ones((batch, seq_len, config.hidden_dim))
        params = ffn.init(jax.random.PRNGKey(0), x, deterministic=True)
        output = ffn.apply(params, x, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_swiglu_activation(self) -> None:
        """SwiGLU should produce non-linear output."""
        config = TINY_CONFIG
        ffn = FeedForward(config=config)

        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, config.hidden_dim))
        params = ffn.init(jax.random.PRNGKey(1), x, deterministic=True)

        # Output should not be a simple linear transformation
        output = ffn.apply(params, x, deterministic=True)
        output_scaled = ffn.apply(params, x * 2, deterministic=True)

        # Non-linearity means output scaling is not proportional
        assert not jnp.allclose(output * 2, output_scaled, atol=0.1)


class TestEmbeddings:
    """Tests for Embeddings layer."""

    def test_embed_shape(self) -> None:
        """Embedding output should be [batch, seq_len, hidden_dim]."""
        config = TINY_CONFIG
        embed = Embeddings(config=config)

        batch, seq_len = 2, 8
        input_ids = jax.random.randint(
            jax.random.PRNGKey(0), (batch, seq_len), 0, config.vocab_size
        )
        params = embed.init(jax.random.PRNGKey(1), input_ids)
        output = embed.apply(params, input_ids)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_unembed_shape(self) -> None:
        """Unembed should produce [batch, seq_len, vocab_size]."""
        config = TINY_CONFIG
        embed = Embeddings(config=config)

        batch, seq_len = 2, 8
        input_ids = jax.random.randint(
            jax.random.PRNGKey(0), (batch, seq_len), 0, config.vocab_size
        )
        params = embed.init(jax.random.PRNGKey(1), input_ids)

        hidden = embed.apply(params, input_ids)
        logits = embed.apply(params, hidden, method=embed.unembed)

        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_weight_tying(self) -> None:
        """With weight tying, embed and unembed should use same weights."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_dim=64,
            max_seq_len=16,
            tie_word_embeddings=True,
        )
        embed = Embeddings(config=config)

        input_ids = jnp.array([[0, 1, 2]])
        params = embed.init(jax.random.PRNGKey(0), input_ids)

        # With weight tying, there should be no separate lm_head
        param_keys = list(params["params"].keys())
        assert "lm_head" not in param_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
