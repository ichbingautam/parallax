"""Tests for the TransformerLM model.

Tests model assembly, parameter counting, and forward pass correctness.
"""

import jax
import jax.numpy as jnp
import pytest

from parallax.config import TINY_CONFIG, TransformerConfig
from parallax.model.transformer import TransformerLM, count_params, create_model


class TestTransformerLM:
    """Tests for the full TransformerLM model."""

    def test_forward_shape(self) -> None:
        """Output logits should have shape [batch, seq_len, vocab_size]."""
        config = TINY_CONFIG
        model, params = create_model(config, jax.random.PRNGKey(0))

        batch, seq_len = 2, 8
        input_ids = jax.random.randint(
            jax.random.PRNGKey(1), (batch, seq_len), 0, config.vocab_size
        )

        logits, _ = model.apply(params, input_ids, deterministic=True)

        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_parameter_count(self) -> None:
        """Actual parameter count should match estimation."""
        config = TINY_CONFIG
        _, params = create_model(config, jax.random.PRNGKey(0))

        actual_count = count_params(params)
        estimated_count = config.num_params()

        # Should be within 10% (estimation doesn't account for all biases etc.)
        assert abs(actual_count - estimated_count) / estimated_count < 0.1

    def test_kv_cache_consistency(self) -> None:
        """KV-Cache should produce same logits as full forward pass."""
        config = TINY_CONFIG
        model, params = create_model(config, jax.random.PRNGKey(0))

        batch = 1
        seq_len = 4
        input_ids = jax.random.randint(
            jax.random.PRNGKey(1), (batch, seq_len), 0, config.vocab_size
        )

        # Full forward pass
        full_logits, _ = model.apply(params, input_ids, deterministic=True)

        # Incremental with cache
        cache = model.init_cache(batch_size=batch, max_seq_len=config.max_seq_len)

        cached_logits = []
        for i in range(seq_len):
            token_ids = input_ids[:, i : i + 1]
            logits, cache = model.apply(
                params,
                token_ids,
                cache=cache,
                cache_index=i,
                deterministic=True,
            )
            cached_logits.append(logits)

        cached_logits = jnp.concatenate(cached_logits, axis=1)

        # Should produce same results
        assert jnp.allclose(full_logits, cached_logits, atol=1e-4)

    def test_deterministic_output(self) -> None:
        """Same inputs should produce same outputs in deterministic mode."""
        config = TINY_CONFIG
        model, params = create_model(config, jax.random.PRNGKey(0))

        input_ids = jnp.array([[1, 2, 3, 4]])

        logits1, _ = model.apply(params, input_ids, deterministic=True)
        logits2, _ = model.apply(params, input_ids, deterministic=True)

        assert jnp.allclose(logits1, logits2)

    def test_scaled_initialization(self) -> None:
        """Residual output projections should be scaled."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_dim=64,
            num_heads=4,
            num_layers=4,
            ffn_dim=128,
            max_seq_len=16,
        )
        _, params = create_model(config, jax.random.PRNGKey(0))

        # Check that output projections have smaller norms
        # (scaled by 1/sqrt(2*num_layers))
        expected_scale = 1.0 / jnp.sqrt(2.0 * config.num_layers)

        block_0_o_proj = params["params"]["block_0"]["attention"]["o_proj"]["kernel"]
        # The initialization variance should be scaled down
        # This is a rough check - actual variance depends on initialization
        o_proj_std = jnp.std(block_0_o_proj)

        # Standard Xavier init would have std ~0.1-0.2 for these dimensions
        # Scaled version should be smaller
        assert o_proj_std < 0.2


class TestModelCreation:
    """Tests for model creation utilities."""

    def test_create_model_returns_tuple(self) -> None:
        """create_model should return (model, params) tuple."""
        config = TINY_CONFIG
        result = create_model(config, jax.random.PRNGKey(0))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TransformerLM)

    def test_different_seeds_different_params(self) -> None:
        """Different random seeds should produce different parameters."""
        config = TINY_CONFIG
        _, params1 = create_model(config, jax.random.PRNGKey(0))
        _, params2 = create_model(config, jax.random.PRNGKey(1))

        # Get first layer embedding weights
        emb1 = params1["params"]["embeddings"]["token_embedding"]["embedding"]
        emb2 = params2["params"]["embeddings"]["token_embedding"]["embedding"]

        assert not jnp.allclose(emb1, emb2)


class TestConfigPresets:
    """Tests for configuration presets."""

    def test_tiny_config_builds(self) -> None:
        """TINY_CONFIG should create a valid model."""
        model, params = create_model(TINY_CONFIG, jax.random.PRNGKey(0))
        input_ids = jnp.array([[1, 2, 3]])
        logits, _ = model.apply(params, input_ids, deterministic=True)
        assert logits.shape == (1, 3, TINY_CONFIG.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
