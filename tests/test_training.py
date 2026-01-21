"""Tests for training components and the overfit test.

Includes the critical "Overfit Test" - training a tiny model to memorize
a single batch, which proves mathematical correctness.
"""

import jax
import jax.numpy as jnp
import pytest

from parallax.config import TransformerConfig
from parallax.model.transformer import create_model
from parallax.training.data import CharacterTokenizer, TextDataset
from parallax.training.loss import cross_entropy_loss, z_loss
from parallax.training.optimizer import (
    create_learning_rate_schedule,
    create_optimizer,
    create_train_state,
    global_norm,
)
from parallax.training.train_step import Metrics, train_step


class TestLossFunctions:
    """Tests for loss functions."""

    def test_cross_entropy_shape(self) -> None:
        """Cross-entropy loss should return a scalar."""
        batch, seq_len, vocab_size = 2, 8, 100
        logits = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, vocab_size))
        targets = jax.random.randint(jax.random.PRNGKey(1), (batch, seq_len), 0, vocab_size)

        loss = cross_entropy_loss(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_perfect_prediction(self) -> None:
        """Loss should be near zero for perfect predictions."""
        batch, seq_len, vocab_size = 1, 4, 10

        # Create logits that strongly predict the targets
        targets = jnp.array([[0, 1, 2, 3]])
        logits = jnp.zeros((batch, seq_len, vocab_size)) - 100
        for i in range(seq_len):
            logits = logits.at[0, i, targets[0, i]].set(100)

        loss = cross_entropy_loss(logits, targets)
        assert loss < 0.01

    def test_cross_entropy_with_ignore_index(self) -> None:
        """Ignored positions should not contribute to loss."""
        batch, seq_len, vocab_size = 1, 4, 10
        logits = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, vocab_size))

        targets_no_ignore = jnp.array([[0, 1, 2, 3]])
        targets_with_ignore = jnp.array([[0, 1, -100, -100]])

        loss_no_ignore = cross_entropy_loss(logits, targets_no_ignore)
        loss_with_ignore = cross_entropy_loss(logits, targets_with_ignore)

        # Losses should be different (ignored positions excluded)
        assert not jnp.allclose(loss_no_ignore, loss_with_ignore)

    def test_z_loss_penalizes_large_logits(self) -> None:
        """Z-loss should be larger for larger logits."""
        batch, seq_len, vocab_size = 1, 4, 10

        small_logits = jnp.ones((batch, seq_len, vocab_size))
        large_logits = jnp.ones((batch, seq_len, vocab_size)) * 100

        small_z = z_loss(small_logits)
        large_z = z_loss(large_logits)

        assert large_z > small_z


class TestOptimizer:
    """Tests for optimizer configuration."""

    def test_schedule_warmup(self) -> None:
        """Learning rate should increase during warmup."""
        schedule = create_learning_rate_schedule(
            learning_rate=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )

        lr_0 = schedule(0)
        lr_50 = schedule(50)
        lr_100 = schedule(100)

        assert lr_0 < lr_50 < lr_100
        assert jnp.isclose(lr_100, 1e-3, rtol=0.01)

    def test_schedule_decay(self) -> None:
        """Learning rate should decay after warmup."""
        schedule = create_learning_rate_schedule(
            learning_rate=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )

        lr_100 = schedule(100)
        lr_500 = schedule(500)
        lr_999 = schedule(999)

        assert lr_100 > lr_500 > lr_999

    def test_global_norm(self) -> None:
        """Global norm should compute L2 norm correctly."""
        tree = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}
        norm = global_norm(tree)
        assert jnp.isclose(norm, 5.0)  # sqrt(9 + 16) = 5


class TestTokenizer:
    """Tests for character tokenizer."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encoding then decoding should return original text."""
        text = "Hello, World!"
        tokenizer = CharacterTokenizer(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_vocab_size_from_text(self) -> None:
        """Vocabulary size should match unique characters."""
        text = "aabbcc"
        tokenizer = CharacterTokenizer(text)
        assert tokenizer.vocab_size == 3  # 'a', 'b', 'c'


class TestDataset:
    """Tests for text dataset."""

    def test_dataset_length(self) -> None:
        """Dataset length should be text_length - seq_len."""
        text = "0123456789"
        dataset = TextDataset(text, seq_len=3)
        # Tokens: [0,1,2,3,4,5,6,7,8,9], num_examples = 10 - 3 = 7
        assert len(dataset) == 7

    def test_batch_shapes(self) -> None:
        """Batches should have correct shapes."""
        text = "abcdefghijklmnopqrstuvwxyz"
        dataset = TextDataset(text, seq_len=4)

        batch_size = 2
        for batch in dataset.batch_iterator(batch_size, shuffle=False):
            assert batch["input_ids"].shape == (batch_size, 4)
            assert batch["targets"].shape == (batch_size, 4)
            break


class TestTrainStep:
    """Tests for the training step."""

    def test_train_step_runs(self) -> None:
        """Train step should execute without errors."""
        config = TransformerConfig(
            vocab_size=26,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_dim=64,
            max_seq_len=16,
            dropout_rate=0.0,
        )

        model, params = create_model(config, jax.random.PRNGKey(0))
        optimizer = create_optimizer(
            learning_rate=1e-3,
            warmup_steps=10,
            total_steps=100,
        )
        state = create_train_state(model, params, optimizer)

        batch = {
            "input_ids": jnp.zeros((2, 8), dtype=jnp.int32),
            "targets": jnp.ones((2, 8), dtype=jnp.int32),
        }
        rng = jax.random.PRNGKey(1)

        new_state, metrics = train_step(state, batch, rng)

        assert new_state.step == 1
        assert isinstance(metrics, Metrics)
        assert not jnp.isnan(metrics.loss)

    def test_train_step_decreases_loss(self) -> None:
        """Multiple train steps should decrease loss."""
        config = TransformerConfig(
            vocab_size=26,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_dim=64,
            max_seq_len=16,
            dropout_rate=0.0,
        )

        model, params = create_model(config, jax.random.PRNGKey(0))
        optimizer = create_optimizer(
            learning_rate=1e-2,  # Higher LR for faster convergence
            warmup_steps=0,
            total_steps=100,
            weight_decay=0.0,
        )
        state = create_train_state(model, params, optimizer)

        # Same batch every time (should overfit)
        batch = {
            "input_ids": jnp.array([[1, 2, 3, 4, 5]]),
            "targets": jnp.array([[2, 3, 4, 5, 6]]),
        }

        losses = []
        for i in range(20):
            rng = jax.random.PRNGKey(i)
            state, metrics = train_step(state, batch, rng)
            losses.append(float(metrics.loss))

        # Loss should decrease overall
        assert losses[-1] < losses[0]


class TestOverfit:
    """The critical "Overfit Test" - proves mathematical correctness."""

    def test_overfit_single_batch(self) -> None:
        """Model should memorize a single repeated pattern to near-zero loss.

        This is the most important test for a new LM implementation.
        If it can't memorize 10 tokens, it won't learn 10 billion.
        """
        # Simple repeated pattern
        text = "AB " * 100  # "AB AB AB AB ..."

        tokenizer = CharacterTokenizer(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=64,
            num_heads=4,
            num_layers=4,
            ffn_dim=128,
            max_seq_len=32,
            dropout_rate=0.0,  # No dropout for deterministic memorization
        )

        model, params = create_model(config, jax.random.PRNGKey(42))
        optimizer = create_optimizer(
            learning_rate=1e-3,
            warmup_steps=0,
            total_steps=1000,
            weight_decay=0.0,  # No regularization
            max_grad_norm=1.0,
        )
        state = create_train_state(model, params, optimizer)

        # Create a single batch to overfit
        tokens = jnp.array([tokenizer.encode(text[:33])])  # seq_len + 1
        batch = {
            "input_ids": tokens[:, :-1],
            "targets": tokens[:, 1:],
        }

        # Train for many steps
        for i in range(500):
            rng = jax.random.PRNGKey(i)
            state, metrics = train_step(state, batch, rng)

            if i % 100 == 0:
                print(f"Step {i}: loss = {metrics.loss:.4f}")

        # Final loss should be very low (near zero)
        assert metrics.loss < 0.1, f"Overfit test failed: loss = {metrics.loss}"

        # Verify the model can reproduce the pattern
        logits, _ = model.apply(
            {"params": state.params},
            batch["input_ids"],
            deterministic=True,
        )
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch["targets"])

        assert accuracy > 0.95, f"Overfit test failed: accuracy = {accuracy}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
