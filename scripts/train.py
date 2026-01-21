#!/usr/bin/env python3
"""Main training script for Parallax Transformer.

Supports both single-device and distributed (pmap) training modes.

Usage:
    # Single device training
    python scripts/train.py --mode single --dataset tiny_shakespeare

    # Overfit test (prove model correctness)
    python scripts/train.py --mode overfit --steps 500

    # Distributed training
    python scripts/train.py --mode distributed --dataset tiny_shakespeare
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from parallax.config import SMALL_CONFIG, TINY_CONFIG, TransformerConfig
from parallax.distributed.pmap_trainer import PmapTrainer, get_device_info
from parallax.model.transformer import create_model
from parallax.training.data import (
    CharacterTokenizer,
    TextDataset,
    download_tiny_shakespeare,
    load_text_file,
)
from parallax.training.optimizer import create_optimizer, create_train_state
from parallax.training.train_step import train_step


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Parallax Transformer")

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "distributed", "overfit"],
        help="Training mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_shakespeare",
        help="Dataset to use (tiny_shakespeare or path to text file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "small"],
        help="Model configuration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (per device for distributed)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs",
        help="Output directory for checkpoints and logs",
    )

    return parser.parse_args()


def train_overfit(args: argparse.Namespace) -> None:
    """Run overfit test to prove model correctness."""
    print("=" * 60)
    print("OVERFIT TEST - Proving Mathematical Correctness")
    print("=" * 60)

    # Simple repeated pattern
    text = "AB CD EF GH " * 100
    tokenizer = CharacterTokenizer(text)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,
        num_heads=4,
        num_layers=4,
        ffn_dim=128,
        max_seq_len=args.seq_len,
        dropout_rate=0.0,  # No dropout for memorization
    )

    print(f"Config: {config.hidden_dim}d, {config.num_layers}L, {config.num_heads}H")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Target: Memorize '{text[:30]}...'")
    print()

    # Create model
    model, params = create_model(config, jax.random.PRNGKey(args.seed))
    optimizer = create_optimizer(
        learning_rate=args.learning_rate,
        warmup_steps=0,
        total_steps=args.steps,
        weight_decay=0.0,
    )
    state = create_train_state(model, params, optimizer)

    # Create single batch to overfit
    tokens = jnp.array([tokenizer.encode(text[: args.seq_len + 1])])
    batch = {
        "input_ids": tokens[:, :-1],
        "targets": tokens[:, 1:],
    }

    # Training loop
    losses = []
    for step in tqdm(range(args.steps), desc="Training"):
        rng = jax.random.PRNGKey(step)
        state, metrics = train_step(state, batch, rng)
        losses.append(float(metrics.loss))

        if step % args.log_interval == 0:
            tqdm.write(
                f"Step {step:4d} | Loss: {metrics.loss:.4f} | Grad Norm: {metrics.grad_norm:.4f}"
            )

    # Final evaluation
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    logits, _ = model.apply(
        {"params": state.params},
        batch["input_ids"],
        deterministic=True,
    )
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = float(jnp.mean(predictions == batch["targets"]))

    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Accuracy: {accuracy * 100:.1f}%")

    # Decode predictions
    pred_text = tokenizer.decode(np.array(predictions[0]))
    target_text = tokenizer.decode(np.array(batch["targets"][0]))
    print(f"Target:     '{target_text}'")
    print(f"Prediction: '{pred_text}'")

    if accuracy >= 0.95:
        print("\n✅ OVERFIT TEST PASSED - Model is mathematically correct!")
    else:
        print(f"\n❌ OVERFIT TEST FAILED - Accuracy {accuracy * 100:.1f}% < 95%")


def train_single(args: argparse.Namespace) -> None:
    """Train on a single device."""
    print("=" * 60)
    print("SINGLE DEVICE TRAINING")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")

    # Load dataset
    if args.dataset == "tiny_shakespeare":
        path = download_tiny_shakespeare()
        text = load_text_file(path)
    else:
        text = load_text_file(args.dataset)

    tokenizer = CharacterTokenizer(text)
    dataset = TextDataset(text, seq_len=args.seq_len, tokenizer=tokenizer)

    print(f"Dataset: {len(text)} chars, {len(dataset)} examples")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Config
    config = TINY_CONFIG if args.config == "tiny" else SMALL_CONFIG

    # Override vocab size
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_dim=config.ffn_dim,
        max_seq_len=args.seq_len,
        dropout_rate=config.dropout_rate,
    )

    print(f"Model: {config.hidden_dim}d, {config.num_layers}L, {config.num_heads}H")
    print(f"Params: ~{config.num_params() / 1e6:.2f}M")
    print()

    # Create model
    model, params = create_model(config, jax.random.PRNGKey(args.seed))
    optimizer = create_optimizer(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )
    state = create_train_state(model, params, optimizer)

    # Training loop
    rng = np.random.default_rng(args.seed)
    batch_iter = dataset.batch_iterator(args.batch_size, shuffle=True, rng=rng)

    step = 0
    start_time = time.time()

    while step < args.steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = dataset.batch_iterator(args.batch_size, shuffle=True, rng=rng)
            batch = next(batch_iter)

        batch = {k: jnp.array(v) for k, v in batch.items()}
        step_rng = jax.random.PRNGKey(step)
        state, metrics = train_step(state, batch, step_rng)

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"Step {step:5d} | Loss: {metrics.loss:.4f} | "
                f"Grad Norm: {metrics.grad_norm:.4f} | "
                f"Steps/sec: {steps_per_sec:.1f}"
            )

        step += 1

    print()
    print(f"Training complete! Final loss: {metrics.loss:.4f}")


def train_distributed(args: argparse.Namespace) -> None:
    """Train with jax.pmap across multiple devices."""
    print("=" * 60)
    print("DISTRIBUTED TRAINING (pmap)")
    print("=" * 60)

    device_info = get_device_info()
    print(f"Devices: {device_info['num_local_devices']}x {device_info['device_kind']}")

    num_devices = device_info["num_local_devices"]
    global_batch_size = args.batch_size * num_devices

    print(f"Local batch size: {args.batch_size}")
    print(f"Global batch size: {global_batch_size}")

    # Load dataset
    if args.dataset == "tiny_shakespeare":
        path = download_tiny_shakespeare()
        text = load_text_file(path)
    else:
        text = load_text_file(args.dataset)

    tokenizer = CharacterTokenizer(text)
    dataset = TextDataset(text, seq_len=args.seq_len, tokenizer=tokenizer)

    # Config
    config = TINY_CONFIG if args.config == "tiny" else SMALL_CONFIG

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_dim=config.ffn_dim,
        max_seq_len=args.seq_len,
        dropout_rate=config.dropout_rate,
    )

    print(f"Model: {config.hidden_dim}d, {config.num_layers}L")
    print()

    # Create model and trainer
    model, params = create_model(config, jax.random.PRNGKey(args.seed))
    optimizer = create_optimizer(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )

    trainer = PmapTrainer(model, params, optimizer)

    # Training loop
    rng = np.random.default_rng(args.seed)
    batch_iter = dataset.batch_iterator(global_batch_size, shuffle=True, rng=rng)

    step = 0
    start_time = time.time()

    while step < args.steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = dataset.batch_iterator(global_batch_size, shuffle=True, rng=rng)
            batch = next(batch_iter)

        batch = {k: jnp.array(v) for k, v in batch.items()}
        step_rng = jax.random.PRNGKey(step)
        metrics = trainer.train_step(batch, step_rng)

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"Step {step:5d} | Loss: {metrics.loss:.4f} | "
                f"Grad Norm: {metrics.grad_norm:.4f} | "
                f"Steps/sec: {steps_per_sec:.1f}"
            )

        step += 1

    print()
    print(f"Training complete! Final loss: {metrics.loss:.4f}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Print JAX info
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    if args.mode == "overfit":
        train_overfit(args)
    elif args.mode == "single":
        train_single(args)
    elif args.mode == "distributed":
        train_distributed(args)


if __name__ == "__main__":
    main()
