#!/usr/bin/env python3
"""Interactive text generation script.

Usage:
    python scripts/generate.py --checkpoint path/to/checkpoint --prompt "To be or not to be"
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from parallax.config import TINY_CONFIG, TransformerConfig
from parallax.inference.generate import generate
from parallax.model.transformer import create_model
from parallax.training.data import CharacterTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with Parallax Transformer")

    parser.add_argument(
        "--prompt",
        type=str,
        default="To be or not to be",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (loop for multiple prompts)",
    )

    return parser.parse_args()


def create_demo_model(vocab: str) -> tuple:
    """Create a tiny demo model for testing.

    In production, this would load a trained checkpoint.
    """
    tokenizer = CharacterTokenizer(vocab)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,
        num_heads=4,
        num_layers=4,
        ffn_dim=128,
        max_seq_len=256,
        dropout_rate=0.0,
    )

    model, params = create_model(config, jax.random.PRNGKey(42))

    return model, params["params"], tokenizer


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Parallax Text Generator")
    print("=" * 60)

    # For demo purposes, create a simple model
    # In production, you would load a trained checkpoint
    vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'\n"
    model, params, tokenizer = create_demo_model(vocab)

    print(f"Model: {model.config.hidden_dim}d, {model.config.num_layers}L")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        print("-" * 40)

        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() == "quit":
                break

            prompt_tokens = jnp.array([tokenizer.encode(prompt)])

            rng = jax.random.PRNGKey(np.random.randint(0, 10000))
            output = generate(
                model,
                params,
                prompt_tokens,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p,
                rng=rng,
            )

            generated_text = tokenizer.decode(np.array(output[0]))
            print(f"\nGenerated:\n{generated_text}")

    else:
        # Single generation
        print(f"Prompt: {args.prompt}")
        print(f"Temperature: {args.temperature}")
        print("-" * 40)

        prompt_tokens = jnp.array([tokenizer.encode(args.prompt)])

        rng = jax.random.PRNGKey(args.seed)
        output = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            rng=rng,
        )

        generated_text = tokenizer.decode(np.array(output[0]))
        print(f"\n{generated_text}")

        print()
        print("-" * 40)
        print("Note: This is an untrained model - output is random.")
        print("Train a model with: python scripts/train.py --mode single")


if __name__ == "__main__":
    main()
