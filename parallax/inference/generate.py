"""Autoregressive text generation with KV-Cache.

Implements efficient token-by-token generation using:
- KV-Cache for O(1) per-token computation
- Various sampling strategies (greedy, temperature, top-k, top-p)
- jax.lax.while_loop for compiled generation
"""

from typing import Any

import jax
import jax.numpy as jnp

from parallax.config import TransformerConfig
from parallax.layers.attention import KVCache
from parallax.model.transformer import TransformerLM

Array = jnp.ndarray
PyTree = Any


def sample_token(
    logits: Array,
    rng: Array,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Array:
    """Sample a single token from logits.

    Args:
        logits: Logits for the next token [vocab_size] or [batch, vocab_size].
        rng: Random key for sampling.
        temperature: Sampling temperature (1.0 = neutral, <1.0 = sharper, >1.0 = flatter).
        top_k: If set, only sample from top k tokens.
        top_p: If set, nucleus sampling threshold (sample from smallest set with prob >= p).

    Returns:
        Sampled token index.
    """
    # Handle batch dimension
    if logits.ndim == 1:
        logits = logits[None, :]
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, vocab_size = logits.shape

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        # Get top-k values and indices
        top_k = min(top_k, vocab_size)
        top_k_logits, _ = jax.lax.top_k(logits, top_k)
        min_top_k = top_k_logits[:, -1:]

        # Mask out tokens below top-k threshold
        logits = jnp.where(
            logits < min_top_k,
            jnp.finfo(logits.dtype).min,
            logits,
        )

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

        # Create mask for tokens to remove
        sorted_mask = cumulative_probs > top_p
        # Shift mask to keep first token above threshold
        sorted_mask = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=bool), sorted_mask[:, :-1]],
            axis=-1,
        )

        # Scatter mask back to original order
        mask = jnp.zeros_like(sorted_mask)
        mask = mask.at[jnp.arange(batch_size)[:, None], sorted_indices].set(sorted_mask)

        logits = jnp.where(mask, jnp.finfo(logits.dtype).min, logits)

    # Sample from distribution
    token = jax.random.categorical(rng, logits, axis=-1)

    if squeeze_output:
        token = token[0]

    return token


def generate(
    model: TransformerLM,
    params: PyTree,
    prompt_tokens: Array,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    rng: Array | None = None,
) -> Array:
    """Generate tokens autoregressively.

    Uses KV-caching for efficient generation (O(1) per token instead of O(N)).

    Args:
        model: TransformerLM model.
        params: Model parameters (just the 'params' dict, not wrapped).
        prompt_tokens: Initial prompt tokens [seq_len] or [batch, seq_len].
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        eos_token_id: End of sequence token (stops generation if produced).
        rng: Random key (uses PRNGKey(0) if not provided).

    Returns:
        Generated tokens including prompt [batch, total_len].
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Handle unbatched input
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens[None, :]

    batch_size, prompt_len = prompt_tokens.shape
    config = model.config

    # Initialize KV-Cache
    cache = model.init_cache(batch_size, max_seq_len=prompt_len + max_new_tokens)

    # Process prompt to fill cache
    logits, cache = model.apply(
        {"params": params},
        prompt_tokens,
        cache=cache,
        cache_index=0,
        deterministic=True,
    )

    # Get logits for last prompt token (next token prediction)
    next_token_logits = logits[:, -1, :]

    # Initialize output with prompt
    output_tokens = prompt_tokens

    # Generate tokens one at a time
    for i in range(max_new_tokens):
        # Split RNG for this step
        rng, step_rng = jax.random.split(rng)

        # Sample next token
        next_token = sample_token(
            next_token_logits,
            step_rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Append to output
        output_tokens = jnp.concatenate(
            [output_tokens, next_token[:, None]], axis=-1
        )

        # Check for EOS
        if eos_token_id is not None and jnp.all(next_token == eos_token_id):
            break

        # Get logits for next token (using cache)
        cache_index = prompt_len + i
        next_logits, cache = model.apply(
            {"params": params},
            next_token[:, None],
            cache=cache,
            cache_index=cache_index,
            deterministic=True,
        )
        next_token_logits = next_logits[:, 0, :]

    return output_tokens


def generate_compiled(
    model: TransformerLM,
    params: PyTree,
    prompt_tokens: Array,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    rng: Array | None = None,
) -> Array:
    """Compiled generation using jax.lax.while_loop.

    This is more efficient for long generations as it compiles the
    entire generation loop.

    Args:
        model: TransformerLM model.
        params: Model parameters.
        prompt_tokens: Initial prompt tokens [batch, seq_len].
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        rng: Random key.

    Returns:
        Generated tokens including prompt.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens[None, :]

    batch_size, prompt_len = prompt_tokens.shape
    total_len = prompt_len + max_new_tokens

    # Initialize KV-Cache
    cache = model.init_cache(batch_size, max_seq_len=total_len)

    # Process prompt
    logits, cache = model.apply(
        {"params": params},
        prompt_tokens,
        cache=cache,
        cache_index=0,
        deterministic=True,
    )

    # Sample first new token
    rng, step_rng = jax.random.split(rng)
    next_token = sample_token(logits[:, -1, :], step_rng, temperature=temperature)

    # Initialize state for while loop
    # State: (tokens, cache, position, rng)
    # tokens has shape [batch, total_len] with padding
    tokens = jnp.zeros((batch_size, total_len), dtype=jnp.int32)
    tokens = tokens.at[:, :prompt_len].set(prompt_tokens)
    tokens = tokens.at[:, prompt_len].set(next_token)

    init_state = (tokens, cache, prompt_len + 1, rng)

    def cond_fn(state):
        _, _, pos, _ = state
        return pos < total_len

    def body_fn(state):
        tokens, cache, pos, rng = state

        # Get last token
        last_token = tokens[:, pos - 1 : pos]

        # Forward pass with cache
        logits, cache = model.apply(
            {"params": params},
            last_token,
            cache=cache,
            cache_index=pos - 1,
            deterministic=True,
        )

        # Sample
        rng, step_rng = jax.random.split(rng)
        next_token = sample_token(logits[:, 0, :], step_rng, temperature=temperature)

        # Update tokens
        tokens = tokens.at[:, pos].set(next_token)

        return (tokens, cache, pos + 1, rng)

    # Run generation loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    tokens, _, _, _ = final_state

    return tokens
