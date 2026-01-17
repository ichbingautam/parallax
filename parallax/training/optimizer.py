"""Optimizer configuration with AdamW, schedules, and gradient clipping.

Implements the optimization stack commonly used in LLM training:
- AdamW (Adam with decoupled weight decay)
- Cosine decay learning rate schedule with warmup
- Gradient clipping by global norm

Reference: https://arxiv.org/abs/1711.05101 (Decoupled Weight Decay)
"""

from typing import Any

import jax.numpy as jnp
import optax
from flax.training import train_state

from parallax.config import TransformerConfig

# Type aliases
PyTree = Any
Schedule = optax.Schedule


def create_learning_rate_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_learning_rate: float = 0.0,
) -> Schedule:
    """Create a cosine decay schedule with linear warmup.

    The schedule:
    1. Linear warmup from 0 to learning_rate over warmup_steps
    2. Cosine decay from learning_rate to min_learning_rate over remaining steps

    Args:
        learning_rate: Peak learning rate after warmup.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_learning_rate: Minimum learning rate at end of decay.

    Returns:
        Optax schedule function.
    """
    # Linear warmup
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )

    # Cosine decay
    decay_steps = total_steps - warmup_steps
    decay_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(decay_steps, 1),
        alpha=min_learning_rate / learning_rate if learning_rate > 0 else 0,
    )

    # Combine: warmup then decay
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps],
    )

    return schedule_fn


def create_optimizer(
    learning_rate: float = 3e-4,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
    min_learning_rate: float = 0.0,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with schedule and gradient clipping.

    This setup follows best practices from GPT/Llama training:
    - AdamW with decoupled weight decay
    - Cosine LR schedule with warmup
    - Gradient clipping by global norm

    Args:
        learning_rate: Peak learning rate.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        beta1: Adam beta1 parameter.
        beta2: Adam beta2 parameter.
        epsilon: Adam epsilon for numerical stability.
        min_learning_rate: Minimum LR at end of schedule.

    Returns:
        Optax gradient transformation.
    """
    # Learning rate schedule
    schedule = create_learning_rate_schedule(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate=min_learning_rate,
    )

    # Chain: clip -> adamw
    optimizer = optax.chain(
        # Gradient clipping by global norm
        optax.clip_by_global_norm(max_grad_norm),
        # AdamW with schedule
        optax.adamw(
            learning_rate=schedule,
            b1=beta1,
            b2=beta2,
            eps=epsilon,
            weight_decay=weight_decay,
        ),
    )

    return optimizer


class TrainState(train_state.TrainState):
    """Extended TrainState with additional fields for tracking."""

    # Drop-in replacement, can extend with more fields later
    pass


def create_train_state(
    model: Any,
    params: PyTree,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Create a TrainState for training.

    Args:
        model: The Flax model (for apply_fn).
        params: Initialized model parameters.
        optimizer: Optax optimizer.

    Returns:
        Initialized TrainState.
    """
    return TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optimizer,
    )


def global_norm(tree: PyTree) -> jnp.ndarray:
    """Compute the global L2 norm of a pytree.

    Args:
        tree: Pytree of arrays.

    Returns:
        Scalar global norm.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))


# Need to import jax for tree operations
import jax
