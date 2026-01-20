"""Pure, JIT-compiled training step.

Implements a stateless training step function that can be compiled with
@jax.jit for maximum performance. The function is pure (no side effects)
and takes all inputs explicitly.

This is the core of Research Engineering: a clean, observable training loop.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from parallax.training.loss import compute_loss
from parallax.training.optimizer import TrainState, global_norm

Array = jnp.ndarray
PyTree = Any


class Metrics(NamedTuple):
    """Training metrics for observability.

    Attributes:
        loss: Total loss (CE + Z-loss).
        ce_loss: Cross-entropy loss component.
        z_loss: Z-loss regularization component.
        grad_norm: L2 norm of gradients (before clipping).
        param_norm: L2 norm of all parameters.
        learning_rate: Current learning rate.
    """

    loss: Array
    ce_loss: Array
    z_loss: Array
    grad_norm: Array
    param_norm: Array
    learning_rate: Array


@jax.jit
def train_step(
    state: TrainState,
    batch: dict[str, Array],
    rng: Array,
    label_smoothing: float = 0.0,
    z_loss_coeff: float = 1e-4,
) -> tuple[TrainState, Metrics]:
    """Execute a single training step.

    This function is pure and JIT-compiled. It:
    1. Computes forward pass and loss
    2. Computes gradients
    3. Updates parameters
    4. Collects metrics for observability

    Args:
        state: Current training state (params, optimizer state).
        batch: Dictionary with 'input_ids' and 'targets' arrays.
        rng: Random key for dropout.
        label_smoothing: Label smoothing factor.
        z_loss_coeff: Z-loss coefficient.

    Returns:
        Tuple of (new_state, metrics).
    """
    input_ids = batch["input_ids"]
    targets = batch["targets"]

    def loss_fn(params: PyTree) -> tuple[Array, dict[str, Array]]:
        """Compute loss and auxiliary outputs."""
        logits, _ = state.apply_fn(
            {"params": params},
            input_ids,
            deterministic=False,
            rngs={"dropout": rng},
        )
        total_loss, loss_dict = compute_loss(
            logits,
            targets,
            label_smoothing=label_smoothing,
            z_loss_coeff=z_loss_coeff,
        )
        return total_loss, loss_dict

    # Compute gradients with auxiliary outputs
    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Compute gradient norm BEFORE clipping
    grad_norm = global_norm(grads)

    # Compute parameter norm
    param_norm = global_norm(state.params)

    # Recompute learning rate from schedule
    # Note: In practice, you'd pass the schedule or store LR differently
    # For now, we'll use the step count to indicate position
    learning_rate = jnp.array(0.0)  # Placeholder - actual LR computed by optax

    # Apply updates
    state = state.apply_gradients(grads=grads)

    # Assemble metrics
    metrics = Metrics(
        loss=loss,
        ce_loss=loss_dict["ce_loss"],
        z_loss=loss_dict["z_loss"],
        grad_norm=grad_norm,
        param_norm=param_norm,
        learning_rate=learning_rate,
    )

    return state, metrics


def create_batch(
    input_ids: Array,
    seq_len: int | None = None,
) -> dict[str, Array]:
    """Create training batch from token IDs.

    For language modeling, targets are inputs shifted left by 1.

    Args:
        input_ids: Token IDs of shape [batch, seq_len + 1] or [batch, seq_len].
        seq_len: If provided, truncate/pad to this length.

    Returns:
        Batch dictionary with 'input_ids' and 'targets'.
    """
    if seq_len is not None and input_ids.shape[1] > seq_len + 1:
        input_ids = input_ids[:, : seq_len + 1]

    # Inputs: all tokens except the last
    inputs = input_ids[:, :-1]
    # Targets: all tokens except the first
    targets = input_ids[:, 1:]

    return {"input_ids": inputs, "targets": targets}
