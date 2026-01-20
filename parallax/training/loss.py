"""Loss functions for language model training.

Cross-entropy loss with optional label smoothing and Z-loss for logit stability.

Z-loss is an auxiliary regularization term that prevents logits from drifting
to extreme values, which is crucial for numerical stability on TPUs.

Reference: https://arxiv.org/abs/2204.02311 (PaLM)
"""

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def cross_entropy_loss(
    logits: Array,
    targets: Array,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> Array:
    """Compute cross-entropy loss with optional label smoothing.

    Args:
        logits: Predicted logits of shape [batch, seq_len, vocab_size].
        targets: Target token indices of shape [batch, seq_len].
        label_smoothing: Label smoothing factor in [0, 1). If > 0, softens
            the target distribution by mixing with uniform.
        ignore_index: Token index to ignore in loss computation (e.g., padding).

    Returns:
        Scalar mean loss.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Create mask for valid positions
    mask = targets != ignore_index

    # Flatten for easier processing
    targets_flat = targets.reshape(-1)

    # Compute log probabilities (numerically stable)
    log_probs = jax.nn.log_softmax(logits, axis=-1).reshape(-1, vocab_size)

    if label_smoothing > 0:
        # Smooth target distribution
        # confidence = 1 - label_smoothing
        # smoothed_targets = confidence * one_hot + label_smoothing / vocab_size
        confidence = 1.0 - label_smoothing
        smooth_value = label_smoothing / vocab_size

        # One-hot targets
        one_hot = jax.nn.one_hot(targets_flat, vocab_size)
        smoothed_targets = confidence * one_hot + smooth_value

        # Cross-entropy with smoothed targets
        loss_per_token = -jnp.sum(smoothed_targets * log_probs, axis=-1)
    else:
        # Standard cross-entropy (more efficient for hard targets)
        # Gather log probabilities at target positions
        loss_per_token = -jnp.take_along_axis(
            log_probs,
            targets_flat[:, None],
            axis=-1,
        ).squeeze(-1)

    # Apply mask and compute mean
    loss_per_token = loss_per_token.reshape(batch_size, seq_len)
    loss_per_token = jnp.where(mask, loss_per_token, 0.0)

    # Mean over valid tokens
    num_valid = jnp.sum(mask)
    # Avoid division by zero
    total_loss = jnp.sum(loss_per_token) / jnp.maximum(num_valid, 1.0)

    return total_loss


def z_loss(logits: Array, coefficient: float = 1e-4) -> Array:
    """Compute Z-loss for logit stability.

    Z-loss penalizes the log of the sum of exponentials of logits,
    which prevents logits from drifting to extreme values.

    Formula: L_z = coefficient * log^2(sum(exp(logits)))

    This is particularly important on TPUs where extreme logits can
    cause numerical issues with softmax.

    Args:
        logits: Predicted logits of shape [batch, seq_len, vocab_size].
        coefficient: Scaling factor for Z-loss (default: 1e-4).

    Returns:
        Scalar Z-loss value.
    """
    # Compute log(sum(exp(z))) for each token position
    # Use logsumexp for numerical stability
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)

    # Square and average
    z_loss_value = coefficient * jnp.mean(log_z**2)

    return z_loss_value


def compute_loss(
    logits: Array,
    targets: Array,
    label_smoothing: float = 0.0,
    z_loss_coeff: float = 1e-4,
    ignore_index: int = -100,
) -> tuple[Array, dict[str, Array]]:
    """Compute total loss with breakdown.

    Combines cross-entropy loss with Z-loss regularization.

    Args:
        logits: Predicted logits [batch, seq_len, vocab_size].
        targets: Target token indices [batch, seq_len].
        label_smoothing: Label smoothing factor.
        z_loss_coeff: Z-loss coefficient.
        ignore_index: Token index to ignore.

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains
        individual loss components.
    """
    ce_loss = cross_entropy_loss(
        logits, targets, label_smoothing=label_smoothing, ignore_index=ignore_index
    )

    z_loss_value = z_loss(logits, coefficient=z_loss_coeff)

    total_loss = ce_loss + z_loss_value

    loss_dict = {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "z_loss": z_loss_value,
    }

    return total_loss, loss_dict
