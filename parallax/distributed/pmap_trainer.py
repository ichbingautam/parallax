"""Distributed training with jax.pmap.

Implements data-parallel training across multiple devices (TPU/GPU):
- Model replication across devices
- Batch sharding across devices
- Gradient averaging with pmean
- RNG key folding for unique dropout masks per device

Reference: https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
"""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from parallax.training.loss import compute_loss
from parallax.training.optimizer import global_norm

Array = jnp.ndarray
PyTree = Any


class DistributedMetrics(NamedTuple):
    """Training metrics from distributed training.

    All metrics are already aggregated across devices with pmean.
    """

    loss: Array
    ce_loss: Array
    z_loss: Array
    grad_norm: Array
    param_norm: Array


def replicate(tree: PyTree) -> PyTree:
    """Replicate a pytree across all local devices.

    Args:
        tree: Pytree to replicate.

    Returns:
        Replicated pytree with leading device dimension.
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (jax.local_device_count(),) + x.shape),
        tree,
    )


def unreplicate(tree: PyTree) -> PyTree:
    """Get the first replica from a replicated pytree.

    Args:
        tree: Replicated pytree with leading device dimension.

    Returns:
        Single replica (first device).
    """
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def shard(batch: PyTree) -> PyTree:
    """Shard a batch across devices.

    Reshapes batch from [global_batch, ...] to [num_devices, local_batch, ...].

    Args:
        batch: Batch with global batch size.

    Returns:
        Sharded batch with device dimension.
    """
    num_devices = jax.local_device_count()

    def _shard(x: Array) -> Array:
        batch_size = x.shape[0]
        assert batch_size % num_devices == 0, (
            f"Batch size {batch_size} not divisible by {num_devices} devices"
        )
        local_batch = batch_size // num_devices
        return x.reshape(num_devices, local_batch, *x.shape[1:])

    return jax.tree_util.tree_map(_shard, batch)


def _pmap_train_step(
    state: train_state.TrainState,
    batch: dict[str, Array],
    rng: Array,
    label_smoothing: float,
    z_loss_coeff: float,
) -> tuple[train_state.TrainState, DistributedMetrics]:
    """Single training step for pmap.

    This function runs on each device with a local batch shard.
    Gradients are averaged across devices before the optimizer step.
    """
    input_ids = batch["input_ids"]
    targets = batch["targets"]

    # Fold device ID into RNG for unique dropout masks per device
    device_id = jax.lax.axis_index("devices")
    rng = jax.random.fold_in(rng, device_id)

    def loss_fn(params: PyTree) -> tuple[Array, dict[str, Array]]:
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

    # Compute local gradients
    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Compute gradient norm before averaging
    grad_norm = global_norm(grads)

    # Average gradients across all devices
    grads = jax.lax.pmean(grads, axis_name="devices")

    # Average metrics across devices
    loss = jax.lax.pmean(loss, axis_name="devices")
    ce_loss = jax.lax.pmean(loss_dict["ce_loss"], axis_name="devices")
    z_loss_val = jax.lax.pmean(loss_dict["z_loss"], axis_name="devices")
    grad_norm = jax.lax.pmean(grad_norm, axis_name="devices")
    param_norm = global_norm(state.params)

    # Apply updates (same on all devices due to pmean)
    state = state.apply_gradients(grads=grads)

    metrics = DistributedMetrics(
        loss=loss,
        ce_loss=ce_loss,
        z_loss=z_loss_val,
        grad_norm=grad_norm,
        param_norm=param_norm,
    )

    return state, metrics


class PmapTrainer:
    """Distributed trainer using jax.pmap.

    Handles:
    - State replication across devices
    - Batch sharding
    - Compiled pmap'd training step
    - Metrics aggregation

    Example:
        trainer = PmapTrainer(model, params, optimizer)
        for batch in dataloader:
            metrics = trainer.train_step(batch, rng)
            print(f"Loss: {metrics.loss}")
    """

    def __init__(
        self,
        model: Any,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 1e-4,
    ) -> None:
        """Initialize distributed trainer.

        Args:
            model: The Flax model.
            params: Initial model parameters.
            optimizer: Optax optimizer.
            label_smoothing: Label smoothing factor.
            z_loss_coeff: Z-loss coefficient.
        """
        self.num_devices = jax.local_device_count()
        self.label_smoothing = label_smoothing
        self.z_loss_coeff = z_loss_coeff

        # Create train state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params["params"],
            tx=optimizer,
        )

        # Replicate state across devices
        self._state = replicate(state)

        # Compile pmap'd train step
        self._pmap_step = jax.pmap(
            functools.partial(
                _pmap_train_step,
                label_smoothing=label_smoothing,
                z_loss_coeff=z_loss_coeff,
            ),
            axis_name="devices",
        )

    @property
    def state(self) -> train_state.TrainState:
        """Get unreplicated state (from first device)."""
        return unreplicate(self._state)

    @property
    def params(self) -> PyTree:
        """Get unreplicated parameters."""
        return self.state.params

    def train_step(
        self,
        batch: dict[str, Array],
        rng: Array,
    ) -> DistributedMetrics:
        """Execute a distributed training step.

        Args:
            batch: Batch with global batch size [global_batch, seq_len].
            rng: Random key.

        Returns:
            Aggregated metrics (already averaged across devices).
        """
        # Shard batch across devices
        batch = shard(batch)

        # Replicate RNG across devices (each device will fold in its ID)
        rng = replicate(rng)

        # Execute pmap'd step
        self._state, metrics = self._pmap_step(self._state, batch, rng)

        # Metrics are already pmean'd, just unreplicate
        return unreplicate(metrics)

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint to disk.

        Args:
            path: Path to save checkpoint.
        """
        import orbax.checkpoint as ocp

        state = self.state
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path, state)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from disk.

        Args:
            path: Path to checkpoint.
        """
        import orbax.checkpoint as ocp

        state = self.state
        checkpointer = ocp.StandardCheckpointer()
        state = checkpointer.restore(path, state)
        self._state = replicate(state)


def get_device_info() -> dict[str, Any]:
    """Get information about available devices.

    Returns:
        Dictionary with device information.
    """
    devices = jax.devices()
    local_devices = jax.local_devices()

    return {
        "num_devices": len(devices),
        "num_local_devices": len(local_devices),
        "device_kind": devices[0].device_kind if devices else "unknown",
        "devices": [str(d) for d in devices],
        "local_devices": [str(d) for d in local_devices],
    }
