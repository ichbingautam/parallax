"""Distributed training package."""

from parallax.distributed.pmap_trainer import (
    PmapTrainer,
    replicate,
    shard,
    unreplicate,
)

__all__ = ["PmapTrainer", "replicate", "unreplicate", "shard"]
