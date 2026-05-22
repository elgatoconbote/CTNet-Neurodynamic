"""Configuration and regime definitions for CTNet Neurodynamic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Regime(IntEnum):
    """Canonical categorical-spectrum phases."""

    EXPLORE = 0
    DECIDE = 1
    REFLECT = 2
    STABILIZE = 3
    VERIFY = 4
    PROJECT = 5


@dataclass(slots=True)
class CTNetConfig:
    """Hyperparameters for a compact CTNet Neurodynamic cell."""

    d_model: int = 64
    state_slots: int = 16
    memory_slots: int = 8
    relation_slots: int = 8
    num_cards: int = 4
    num_regimes: int = 6
    hidden_mult: int = 2
    min_speed: float = 0.02
    max_speed: float = 0.35
    memory_write_rate: float = 0.08
    relation_write_rate: float = 0.05
    incoherence_decay: float = 0.92
    incoherence_gain: float = 0.15
    coherence_relief: float = 0.08
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even because the state is partitioned as u/p")
        if self.state_slots <= 0 or self.memory_slots <= 0 or self.relation_slots <= 0:
            raise ValueError("state_slots, memory_slots and relation_slots must be positive")
        if self.num_regimes != len(Regime):
            raise ValueError(f"num_regimes must be {len(Regime)} for the canonical phase set")
        if not (0.0 < self.min_speed <= self.max_speed):
            raise ValueError("speed bounds must satisfy 0 < min_speed <= max_speed")
