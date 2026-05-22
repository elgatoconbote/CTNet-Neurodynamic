"""Auditing utilities for CTNet membership conditions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .metrics import effective_rank
from .model import CTNetNeurodynamicCell, CTNetState


@dataclass(slots=True)
class AuditReport:
    passed: bool
    checks: dict[str, bool]
    values: dict[str, float]


def audit_ctnet_membership(
    model: CTNetNeurodynamicCell,
    x: Tensor,
    state: CTNetState,
    *,
    reversibility_eps: float = 1e-4,
    min_effective_rank: float = 1.1,
) -> AuditReport:
    """Check the compact CTNet invariants on one batch."""

    with torch.no_grad():
        z_in = model.norm(state.z + model.input(x))
        z_core = model.core(z_in)
        z_back = model.core.inverse(z_core)
        rev_error = (z_back - z_in).pow(2).mean().sqrt().item()
        y, next_state, trace = model(x, state)
        memory_shape_fixed = next_state.memory.shape == state.memory.shape
        relation_shape_fixed = next_state.relations.shape == state.relations.shape
        memory_rank = effective_rank(next_state.memory).mean().item()
        relation_rank = effective_rank(next_state.relations).mean().item()
        output_is_projection = y.shape[:2] == next_state.z.shape[:2] and y.data_ptr() != next_state.z.data_ptr()
        coherence_finite = torch.isfinite(trace["coherence"]).all().item()
        regime_ok = trace["phase_distribution"].shape[-1] == model.cfg.num_regimes
        speed_in_bounds = (
            (trace["speed"] >= model.cfg.min_speed).all()
            and (trace["speed"] <= model.cfg.max_speed).all()
        ).item()

    checks = {
        "reversible_core": rev_error < reversibility_eps,
        "fixed_memory_shape": memory_shape_fixed,
        "fixed_relation_shape": relation_shape_fixed,
        "memory_effective_rank": memory_rank >= min_effective_rank,
        "relation_effective_rank": relation_rank >= min_effective_rank,
        "projective_readout": bool(output_is_projection),
        "coherence_finite": bool(coherence_finite),
        "regime_distribution": bool(regime_ok),
        "speed_in_bounds": bool(speed_in_bounds),
    }
    values = {
        "rev_error": rev_error,
        "memory_effective_rank": memory_rank,
        "relation_effective_rank": relation_rank,
        "coherence_mean": trace["coherence"].mean().item(),
        "speed_mean": trace["speed"].mean().item(),
        "incoherence_debt_mean": trace["incoherence_debt"].mean().item(),
    }
    return AuditReport(passed=all(checks.values()), checks=checks, values=values)
