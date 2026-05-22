"""Metrics for CTNet Neurodynamic health and audit."""

from __future__ import annotations

import torch
from torch import Tensor


def effective_rank(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Entropy-based effective rank for a batch of matrices/tensors."""

    if x.ndim < 3:
        raise ValueError("effective_rank expects at least 3 dimensions: B x N x D")
    b = x.shape[0]
    matrix = x.reshape(b, -1, x.shape[-1])
    ranks = []
    for item in matrix:
        s = torch.linalg.svdvals(item.float()).clamp_min(eps)
        p = s / s.sum()
        ranks.append(torch.exp(-(p * p.log()).sum()))
    return torch.stack(ranks).to(device=x.device, dtype=x.dtype)


def phase_entropy(pi: Tensor, eps: float = 1e-8) -> Tensor:
    return -(pi * pi.clamp_min(eps).log()).sum(dim=-1)


def phase_sovereignty(pi: Tensor) -> Tensor:
    return pi.max(dim=-1).values


def base_diversity(pi: Tensor) -> Tensor:
    return torch.exp(phase_entropy(pi))


def contextual_vitality(
    lyapunov_proxy: Tensor,
    coherence: Tensor,
    diversity: Tensor,
    incoherence_debt: Tensor,
    speed: Tensor,
    *,
    max_speed: float,
) -> Tensor:
    """Vitality proxy: alive if sensitive, coherent, diverse and not debt-saturated."""

    speed_saturation = (speed / max_speed).clamp(0, 1)
    raw = (
        0.7 * lyapunov_proxy
        + 1.2 * coherence
        + 0.4 * torch.log1p(diversity)
        - 0.8 * incoherence_debt
        - 0.3 * speed_saturation
    )
    return torch.sigmoid(raw)


def chaos_rescue_ratio(info_after_basis_change: Tensor, info_before: Tensor, eps: float = 1e-6) -> Tensor:
    """R_nu > 1 means a basis change revealed structure in a residual."""

    return info_after_basis_change / (info_before + eps)
