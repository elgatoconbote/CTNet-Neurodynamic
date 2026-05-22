"""Neurodynamic CTNet modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import CTNetConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def mlp(in_dim: int, out_dim: int, hidden_mult: int = 2) -> nn.Sequential:
    hidden = max(in_dim, out_dim) * hidden_mult
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


class ReversibleUPCore(nn.Module):
    """Additive reversible coupling over the u/p partition."""

    def __init__(self, cfg: CTNetConfig) -> None:
        super().__init__()
        half = cfg.d_model // 2
        self.f = mlp(half, half, cfg.hidden_mult)
        self.g = mlp(half, half, cfg.hidden_mult)

    def forward(self, z: Tensor) -> Tensor:
        u, p = z.chunk(2, dim=-1)
        u2 = u + self.f(p)
        p2 = p + self.g(u2)
        return torch.cat([u2, p2], dim=-1)

    def inverse(self, z: Tensor) -> Tensor:
        u2, p2 = z.chunk(2, dim=-1)
        p = p2 - self.g(u2)
        u = u2 - self.f(p)
        return torch.cat([u, p], dim=-1)


class TopologicalMemory(nn.Module):
    """Fixed-support topological memory update: never append, only deform."""

    def __init__(self, cfg: CTNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.write = mlp(d * 2, d, cfg.hidden_mult)
        self.norm = RMSNorm(d, cfg.eps)

    def read(self, z: Tensor, memory: Tensor) -> Tensor:
        q = self.query(z)
        k = self.key(memory)
        v = self.value(memory)
        attn = torch.softmax(torch.einsum("bnd,bmd->bnm", q, k) / (z.shape[-1] ** 0.5), dim=-1)
        return torch.einsum("bnm,bmd->bnd", attn, v)

    def forward(self, z: Tensor, memory: Tensor, coherence: Tensor) -> tuple[Tensor, Tensor]:
        read = self.read(z, memory)
        summary = z.mean(dim=1, keepdim=True).expand(-1, memory.shape[1], -1)
        proposal = torch.tanh(self.write(torch.cat([memory, summary], dim=-1)))
        gate = coherence.view(coherence.shape[0], 1, 1).clamp(0.0, 1.0)
        next_memory = self.norm(memory + self.cfg.memory_write_rate * gate * proposal)
        return next_memory, read


class ReifiedRelations(nn.Module):
    """Fixed-support relation bank."""

    def __init__(self, cfg: CTNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.update = mlp(d * 3, d, cfg.hidden_mult)
        self.norm = RMSNorm(d, cfg.eps)

    def forward(self, z: Tensor, memory: Tensor, relations: Tensor, coherence: Tensor) -> Tensor:
        z_summary = z.mean(dim=1, keepdim=True).expand(-1, relations.shape[1], -1)
        m_summary = memory.mean(dim=1, keepdim=True).expand(-1, relations.shape[1], -1)
        proposal = torch.tanh(self.update(torch.cat([relations, z_summary, m_summary], dim=-1)))
        gate = coherence.view(coherence.shape[0], 1, 1).clamp(0.0, 1.0)
        return self.norm(relations + self.cfg.relation_write_rate * gate * proposal)


@dataclass(slots=True)
class CoherenceTrace:
    coherence: Tensor
    energy: Tensor
    structural_information: Tensor
    speed: Tensor
    fragility: Tensor
    saturation: Tensor
    phase_entropy: Tensor | None = None


class CoherenceTensor(nn.Module):
    """Causal coherence field: curves chaos instead of suppressing it."""

    def __init__(self, cfg: CTNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scorer = mlp(7, 4, cfg.hidden_mult)

    def forward(self, z_bar: Tensor, z_prev: Tensor, memory: Tensor, relations: Tensor, debt: Tensor) -> CoherenceTrace:
        delta = z_bar - z_prev
        energy = delta.pow(2).mean(dim=(1, 2))
        z_var = z_bar.var(dim=1).mean(dim=-1)
        m_var = memory.var(dim=1).mean(dim=-1)
        r_var = relations.var(dim=1).mean(dim=-1)
        structural_information = torch.log1p(z_var + m_var + r_var)
        saturation = z_bar.abs().tanh().mean(dim=(1, 2))
        fragility = torch.relu(debt + energy - structural_information)
        entropy_proxy = torch.log1p(z_bar.std(dim=1).mean(dim=-1))
        features = torch.stack([structural_information, entropy_proxy, energy, fragility, saturation, debt, m_var + r_var], dim=-1)
        out = self.scorer(features)
        coherence = torch.sigmoid(out[:, 0])
        speed = torch.exp(0.35 * out[:, 1]).clamp(self.cfg.min_speed, self.cfg.max_speed)
        return CoherenceTrace(coherence, energy, structural_information, speed, fragility, saturation)


class RegimeController(nn.Module):
    """Categorical-spectrum regime controller: softmax spectrum + argmax phase."""

    def __init__(self, cfg: CTNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.to_logits = mlp(7, cfg.num_regimes, cfg.hidden_mult)
        self.embedding = nn.Embedding(cfg.num_regimes, cfg.d_model)

    def forward(self, trace: CoherenceTrace, memory: Tensor, relations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        m_norm = memory.pow(2).mean(dim=(1, 2)).sqrt()
        r_norm = relations.pow(2).mean(dim=(1, 2)).sqrt()
        features = torch.stack([trace.structural_information, trace.energy, trace.fragility, trace.saturation, trace.coherence, m_norm, r_norm], dim=-1)
        logits = self.to_logits(features)
        pi = torch.softmax(logits, dim=-1)
        rho = torch.argmax(pi, dim=-1)
        trace.phase_entropy = -(pi * pi.clamp_min(1e-8).log()).sum(dim=-1)
        return pi, rho, self.embedding(rho)


class AdmissibilityField(nn.Module):
    """Smooth legal-transition field, not a binary wall."""

    def __init__(self, cfg: CTNetConfig) -> None:
        super().__init__()
        self.field = mlp(cfg.d_model * 2, cfg.d_model, cfg.hidden_mult)

    def forward(self, z_bar: Tensor, regime_embedding: Tensor) -> Tensor:
        regime = regime_embedding.unsqueeze(1).expand_as(z_bar)
        return torch.sigmoid(self.field(torch.cat([z_bar, regime], dim=-1)))


class MultiCardReadout(nn.Module):
    """Projective multi-card readout; output is not the state identity."""

    def __init__(self, cfg: CTNetConfig, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim or cfg.d_model
        self.cards = nn.ModuleList([nn.Sequential(RMSNorm(cfg.d_model), nn.Linear(cfg.d_model, out_dim)) for _ in range(cfg.num_cards)])
        self.base = nn.Linear(cfg.d_model, out_dim)
        self.selector = mlp(cfg.d_model * 3, cfg.num_cards, cfg.hidden_mult)
        self.mix_logits = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, z: Tensor, memory: Tensor, relations: Tensor) -> tuple[Tensor, Tensor]:
        z_summary = z.mean(dim=1)
        m_summary = memory.mean(dim=1)
        r_summary = relations.mean(dim=1)
        weights = torch.softmax(self.selector(torch.cat([z_summary, m_summary, r_summary], dim=-1)), dim=-1)
        card_outputs = torch.stack([card(z) for card in self.cards], dim=2)
        y_cards = torch.einsum("bk,bnkd->bnd", weights, card_outputs)
        y_base = self.base(z)
        alpha = torch.softmax(self.mix_logits, dim=0)
        return alpha[0] * y_cards + alpha[1] * y_base, weights
