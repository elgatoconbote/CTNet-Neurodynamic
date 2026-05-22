"""CTNet Neurodynamic cell."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import CTNetConfig
from .modules import (
    AdmissibilityField,
    CoherenceTensor,
    MultiCardReadout,
    ReifiedRelations,
    RegimeController,
    ReversibleUPCore,
    RMSNorm,
    TopologicalMemory,
    mlp,
)


@dataclass(slots=True)
class CTNetState:
    """Persistent CTNet state."""

    z: Tensor
    memory: Tensor
    relations: Tensor
    incoherence_debt: Tensor


class CTNetNeurodynamicCell(nn.Module):
    """Compact CTNet Neuronal reference cell.

    Implements Omega_t -> Omega_{t+1} -> Y_t, where Y is a projective output and
    does not exhaust Omega.
    """

    def __init__(self, cfg: CTNetConfig, out_dim: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.d_model, cfg.d_model)
        self.core = ReversibleUPCore(cfg)
        self.memory = TopologicalMemory(cfg)
        self.relations = ReifiedRelations(cfg)
        self.coherence = CoherenceTensor(cfg)
        self.regime = RegimeController(cfg)
        self.admissibility = AdmissibilityField(cfg)
        self.delta_memory = nn.Linear(cfg.d_model, cfg.d_model)
        self.delta_regime = nn.Linear(cfg.d_model, cfg.d_model)
        self.delta_adm = nn.Linear(cfg.d_model, cfg.d_model)
        self.delta_coherence = mlp(1, cfg.d_model, cfg.hidden_mult)
        self.norm = RMSNorm(cfg.d_model, cfg.eps)
        self.readout = MultiCardReadout(cfg, out_dim=out_dim)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CTNetState:
        kwargs = {"device": device, "dtype": dtype or torch.float32}
        z = torch.zeros(batch_size, self.cfg.state_slots, self.cfg.d_model, **kwargs)
        memory = 0.01 * torch.randn(batch_size, self.cfg.memory_slots, self.cfg.d_model, **kwargs)
        relations = 0.01 * torch.randn(batch_size, self.cfg.relation_slots, self.cfg.d_model, **kwargs)
        debt = torch.zeros(batch_size, **kwargs)
        return CTNetState(z=z, memory=memory, relations=relations, incoherence_debt=debt)

    def forward(self, x: Tensor, state: CTNetState) -> tuple[Tensor, CTNetState, dict[str, Tensor]]:
        if x.shape != state.z.shape:
            raise ValueError(f"x must have shape {tuple(state.z.shape)}, got {tuple(x.shape)}")

        z_in = self.norm(state.z + self.input(x))
        z_bar = self.core(z_in)

        coh = self.coherence(z_bar, z_in, state.memory, state.relations, state.incoherence_debt)
        pi, rho, regime_emb = self.regime(coh, state.memory, state.relations)
        adm = self.admissibility(z_bar, regime_emb)
        z_adm = adm * z_bar + (1.0 - adm) * z_in

        memory_next, memory_read = self.memory(z_adm, state.memory, coh.coherence)
        relations_next = self.relations(z_adm, memory_next, state.relations, coh.coherence)

        delta_m = self.delta_memory(memory_read)
        delta_rho = self.delta_regime(regime_emb).unsqueeze(1).expand_as(z_adm)
        delta_a = self.delta_adm(z_adm - z_bar)
        delta_c = self.delta_coherence(coh.coherence.unsqueeze(-1)).unsqueeze(1).expand_as(z_adm)
        delta = delta_m + delta_rho + delta_a + delta_c

        speed = coh.speed.view(coh.speed.shape[0], 1, 1)
        z_next = self.norm(z_bar + speed * delta)
        y, card_weights = self.readout(z_next, memory_next, relations_next)

        new_debt = (
            self.cfg.incoherence_decay * state.incoherence_debt
            + self.cfg.incoherence_gain * coh.energy
            - self.cfg.coherence_relief * coh.coherence
        ).clamp_min(0.0)

        next_state = CTNetState(z=z_next, memory=memory_next, relations=relations_next, incoherence_debt=new_debt)
        trace = {
            "coherence": coh.coherence,
            "coherence_energy": coh.energy,
            "structural_information": coh.structural_information,
            "speed": coh.speed,
            "fragility": coh.fragility,
            "saturation": coh.saturation,
            "phase_distribution": pi,
            "active_regime": rho,
            "phase_entropy": coh.phase_entropy,
            "admissibility_mean": adm.mean(dim=(1, 2)),
            "card_weights": card_weights,
            "incoherence_debt": new_debt,
        }
        return y, next_state, trace
