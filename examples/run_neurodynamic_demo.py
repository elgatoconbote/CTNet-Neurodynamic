from __future__ import annotations

import torch

from ctnet_neurodynamic import CTNetConfig, CTNetNeurodynamicCell, audit_ctnet_membership
from ctnet_neurodynamic.metrics import base_diversity, contextual_vitality, phase_sovereignty


def main() -> None:
    torch.manual_seed(7)
    cfg = CTNetConfig(d_model=64, state_slots=16, memory_slots=8, relation_slots=8)
    cell = CTNetNeurodynamicCell(cfg)
    state = cell.init_state(batch_size=2)

    for step in range(5):
        shock = 1.0 + 0.4 * step
        x = shock * torch.randn(2, cfg.state_slots, cfg.d_model)
        y, state, trace = cell(x, state)
        diversity = base_diversity(trace["phase_distribution"])
        sovereignty = phase_sovereignty(trace["phase_distribution"])
        vitality = contextual_vitality(
            lyapunov_proxy=trace["coherence_energy"].sqrt(),
            coherence=trace["coherence"],
            diversity=diversity,
            incoherence_debt=trace["incoherence_debt"],
            speed=trace["speed"],
            max_speed=cfg.max_speed,
        )
        print(
            f"step={step} y={tuple(y.shape)} rho={trace['active_regime'].tolist()} "
            f"coh={trace['coherence'].mean().item():.3f} "
            f"div={diversity.mean().item():.3f} "
            f"sov={sovereignty.mean().item():.3f} "
            f"vitality={vitality.mean().item():.3f}"
        )

    report = audit_ctnet_membership(cell, torch.randn(2, cfg.state_slots, cfg.d_model), state)
    print("audit passed:", report.passed)
    print(report.values)


if __name__ == "__main__":
    main()
