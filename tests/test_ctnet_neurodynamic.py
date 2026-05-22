from __future__ import annotations

import torch

from ctnet_neurodynamic import CTNetConfig, CTNetNeurodynamicCell, audit_ctnet_membership
from ctnet_neurodynamic.metrics import base_diversity, effective_rank, phase_entropy, phase_sovereignty


def test_reversible_core_roundtrip() -> None:
    torch.manual_seed(0)
    cfg = CTNetConfig(d_model=32, state_slots=4, memory_slots=3, relation_slots=3)
    cell = CTNetNeurodynamicCell(cfg)
    z = torch.randn(2, cfg.state_slots, cfg.d_model)
    z2 = cell.core(z)
    z_back = cell.core.inverse(z2)
    assert torch.allclose(z_back, z, atol=1e-5)


def test_forward_preserves_fixed_support_shapes() -> None:
    torch.manual_seed(1)
    cfg = CTNetConfig(d_model=32, state_slots=5, memory_slots=4, relation_slots=4)
    cell = CTNetNeurodynamicCell(cfg)
    state = cell.init_state(batch_size=2)
    x = torch.randn(2, cfg.state_slots, cfg.d_model)
    y, next_state, trace = cell(x, state)
    assert y.shape == (2, cfg.state_slots, cfg.d_model)
    assert next_state.z.shape == state.z.shape
    assert next_state.memory.shape == state.memory.shape
    assert next_state.relations.shape == state.relations.shape
    assert trace["phase_distribution"].shape == (2, cfg.num_regimes)
    assert trace["card_weights"].shape == (2, cfg.num_cards)


def test_metrics_are_well_formed() -> None:
    pi = torch.tensor([[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]])
    assert phase_entropy(pi).item() > 0
    assert phase_sovereignty(pi).item() == torch.tensor(0.6).item()
    assert base_diversity(pi).item() > 1
    x = torch.randn(2, 4, 8)
    assert effective_rank(x).shape == (2,)


def test_audit_passes_core_invariants() -> None:
    torch.manual_seed(2)
    cfg = CTNetConfig(d_model=32, state_slots=6, memory_slots=6, relation_slots=6)
    cell = CTNetNeurodynamicCell(cfg)
    state = cell.init_state(batch_size=2)
    x = torch.randn(2, cfg.state_slots, cfg.d_model)
    report = audit_ctnet_membership(cell, x, state)
    assert report.checks["reversible_core"]
    assert report.checks["fixed_memory_shape"]
    assert report.checks["fixed_relation_shape"]
    assert report.checks["projective_readout"]


def test_individuation_changes_shock_response() -> None:
    torch.manual_seed(3)
    cfg = CTNetConfig(d_model=32, state_slots=4, memory_slots=4, relation_slots=4)
    cell = CTNetNeurodynamicCell(cfg)
    a = cell.init_state(batch_size=1)
    b = cell.init_state(batch_size=1)
    b.memory = b.memory + 0.5
    shock = 3.0 * torch.randn(1, cfg.state_slots, cfg.d_model)
    ya, _, _ = cell(shock, a)
    yb, _, _ = cell(shock, b)
    assert not torch.allclose(ya, yb)
