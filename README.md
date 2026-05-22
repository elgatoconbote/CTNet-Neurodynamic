# CTNet Neurodynamic

Reference implementation of **CTNet Neuronal**: fixed-support topological memory, reified relations, reversible `u/p` core, coherence tensor, categorical-spectrum regime, smooth admissibility and projective readout.

CTNet does not append context as a growing history. It folds context into a persistent internal topology.

```text
surface input -> distributed state -> u/p reversible core
-> fixed memory M -> fixed relations R -> deterministic chaos
-> coherence tensor -> regime -> admissibility -> multi-card readout
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
python examples/run_neurodynamic_demo.py
```

## Minimal use

```python
import torch
from ctnet_neurodynamic import CTNetConfig, CTNetNeurodynamicCell

cfg = CTNetConfig(d_model=64, state_slots=16, memory_slots=8, relation_slots=8)
cell = CTNetNeurodynamicCell(cfg)
state = cell.init_state(batch_size=2)
x = torch.randn(2, 16, 64)
y, state, trace = cell(x, state)
print(y.shape, trace["active_regime"])
```

## CTNet invariants

```text
reversible core: ||F^-1(F(Z)) - Z|| < eps
fixed memory: shape(M_t+1) = shape(M_t)
fixed relations: shape(R_t+1) = shape(R_t)
readout is projective: Y_t != Omega_t
coherence affects transition
regime affects transition
admissibility is smooth, not binary
```

If a change requires growing memory, output-as-identity or a broken reversible core, it is not yet in CTNet form.
