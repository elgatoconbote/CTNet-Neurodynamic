# CTNet-Neurodynamic

Seed implementation of CTNet-Neurodynamic: a non-LLM neurocomputational architecture based on CTNet and Neuro-EMQR principles.

This repository starts with a small CPU-runnable Python prototype. It does not call external model APIs. It implements a distributed state, fixed-support memory, reified relations, coherence metrics, executive regimes, admissibility, and projective readout.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e . pytest
pytest -q
```

## Core state

```text
Omega_t = (Z_t, M_t, R_t, C_t, rho_t, pi_t, A_t, nu_t, Theta_t)
Z_t = [u_t, p_t]
Omega_{t+1} = U_CT(Omega_t, x_t)
y_t = P_out(Pi_rho(Omega_{t+1}))
```

## Status

Initial scaffold. The architecture is intentionally small and auditable. It is not pretrained and must learn from local streams.
