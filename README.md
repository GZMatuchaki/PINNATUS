![PINNATUS - Scientific research framework for VPINNs](assets/pinnatus-banner.svg)

<p align="center">
  <img alt="Development status: early development" src="https://img.shields.io/badge/status-early%20development-E2A54A?style=flat-square" />
  <img alt="Research focus: VPINNs" src="https://img.shields.io/badge/research-VPINNs-8B73E6?style=flat-square" />
  <img alt="Reference method: PINNs" src="https://img.shields.io/badge/baseline-PINNs-4C9ACD?style=flat-square" />
  <img alt="Python" src="https://img.shields.io/badge/language-Python-3776AB?style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/backend-PyTorch-EE4C2C?style=flat-square" />
</p>

<p align="center">
  <strong>A research-oriented framework for the study, development, and comparison of physics-informed neural methods.</strong>
</p>

> [!IMPORTANT]
> **Early-stage research project.** PINNATUS is under active development. Its interfaces, numerical implementations, and documentation may change. It is not yet intended as a production-ready or generally validated solver.

## Overview

**PINNATUS** is a scientific framework being developed in **Python** and **PyTorch** as a computational laboratory for research on **Variational Physics-Informed Neural Networks (VPINNs)**. Conventional **Physics-Informed Neural Networks (PINNs)** are included in its intended scope as reference methods for systematic comparisons.

The project's primary purpose is to support methodological investigation during doctoral research, not to reproduce every capability of an established general-purpose framework.

## Research scope

- **Variational formulations:** weak forms, test functions, numerical quadrature, and related approximation strategies.
- **PINN baselines:** strong-form approaches for controlled numerical comparisons.
- **Scientific benchmarking:** common physical problems, evaluation metrics, and reproducible experiments.
- **Methodological flexibility:** access to the mathematical and computational components needed to investigate and modify algorithms.

These are **development goals**, not a declaration that every feature is currently available.

## Architectural vision

The long-term goal is a modular interface that separates the physical problem, neural network, solution method, and training procedure.

```python
# Illustrative future API - not yet an implemented/stable interface.
problem = Poiseuille()
network = MLP(layers=[1, 32, 32, 1])

model = VPINN(
    problem=problem,
    network=network,
    test_functions=Legendre(degree=5),
)

trainer = Trainer(model=model, optimizer="Adam")
trainer.train()
```

The same physical problem should also support a conventional PINN formulation, enabling comparisons without redefining the governing problem.

## Design principles

| Principle | Intended approach |
| --- | --- |
| **Modularity** | Separate problem definitions, neural networks, numerical formulations, and training. |
| **Transparency** | Keep residuals, variational integrals, losses, and optimization procedures inspectable. |
| **Reproducibility** | Make experimental configurations and numerical comparisons traceable. |
| **Extensibility** | Support new VPINN formulations without rewriting the whole research environment. |

## Project status

PINNATUS is currently in **early development**. The architecture and examples are evolving alongside the research. Installation instructions, validated examples, and API documentation will be added as the relevant components become available.

---

<p align="center"><sub>PINNATUS · Scientific research in progress</sub></p>
