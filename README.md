# VPINN com PyTorch

Base leve para experimentar **Variational Physics-Informed Neural Networks** (VPINNs) em problemas 1D. O exemplo resolve a equação de Poisson

`-u''(x) = f(x),  x ∈ [-1, 1],  u(-1)=u(1)=0`

com a solução de referência `u(x)=sin(πx)`. A perda usa a forma fraca: integra os resíduos contra funções-teste de Legendre via quadratura de Gauss–Legendre.

## Preparação

Requer Python 3.10 ou superior.

```bash
cd vpinn-pytorch
./scripts/setup.sh
source .venv/bin/activate
python -m vpinn.train --config configs/poisson_1d.yaml
```

Os resultados são salvos em `outputs/poisson_1d/`. Para testar a instalação:

```bash
pytest
```

## Estrutura

```text
configs/       parâmetros dos experimentos
notebooks/     exploração e gráficos ad hoc
scripts/       preparação do ambiente
src/vpinn/     implementação reutilizável
tests/         verificações numéricas
outputs/       artefatos gerados (ignorado pelo Git)
```

O ponto natural para novos problemas é adicionar um módulo em `src/vpinn/problems/` e um arquivo correspondente em `configs/`.
