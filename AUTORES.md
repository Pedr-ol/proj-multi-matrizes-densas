# Autores e Responsabilidades

**Projeto:** Multiplicação de matrizes densas (`C = A × B`) — Problema nº 1  
**Disciplina:** Programação Paralela (CCO085) — IESB 2026/2

| Integrante | Frente obrigatória | Artefatos principais |
|---|---|---|
| [Pedro Rocha/Mauricio Saboia] | Baseline e Perfilamento | `src/sequencial/01_matmul.cpp`, `scripts/medir.py`, análise de complexidade/hotspots, estimativa de Amdahl |
| [Pedro Rocha] | Versão OpenMP (memória compartilhada) | `src/openmp/matmul_openmp.cpp`, `src/openmp/matmul_openmp_blocked.cpp`, tratamento de corrida, estudo de escalonamento, curvas de speedup por threads |
| [Mauricio Saboia] | Versão MPI (memória distribuída) | `src/mpi/matmul_mpi.cpp`, decomposição de domínio, operações coletivas, curvas de speedup por processos |

