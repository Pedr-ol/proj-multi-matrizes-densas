#!/usr/bin/env python3
"""Gera os graficos de analise experimental a partir dos CSVs.

Graficos produzidos:
  tempo_execucao.png        tempo x tamanho da matriz
  speedup.png               speedup x tamanho
  speedup_openmp.png        speedup x threads
  speedup_mpi.png           speedup x processos
  overhead_mpi.png          overhead de comunicacao x processos
  eficiencia.png            eficiencia x threads/processos
  escalabilidade_forte.png  speedup com N fixo e p variavel
  escalonamento.png         static x dynamic x guided
  tamanho_blocos.png        impacto do tamanho do bloco
  escalabilidade_fraca.png  eficiencia em escalabilidade fraca
  amdahl.png                medicoes x limite teorico da Lei de Amdahl
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"
GRAFICOS = BASE / "graficos"

GRAFICOS.mkdir(exist_ok=True)


def carregar_csv(nome):
    caminho = RESULTADOS / nome
    if not caminho.exists():
        return []
    with open(caminho, newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def num(linha):
    return int(linha["N"])


def tempo(linha):
    return float(linha["tempo"])


def paralelismo(linha):
    return int(linha["configuracao"].split()[0])


def escala(linha):
    partes = linha["configuracao"].split()
    if partes[-1] in ("static", "dynamic", "guided"):
        return partes[-1]
    return "static"


def por_n(dados):
    grupos = {}
    for linha in dados:
        grupos.setdefault(num(linha), []).append(linha)
    return grupos


def melhores(dados):
    """Mantem apenas a melhor configuracao para cada (N, paralelismo)."""
    melhores_por_chave = {}
    for linha in dados:
        chave = (num(linha), paralelismo(linha))
        atual = melhores_por_chave.get(chave)
        if atual is None or tempo(linha) < tempo(atual):
            melhores_por_chave[chave] = linha
    return list(melhores_por_chave.values())


def melhor_por_n(dados):
    """Mantem apenas a configuracao mais rapida para cada N."""
    melhores_por_n = {}
    for linha in dados:
        atual = melhores_por_n.get(num(linha))
        if atual is None or tempo(linha) < tempo(atual):
            melhores_por_n[num(linha)] = linha
    return sorted(melhores_por_n.values(), key=num)


baseline = carregar_csv("baseline.csv")
openmp = carregar_csv("openmp.csv")
mpi = carregar_csv("mpi.csv")
blocked = carregar_csv("openmp_blocked.csv")
fraca = carregar_csv("escalabilidade_fraca.csv")
overhead = carregar_csv("mpi_overhead.csv")

tempos_base = {num(linha): tempo(linha) for linha in baseline}

openmp_static = [x for x in openmp if escala(x) == "static"]
blocked_melhores = melhores(blocked)


def salvar(nome):
    plt.tight_layout()
    plt.savefig(GRAFICOS / nome, dpi=300)
    plt.close()


# 1. Tempo de execucao
plt.figure(figsize=(12, 7))
if baseline:
    ordenado = sorted(baseline, key=num)
    plt.plot([num(x) for x in ordenado], [tempo(x) for x in ordenado],
             marker="o", label="Sequencial")

for nome, dados in [("OpenMP", openmp_static), ("MPI", mpi)]:
    for p in sorted({paralelismo(x) for x in dados}):
        pontos = sorted([x for x in dados if paralelismo(x) == p], key=num)
        if pontos:
            rotulo = "threads" if nome == "OpenMP" else "processos"
            plt.plot([num(x) for x in pontos], [tempo(x) for x in pontos],
                     marker="o", label=f"{nome} ({p} {rotulo})")

if blocked_melhores:
    melhor_geral = melhor_por_n(blocked_melhores)
    plt.plot([num(x) for x in melhor_geral], [tempo(x) for x in melhor_geral],
             marker="s", linestyle="--", label="Bloqueado (melhor)")

plt.xlabel("Tamanho da matriz (N)")
plt.ylabel("Tempo (segundos)")
plt.title("Tempo de execucao das implementacoes")
plt.legend(ncol=2)
plt.grid(True)
salvar("tempo_execucao.png")


# 2. Speedup em relacao ao sequencial (por tamanho)
plt.figure(figsize=(12, 7))
for nome, dados in [("OpenMP", openmp_static), ("MPI", mpi)]:
    for p in sorted({paralelismo(x) for x in dados}):
        pontos = [
            x for x in dados
            if paralelismo(x) == p and num(x) in tempos_base
        ]
        pontos.sort(key=num)
        if pontos:
            rotulo = "threads" if nome == "OpenMP" else "processos"
            plt.plot(
                [num(x) for x in pontos],
                [tempos_base[num(x)] / tempo(x) for x in pontos],
                marker="o", label=f"{nome} ({p} {rotulo})"
            )

plt.axhline(1, linestyle=":", linewidth=1, color="black")
plt.xlabel("Tamanho da matriz (N)")
plt.ylabel("Speedup")
plt.title("Speedup em relacao a versao sequencial")
plt.legend(ncol=2)
plt.grid(True)
salvar("speedup.png")


# 3. Speedup OpenMP por numero de threads
plt.figure(figsize=(10, 6))
for n, linhas in sorted(por_n(openmp_static).items()):
    if n not in tempos_base:
        continue
    linhas.sort(key=paralelismo)
    plt.plot([paralelismo(x) for x in linhas],
             [tempos_base[n] / tempo(x) for x in linhas],
             marker="o", label=f"N={n}")
plt.axhline(1, linestyle=":", linewidth=1, color="black")
plt.xlabel("Numero de threads")
plt.ylabel("Speedup")
plt.title("OpenMP: speedup por numero de threads (static)")
plt.xticks(sorted({paralelismo(x) for x in openmp_static}))
plt.legend()
plt.grid(True)
salvar("speedup_openmp.png")


# 4. Speedup MPI por numero de processos
plt.figure(figsize=(10, 6))
for n, linhas in sorted(por_n(mpi).items()):
    if n not in tempos_base:
        continue
    linhas.sort(key=paralelismo)
    plt.plot([paralelismo(x) for x in linhas],
             [tempos_base[n] / tempo(x) for x in linhas],
             marker="o", label=f"N={n}")
plt.axhline(1, linestyle=":", linewidth=1, color="black")
plt.xlabel("Numero de processos")
plt.ylabel("Speedup")
plt.title("MPI: speedup por numero de processos")
plt.xticks(sorted({paralelismo(x) for x in mpi}))
plt.legend()
plt.grid(True)
salvar("speedup_mpi.png")


# 5. Overhead de comunicacao do MPI
if overhead:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for n, linhas in sorted(por_n(overhead).items()):
        linhas.sort(key=lambda x: int(x["processos"]))
        processos = [int(x["processos"]) for x in linhas]
        tempos_total = [float(x["tempo_total"]) for x in linhas]
        tempos_comp = [float(x["tempo_computacao"]) for x in linhas]
        valores = [float(x["overhead"]) for x in linhas]
        percentuais = [
            100.0 * o / t if t else 0.0 for o, t in zip(valores, tempos_total)
        ]
        axes[0].plot(processos, valores, marker="o", label=f"N={n}")
        axes[1].plot(processos, percentuais, marker="o", label=f"N={n}")
    axes[0].set_title("Overhead absoluto")
    axes[0].set_xlabel("Numero de processos")
    axes[0].set_ylabel("Overhead (segundos)")
    axes[1].set_title("Overhead relativo")
    axes[1].set_xlabel("Numero de processos")
    axes[1].set_ylabel("Overhead (% do tempo total)")
    for ax in axes:
        ax.grid(True)
        ax.set_xticks(sorted({int(x["processos"]) for x in overhead}))
    axes[0].legend()
    fig.suptitle("MPI: custo de comunicacao em funcao do numero de processos")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "overhead_mpi.png", dpi=300)
    plt.close(fig)


# 6. Eficiencia paralela
plt.figure(figsize=(10, 6))
for nome, dados in [("OpenMP", openmp_static), ("MPI", mpi)]:
    for n, linhas in sorted(por_n(dados).items()):
        t1 = next((tempo(x) for x in linhas if paralelismo(x) == 1), None)
        if t1 is None:
            continue
        linhas.sort(key=paralelismo)
        eficiencia = [
            (t1 / tempo(x)) / paralelismo(x) * 100
            for x in linhas
        ]
        plt.plot([paralelismo(x) for x in linhas], eficiencia,
                 marker="o", label=f"{nome} N={n}")

plt.xlabel("Numero de processos ou threads")
plt.ylabel("Eficiencia (%)")
plt.title("Eficiencia paralela")
plt.legend()
plt.grid(True)
salvar("eficiencia.png")


# 7. Escalabilidade forte (N fixo, p variavel)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (nome, dados) in zip(axes, [("OpenMP", openmp_static), ("MPI", mpi)]):
    for n, linhas in sorted(por_n(dados).items()):
        if n not in tempos_base:
            continue
        linhas.sort(key=paralelismo)
        processos = [paralelismo(x) for x in linhas]
        speedup = [tempos_base[n] / tempo(x) for x in linhas]
        ax.plot(processos, speedup, marker="o", label=f"N={n}")
    if dados:
        p_max_local = max(paralelismo(x) for x in dados)
        ax.plot([1, p_max_local], [1, p_max_local],
                linestyle=":", color="black", label="Ideal")
        ax.set_xticks(sorted({paralelismo(x) for x in dados}))
    ax.set_title(f"Escalabilidade forte - {nome}")
    ax.set_xlabel("Numero de processos ou threads")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True)
fig.tight_layout()
fig.savefig(GRAFICOS / "escalabilidade_forte.png", dpi=300)
plt.close(fig)


# 8. Comparacao das politicas de escalonamento
plt.figure(figsize=(10, 6))
if openmp:
    n_alvo = max({num(x) for x in openmp})
    for politica in ("static", "dynamic", "guided"):
        pontos = sorted(
            [x for x in openmp if num(x) == n_alvo and escala(x) == politica],
            key=paralelismo
        )
        if pontos:
            plt.plot([paralelismo(x) for x in pontos],
                     [tempo(x) for x in pontos],
                     marker="o", label=politica)
    plt.title(f"OpenMP: politicas de escalonamento (N={n_alvo})")
    plt.xlabel("Numero de threads")
    plt.ylabel("Tempo (segundos)")
    plt.xticks(sorted({paralelismo(x) for x in openmp}))
    plt.legend()
    plt.grid(True)
    salvar("escalonamento.png")


# 9. Impacto do tamanho do bloco
if blocked:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tamanhos = sorted({num(x) for x in blocked})[:2]
    if len(tamanhos) == 1:
        axes = [axes]
    for ax, n in zip(axes, tamanhos):
        for bloco in sorted({x["configuracao"].split("BS=")[-1] for x in blocked
                             if "BS=" in x["configuracao"]}):
            pontos = sorted(
                [x for x in blocked
                 if num(x) == n and f"BS={bloco}" in x["configuracao"]],
                key=paralelismo
            )
            if pontos:
                ax.plot([paralelismo(x) for x in pontos],
                        [tempo(x) for x in pontos],
                        marker="o", label=f"BS={bloco}")
        ax.set_title(f"Matriz N={n}")
        ax.set_xlabel("Numero de threads")
        ax.set_ylabel("Tempo (segundos)")
        ax.set_xticks(sorted({paralelismo(x) for x in blocked}))
        ax.grid(True)
    axes[0].legend()
    fig.suptitle("Impacto do tamanho dos blocos")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "tamanho_blocos.png", dpi=300)
    plt.close(fig)


# 10. Escalabilidade fraca
plt.figure(figsize=(10, 6))
for tipo, nome_curto in [("OpenMP Fraca", "OpenMP"), ("MPI Fraca", "MPI")]:
    linhas = sorted([x for x in fraca if x["paralelismo"] == tipo],
                    key=paralelismo)
    if not linhas:
        continue
    t1 = tempo(linhas[0])
    plt.plot([paralelismo(x) for x in linhas],
             [t1 / tempo(x) * 100 for x in linhas],
             marker="o", label=nome_curto)
if fraca:
    plt.axhline(100, linestyle=":", linewidth=1, color="black", label="Ideal")
    plt.xlabel("Numero de processos ou threads")
    plt.ylabel("Eficiencia fraca (%)")
    plt.title("Escalabilidade fraca (trabalho por unidade constante)")
    plt.legend()
    plt.grid(True)
    salvar("escalabilidade_fraca.png")


# 11. Lei de Amdahl x medicoes
plt.figure(figsize=(10, 6))
p_max = max([1] + [paralelismo(x) for x in openmp + mpi])
ps = list(range(1, max(16, p_max) + 1))

for serial in (0.01, 0.05, 0.10, 0.20):
    plt.plot(ps, [1 / (serial + (1 - serial) / p) for p in ps],
             linestyle="--", label=f"Amdahl s={serial:.2f}")

if openmp_static:
    n_alvo = max({num(x) for x in openmp_static})
    pontos = sorted([x for x in openmp_static if num(x) == n_alvo],
                    key=paralelismo)
    plt.plot([paralelismo(x) for x in pontos],
             [tempos_base[n_alvo] / tempo(x) for x in pontos],
             marker="o", linewidth=2, label=f"OpenMP medido (N={n_alvo})")
if mpi:
    n_alvo = max({num(x) for x in mpi})
    pontos = sorted([x for x in mpi if num(x) == n_alvo], key=paralelismo)
    plt.plot([paralelismo(x) for x in pontos],
             [tempos_base[n_alvo] / tempo(x) for x in pontos],
             marker="s", linewidth=2, label=f"MPI medido (N={n_alvo})")

plt.xlabel("Numero de unidades de processamento")
plt.ylabel("Speedup")
plt.title("Medicoes x limite teorico da Lei de Amdahl")
plt.legend()
plt.grid(True)
salvar("amdahl.png")

print("Graficos gerados com sucesso em:", GRAFICOS)
