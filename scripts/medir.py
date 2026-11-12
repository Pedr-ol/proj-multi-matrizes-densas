#!/usr/bin/env python3
"""Gera os dados de medicao em CSV.

Protocolo:
  - cada configuracao executa AQUECIMENTO vezes antes das medicoes;
  - o tempo reportado e a MEDIANA de REPETICOES execucoes;
  - a entrada e deterministica (definida pelo baseline do professor).

Variaveis de ambiente (opcionais):
  REPETICOES=3 AQUECIMENTO=1
  NS_BASE=256,512,1024,2048
  NS_PAR=1024,2048
  NS_BLOCKED=1024,2048
  THREADS=1,2,4,8  PROCS=1,2,4,8
  SCHEDULES=static,dynamic,guided
  BLOCOS=32,64,128
  WEAK_BASE=512
  INCLUIR_BLOQUEADO=1 INCLUIR_ESCALONAMENTO=1 INCLUIR_FRACA=1
  MPI_RUNNER=mpirun
"""
import csv
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BIN = BASE / "bin"
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(exist_ok=True)

REPETICOES = int(os.getenv("REPETICOES", "3"))
AQUECIMENTO = int(os.getenv("AQUECIMENTO", "1"))
NS_BASE = [int(x) for x in os.getenv("NS_BASE", "256,512,1024,2048").split(",")]
NS_PAR = [int(x) for x in os.getenv("NS_PAR", "1024,2048").split(",")]
NS_BLOCKED = [int(x) for x in os.getenv("NS_BLOCKED", "1024,2048").split(",")]
THREADS = [int(x) for x in os.getenv("THREADS", "1,2,4,8").split(",")]
PROCS = [int(x) for x in os.getenv("PROCS", "1,2,4,8").split(",")]
SCHEDULES = [x.strip() for x in os.getenv("SCHEDULES", "static,dynamic,guided").split(",")]
BLOCOS = [int(x) for x in os.getenv("BLOCOS", "32,64,128").split(",")]
WEAK_BASE = int(os.getenv("WEAK_BASE", "512"))
INCLUIR_BLOQUEADO = os.getenv("INCLUIR_BLOQUEADO", "1") == "1"
INCLUIR_ESCALONAMENTO = os.getenv("INCLUIR_ESCALONAMENTO", "1") == "1"
INCLUIR_FRACA = os.getenv("INCLUIR_FRACA", "1") == "1"
MPI_RUNNER = os.getenv("MPI_RUNNER", "mpirun")

SAIDA = re.compile(
    r"checksum=([-+0-9.eE]+).*?tempo=([-+0-9.eE]+)"
    r"(?:.*?tempo_computacao=([-+0-9.eE]+))?"
)


def exigir_binarios():
    necessarios = ["baseline", "openmp", "mpi"]
    if INCLUIR_BLOQUEADO:
        necessarios.append("openmp_blocked")
    faltando = [n for n in necessarios if not (BIN / n).exists()]
    if faltando:
        sys.exit(
            "Binarios ausentes em bin/: " + ", ".join(faltando) +
            "\nRode 'make all' antes de medir."
        )


def prefixo_mpi(np):
    cmd = [MPI_RUNNER]
    try:
        versao = subprocess.run(
            [MPI_RUNNER, "--version"],
            capture_output=True, text=True, timeout=15
        )
        texto = versao.stdout + versao.stderr
        if "Open MPI" in texto or "OpenRTE" in texto:
            cmd += ["--allow-run-as-root", "--oversubscribe"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    cmd += ["-np", str(np)]
    return cmd


def executar(cmd, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            "Falha ao executar: " + " ".join(cmd) +
            "\nstdout:\n" + proc.stdout +
            "\nstderr:\n" + proc.stderr
        )
    m = SAIDA.search(proc.stdout)
    if not m:
        raise RuntimeError(
            "Saida nao reconhecida de: " + " ".join(cmd) +
            "\nstdout:\n" + proc.stdout +
            "\nstderr:\n" + proc.stderr
        )
    return (
        float(m.group(1)),
        float(m.group(2)),
        float(m.group(3)) if m.group(3) else None,
    )


def medir(cmd, env_extra=None):
    checksum = None
    tempos = []
    computacoes = []
    for i in range(AQUECIMENTO + REPETICOES):
        checksum, tempo, computacao = executar(cmd, env_extra)
        if i >= AQUECIMENTO:
            tempos.append(tempo)
            if computacao is not None:
                computacoes.append(computacao)
    computacao = statistics.median(computacoes) if computacoes else None
    return statistics.median(tempos), checksum, computacao


def gravar_csv(nome, linhas):
    destino = RESULTADOS / nome
    with open(destino, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "paralelismo", "configuracao", "tempo", "checksum"])
        for n, par, config, tempo, checksum in linhas:
            w.writerow([n, par, config, f"{tempo:.6f}", f"{checksum:.6f}"])
    print(f"  -> {destino}")


def gravar_overhead(linhas):
    destino = RESULTADOS / "mpi_overhead.csv"
    with open(destino, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "N", "processos", "tempo_total", "tempo_computacao", "overhead"
        ])
        for n, p, total, computacao in linhas:
            overhead = total - computacao
            w.writerow([
                n, p, f"{total:.6f}", f"{computacao:.6f}", f"{overhead:.6f}"
            ])
    print(f"  -> {destino}")


def medir_baseline():
    print("[1/5] Baseline sequencial")
    linhas = []
    for n in NS_BASE:
        tempo, checksum, _ = medir([str(BIN / "baseline"), str(n)])
        print(f"  N={n:<5} {tempo:9.6f} s  checksum={checksum:.6f}")
        linhas.append((n, "Sequencial", "1 processo", tempo, checksum))
    gravar_csv("baseline.csv", linhas)


def medir_openmp():
    print("[2/5] OpenMP (strong scaling)")
    linhas = []
    for n in NS_PAR:
        for escala in SCHEDULES:
            for t in THREADS:
                tempo, checksum, _ = medir(
                    [str(BIN / "openmp"), str(n), str(t), escala],
                    {"OMP_NUM_THREADS": str(t)}
                )
                print(f"  N={n:<5} {t} threads {escala:<7} {tempo:9.6f} s")
                linhas.append(
                    (n, "OpenMP", f"{t} threads - {escala}", tempo, checksum)
                )
    gravar_csv("openmp.csv", linhas)


def medir_mpi():
    print("[3/5] MPI (strong scaling)")
    linhas = []
    overhead = []
    for n in NS_PAR:
        for p in PROCS:
            tempo, checksum, computacao = medir(
                prefixo_mpi(p) + [str(BIN / "mpi"), str(n)]
            )
            print(f"  N={n:<5} {p} processos           {tempo:9.6f} s")
            linhas.append((n, "MPI", f"{p} processos", tempo, checksum))
            if computacao is not None:
                overhead.append((n, p, tempo, computacao))
    gravar_csv("mpi.csv", linhas)
    gravar_overhead(overhead)


def medir_bloqueado():
    print("[4/5] OpenMP com blocagem")
    linhas = []
    for n in NS_BLOCKED:
        for bs in BLOCOS:
            for t in THREADS:
                tempo, checksum, _ = medir(
                    [str(BIN / "openmp_blocked"), str(n), str(t), str(bs)],
                    {"OMP_NUM_THREADS": str(t)}
                )
                print(f"  N={n:<5} {t} threads BS={bs:<3}   {tempo:9.6f} s")
                linhas.append(
                    (n, "OpenMP Bloqueado", f"{t} threads - BS={bs}", tempo, checksum)
                )
    gravar_csv("openmp_blocked.csv", linhas)


def escalabilidade_fraca():
    print("[5/5] Escalabilidade fraca (trabalho por unidade constante)")
    linhas = []

    for t in THREADS:
        n = max(128, int(round(WEAK_BASE * (t ** (1.0 / 3.0)))))
        tempo, checksum, _ = medir(
            [str(BIN / "openmp"), str(n), str(t), "static"],
            {"OMP_NUM_THREADS": str(t)}
        )
        print(f"  OpenMP {t} threads  N={n:<5} {tempo:9.6f} s")
        linhas.append((n, "OpenMP Fraca", f"{t} threads", tempo, checksum))

    for p in PROCS:
        n = max(128, int(round(WEAK_BASE * (p ** (1.0 / 3.0)))))
        tempo, checksum, _ = medir(prefixo_mpi(p) + [str(BIN / "mpi"), str(n)])
        print(f"  MPI    {p} processos N={n:<5} {tempo:9.6f} s")
        linhas.append((n, "MPI Fraca", f"{p} processos", tempo, checksum))

    gravar_csv("escalabilidade_fraca.csv", linhas)


def gravar_metodologia():
    destino = RESULTADOS / "metodologia.txt"
    with open(destino, "w") as f:
        f.write("Metodologia de medicao\n")
        f.write("=====================\n")
        f.write(f"Data: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Maquina: {socket.gethostname()}\n")
        f.write(f"Plataforma: {platform.platform()}\n")
        f.write(f"Processadores logicos: {os.cpu_count()}\n")
        f.write(f"Repeticoes (mediana): {REPETICOES}\n")
        f.write(f"Execucoes de aquecimento descartadas: {AQUECIMENTO}\n")
        f.write(f"Tamanhos do baseline: {NS_BASE}\n")
        f.write(f"Tamanhos paralelos: {NS_PAR}\n")
        f.write(f"Threads OpenMP: {THREADS}\n")
        f.write(f"Processos MPI: {PROCS}\n")
        f.write(f"Escalonamentos: {SCHEDULES}\n")
        f.write(f"Blocos: {BLOCOS}\n")
        f.write(f"Base da escalabilidade fraca: {WEAK_BASE}\n")
    print(f"  -> {destino}")


def main():
    exigir_binarios()
    print(
        f"Medindo com {REPETICOES} repeticoes + {AQUECIMENTO} aquecimento(s)\n"
    )
    medir_baseline()
    if INCLUIR_ESCALONAMENTO:
        medir_openmp()
        medir_mpi()
    if INCLUIR_BLOQUEADO:
        medir_bloqueado()
    if INCLUIR_FRACA:
        escalabilidade_fraca()
    gravar_metodologia()
    print("Medicoes concluidas.")


if __name__ == "__main__":
    main()
