#!/usr/bin/env python3
"""Valida a corretude das versoes paralelas contra o baseline.

O baseline define o gabarito. A comparacao usa erro relativo, porque
versoes com blocagem/reduction alteram apenas a ordem das somas em
ponto flutuante.

Variaveis de ambiente:
  N_VALIDACAO=512,1024
  THREADS_VALIDACAO=1,2,4,8
  PROCS_VALIDACAO=1,2,4,8
  TOLERANCIA=1e-9
"""
import os
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BIN = BASE / "bin"

NS = [int(x) for x in os.getenv("N_VALIDACAO", "512,1024").split(",")]
THREADS = [int(x) for x in os.getenv("THREADS_VALIDACAO", "1,2,4,8").split(",")]
PROCS = [int(x) for x in os.getenv("PROCS_VALIDACAO", "1,2,4,8").split(",")]
TOLERANCIA = float(os.getenv("TOLERANCIA", "1e-9"))
MPI_RUNNER = os.getenv("MPI_RUNNER", "mpirun")

SAIDA = re.compile(r"checksum=([-+0-9.eE]+)")


def exigir_binarios():
    faltando = [n for n in ("baseline", "openmp", "mpi", "openmp_blocked")
                if not (BIN / n).exists()]
    if faltando:
        sys.exit(
            "Binarios ausentes em bin/: " + ", ".join(faltando) +
            "\nRode 'make all' antes de validar."
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


def checksum(cmd, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            "Falha ao executar: " + " ".join(cmd) + "\n" + proc.stderr
        )
    m = SAIDA.search(proc.stdout)
    if not m:
        raise RuntimeError(
            "Saida sem checksum: " + " ".join(cmd) + "\n" + proc.stdout
        )
    return float(m.group(1))


def erro_relativo(referencia, valor):
    return abs(valor - referencia) / max(abs(referencia), 1.0)


def main():
    exigir_binarios()
    print(f"Tolerancia relativa: {TOLERANCIA:g}\n")

    falhas = 0

    for n in NS:
        referencia = checksum([str(BIN / "baseline"), str(n)])
        print(f"N={n}  baseline checksum={referencia:.6f}")

        casos = []

        for t in THREADS:
            casos.append((
                f"OpenMP {t} threads",
                checksum([str(BIN / "openmp"), str(n), str(t), "static"],
                         {"OMP_NUM_THREADS": str(t)})
            ))

        for escala in ("dynamic", "guided"):
            casos.append((
                f"OpenMP 4 threads {escala}",
                checksum([str(BIN / "openmp"), str(n), "4", escala],
                         {"OMP_NUM_THREADS": "4"})
            ))

        for p in PROCS:
            casos.append((
                f"MPI {p} processos",
                checksum(prefixo_mpi(p) + [str(BIN / "mpi"), str(n)])
            ))

        casos.append((
            "OpenMP bloqueado BS=64",
            checksum([str(BIN / "openmp_blocked"), str(n), "4", "64"],
                     {"OMP_NUM_THREADS": "4"})
        ))

        for nome, valor in casos:
            erro = erro_relativo(referencia, valor)
            ok = erro <= TOLERANCIA
            if not ok:
                falhas += 1
            print(
                f"  [{'OK' if ok else 'FALHA'}] {nome:<28} "
                f"checksum={valor:.6f}  erro_rel={erro:.2e}"
            )
        print()

    if falhas:
        print(f"Validacao REPROVADA: {falhas} caso(s) fora da tolerancia.")
        return 1

    print("Validacao APROVADA: todos os casos reproduzem o baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
