# Projeto Acadêmico de Programação Paralela — Multiplicação de Matrizes Densas

**Instituição:** Centro Universitário IESB  
**Disciplina:** Programação Paralela (CCO085) — 2026/2  
**Problema:** nº 1 — Multiplicação de matrizes densas `C = A × B`  
**Ambiente de referência:** Linux, `g++`, OpenMP e OpenMPI (ou o contêiner Docker do projeto)

O baseline sequencial é o código fornecido pelo professor e **não foi alterado**:
mesma entrada determinística, mesma ordem de cálculo e mesmo checksum de referência.
As versões OpenMP e MPI reproduzem esse resultado e são comparadas por tempo,
speedup, eficiência e escalabilidade.

---

## 1. Estrutura do repositório

```text
projeto-paralelo/
├── Makefile                  # compila tudo no ambiente da disciplina
├── README.md                 # este arquivo
├── AUTORES.md                # frentes de cada integrante
├── dockerfile                # ambiente Linux com g++, OpenMPI, Python e matplotlib
├── compose.yaml              # sobe o ambiente Docker
├── src/
│   ├── sequencial/
│   │   └── 01_matmul.cpp     # baseline do professor (lógica intacta)
│   ├── openmp/
│   │   ├── matmul_openmp.cpp         # paralelização por linhas + escalonamento configurável
│   │   └── matmul_openmp_blocked.cpp # versão com blocagem de cache
│   ├── mpi/
│   │   └── matmul_mpi.cpp    # decomposição por linhas + Scatterv/Bcast/Gatherv
│   └── tests/
│       ├── test_openmp.cpp
│       └── test_mpi.cpp
├── scripts/
│   ├── medir.py              # gera os CSVs de medição
│   ├── validar.py            # compara checksums contra o baseline
│   └── gerar_graficos.py     # gera todos os gráficos
├── resultados/               # CSVs medidos + metodologia.txt
├── graficos/                 # PNGs gerados
└── bin/                      # binários compilados
```

---

## 2. Rodando do zero

### Opção A — Docker (recomendada para casa)

```bash
docker compose build
docker compose run --rm ambiente
```

Dentro do contêiner:

```bash
make all          # compila baseline, OpenMP, OpenMP bloqueado, MPI e testes
make validar      # prova que as versões paralelas reproduzem o baseline
make medir        # gera resultados/*.csv
make graficos     # gera graficos/*.png
```

Para abrir um shell no ambiente já criado:

```bash
docker compose exec ambiente bash
```

### Opção B — Linux nativo (laboratório)

Requisitos: `g++` com OpenMP, `mpicxx`/`mpirun` (OpenMPI), `make` e Python 3 com
`matplotlib`.

```bash
make all
make validar
make medir
make graficos
```

---

## 3. Como executar cada versão

### Baseline sequencial

```bash
./bin/baseline 1024
```

Saída: `N=1024  checksum=...  tempo=... s`

### OpenMP

```bash
./bin/openmp <N> [threads] [static|dynamic|guided]
./bin/openmp 2048 8 static
./bin/openmp 2048 8 dynamic
./bin/openmp 2048 8 guided
```

Também é possível controlar pelo ambiente:

```bash
OMP_NUM_THREADS=8 ./bin/openmp 2048 8 static
```

### OpenMP com blocagem

```bash
./bin/openmp_blocked <N> [threads] [BS]
./bin/openmp_blocked 2048 8 64
```

### MPI

```bash
mpirun -np 4 ./bin/mpi 1024
```

No contêiner o `Makefile`/scripts já passam `--allow-run-as-root` e
`--oversubscribe` quando o OpenMPI é detectado.

---

## 4. Protocolo de medição

O script `scripts/medir.py` implementa o protocolo exigido:

- **Aquecimento:** 1 execução descartada por configuração (configurável em `AQUECIMENTO`);
- **Repetições:** 3 execuções medidas por configuração (configurável em `REPETICOES`);
- **Estatística:** mediana dos tempos medidos;
- **Entrada determinística:** a mesma do baseline;
- **Tamanhos:** baseline em `N = 256, 512, 1024, 2048`; versões paralelas em
  `N = 1024, 2048` (configurável);
- **Threads/procs:** `1, 2, 4, 8`;
- **Escalonamentos OpenMP:** `static`, `dynamic` e `guided`;
- **Blocos:** `32, 64, 128`;
- **Máquina:** dados de plataforma, host e número de CPUs gravados em
  `resultados/metodologia.txt`.

Exemplo de execução mais curta:

```bash
REPETICOES=1 AQUECIMENTO=0 NS_PAR=1024 python3 scripts/medir.py
```

Arquivos gerados:

| Arquivo | Conteúdo |
|---|---|
| `resultados/baseline.csv` | tempos do sequencial |
| `resultados/openmp.csv` | OpenMP por thread e escalonamento |
| `resultados/mpi.csv` | MPI por número de processos |
| `resultados/openmp_blocked.csv` | OpenMP com blocagem |
| `resultados/mpi_overhead.csv` | tempo total, tempo de computação e overhead de comunicação do MPI |
| `resultados/escalabilidade_fraca.csv` | escalabilidade fraca OpenMP/MPI |
| `resultados/metodologia.txt` | máquina, repetições e parâmetros |

Formato dos CSVs: `N,paralelismo,configuracao,tempo,checksum`.

---

## 5. Validação de corretude

```bash
make validar
# ou
python3 scripts/validar.py
```

O script compara o checksum do baseline com:

- OpenMP em `1, 2, 4, 8` threads (`static`);
- OpenMP com `dynamic` e `guided`;
- MPI com `1, 2, 4, 8` processos;
- OpenMP bloqueado com `BS=64`.

A tolerância relativa padrão é `1e-9` (configurável em `TOLERANCIA`), pois versões
com blocagem/reduction mudam apenas a ordem das somas em ponto flutuante. A versão
OpenMP principal mantém o checksum sequencial e reproduz o baseline bit a bit.

---

## 6. Análise do algoritmo sequencial

O baseline calcula, para cada linha `i` de `C`:

```text
para i em 0..N-1
  para k em 0..N-1
    a = A[i][k]
    para j em 0..N-1
      C[i][j] += a * B[k][j]
```

- **Complexidade de tempo:** `O(N³)` (dois laços externos de `N` e o laço interno de `N`);
- **Complexidade de memória:** `O(N²)` para `A`, `B` e `C`;
- **Hotspot:** o laço interno `j` é o ponto quente; ele executa `N³` multiplicações
  e adições e percorre `B` linha a linha, o que favorece a localidade de cache;
- **Operações de ponto flutuante:** aproximadamente `2N³` (uma multiplicação e uma
  adição por elemento).

A entrada é determinística (`A[i] = (i % 100) * 0.01` e
`B[i] = ((i * 7) % 100) * 0.01`), portanto o checksum é o gabarito.

### Perfilamento e baseline

Tempos já registrados no repositório (remedir na máquina da apresentação):

| N | Tempo sequencial (s) | Checksum |
|---:|---:|---:|
| 256 | 0,009916 | 4.109.165,296000 |
| 512 | 0,071757 | 32.883.431,310401 |
| 1024 | 0,904321 | 263.088.437,159955 |
| 2048 | 8,919968 | 2.104.745.207,318985 |

O crescimento de `N=1024` para `N=2048` (≈ 9,9× quando `N` dobra) confirma o custo
cúbico. A tabela de tempos por configuração gerada por `scripts/medir.py` é a
evidência de perfilamento por trecho/versão; o hotspot é o produto interno
`C[i][j] += A[i][k] * B[k][j]`, que corresponde ao laço mais interno.

### Lei de Amdahl

Com fração serial `s`, o speedup máximo é:

```text
S(p) = 1 / (s + (1 - s) / p)      e      S_max = 1 / s
```

A fração paralelizável pode ser estimada a partir do speedup medido `S` com `p`
unidades:

```text
s = (1/S - 1/p) / (1 - 1/p)
```

No gráfico `graficos/amdahl.png` as curvas teóricas são comparadas com os speedups
medidos. Exemplo com os dados atuais de OpenMP em `N=2048`, `p=8` (`S≈2,13`):
`s ≈ 0,395`, o que prevê `S_max ≈ 2,5`. O afastamento em relação ao ideal decorre
principalmente de **limite de banda de memória** (o algoritmo relê `B` muitas vezes),
overhead de sincronização e, no MPI, comunicação (`Bcast` de `N²` doubles a cada
execução). A versão com blocagem reduz esse efeito de cache e chega a speedup
próximo de linear (`≈ 7,7×` com 8 threads), o que mostra que a fração serial
estimada por Amdahl depende do modelo e da hierarquia de memória. Os dados acima
foram medidos no contêiner Docker do projeto (WSL2, 8 processadores lógicos);
recompute na máquina do laboratório antes da apresentação.

---

## 7. Estratégias de paralelização

### OpenMP (memória compartilhada)

- **Padrão:** `parallel for` sobre as linhas de `C` (decomposição de domínio).
- **Condição de corrida:** não existe escrita concorrente na mesma posição de `C`,
  pois cada thread escreve apenas nas suas linhas. Por isso não é necessário
  `critical` nem `atomic` na região principal.
- **Escolha entre critical/atomic/reduction:** `critical` e `atomic` serializariam
  a escrita e destruiriam o desempenho; `reduction` é usado no checksum da versão
  com blocagem, onde a soma é o único ponto de redução.
- **Escalonamento:** `schedule(runtime)` permite comparar `static`, `dynamic` e
  `guided` sem recompilar; a carga é uniforme por linha, então `static` tende a ser
  suficiente, mas a comparação está no gráfico `escalonamento.png`.
- **First-touch:** a versão bloqueada inicializa as matrizes em paralelo para
  reduzir o custo de alocação de páginas.

### MPI (memória distribuída)

- **Decomposição de domínio por linhas:** cada processo recebe um bloco de linhas
  de `A` e calcula o bloco correspondente de `C`.
- **Operações coletivas:**
  - `MPI_Scatterv` distribui as linhas de `A` (contagens e deslocamentos variáveis);
  - `MPI_Bcast` replica `B` para todos os processos;
  - `MPI_Gatherv` remonta `C` no processo 0;
  - `MPI_Reduce` coleta o maior tempo de computação para a análise de overhead.
- **Balanceamento:** `N % size` linhas extras são distribuídas entre os primeiros
  processos, evitando desbalanceamento de até uma linha.
- **Overhead:** a comunicação cresce com `N²` (broadcast de `B`) e com o número de
  processos; `resultados/mpi_overhead.csv` e o gráfico `overhead_mpi.png` mostram
  a diferença entre o tempo total e o tempo de computação por processo. Por isso
  o speedup MPI é menor que o OpenMP para estes tamanhos.

---

## 8. Gráficos gerados

`make graficos` produz em `graficos/`:

- `tempo_execucao.png` — tempo × tamanho;
- `speedup.png` — speedup × tamanho;
- `speedup_openmp.png` — speedup × threads;
- `speedup_mpi.png` — speedup × processos;
- `overhead_mpi.png` — overhead de comunicação absoluto e relativo × processos;
- `eficiencia.png` — eficiência × threads/processos;
- `escalabilidade_forte.png` — speedup com `N` fixo e `p` variável;
- `escalonamento.png` — comparação static/dynamic/guided;
- `tamanho_blocos.png` — impacto de `BS`;
- `escalabilidade_fraca.png` — eficiência em escalabilidade fraca;
- `amdahl.png` — medições × limite teórico de Amdahl.

A escalabilidade forte é representada por `escalabilidade_forte.png` (e pelos
gráficos de speedup com `N` fixo e `p` variável). A escalabilidade fraca usa `N`
crescente com `p` de modo que o trabalho por unidade (`N³/p`) permaneça constante.

---

## 9. Checklist dos requisitos do PDF

- [x] Baseline sequencial fornecido, sem alteração de lógica/entrada;
- [x] Perfilamento e identificação do hotspot (`O(N³)`, laço `j`);
- [x] Baseline para pelo menos três tamanhos, com metodologia de repetições e
      aquecimento documentada;
- [x] Estimativa da fração paralelizável e discussão da Lei de Amdahl;
- [x] OpenMP com tratamento explícito de corrida, `reduction`, escolha justificada
      e comparação de escalonamentos;
- [x] Curva de speedup OpenMP por threads;
- [x] MPI com decomposição de domínio, coletivas (`Scatterv`, `Bcast`, `Gatherv`,
      `Reduce`), resultado independente do número de processos e análise de
      overhead de comunicação;
- [x] Curva de speedup MPI por processos;
- [x] Validação de corretude contra o baseline (`scripts/validar.py`);
- [x] Análise experimental com gráficos de tempo, speedup, eficiência,
      escalabilidade forte/fraca e Amdahl;
- [x] Código das três versões em diretórios separados;
- [x] `Makefile` que compila tudo;
- [x] Script que gera os dados de medição em CSV;
- [x] `README.md` que roda o projeto do zero;
- [x] `AUTORES.md` com a frente de cada integrante.

---

## 10. Observações

- Os tempos dependem da máquina. Rode `make medir` no laboratório antes da
  apresentação e atualize os gráficos.
- O baseline deve permanecer exatamente como fornecido pelo professor.
- Para dúvidas de ambiente, use o contêiner Docker — ele já contém todas as
  dependências do projeto.
