/* Versao MPI — Multiplicacao de matrizes densas C = A x B
 *
 * Decomposicao de dominio por linhas de A/C:
 *   - MPI_Scatterv distribui blocos de linhas de A para cada processo;
 *   - MPI_Bcast replica B integralmente em todos os processos;
 *   - cada processo calcula o seu bloco de C;
 *   - MPI_Gatherv remonta C no processo 0.
 *
 * O numero de linhas por processo pode variar em 1 (N nao divisivel),
 * por isso sao usados Scatterv/Gatherv em vez de Scatter/Gather.
 *
 * Uso: mpirun -np <P> ./matmul_mpi <N>
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 512;

    int linhas = N / size;
    int resto = N % size;

    vector<int> contagem(size);
    vector<int> deslocamento(size);

    int inicio = 0;

    for (int p = 0; p < size; p++) {
        int qtd = linhas + (p < resto ? 1 : 0);

        contagem[p] = qtd * N;
        deslocamento[p] = inicio * N;

        inicio += qtd;
    }

    int minhas_linhas = linhas + (rank < resto ? 1 : 0);

    vector<double> A, B(N*N), C_local(minhas_linhas * N, 0.0);
    vector<double> C;

    if (rank == 0) {
        A.resize(N*N);
        C.resize(N*N, 0.0);

        for (int i = 0; i < N*N; i++) {
            A[i] = (i % 100) * 0.01;
            B[i] = ((i * 7) % 100) * 0.01;
        }
    }

    vector<double> A_local(minhas_linhas * N);

    // Barreira para alinhar o inicio da medicao entre os processos.
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        contagem.data(),
        deslocamento.data(),
        MPI_DOUBLE,
        A_local.data(),
        minhas_linhas * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        B.data(),
        N*N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double tc0 = MPI_Wtime();

    for (int i = 0; i < minhas_linhas; i++) {
        for (int k = 0; k < N; k++) {

            double a = A_local[i*N + k];

            for (int j = 0; j < N; j++) {
                C_local[i*N + j] += a * B[k*N + j];
            }
        }
    }

    double tc1 = MPI_Wtime();

    MPI_Gatherv(
        C_local.data(),
        minhas_linhas * N,
        MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr,
        contagem.data(),
        deslocamento.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double t_computacao = tc1 - tc0;
    double t_computacao_max = 0.0;
    MPI_Reduce(&t_computacao, &t_computacao_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double t1 = MPI_Wtime();

        double soma = 0.0;

        for (int i = 0; i < N*N; i++) {
            soma += C[i];
        }

        printf("N=%d  procs=%d  checksum=%.6f  tempo=%.6f s  tempo_computacao=%.6f s\n",
               N, size, soma, t1 - t0, t_computacao_max);
    }

    MPI_Finalize();

    return 0;
}
