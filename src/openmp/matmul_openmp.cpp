/* Versao OpenMP — Multiplicacao de matrizes densas C = A x B
 *
 * Decomposicao de dominio por linhas de C: cada thread calcula um
 * subconjunto disjunto de linhas. Como nao ha escrita concorrente na
 * mesma posicao de memoria, a regiao paralela nao precisa de critical
 * nem atomic. O checksum e mantido sequencial para preservar bit a bit
 * o resultado do baseline.
 *
 * Uso: ./matmul_openmp <N> [threads] [static|dynamic|guided]
 *   N        tamanho da matriz (padrao 512)
 *   threads  numero de threads (padrao: omp_get_max_threads())
 *   escala   politica de escalonamento (padrao: static)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;

static double agora() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {

    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    const char* escala = (argc > 3) ? argv[3] : "static";

    if (threads <= 0) {
        threads = omp_get_max_threads();
    }
    omp_set_num_threads(threads);

    omp_sched_t politica = omp_sched_static;
    if (strcmp(escala, "dynamic") == 0) {
        politica = omp_sched_dynamic;
    } else if (strcmp(escala, "guided") == 0) {
        politica = omp_sched_guided;
    } else if (strcmp(escala, "static") != 0) {
        fprintf(stderr, "Escalonamento invalido: %s (use static, dynamic ou guided)\n", escala);
        return 1;
    }
    omp_set_schedule(politica, 0);

    vector<double> A(N*N), B(N*N), C(N*N, 0.0);

    // Entrada deterministica identica ao baseline.
    for (int i = 0; i < N*N; i++) {
        A[i] = (i % 100) * 0.01;
        B[i] = ((i * 7) % 100) * 0.01;
    }

    double t0 = agora();

    // Cada linha de C e escrita por exatamente uma thread.
    // A ordem dos lacos k e j e a mesma do baseline, portanto o
    // resultado numerico e identico ao sequencial.
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {

            double a = A[i*N + k];

            for (int j = 0; j < N; j++) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }

    double t1 = agora();

    // Checksum sequencial: mantem o mesmo arredondamento do baseline.
    double soma = 0.0;

    for (int i = 0; i < N*N; i++) {
        soma += C[i];
    }

    printf("N=%d  threads=%d  schedule=%s  checksum=%.6f  tempo=%.6f s\n",
           N, threads, escala, soma, t1 - t0);

    return 0;
}
