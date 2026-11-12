/* Versao OpenMP com blocagem (cache blocking) — C = A x B
 *
 * O laco interno e dividido em blocos para melhorar a localidade de
 * cache. A paralelizacao continua sobre as linhas de C: cada thread
 * escreve apenas nas suas proprias linhas (sem corrida).
 *
 * Uso: ./matmul_openmp_blocked <N> [threads] [BS]
 *   N        tamanho da matriz (padrao 512)
 *   threads  numero de threads (padrao: omp_get_max_threads())
 *   BS       tamanho do bloco (padrao 128; teste 32, 64 e 128)
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <omp.h>
using namespace std;

static double agora() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    int BS = (argc > 3) ? atoi(argv[3]) : 128;

    if (threads <= 0) {
        threads = omp_get_max_threads();
    }
    if (BS <= 0) {
        BS = 128;
    }
    omp_set_num_threads(threads);

    double* __restrict A = new double[(size_t)N*N];
    double* __restrict B = new double[(size_t)N*N];
    double* __restrict C = new double[(size_t)N*N];

    // First-touch paralelo + entrada deterministica (mesma do professor)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            size_t idx = (size_t)i*N + j;
            A[idx] = (idx % 100) * 0.01;
            B[idx] = ((idx * 7) % 100) * 0.01;
            C[idx] = 0.0;
        }

    double t0 = agora();

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BS) {
        int iMax = min(ii + BS, N);
        for (int kk = 0; kk < N; kk += BS) {
            int kMax = min(kk + BS, N);
            for (int jj = 0; jj < N; jj += BS) {
                int jMax = min(jj + BS, N);
                for (int i = ii; i < iMax; i++) {
                    for (int k = kk; k < kMax; k++) {
                        double a = A[(size_t)i*N + k];
                        #pragma omp simd
                        for (int j = jj; j < jMax; j++)
                            C[(size_t)i*N + j] += a * B[(size_t)k*N + j];
                    }
                }
            }
        }
    }

    double t1 = agora();

    // Reduction elimina a corrida na soma do checksum.
    double soma = 0.0;
    #pragma omp parallel for reduction(+:soma)
    for (size_t i = 0; i < (size_t)N*N; i++) soma += C[i];

    printf("N=%d  threads=%d  BS=%d  checksum=%.6f  tempo=%.6f s\n",
           N, threads, BS, soma, t1 - t0);

    delete[] A; delete[] B; delete[] C;
    return 0;
}
