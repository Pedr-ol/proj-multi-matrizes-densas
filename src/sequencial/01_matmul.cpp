/* Baseline 01 — Multiplicacao de matrizes densas C = A x B
 * Programacao Paralela (CCO085) — IESB 2026/2 — Prof. Rodrigo Goncalves Pinto
 *
 * Ponto de partida sequencial. O trio deve paralelizar em OpenMP e MPI.
 * Aspecto interessante: efeito de cache e ordem de lacos.
 *
 * Compilar: g++ -O2 -o 01_matmul 01_matmul.cpp
 * Executar: ./01_matmul 512        (matriz N x N; padrao 512)
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
using namespace std;

static double agora() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    vector<double> A(N*N), B(N*N), C(N*N, 0.0);

    // entrada deterministica (mesma para todos os trios)
    for (int i = 0; i < N*N; i++) { A[i] = (i % 100) * 0.01; B[i] = ((i * 7) % 100) * 0.01; }

    double t0 = agora();
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = A[i*N + k];
            for (int j = 0; j < N; j++)
                C[i*N + j] += a * B[k*N + j];
        }
    double t1 = agora();

    // checksum para validar corretude apos paralelizar
    double soma = 0.0;
    for (int i = 0; i < N*N; i++) soma += C[i];
    printf("N=%d  checksum=%.6f  tempo=%.6f s\n", N, soma, t1 - t0);
    return 0;
}
