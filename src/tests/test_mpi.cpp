
#include <iostream>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int valor = rank;
    int soma = 0;

    MPI_Allreduce(
        &valor,
        &soma,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    cout << "Processo " << rank
         << " de " << size
         << " | Soma coletiva = " << soma
         << endl;

    MPI_Finalize();

    return 0;
}
