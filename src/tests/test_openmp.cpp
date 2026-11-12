
#include <iostream>
#include <omp.h>

using namespace std;

int main() {

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int total = omp_get_num_threads();

        #pragma omp critical
        {
            cout << "Thread " << id
                 << " de " << total
                 << " executando." << endl;
        }
    }

    return 0;
}
