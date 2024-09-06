#include <iostream>
#include <cstdlib>
#include <cstdio>
using namespace std;

int main() {

    for (int j = 0; j < 15; j++)
    {
        // Numeri di threads da testare
        int thread_counts[] = {1, 2, 4, 8, 16};
        int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

        for (int i = 0; i < num_tests; ++i) {
            int num_threads = thread_counts[i];
            string command = "./Mat_Mul_OpenMP_Threads " + to_string(num_threads);
            cout << "Number of threads: " << to_string(num_threads) << endl;

            // Eseguire il comando
            int ret = system(command.c_str());
            if (ret != 0) {
                cerr << "Error: command failed with code " << ret << endl;
            }
        }
    }


    return 0;
}
