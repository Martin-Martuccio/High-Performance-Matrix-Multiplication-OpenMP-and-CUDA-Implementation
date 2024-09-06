#include <iostream>
#include <cstdlib> // Per la funzione system()

int main() {
    const int num_executions = 15; // Numero di volte da eseguire il programma

    system("icc -I /home/stud/esame/Desktop/HPC/eigen-3.4.0 Mat_Mul_OpenMP_SIMD_tiling.cpp -o Mat_Mul_OpenMP_SIMD_tiling -Ofast -diag-disable=10441 -fopenmp -qopt-report=5 -xHost");

    for (int i = 0; i < num_executions; ++i) {
        system("./Mat_Mul_OpenMP_SIMD_tiling");
    }

    return 0;
}