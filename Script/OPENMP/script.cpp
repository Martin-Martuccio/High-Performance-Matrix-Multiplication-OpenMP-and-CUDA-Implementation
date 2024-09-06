#include <iostream>
#include <cstdlib> // Per la funzione system()

int main() {
    const int num_executions = 15; // Numero di volte da eseguire il programma

    system("icc Mat_Mul_allocation_loop.cpp -o Mat_Mul_allocation_loop -Ofast -diag-disable=10441 -I /home/stud/esame/Desktop/HPC/eigen-3.4.0 -xHost");

    for (int i = 0; i < num_executions; ++i) {
        system("./Mat_Mul_allocation_loop");
    }

    return 0;
}