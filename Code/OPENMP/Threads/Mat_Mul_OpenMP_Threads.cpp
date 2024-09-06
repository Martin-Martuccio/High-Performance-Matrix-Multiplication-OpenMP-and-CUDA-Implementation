#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <omp.h>
#include "omp.h"
#include <immintrin.h>
using namespace std;

#define rows_and_cols 5000
#define TILE_SIZE 256


int main(int argc, char* argv[])
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <num_threads>" << endl;
        return 1;
    }

    // Set the number of threads from the script
    int num_threads = atoi(argv[1]);

    const int r1 = rows_and_cols;
    const int c1 = rows_and_cols;
    const int r2 = rows_and_cols;
    const int c2 = rows_and_cols;


    int i, j, k;

    // Dichiarazione e allocazione contigua delle matrici con allineamento
    int* a = (int*) aligned_alloc(32, r1 * c1 * sizeof(int));
    int* b = (int*) aligned_alloc(32, r2 * c2 * sizeof(int));
    int* mult = (int*) aligned_alloc(32, r1 * c2 * sizeof(int));

    

    // If column of first matrix in not equal to row of second matrix,
    // ask the user to enter the size of matrix again.
    while (c1!=r2)
    {
        cout << "Error! Column of first matrix not equal to row of second.";
    }

    // Storing elements of first matrix.
    for(int i = 0; i < r1; ++i)
        for(int j = 0; j < c1; ++j)
        {
            a[i * c1 + j] = rand() % 10;
        }

    // Storing elements of second matrix.
    for(int i = 0; i < r2; ++i)
        for(int j = 0; j < c2; ++j)
        {
            b[i * c2 + j] = rand() % 10;
        }

    // Initializing elements of matrix mult to 0.
    fill(mult, mult + r1 * c2, 0);

    // -------------------- HOTSPOT -------------------------------

    const auto start = chrono::steady_clock::now();

    // Multiplying matrix a and b and storing in array mult using OpenMP and SIMD.
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int ii = 0; ii < r1; ii += TILE_SIZE) {
        for (int jj = 0; jj < c2; jj += TILE_SIZE) {
            for (int kk = 0; kk < c1; kk += TILE_SIZE) {
                for (int i = ii; i < min(ii + TILE_SIZE, r1); ++i) {
                    for (int k = kk; k < min(kk + TILE_SIZE, c1); ++k) {
                        // Utilizza vettorizzazione SIMD per calcolare il prodotto della riga i di a e la colonna j di b
                        #pragma omp simd
                        for (int j = jj; j < min(jj + TILE_SIZE, c2); ++j) {
                            mult[i * c2 + j] += a[i * c1 + k] * b[k * c2 + j];
                        }
                    }
                }
            }
        }
    }

    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " milliseconds." << endl;        

    // ------------------------------------------------------------

    // Creazione di matrici Eigen utilizzando i dati esistenti
    Eigen::MatrixXi matrix1(r1, c1);
    Eigen::MatrixXi matrix2(r2, c2);

    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c1; ++j) {
            matrix1(i, j) = a[i * c1 + j];
        }
    }

    for (int i = 0; i < r2; ++i) {
        for (int j = 0; j < c2; ++j) {
            matrix2(i, j) = b[i * c2 + j];
        }
    }

    // Calcolo del prodotto tra le due matrici
    Eigen::MatrixXd result1 = matrix1.cast<double>() * matrix2.cast<double>();

    // Creazione della matrice mult_double per il confronto
    Eigen::MatrixXd mult_double(r1, c2);
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c2; ++j) {
            mult_double(i, j) = static_cast<double>(mult[i * c2 + j]);
        }
    }

    /*
    // Stampa dei risultati
    cout << "Result with Eigen:" << endl << result1 << endl;
    cout << "Result with program:" << endl << mult_double << endl;
    */

    // Confronto tra le due matrici
    if (result1.isApprox(mult_double)) {
        cout << "Matrices are equal." << endl;
    } else {
        cout << "Matrices are not equal." << endl;
    }

    // Deallocazione della memoria
    free(a);
    free(b);
    free(mult);

    return 0;
}