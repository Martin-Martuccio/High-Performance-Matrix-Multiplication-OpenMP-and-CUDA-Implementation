#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <omp.h>
#include "omp.h"
using namespace std;

#define rows_and_cols 5000

int main()
{

    const int r1 = rows_and_cols;
    const int c1 = rows_and_cols;
    const int r2 = rows_and_cols;
    const int c2 = rows_and_cols;


    int i, j, k;

    // Dichiarazione e allocazione contigua delle matrici
    vector<int> a(r1 * c1);
    vector<int> b(r2 * c2);
    vector<int> mult(r1 * c2, 0);

    

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
    fill(mult.begin(), mult.end(), 0);

    // -------------------- HOTSPOT -------------------------------

    const auto start = chrono::steady_clock::now();

    // Multiplying matrix a and b and storing in array mult.
    for(int i = 0; i < r1; ++i)
        for(int k = 0; k < c1; ++k)
            for(int j = 0; j < c2; ++j)
            {
                mult[i * c2 + j] += a[i * c1 + k] * b[k * c2 + j];
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

    return 0;
}