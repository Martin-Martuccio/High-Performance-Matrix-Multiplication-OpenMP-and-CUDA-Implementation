#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#define ROW1 5000
#define COL1 5000
#define COL2 5000
#define ROW2 5000
#define BLOCK_SIZE  32 

using namespace std;

// Get a matrix element
__device__ int GetElement(const int* elements, int stride, int row, int col) {
    return elements[row * stride + col];
}

// Set a matrix element
__device__ void SetElement(int* elements, int stride, int row, int col, int value) {
    elements[row * stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ void GetSubMatrix(const int* elements, int* subElements, int stride, int blockRow, int blockCol, int row, int col) {
    int globalRow = blockRow * BLOCK_SIZE + row;
    int globalCol = blockCol * BLOCK_SIZE + col;
    if (globalRow < stride && globalCol < stride) {
        subElements[row * BLOCK_SIZE + col] = elements[globalRow * stride + globalCol];
    } else {
        subElements[row * BLOCK_SIZE + col] = 0;
    }
}

__global__ void matrixMultiply(int *A, int *B, int *C, int numARows, int numACols, int numBCols) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    int Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (numACols + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        GetSubMatrix(A, &As[0][0], numACols, blockRow, m, row, col);
        GetSubMatrix(B, &Bs[0][0], numBCols, m, blockCol, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if ((blockRow * BLOCK_SIZE + row) < numARows && (blockCol * BLOCK_SIZE + col) < numBCols) {
        SetElement(C, numBCols, blockRow * BLOCK_SIZE + row, blockCol * BLOCK_SIZE + col, Cvalue);
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        cout << msg << ": " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
 
    // Dimensioni della matrice
    const int r1 = ROW1, c1 = COL1, r2 = ROW2, c2 = COL2;

    // Allocazione della memoria unificata
    int *a, *b, *mult;
    checkCudaError(cudaMallocManaged(&a, r1 * c1 * sizeof(int)), "Failed to allocate unified memory for matrix A");
    checkCudaError(cudaMallocManaged(&b, r2 * c2 * sizeof(int)), "Failed to allocate unified memory for matrix B");
    checkCudaError(cudaMallocManaged(&mult, r1 * c2 * sizeof(int)), "Failed to allocate unified memory for matrix C");

    // Inizializzazione delle matrici
    for (int i = 0; i < r1 * c1; ++i) {
        a[i] = rand() % 10;
    }
    for (int i = 0; i < r2 * c2; ++i) {
        b[i] = rand() % 10;
    }
    for (int i = 0; i < r1 * c2; ++i) {
        mult[i] = 0;
    }

    // If column of first matrix in not equal to row of second matrix,
    if (c1 != r2) {
        cout << "Error! Column of first matrix not equal to row of second.";
        return -1;
    }

   
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((c2 + blockSize.x - 1) / blockSize.x, (r1 + blockSize.y - 1) / blockSize.y);

    // Timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");

    // Avvia il timer
    checkCudaError(cudaEventRecord(start, 0), "Failed to record start event");

    // Lunch kernel function
    matrixMultiply<<<gridSize, blockSize>>>(a, b, mult, r1, c1, c2);    
    checkCudaError(cudaGetLastError(), "Error after kernel launch");
    cudaDeviceSynchronize();

    // Ferma il timer
    checkCudaError(cudaEventRecord(stop, 0), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to calculate elapsed time");

    std::cout << "Time elapsed: " << milliseconds << " milliseconds." << std::endl;

    // Creazione di matrici Eigen utilizzando i dati esistenti
    // Eigen::MatrixXi matrix1(r1, c1);
    // Eigen::MatrixXi matrix2(r2, c2);

    // for (int i = 0; i < r1; ++i) {
    //     for (int j = 0; j < c1; ++j) {
    //         matrix1(i, j) = a[i * c1 + j];
    //     }
    // }

    // for (int i = 0; i < r2; ++i) {
    //     for (int j = 0; j < c2; ++j) {
    //         matrix2(i, j) = b[i * c2 + j];
    //     }
    // }

    // Calcolo del prodotto tra le due matrici
    // Eigen::MatrixXd result1 = matrix1.cast<double>() * matrix2.cast<double>();

    // // Creazione della matrice mult_double per il confronto
    // Eigen::MatrixXd mult_double(r1, c2);
    // for (int i = 0; i < r1; ++i) {
    //     for (int j = 0; j < c2; ++j) {
    //         mult_double(i, j) = mult[i * c2 + j];
    //     }
    // }

    // Stampa dei risultati
    //std::cout << "Result with Eigen:" << std::endl << result1 << std::endl;
    //std::cout << "Result with program:" << std::endl << mult_double << std::endl;

    // Confronto tra le due matrici
    // if (result1.isApprox(mult_double, 1e-5)) {
    //     std::cout << "Matrices are equal." << std::endl;
    // } else {
    //     std::cout << "Matrices are not equal." << std::endl;
    // }

    // Free device memory
    cudaFree(mult);
    cudaFree(a);
    cudaFree(b);

    return 0;
}
