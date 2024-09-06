#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#define ROW1 1024
#define COL1 1024
#define COL2 1024
#define ROW2 1024

using namespace std;


__global__ void matrixMultiply(int *A, int *B, int *C, int numARows, int numACols, int numBCols, size_t pitchA, size_t pitchB, size_t pitchC)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < numARows && col < numBCols) {
            int sum = 0;
            for (int k = 0; k < numACols; ++k) {
                sum += A[row * numACols + k] * B[k * numBCols + col];
            }
            C[row * numBCols + col] = sum;
        }

}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        cout << msg << ": " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    // Controllo dei parametri da riga di comando
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <threadSizeX> <threadSizeY>" << std::endl;
        return 1;
    }

    int threadSizeX = std::atoi(argv[1]);
    int threadSizeY = std::atoi(argv[2]);

    // Dimensioni della matrice
    const int r1 = ROW1, c1 = COL1, r2 = ROW2, c2 = COL2;

        size_t pitchA = 0, pitchB = 0, pitchC = 0;

        // Allocate Unified Memory -- accessible from CPU or GPU
        int *a, *b, *mult;
        checkCudaError(cudaMallocManaged(&a, r1 * c1 * sizeof(int)), "Failed to allocate unified memory for matrix A");
        checkCudaError(cudaMallocManaged(&b, r2 * c2 * sizeof(int)), "Failed to allocate unified memory for matrix B");
        checkCudaError(cudaMallocManaged(&mult, r1 * c2 * sizeof(int)), "Failed to allocate unified memory for matrix C");

        // Init randomly the matrices
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
    if (c1!=r2)
        {
            cout << "Error! Column of first matrix not equal to row of second.";
        }


// --------------------- Configuration of # Threads and # Block for each implementation --------------------------------------------------------------------------


    dim3 threadSize(threadSizeX, threadSizeY);
    dim3 blockSize((r1 + threadSize.x - 1) / threadSize.x, (c2 + threadSize.y - 1) / threadSize.y);

    int totalThreads = blockSize.x * blockSize.y * blockSize.z * threadSize.x * threadSize.y * threadSize.z;
    int totalElements = r1 * c2;
    cout << "threadSize: (" << threadSize.x << ", " << threadSize.y << ", " << threadSize.z << ")" <<endl;
    cout << "blockSize: (" << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << ")" <<endl;
    cout << "Total threads: " << totalThreads <<endl;
    cout << "Total threads for each block: " << threadSize.x * threadSize.y * threadSize.z <<endl;
    cout << "Total number of blocks (must be less or equal than 1024): " << blockSize.x * blockSize.y * blockSize.z  <<endl;
    int max_limit_blocks = 1024;
    int nBlocks = blockSize.x * blockSize.y * blockSize.z;
    if ( nBlocks > max_limit_blocks )
    {
        int surplus = nBlocks - max_limit_blocks;
        cout << "ALLERT! This configutation is not supported by this GPU" << endl;
    }

    cout << "Total elements to calculate: " << totalElements <<endl;

    // ----------------------Cuda implementation-----------------------

    // Timing
    // Usa cudaEvent_t per misurare il tempo CUDA
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");

    // Avvia il timer
    checkCudaError(cudaEventRecord(start, 0), "Failed to record start event");

    // Lunch kernel function
    matrixMultiply<<<threadSize,blockSize>>>(a,b,mult, r1, c1, c2, pitchA, pitchB, pitchC);
    checkCudaError(cudaGetLastError(), "Error after kernel launch");
    cudaDeviceSynchronize();

    // Ferma il timer
    checkCudaError(cudaEventRecord(stop, 0), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to calculate elapsed time");

    std::cout << "Time elapsed: " << milliseconds << " milliseconds." << std::endl;


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
            mult_double(i, j) = mult[i * c2 + j];
        }
    }

    // Stampa dei risultati
    //std::cout << "Result with Eigen:" << std::endl << result1 << std::endl;
    //std::cout << "Result with program:" << std::endl << mult_double << std::endl;

    // Confronto tra le due matrici
    if (result1.isApprox(mult_double, 1e-5)) {
        std::cout << "Matrices are equal." << std::endl;
    } else {
        std::cout << "Matrices are not equal." << std::endl;
    }

    // Free device memory
    cudaFree(mult);
    cudaFree(a);
    cudaFree(b);

    return 0;
}