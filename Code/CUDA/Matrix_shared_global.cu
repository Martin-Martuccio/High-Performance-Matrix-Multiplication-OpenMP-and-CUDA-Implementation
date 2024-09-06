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


__global__ void matrixMultiply(int *A, int *B, int *C, int numARows, int numACols, int numBCols, size_t pitchA, size_t pitchB, size_t pitchC)
{

 #ifdef USE_SHARED_MEMORY
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
              // C[(blockRow * BLOCK_SIZE + row) * numBCols + (blockCol * BLOCK_SIZE + col)] = Cvalue;
            }
    #endif

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

    #ifdef USE_SHARED_MEMORY
        size_t pitchA = 0, pitchB = 0, pitchC = 0;

        // Vector A,B, Mult
        vector<int> a(r1 * c1); // Matrice A
        vector<int> b(r2 * c2); // Matrice B
        vector<int> mult(r1 * c2, 0); // Matrice risultato

            // Init randomly the matrixes
        for (int i = 0; i < r1; ++i)
            for (int j = 0; j < c1; ++j)
            {
                a[i * c1 + j] = rand() % 10;
            }

        for (int i = 0; i < r2; ++i)
            for (int j = 0; j < c2; ++j)
            {
                b[i * c2 + j] = rand() % 10;
            }

        for (int i = 0; i < r1; ++i)
            for (int j = 0; j < c2; ++j)
            {
                mult[i * c2 + j] = 0;
            }

        // cout << "this is matrix A: " << endl;

        //print matrix A
        // for (int i = 0; i < r1; ++i)
        // {
        //     for (int j = 0; j < c1; ++j)
        //     {
        //         cout << a[i * c1 + j] << " ";
        //     }
        //     cout << endl;
        // }

        // cout << "this is matrix B: " << endl;

        //print matrix B
        // for (int i = 0; i < r1; ++i)
        // {
        //     for (int j = 0; j < c1; ++j)
        //     {
        //         cout << b[i * c1 + j] << " ";
        //     }
        //     cout << endl;
        // }

    #endif



    // If column of first matrix in not equal to row of second matrix,
    while (c1!=r2)
        {
            cout << "Error! Column of first matrix not equal to row of second.";
        }


//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Parameter for the kernel function

// Allocate device memory
    int *d_mult, *d_a, *d_b;

// Configuration

    #ifdef USE_SHARED_MEMORY
    // Creating space in the GPU (checkCudaError for handling the error)
        checkCudaError(cudaMalloc(&d_a, r1 * c1 * sizeof(int)), "Errore allocazione memoria per d_a");
        checkCudaError(cudaGetLastError(), "Error after cudaMemcpy d_a");

        checkCudaError(cudaMalloc(&d_b, r2 * c2 * sizeof(int)), "Errore allocazione memoria per d_b");
        checkCudaError(cudaGetLastError(), "Error after cudaMemcpy d_b");

        checkCudaError(cudaMalloc(&d_mult, r1 * c2 * sizeof(int)), "Errore allocazione memoria per d_mult");
        checkCudaError(cudaGetLastError(), "Error after cudaMemcpy d_mult");

        checkCudaError(cudaMemcpy(d_a, a.data(), r1 * c1 * sizeof(int), cudaMemcpyHostToDevice), "Errore copia dati per d_a");
        checkCudaError(cudaMemcpy(d_b, b.data(), r2 * c2 * sizeof(int), cudaMemcpyHostToDevice), "Errore copia dati per d_b");
        checkCudaError(cudaMemcpy(d_mult, mult.data(), r1 * c2 * sizeof(int), cudaMemcpyHostToDevice), "Errore copia dati per d_mult");
        cudaDeviceSynchronize();
    #endif

//----------------------Check the correctness--------------

// Check the correctness of copy

    // Verifica che i dati siano copiati correttamente dalla CPU alla GPU e viceversa

        int *h_a_check = new int[r1 * c1];
        int *h_b_check = new int[r2 * c2];
        int *h_mult_check = new int[r1 * c2];
        bool is_correct =  true;

    #ifdef USE_SHARED_MEMORY

        checkCudaError(cudaMemcpy(h_a_check, d_a, r1 * c1 * sizeof(int), cudaMemcpyDeviceToHost), "Errore copia dati da d_a");
        checkCudaError(cudaMemcpy(h_b_check, d_b, r2 * c2 * sizeof(int), cudaMemcpyDeviceToHost), "Errore copia dati da d_b");
        checkCudaError(cudaMemcpy(h_mult_check, d_mult, r1 * c2 * sizeof(int), cudaMemcpyDeviceToHost), "Errore copia dati da d_mult");
        cudaDeviceSynchronize();

        // Verifica per la matrice A
        for (int i = 0; i < r1 * c1; ++i) {
            if (a[i] != h_a_check[i]) {
                cout << "Errore nella copia dei dati per la matrice A all'elemento " << i
                    << ": atteso " << a[i] << ", ottenuto " << h_a_check[i] << endl;
                is_correct = false;
            }
        }

        // Verifica per la matrice B
        for (int i = 0; i < r2 * c2; ++i) {
            if (b[i] != h_b_check[i]) {
                cout << "Errore nella copia dei dati per la matrice B all'elemento " << i
                    << ": atteso " << b[i] << ", ottenuto " << h_b_check[i] << endl;
                is_correct = false;
            }
        }

        if (is_correct) {
            cout << "Dati copiati correttamente." << endl;
        } else {
            cout << "Errore nella copia dei dati." << endl;
        }

    #endif


// --------------------- Configuration of # Threads and # Block for each implementation --------------------------------------------------------------------------

    #ifdef USE_SHARED_MEMORY
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((c2 + blockSize.x - 1) / blockSize.x, (r1 + blockSize.y - 1) / blockSize.y);

    #endif

    /*
    int totalThreads = blockSize.x * blockSize.y * blockSize.z * threadSize.x * threadSize.y * threadSize.z;
    int totalElements = r1 * c2;
    cout << "threadSize: (" << threadSize.x << ", " << threadSize.y << ", " << threadSize.z << ")" <<endl;
    */
    cout << "blockSize: (" << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << ")" <<endl;
    // cout << "Total threads: " << totalThreads <<endl;
    // cout << "Total threads for each block: " << threadSize.x * threadSize.y * threadSize.z <<endl;
    cout << "Total number of blocks (must be less or equal than 1024): " << blockSize.x * blockSize.y * blockSize.z  <<endl;

    // Timing
    // Usa cudaEvent_t per misurare il tempo CUDA
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");

    // Avvia il timer
    checkCudaError(cudaEventRecord(start, 0), "Failed to record start event");

    // Lunch kernel function
    // matrixMultiply<<<threadSize,blockSize>>>(d_a, d_b, d_mult, r1, c1, c2, pitchA, pitchB, pitchC);
        matrixMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_mult, r1, c1, c2, pitchA, pitchB, pitchC);
        checkCudaError(cudaGetLastError(), "Error after kernel launch");
        cudaDeviceSynchronize();

    // Ferma il timer
    checkCudaError(cudaEventRecord(stop, 0), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to calculate elapsed time");

    std::cout << "Time elapsed: " << milliseconds << " milliseconds." << std::endl;

    #ifdef USE_SHARED_MEMORY
        // Copy result from device to host
        cudaMemcpy(mult.data(), d_mult, r1 * c2 * sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaError(cudaGetLastError(), "Error after cudaMemcpy d_mult");
    #endif


    // ------------------------------------------------------------

    // Creazione di matrici Eigen utilizzando i dati esistenti
    // Eigen::MatrixXi matrix1(r1, c1);
    // Eigen::MatrixXi matrix2(r2, c2);


    // for (int i = 0; i < r1; ++i) {
    //     for (int j = 0; j < c1; ++j) {

    //         #ifdef USE_SHARED_MEMORY
    //             matrix1(i, j) = a[i * c1 + j];
    //         #endif
    //     }
    // }

    // for (int i = 0; i < r2; ++i) {
    //     for (int j = 0; j < c2; ++j) {

    //             #ifdef USE_SHARED_MEMORY
    //                 matrix2(i, j) = b[i * c2 + j];
    //             #endif
    //     }
    // }

    // Calcolo del prodotto tra le due matrici
    // Eigen::MatrixXd result1 = matrix1.cast<double>() * matrix2.cast<double>();

    // Creazione della matrice mult_double per il confronto
    // Eigen::MatrixXd mult_double(r1, c2);
    // for (int i = 0; i < r1; ++i) {
    //     for (int j = 0; j < c2; ++j) {

    //         #ifdef USE_SHARED_MEMORY
    //             mult_double(i, j) = mult[i * c2 + j];
    //         #endif

    //     }
    // }


    // Stampa dei risultati
    //cout << "Result with Eigen:" << endl << result1 << endl;
    //cout << "Result with program:" << endl << mult_double << endl;

    // Confronto tra le due matrici
    // if (result1.isApprox(mult_double, 1e-5)) {
    //     cout << "Matrices are equal." << endl;
    // } else {
    //     cout << "Matrices are not equal." << endl;
    // }

    // Free device memory
    cudaFree(d_mult);
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a_check;
    delete[] h_b_check;
    delete[] h_mult_check;

    return 0;
}