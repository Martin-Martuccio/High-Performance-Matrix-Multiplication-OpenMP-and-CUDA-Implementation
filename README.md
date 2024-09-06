# High Performance Matrix Multiplication: OpenMP and CUDA Implementation

## Overview

This repository contains a comprehensive report detailing the implementation and optimization of matrix multiplication using OpenMP and CUDA. The goal of the project was to enhance the performance of matrix multiplication, which is a fundamental operation in many scientific computing fields, using modern parallel computing techniques. By leveraging both CPU and GPU parallelism, significant speedup and efficiency improvements were achieved.

## Contents

- **Introduction**: Provides an overview of the importance of matrix multiplication in scientific computing and the performance benefits achievable through parallelization.
  
- **Methodology**: Describes the computational environment and tools used:
  - **CPU**: Intel Core i9-12900K with 24 logical CPUs and AVX2 vectorization support.
  - **GPU**: NVIDIA T400 with 384 CUDA cores and 2GB of GDDR6 memory.
  - **Technologies**: 
    - **OpenMP**: For CPU-based parallelism.
    - **CUDA**: For GPU-based parallelism.

- **Matrix Multiplication Algorithm**: Discusses the basic matrix multiplication algorithm and identifies computational hotspots that were targeted for optimization.

- **Optimizations**: 
  - **Basic Sequential Algorithm**: Initial implementation with optimizations based on compiler flags.
  - **OpenMP Optimization**:
    - Use of loop swapping, contiguous allocation, and vectorization to improve cache efficiency and computational performance.
    - Use of multi-threading and SIMD (Single Instruction, Multiple Data) to improve performance on CPU.
    - Loop tiling to enhance cache locality.
  - **CUDA Implementation**:
    - **Global Memory**: Basic implementation utilizing global memory.
    - **Pitched Memory**: Improved versions leveraging CUDA memory management techniques.
    - **Unified Memory**: Used to compare execution time with other memory management techniques.
    - **Tiling Version**: Exploits shared memory to load portions (tiles) of the matrices for faster computations.
  
- **Performance Analysis**: 
  - Detailed comparison of execution times across different optimization techniques for both OpenMP and CUDA implementations.
  - The analysis includes speedup graphs, efficiency calculations, and a breakdown of improvements obtained from various strategies.

- **Final Comparison**: A side-by-side comparison between OpenMP and CUDA, showing the strengths and weaknesses of each approach. For large matrices, OpenMP showed significant improvements, while CUDA had limitations due to hardware constraints but demonstrated strong performance in GPU-specific tasks.

## Key Findings

- **OpenMP**: Achieved substantial improvements in execution times, especially for large matrices, by utilizing multi-threading, SIMD, and loop tiling.
- **CUDA**: Offered significant speedups, particularly with shared memory, though it was constrained by the hardware's CUDA core limitations.

## How to Run the Code

This project includes source code for both OpenMP and CUDA implementations. Follow the instructions below to compile and run the code:

### OpenMP
1. Compile the code using the Intel C Compiler (icc):
   ```bash
   icc -O2 matrix_multiplication.cpp -o matrix_omp
2. Run the program
   ```bash
   icc -O2 matrix_multiplication.cpp -o matrix_omp

### OpenMP
1. Compile the CUDA code using the NVIDIA Compiler (nvcc):
   ```bash
   nvcc matrix_multiplication.cu -o matrix_cuda
2. Run the program
   ```bash
   ./matrix_cuda

#### Make sure you have the necessary compilers and CUDA toolkit installed before running the code.
   
## Conclusion

This project demonstrated that significant performance improvements can be achieved in matrix multiplication using parallel computing techniques such as OpenMP and CUDA. The results indicate that while OpenMP is particularly effective for CPU-bound tasks, CUDA excels in leveraging GPU resources for parallel computation, albeit with some limitations related to hardware constraints (in our case).

For more details on the performance analysis, graphs, and implementation specifics, please refer to the full report available in this repository. Report : [Report Link]( ... )

## Authors
- [Martin Martuccio](https://github.com/Martin-Martuccio) - Project Author
- [Lorenzo Mesi](https://github.com/LorenzoMesi) - Project Author



![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
