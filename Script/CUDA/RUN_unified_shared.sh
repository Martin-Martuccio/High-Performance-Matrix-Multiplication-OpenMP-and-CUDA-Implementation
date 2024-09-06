#!/bin/bash

# Configurazione dell'ambiente per il compilatore NVHPC
NVARCH=$(uname -s)_$(uname -m)
export NVARCH
NVCOMPILERS=/opt/nvidia/hpc_sdk
export NVCOMPILERS
MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/23.7/compilers/man
export MANPATH
PATH=$NVCOMPILERS/$NVARCH/23.7/compilers/bin:$PATH
export PATH

# Compilazione del codice con USE_SHARED_MEMORY
nvc++ -I/home/stud/esame/hpc/eigen --diag_suppress code_is_unreachable -o matrix_unified_shared Matrix_unified_shared.cu

# File di output per i risultati
output_file="risultati_use_shared_unified_memory_32_5000.txt"
> $output_file

# Definire le dimensioni delle tile candidate
tile_sizes=(32)
matrix_sizes=(5000)

# Eseguire i test per ogni combinazione di tile size e matrix size
for tile_size in "${tile_sizes[@]}"
do
    for matrix_size in "${matrix_sizes[@]}"
    do
        echo -e "\n============================================\n" >> $output_file
        echo "Testing tile_size=${tile_size} with matrix_size=${matrix_size}" >> $output_file
        echo "Testing tile_size=${tile_size} with matrix_size=${matrix_size}"

        for i in {1..10}
        do
            echo -e "\nEsecuzione con tile_size=${tile_size} - Matrix size ${matrix_size} - Esecuzione $i" >> $output_file
            echo "Timestamp: $(date)" >> $output_file
            ./matrix_unified_shared >> temp_output.txt 2>&1
            cat temp_output.txt >> $output_file
            rm temp_output.txt
        done
    done
done

echo -e "\nEsecuzioni completate. I risultati sono stati salvati in $output_file." >> $output_file
