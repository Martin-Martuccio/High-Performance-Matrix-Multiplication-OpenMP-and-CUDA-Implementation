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

# Compilazione del codice con il comando specificato
nvc++ -I/home/stud/esame/hpc/eigen --diag_suppress code_is_unreachable -o matrix_unified unified_matrix.cu

# File di output per i risultati
output_file="risultati_unified_memory_1024.txt"
> $output_file

# Limiti della GPU
max_threads=1024
r1=1024
c2=1024

# Eseguire per configurazioni da 2^0 a 2^10
for x in {0..10}
do
    for y in {0..10}
    do
        threadSizeX=$((2**x))
        threadSizeY=$((2**y))
        
        # Calcolare blockSize
        blockSizeX=$(((r1 + threadSizeX - 1) / threadSizeX))
        blockSizeY=$(((c2 + threadSizeY - 1) / threadSizeY))
        total_blocks=$((blockSizeX * blockSizeY))
        
        # Controllare se la configurazione è valida
        total_threads=$((threadSizeX * threadSizeY))
        echo "Controllando configurazione: threadSizeX=$threadSizeX, threadSizeY=$threadSizeY, total_threads=$total_threads, blockSizeX=$blockSizeX, blockSizeY=$blockSizeY, total_blocks=$total_blocks"
        
        if [[ $total_threads -le $max_threads && $threadSizeX -le 1024 && $threadSizeY -le 1024 && $total_blocks -le $max_threads ]]
        then
            echo -e "\n============================================\n" >> $output_file
            echo "Configurazione valida: ($threadSizeX, $threadSizeY). Eseguendo 10 esecuzioni." >> $output_file
            for i in {1..10}
            do
                echo "Esecuzione con threadSize ($threadSizeX, $threadSizeY) - Esecuzione $i" >> $output_file
                echo "Timestamp: $(date)" >> $output_file
                ./matrix_unified $threadSizeX $threadSizeY >> temp_output.txt 2>&1
                cat temp_output.txt >> $output_file
                rm temp_output.txt
            done
              else
            echo -e "\n============================================\n" >> $output_file
            echo "Configurazione non valida: ($threadSizeX, $threadSizeY). Saltando." >> $output_file
            if [[ $total_threads -gt $max_threads ]]; then
                echo "Motivo: Numero totale di threads ($total_threads) supera il limite di $max_threads." >> $output_file
                echo "Numero totale di threads ($total_threads) supera il limite di $max_threads."
            fi
            if [[ $threadSizeX -gt 1024 ]]; then
                echo "Motivo: Numero di threads per la dimensione X ($threadSizeX) supera il limite di 1024." >> $output_file
                echo "Numero di threads per la dimensione X ($threadSizeX) supera il limite di 1024."
            fi
            if [[ $threadSizeY -gt 1024 ]]; then
                echo "Motivo: Numero di threads per la dimensione Y ($threadSizeY) supera il limite di 1024." >> $output_file
                echo "Numero di threads per la dimensione Y ($threadSizeY) supera il limite di 1024."
            fi
            if [[ $total_blocks -gt $max_threads ]]; then
                echo "Motivo: Numero totale di blocchi ($total_blocks) supera il limite di $max_threads." >> $output_file
                echo "Numero totale di blocchi ($total_blocks) supera il limite di $max_threads."
            fi
        fi
    done
done


echo "Esecuzioni completate. I risultati sono stati salvati in $output_file."
