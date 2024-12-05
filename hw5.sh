#!/bin/bash

module load cuda/11.7.0

nvcc HW5_base.cu -o HW5_base 
# # HW5_base 1000 4 /scratch/ualclsd0142/gpubase1000.txt
HW5_base 5000 5000 /scratch/ualclsd0142/gpubase5k.txt
HW5_base 10000 5000 /scratch/ualclsd0142/gpubase10k.txt

nvcc HW5_shared.cu -o HW5_shared
# # ./HW5_shared 100 4 /scratch/ualclsd0142/gpushared100.txt
HW5_shared 5000 5000 /scratch/ualclsd0142/gpushared5k.txt
HW5_shared 10000 5000 /scratch/ualclsd0142/gpushared10k.txt

# nvcc HW5_stride.cu -o HW5_stride
# # HW5_stride 1000 4 /scratch/ualclsd0142/gpustride1000.txt
# HW5_stride 5000 5000 /scratch/ualclsd0142/gpustride5k.txt
# HW5_stride 10000 5000 /scratch/ualclsd0142/gpustride10k.txt

cd /scratch/ualclsd0142/
# diff gpubase100.txt gpushared100.txt
# diff gpubase1000.txt gpustride1000.txt
diff gpubase5k.txt gpushared5k.txt
# # diff gpubase5k.txt gpustride5k.txt
diff gpubase10k.txt gpushared10k.txt
diff gpubase5k.txt output.5000.5000.20.txt
diff gpubase10k.txt output.10000.5000.20.txt
