#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00 ## wall-clock time limit
#SBATCH --partition=standard ## can be "standard" or "cpu"
echo "Job started at `date`"
echo "Starting executions with data_64_64_64_3.bin"
mpirun -np 8 ./src data_64_64_64_3.bin 2 2 2 64 64 64 3 output_64_64_64_3_8.txt
echo "All jobs completed at `date`"
