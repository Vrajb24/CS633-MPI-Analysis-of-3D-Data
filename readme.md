# 3D Time Series Data Analysis with MPI

This project implements a parallel MPI application to analyze 3D time series data, finding local minima/maxima counts and global min/max values for each time step.

## Files Included

- `src.c` - Main MPI C source code implementing 3D domain decomposition
- `job.sh` - SLURM job script for running the application on HPC clusters
- `JobScriptGen.py` - Python utility to generate custom job scripts
- `PlotGen.py` - Performance analysis script that generates visualization graphs
- `Output/output_*_*_*_*_*.txt` - Output files from various test runs

## Setup Instructions

### Prerequisites

- MPI implementation (MPICH or OpenMPI)
- C compiler (gcc/icc)
- Python 3 with matplotlib and numpy for visualization

### Compilation

Compile the source code using:

```bash
mpicc -o src src.c -lm
```

## Implementation Approach

- Manual calculation of process coordinates in 3D grid
- Direct neighbor rank computation
- Parallel I/O using MPI-IO for efficient file access
- Ghost cell exchange with MPI_Sendrecv
- Distributed local minima/maxima detection
- Global reduction operations for result aggregation

## Running the Application

### Input Files

Input files should be placed in the root directory of the project. The expected format is:
- Binary files named as `data_NX_NY_NZ_NC.bin`
- Where NX, NY, NZ are grid dimensions and NC is the number of time steps

### Manual Execution

Run the application with:

```bash
mpirun -np P ./src input_file.bin PX PY PZ NX NY NZ NC output_file.txt
```

Where:
- `P` is the total number of processes (must equal PX×PY×PZ)
- `PX`, `PY`, `PZ` are process grid dimensions
- `NX`, `NY`, `NZ` are data grid dimensions
- `NC` is the number of time steps
- `output_file.txt` is where results will be written

### Using Job Scripts

#### Predefined Job Script

The included `job.sh` can be submitted directly to SLURM:

```bash
sbatch job.sh
```

By default, the job script runs a single configuration. You can modify it to run multiple test cases.

#### Generating Custom Job Scripts

Use the JobScriptGen.py script to create customized job scripts:

```bash
python3 JobScriptGen.py
```

Follow the interactive prompts to specify:
- Number of nodes and tasks per node
- Process count and grid decomposition (PX, PY, PZ)
- Data dimensions and time steps
- Wall time and partition

## Performance Analysis

After running the application, use the PlotGen.py script to analyze performance:

```bash
python3 PlotGen.py
```

This script automatically:
- Detects all output files matching the pattern `output_NX_NY_NZ_NC_P.txt`
- Parses timing information and generates four plots:
  1. Total execution time vs number of processes
  2. Speedup vs number of processes
  3. Parallel efficiency vs number of processes
  4. Execution time breakdown (read time vs computation time)
- Generates a `performance_analysis.png` file with all plots
- Prints strong scaling metrics to the console

## Important Notes

- Make sure the total number of processes (P) equals PX×PY×PZ
- Data dimensions (NX, NY, NZ) must be divisible by process dimensions (PX, PY, PZ)
- For optimal performance, adjust process decomposition based on your dataset
- The plot script handles comma-separated timing values in output files

## Example Workflow

1. Copy input bin files to the root folder
2. Compile the source code: `mpicc -o src src.c -lm`
3. Either:
   - Use `JobScriptGen.py` to create a custom job script
   - Manually edit the existing `job.sh`
4. Submit the job: `sbatch job.sh`
5. After completion, run `python3 PlotGen.py` to analyze performance
6. Include the plots and analysis in your report

## Output Format

- Line 1: Count of local minima and maxima for each time step
- Line 2: Global minimum and maximum values for each time step
- Line 3: Read time, computation time, and total execution time