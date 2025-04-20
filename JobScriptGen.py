#!/usr/bin/env python3

"""
Script to generate a SLURM job submission file by asking for parameters
interactively and populating a template job script.
"""

import os
import sys

def get_numeric_input(prompt, default=None):
    """Get valid numeric input from user."""
    while True:
        if default:
            user_input = input(f"{prompt} [default: {default}]: ")
            if user_input == "":
                return default
        else:
            user_input = input(f"{prompt}: ")
        
        try:
            value = int(user_input)
            return value
        except ValueError:
            print("Error: Please enter a valid number.")


def get_string_input(prompt, default=None):
    """Get string input from user with optional default."""
    if default:
        user_input = input(f"{prompt} [default: {default}]: ")
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ")


def main():
    print("SLURM Job Script Generator")
    print("==========================\n")
    
    # Get node information
    nodes = get_numeric_input("Enter number of nodes")
    tasks_per_node = get_numeric_input("Enter number of tasks per node")
    
    # Calculate total number of processes
    num_processes = get_numeric_input("Enter number of processes")
    
    # Get data dimensions
    x = get_numeric_input("Enter data dimension x")
    y = get_numeric_input("Enter data dimension y")
    z = get_numeric_input("Enter data dimension z")
    c = get_numeric_input("Enter number of timestamps (c)")
    
    # Get process dimensions
    px = get_numeric_input("Enter process dimension px")
    py = get_numeric_input("Enter process dimension py")
    pz = get_numeric_input("Enter process dimension pz")
    
    # Get wall time
    wall_time = get_string_input("Enter wall time limit (format HH:MM:SS)", "00:10:00")
    
    # Get partition
    partition = get_string_input("Enter partition (standard/cpu)", "standard")
    
    # Create the job script
    job_script = f"""#!/bin/bash
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time={wall_time} ## wall-clock time limit
#SBATCH --partition={partition} ## can be "standard" or "cpu"
echo "Job started at `date`"
echo "Starting executions with data_{x}_{y}_{z}_{c}.bin"
mpirun -np {num_processes} ./src data_{x}_{y}_{z}_{c}.bin {px} {py} {pz} {x} {y} {z} {c} output_{x}_{y}_{z}_{c}_{num_processes}.txt
echo "All jobs completed at `date`"
"""
    
    # Write to file
    filename = "job.sh"
    with open(filename, "w") as f:
        f.write(job_script)
    
    # Make file executable
    os.chmod(filename, 0o755)
    
    print(f"\nJob script '{filename}' has been generated successfully!")
    print(f"You can submit it using: sbatch {filename}")


if __name__ == "__main__":
    main()