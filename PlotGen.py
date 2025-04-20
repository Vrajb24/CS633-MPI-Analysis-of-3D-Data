import matplotlib.pyplot as plt
import numpy as np
import re
import os
from collections import defaultdict

# Regular expression to extract parameters from filenames
# Matches: output_NX_NY_NZ_NC_P.txt
file_pattern = re.compile(r'output_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.txt')

# Scan directory for output files
output_files = []
for filename in os.listdir('.'):
    match = file_pattern.match(filename)
    if match:
        nx, ny, nz, nc, p = map(int, match.groups())
        output_files.append({
            'filename': filename,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'nc': nc,
            'p': p,
            'dataset': f"{nx}_{ny}_{nz}_{nc}"
        })

if not output_files:
    print("No output files found matching the pattern output_NX_NY_NZ_NC_P.txt")
    exit(1)

# Group files by dataset
datasets = {}
for file_info in output_files:
    dataset = file_info['dataset']
    if dataset not in datasets:
        datasets[dataset] = []
    datasets[dataset].append(file_info)

# Sort each dataset's files by process count
for dataset in datasets:
    datasets[dataset].sort(key=lambda x: x['p'])

# Create data structures for timing information
timing_data = {
    dataset: {
        'process_counts': [],
        'read_times': [],
        'compute_times': [],
        'total_times': []
    } for dataset in datasets
}

# Extract timing data from files
for dataset, files in datasets.items():
    for file_info in files:
        filename = file_info['filename']
        p = file_info['p']
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    # Clean up the values by removing commas and any other non-numeric characters
                    # except decimal points and negative signs
                    time_values = re.findall(r'-?\d+\.\d+', lines[2])
                    if len(time_values) >= 3:
                        timing_data[dataset]['process_counts'].append(p)
                        timing_data[dataset]['read_times'].append(float(time_values[0]))
                        timing_data[dataset]['compute_times'].append(float(time_values[1]))
                        timing_data[dataset]['total_times'].append(float(time_values[2]))
                        print(f"Successfully read data from {filename}: {time_values}")
                    else:
                        print(f"Warning: Invalid timing data format in {filename}")
                else:
                    print(f"Warning: Not enough lines in {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")

# Check if we have data to plot
has_data = False
for dataset in datasets:
    if timing_data[dataset]['process_counts']:
        has_data = True
        break

if not has_data:
    print("No valid timing data found in any of the output files.")
    exit(1)

# Create plots
plt.figure(figsize=(12, 10))

# Plot 1: Execution Time vs Number of Processes
plt.subplot(2, 2, 1)
for dataset in datasets:
    if timing_data[dataset]['process_counts']:  # Check if we have data
        plt.plot(
            timing_data[dataset]['process_counts'], 
            timing_data[dataset]['total_times'], 
            marker='o', 
            label=f"Dataset {dataset}"
        )
plt.title('Total Execution Time vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.legend()

# Plot 2: Speedup vs Number of Processes
plt.subplot(2, 2, 2)
for dataset in datasets:
    if timing_data[dataset]['process_counts']:  # Check if we have data
        process_counts = timing_data[dataset]['process_counts']
        times = timing_data[dataset]['total_times']
        if times:
            baseline = times[0]  # Time with smallest process count
            p_baseline = process_counts[0]
            speedup = [baseline/time for time in times]
            plt.plot(process_counts, speedup, marker='o', label=f"Dataset {dataset}")
            
            # Add linear speedup reference line for the first dataset only
            if dataset == list(datasets.keys())[0]:
                ideal_speedup = [p/p_baseline for p in process_counts]
                plt.plot(process_counts, ideal_speedup, 'k--', label='Linear Speedup')
plt.title('Speedup vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()

# Plot 3: Efficiency vs Number of Processes
plt.subplot(2, 2, 3)
for dataset in datasets:
    if timing_data[dataset]['process_counts']:  # Check if we have data
        process_counts = timing_data[dataset]['process_counts']
        times = timing_data[dataset]['total_times']
        if times:
            baseline = times[0]  # Time with smallest process count
            p_baseline = process_counts[0]
            efficiency = [baseline*p_baseline/(time*p) for p, time in zip(process_counts, times)]
            plt.plot(process_counts, efficiency, marker='o', label=f"Dataset {dataset}")
plt.title('Parallel Efficiency vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency')
plt.grid(True)
plt.axhline(y=1, color='k', linestyle='--', label='Ideal Efficiency')
plt.legend()

# Plot 4: Breakdown of execution time components
plt.subplot(2, 2, 4)
for dataset_idx, dataset in enumerate(datasets):
    if not timing_data[dataset]['process_counts']:
        continue
        
    process_counts = timing_data[dataset]['process_counts']
    read_times = timing_data[dataset]['read_times']
    compute_times = timing_data[dataset]['compute_times']
    
    # Calculate positions for bars
    x = np.arange(len(process_counts))
    width = 0.35
    offset = 0 if dataset_idx == 0 else width
    
    # Plot read time and compute time as stacked bars
    plt.bar(x + offset - width/2, read_times, width, 
            label=f"Read Time ({dataset})" if dataset_idx == 0 else None)
    plt.bar(x + offset - width/2, compute_times, width, bottom=read_times, 
            label=f"Compute Time ({dataset})" if dataset_idx == 0 else None)

plt.title('Time Breakdown by Process Count')
plt.xlabel('Number of Processes')
plt.ylabel('Time (seconds)')
plt.xticks(x, process_counts)
plt.legend()

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=300)
print("Plot saved as 'performance_analysis.png'")

# Print strong scaling analysis
print("\nStrong Scaling Analysis:")
print("=======================")
for dataset in datasets:
    print(f"\nDataset: {dataset}")
    if timing_data[dataset]['process_counts']:
        process_counts = timing_data[dataset]['process_counts']
        times = timing_data[dataset]['total_times']
        if times:
            print("Processes\tTime(s)\tSpeedup\tEfficiency")
            print("---------\t------\t-------\t----------")
            baseline = times[0]
            p_baseline = process_counts[0]
            for p, time in zip(process_counts, times):
                speedup = baseline/time
                efficiency = speedup/(p/p_baseline)
                print(f"{p}\t\t{time:.4f}\t{speedup:.4f}\t{efficiency:.4f}")