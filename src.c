#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>


#define DEBUG 0   // turn on for debug msgs

int main(int argc, char *argv[]) {
    int rank, size;
    double t1, t2, t3; 
    double read_time, comp_time, total_time;
    double max_read_time, max_comp_time, max_total_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // check args
    if (argc != 10) {
        if (rank == 0) {
            printf("Usage: %s <input_file> <PX> <PY> <PZ> <NX> <NY> <NZ> <NC> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    // parse cmd line args
    char *input_file = argv[1];
    int px = atoi(argv[2]);
    int py = atoi(argv[3]);
    int pz = atoi(argv[4]);
    int nx = atoi(argv[5]);
    int ny = atoi(argv[6]);
    int nz = atoi(argv[7]);
    int nc = atoi(argv[8]);  // num time steps
    char *output_file = argv[9];
    
    // make sure we have right # of processes
    if (size != px * py * pz) {
        if (rank == 0) {
            printf("Error: Number of processes (%d) must equal PX*PY*PZ (%d*%d*%d=%d)\n", 
                  size, px, py, pz, px*py*pz);
        }
        MPI_Finalize();
        return 1;
    }
    
    t1 = MPI_Wtime(); // start timing
    
    
    // figure out process coords from rank
    int coords[3];
    coords[2] = rank / (px * py);                       
    int remainder = rank % (px * py);
    coords[1] = remainder / px;                         
    coords[0] = remainder % px;                         

    if (DEBUG && rank == 0) {
        printf("Starting decomposition for grid size: %dx%dx%d with %d processes\n", nx, ny, nz, size);
    }
    
    // local dimensions - handle uneven splits
    int local_nx = nx / px + (coords[0] < nx % px ? 1 : 0);
    int local_ny = ny / py + (coords[1] < ny % py ? 1 : 0);
    int local_nz = nz / pz + (coords[2] < nz % pz ? 1 : 0);

    if (DEBUG) {
        printf("Rank %d: Coords (%d,%d,%d), Local size: %dx%dx%d\n", 
               rank, coords[0], coords[1], coords[2], local_nx, local_ny, local_nz);
    }
    
    // starting global index for each dimension
    int start_x = 0;
    for (int i = 0; i < coords[0]; i++) {
        start_x += nx / px + (i < nx % px ? 1 : 0);
    }

    int start_y = 0;
    for (int i = 0; i < coords[1]; i++) {
        start_y += ny / py + (i < ny % py ? 1 : 0);
    }

    int start_z = 0;
    for (int i = 0; i < coords[2]; i++) {
        start_z += nz / pz + (i < nz % pz ? 1 : 0);
    }

    
    // neighbor ranks in each direction
    int neighbors[6];

    
    // left neighbor (x-)
    int left_coords[3] = {coords[0]-1, coords[1], coords[2]};
    neighbors[0] = (left_coords[0] >= 0) ? 
        left_coords[0] + left_coords[1]*px + left_coords[2]*px*py : MPI_PROC_NULL;

    
    // right (x+)
    int right_coords[3] = {coords[0]+1, coords[1], coords[2]};
    neighbors[1] = (right_coords[0] < px) ? 
        right_coords[0] + right_coords[1]*px + right_coords[2]*px*py : MPI_PROC_NULL;

    
    // down (y-)
    int down_coords[3] = {coords[0], coords[1]-1, coords[2]};
    neighbors[2] = (down_coords[1] >= 0) ? 
        down_coords[0] + down_coords[1]*px + down_coords[2]*px*py : MPI_PROC_NULL;

    
    // up
    int up_coords[3] = {coords[0], coords[1]+1, coords[2]};
    neighbors[3] = (up_coords[1] < py) ? 
        up_coords[0] + up_coords[1]*px + up_coords[2]*px*py : MPI_PROC_NULL;

    
    // back (z-)
    int back_coords[3] = {coords[0], coords[1], coords[2]-1};
    neighbors[4] = (back_coords[2] >= 0) ? 
        back_coords[0] + back_coords[1]*px + back_coords[2]*px*py : MPI_PROC_NULL;

    
    // front (z+)
    int front_coords[3] = {coords[0], coords[1], coords[2]+1};
    neighbors[5] = (front_coords[2] < pz) ? 
        front_coords[0] + front_coords[1]*px + front_coords[2]*px*py : MPI_PROC_NULL;

    
    // allocate local array w/ ghost cells
    float *local_data = (float *)malloc((local_nx + 2) * (local_ny + 2) * (local_nz + 2) * nc * sizeof(float));
    if (!local_data) {
        printf("Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    
    // init with sentinel vals
    for (int t = 0; t < nc; t++) {
        for (int k = 0; k < local_nz + 2; k++) {
            for (int j = 0; j < local_ny + 2; j++) {
                for (int i = 0; i < local_nx + 2; i++) {
                    int idx = t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                            k * (local_nx + 2) * (local_ny + 2) + 
                            j * (local_nx + 2) + i;
                    local_data[idx] = FLT_MAX; 
                }
            }
        }
    }

    
    // temp buffer for file read
    float *read_buffer = (float *)malloc(local_nx * local_ny * local_nz * nc * sizeof(float));
    if (!read_buffer) {
        printf("Process %d: Memory allocation failed for read_buffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    
    // --- FILE I/O SECTION ---
    
    // sync before file ops
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (DEBUG && rank == 0) {
        printf("Opening file: %s\n", input_file);
    }

    
    // open file
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    
    // Calculate file offset for this process's data
    MPI_Offset base_offset = 0;
    
    
    // figure out where our chunk starts in file
    long long start_point_idx = 0;
    
    
    // add z offset
    start_point_idx += (long long)start_z * nx * ny;
    
    
    // add y offset
    start_point_idx += (long long)start_y * nx;
    
    
    // add x offset
    start_point_idx += start_x;
    
    
    base_offset = (MPI_Offset)(start_point_idx * nc * sizeof(float));
    
    if (DEBUG) {
        printf("Rank %d: Reading from file offset %lld\n", rank, (long long)base_offset);
    }

    
    // setup parallel read
    int global_sizes[4] = {nz, ny, nx, nc};  
    
    
    // our chunk size
    int local_sizes[4] = {local_nz, local_ny, local_nx, nc};
    
    
    // where our chunk starts in global array
    int starts[4] = {start_z, start_y, start_x, 0};  
    
    
    // make datatype to describe our part
    MPI_Datatype subarray_type;
    MPI_Type_create_subarray(4, global_sizes, local_sizes, starts, 
                           MPI_ORDER_C, MPI_FLOAT, &subarray_type);
    
    MPI_Type_commit(&subarray_type);
    
    
    // tell MPI how to view the file
    MPI_File_set_view(fh, 0, MPI_FLOAT, subarray_type, "native", MPI_INFO_NULL);
    
    
    // actually read the data
    MPI_Status status;
    MPI_File_read_all(fh, read_buffer, local_nx * local_ny * local_nz * nc, 
                    MPI_FLOAT, &status);
    
    
    // did we get everything?
    int items_read;
    MPI_Get_count(&status, MPI_FLOAT, &items_read);
    
    if (items_read != local_nx * local_ny * local_nz * nc) {
        printf("Rank %d: Expected to read %d items, but read %d items\n", 
              rank, local_nx * local_ny * local_nz * nc, items_read);
    }
    
    
    // cleanup
    MPI_Type_free(&subarray_type);
    MPI_File_close(&fh);
    
    
    // wait for everyone
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (DEBUG && rank == 0) {
        printf("All processes completed file reading\n");
    }

    
    // Move data to our format with ghost cells
    // File has time as inner dim, we want it outer
    int idx = 0;
    for (int z = 0; z < local_nz; z++) {
        for (int y = 0; y < local_ny; y++) {
            for (int x = 0; x < local_nx; x++) {
                for (int t = 0; t < nc; t++) {
                    
                    float value = read_buffer[(z * local_ny * local_nx + y * local_nx + x) * nc + t];
                    
                    
                    // +1 to skip ghost layer
                    int local_idx = t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                                   (z + 1) * (local_nx + 2) * (local_ny + 2) + 
                                   (y + 1) * (local_nx + 2) + (x + 1);
                    
                    local_data[local_idx] = value;
                }
            }
        }
    }
    
    free(read_buffer);
    
    t2 = MPI_Wtime();  // end of read timing
    read_time = t2 - t1;

    
    // Exchange ghost cells - for each timestep
    for (int t = 0; t < nc; t++) {
        
        // buffers for ghost exchange - need separate send/recv
        float *send_buf_x = (float *)malloc(local_ny * local_nz * sizeof(float));
        float *recv_buf_x = (float *)malloc(local_ny * local_nz * sizeof(float));
        float *send_buf_y = (float *)malloc(local_nx * local_nz * sizeof(float));
        float *recv_buf_y = (float *)malloc(local_nx * local_nz * sizeof(float));
        float *send_buf_z = (float *)malloc(local_nx * local_ny * sizeof(float));
        float *recv_buf_z = (float *)malloc(local_nx * local_ny * sizeof(float));
        
        
        if (neighbors[0] != MPI_PROC_NULL) { // left neighbor exists
            
            // pack data from inner x face
            for (int k = 0; k < local_nz; k++) {
                for (int j = 0; j < local_ny; j++) {
                    send_buf_x[k * local_ny + j] = local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                                                            (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                                                            (j + 1) * (local_nx + 2) + 1];
                }
            }
            MPI_Sendrecv(send_buf_x, local_ny * local_nz, MPI_FLOAT, neighbors[0], 0,
                        recv_buf_x, local_ny * local_nz, MPI_FLOAT, neighbors[0], 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            
            // unpack to ghost layer
            for (int k = 0; k < local_nz; k++) {
                for (int j = 0; j < local_ny; j++) {
                    local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                            (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                            (j + 1) * (local_nx + 2) + 0] = recv_buf_x[k * local_ny + j];
                }
            }
        }
        
        if (neighbors[1] != MPI_PROC_NULL) { // right neighbor
            
            // pack right face
            for (int k = 0; k < local_nz; k++) {
                for (int j = 0; j < local_ny; j++) {
                    send_buf_x[k * local_ny + j] = local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                                                            (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                                                            (j + 1) * (local_nx + 2) + local_nx];
                }
            }
            MPI_Sendrecv(send_buf_x, local_ny * local_nz, MPI_FLOAT, neighbors[1], 0,
                        recv_buf_x, local_ny * local_nz, MPI_FLOAT, neighbors[1], 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            
            // unpack to right ghost
            for (int k = 0; k < local_nz; k++) {
                for (int j = 0; j < local_ny; j++) {
                    local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                            (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                            (j + 1) * (local_nx + 2) + (local_nx + 1)] = recv_buf_x[k * local_ny + j];
                }
            }
        }
        
        
        if (neighbors[2] != MPI_PROC_NULL) { // down neighbor
            
            // pack bottom face
            for (int k = 0; k < local_nz; k++) {
                for (int i = 0; i < local_nx; i++) {
                    send_buf_y[k * local_nx + i] = local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                                                           (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                                                           1 * (local_nx + 2) + (i + 1)];
                }
            }
            MPI_Sendrecv(send_buf_y, local_nx * local_nz, MPI_FLOAT, neighbors[2], 0,
                        recv_buf_y, local_nx * local_nz, MPI_FLOAT, neighbors[2], 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            
            // store in bottom ghost
            for (int k = 0; k < local_nz; k++) {
                for (int i = 0; i < local_nx; i++) {
                    local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                             (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                             0 * (local_nx + 2) + (i + 1)] = recv_buf_y[k * local_nx + i];
                }
            }
        }

        if (neighbors[3] != MPI_PROC_NULL) { // up
            
            // pack top face
            for (int k = 0; k < local_nz; k++) {
                for (int i = 0; i < local_nx; i++) {
                    send_buf_y[k * local_nx + i] = local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                                                           (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                                                           local_ny * (local_nx + 2) + (i + 1)];
                }                       
            }
            MPI_Sendrecv(send_buf_y, local_nx * local_nz, MPI_FLOAT, neighbors[3], 0,
                        recv_buf_y, local_nx * local_nz, MPI_FLOAT, neighbors[3], 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            
            // store in top ghost
            for (int k = 0; k < local_nz; k++) {
                for (int i = 0; i < local_nx; i++) {
                    local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                             (k + 1) * (local_nx + 2) * (local_ny + 2) + 
                             (local_ny + 1) * (local_nx + 2) + (i + 1)] = recv_buf_y[k * local_nx + i];
                }
            }
        }

        
        if (neighbors[4] != MPI_PROC_NULL) { // back (z-)
            
            // pack back face
            for (int j = 0; j < local_ny; j++) {
                for (int i = 0; i < local_nx; i++) {
                    send_buf_z[j * local_nx + i] = local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                                                           1 * (local_nx + 2) * (local_ny + 2) + 
                                                           (j + 1) * (local_nx + 2) + (i + 1)];
                }
            }
            MPI_Sendrecv(send_buf_z, local_nx * local_ny, MPI_FLOAT, neighbors[4], 0,
                        recv_buf_z, local_nx * local_ny, MPI_FLOAT, neighbors[4], 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            
            // store in back ghost
            for (int j = 0; j < local_ny; j++) {
                for (int i = 0; i < local_nx; i++) {
                    local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                             0 * (local_nx + 2) * (local_ny + 2) + 
                             (j + 1) * (local_nx + 2) + (i + 1)] = recv_buf_z[j * local_nx + i];
                }
            }
        }

        if (neighbors[5] != MPI_PROC_NULL) { // front
            
            // pack front face
            for (int j = 0; j < local_ny; j++) {
                for (int i = 0; i < local_nx; i++) {
                    send_buf_z[j * local_nx + i] = local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                                                           local_nz * (local_nx + 2) * (local_ny + 2) + 
                                                           (j + 1) * (local_nx + 2) + (i + 1)];
                }
            }
            MPI_Sendrecv(send_buf_z, local_nx * local_ny, MPI_FLOAT, neighbors[5], 0,
                        recv_buf_z, local_nx * local_ny, MPI_FLOAT, neighbors[5], 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            
            // store in front ghost
            for (int j = 0; j < local_ny; j++) {
                for (int i = 0; i < local_nx; i++) {
                    local_data[t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                             (local_nz + 1) * (local_nx + 2) * (local_ny + 2) + 
                             (j + 1) * (local_nx + 2) + (i + 1)] = recv_buf_z[j * local_nx + i];
                }
            }
        }
        
        // clean up ghost bufs
        free(send_buf_x);
        free(recv_buf_x);
        free(send_buf_y);
        free(recv_buf_y);
        free(send_buf_z);
        free(recv_buf_z);
    }

    
    // Look for local min/max points and count them
    int *local_min_count = (int *)malloc(nc * sizeof(int));
    int *local_max_count = (int *)malloc(nc * sizeof(int));
    float *local_min_val = (float *)malloc(nc * sizeof(float));
    float *local_max_val = (float *)malloc(nc * sizeof(float));

    for (int t = 0; t < nc; t++) {
        local_min_count[t] = 0;
        local_max_count[t] = 0;
        local_min_val[t] = FLT_MAX;  // init to extremes
        local_max_val[t] = -FLT_MAX;
        
        // check all real points (skip ghosts)
        for (int k = 1; k <= local_nz; k++) {
            for (int j = 1; j <= local_ny; j++) {
                for (int i = 1; i <= local_nx; i++) {
                    int idx = t * (local_nx + 2) * (local_ny + 2) * (local_nz + 2) + 
                            k * (local_nx + 2) * (local_ny + 2) + 
                            j * (local_nx + 2) + i;
                    
                    float center = local_data[idx];
                    
                    // track global min/max
                    if (center < local_min_val[t]) local_min_val[t] = center;
                    if (center > local_max_val[t]) local_max_val[t] = center;
                    
                    // Check if it's a local minimum with boundary consideration
                    bool is_min = true;

                    
                    // left - only if not global boundary
                    if (!(i == 1 && neighbors[0] == MPI_PROC_NULL))
                        is_min = is_min && (center < local_data[idx - 1]);

                    
                    // right check
                    if (!(i == local_nx && neighbors[1] == MPI_PROC_NULL))
                        is_min = is_min && (center < local_data[idx + 1]);

                    
                    // down check
                    if (!(j == 1 && neighbors[2] == MPI_PROC_NULL))
                        is_min = is_min && (center < local_data[idx - (local_nx + 2)]);

                    
                    // up
                    if (!(j == local_ny && neighbors[3] == MPI_PROC_NULL))
                        is_min = is_min && (center < local_data[idx + (local_nx + 2)]);

                    
                    // back
                    if (!(k == 1 && neighbors[4] == MPI_PROC_NULL))
                        is_min = is_min && (center < local_data[idx - (local_nx + 2) * (local_ny + 2)]);

                    
                    // front
                    if (!(k == local_nz && neighbors[5] == MPI_PROC_NULL))
                        is_min = is_min && (center < local_data[idx + (local_nx + 2) * (local_ny + 2)]);

                    if (is_min) {
                        local_min_count[t]++;
                    }
                    
                    // same for max
                    bool is_max = true;

                    
                    if (!(i == 1 && neighbors[0] == MPI_PROC_NULL))
                        is_max = is_max && (center > local_data[idx - 1]);

                    
                    if (!(i == local_nx && neighbors[1] == MPI_PROC_NULL))
                        is_max = is_max && (center > local_data[idx + 1]);

                    
                    if (!(j == 1 && neighbors[2] == MPI_PROC_NULL))
                        is_max = is_max && (center > local_data[idx - (local_nx + 2)]);

                    
                    if (!(j == local_ny && neighbors[3] == MPI_PROC_NULL))
                        is_max = is_max && (center > local_data[idx + (local_nx + 2)]);

                    
                    if (!(k == 1 && neighbors[4] == MPI_PROC_NULL))
                        is_max = is_max && (center > local_data[idx - (local_nx + 2) * (local_ny + 2)]);

                    
                    if (!(k == local_nz && neighbors[5] == MPI_PROC_NULL))
                        is_max = is_max && (center > local_data[idx + (local_nx + 2) * (local_ny + 2)]);

                    if (is_max) {
                        local_max_count[t]++;
                    }
                }
            }
        }
    }

    
    // alloc mem for results on rank 0
    int *global_min_count = NULL;
    int *global_max_count = NULL;
    float *global_min_val = NULL;
    float *global_max_val = NULL;
    
    if (rank == 0) {
        global_min_count = (int *)malloc(nc * sizeof(int));
        global_max_count = (int *)malloc(nc * sizeof(int));
        global_min_val = (float *)malloc(nc * sizeof(float));
        global_max_val = (float *)malloc(nc * sizeof(float));
    }

    // combine results
    MPI_Reduce(local_min_count, global_min_count, nc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_count, global_max_count, nc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_min_val, global_min_val, nc, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_val, global_max_val, nc, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    
    // calc final times
    t3 = MPI_Wtime();
    comp_time = t3 - t2;
    total_time = t3 - t1;

    
    // get max times (slowest proc)
    MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    // rank0 writes the results
    if (rank == 0) {
        FILE *fp = fopen(output_file, "w");
        if (!fp) {
            printf("Error opening output file %s\n", output_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        
        // write min/max counts
        for (int t = 0; t < nc; t++) {
            fprintf(fp, "(%d, %d)", global_min_count[t], global_max_count[t]);
            if (t < nc - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "\n");
        
        
        // write actual min/max values
        for (int t = 0; t < nc; t++) {
            fprintf(fp, "(%.4f, %.4f)", global_min_val[t], global_max_val[t]);
            if (t < nc - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "\n");
        
        
        // write timing stats
        fprintf(fp, "%.6f, %.6f, %.6f\n", max_read_time, max_comp_time, max_total_time);
        
        fclose(fp);
        
        printf("Results written to %s\n", output_file);
    }

    
    // free memory
    free(local_data);
    free(local_min_count);
    free(local_max_count);
    free(local_min_val);
    free(local_max_val);
    if (rank == 0) {
        free(global_min_count);
        free(global_max_count);
        free(global_min_val);
        free(global_max_val);
    }

    MPI_Finalize();
    return 0;
}