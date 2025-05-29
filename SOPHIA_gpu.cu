#include <stdio.h>
#include <string>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <cub/cub.cuh>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Variable_Type.cuh"
#include "Parameters.cuh"
#include "class_Cuda_Particle_Array.cuh"

//------------------------------------------------------------------
// Global Variable Declarations
//------------------------------------------------------------------

// Solver parameters (see function_init.cuh)
// Host
int_t vii[vii_size];
Real vif[vif_size];

// Device (solver options)
__constant__ int_t k_vii[vii_size];
__constant__ Real k_vif[vif_size];

// Device (property tables)
__constant__ Real k_Tab_T[table_size];    // Temperature table
__constant__ Real k_Tab_h[table_size];    // Enthalpy table
__constant__ Real k_Tab_k[table_size];    // Conductivity table
__constant__ Real k_Tab_cp[table_size];   // Specific heat table
__constant__ Real k_Tab_vis[table_size];  // Viscosity table
__constant__ int k_table_index[10];       // Start index for each table
__constant__ int k_table_size[10];        // Number of entries for each table

// Particle arrays
part1* HP1;                       // All host particles
part1* DHP1[Max_GPU];             // Host-side arrays per GPU (for partitioning)

// Host-side tables
Real host_Tab_T[table_size];
Real host_Tab_h[table_size];
Real host_Tab_k[table_size];
Real host_Tab_cp[table_size];
Real host_Tab_vis[table_size];

int host_table_index[10];
int host_table_size[10];

// Open boundary (buffer zone for inflow/outflow)
Real space;           // Virtual grid spacing for open boundary inlet
int Nsx = 0;          // Number of grid points along x for open boundary
int Nsz = 0;          // Number of grid points along z (or y) for open boundary
int buffer_size = 0;  // Additional memory for generated buffer particles

__device__ int num_buffer[1];     // Total number of buffer particles (needed for Open Boundary/APS models)

// Plot Data
int num_plot_data;                // Number of plot variables
char plot_data[20][20];           // Names of variables to plot

// Function header includes (simulation core logic)
#include "function_init.cuh"
#include "functions_NNPS.cuh"
#include "functions_PROP.cuh"
#include "functions_KNL.cuh"
#include "functions_PREP.cuh"
#include "functions_TIME_ISPH.cuh"
#include "functions_ALE.cuh"
#include "functions_BC.cuh"
#include "functions_PPE.cuh"
#include "functions_OUTPUT.cuh"
#include "functions_PST.cuh"
#include "ISPH_Calc.cuh"

int main(int argc, char **argv)
{
    // Initialize solver parameters
    memset(vii, 0, sizeof(int_t) * vii_size);
    memset(vif, 0, sizeof(Real) * vif_size);

    ngpu = atoi(argv[1]); // Number of GPUs (from command-line)

    char fn[64], fn2[64];
    strcpy(fn, "./input/solv.txt");
    strcpy(fn2, "./input/data.txt");

    read_solv_input(vii, vif, fn);    // Read solver option input file
    read_table(fn2);                  // Read property table

    // Print GPU device properties
    {
        int_t gcount, i;
        struct cudaDeviceProp prop;
        cudaGetDeviceCount(&gcount);

        for (i = 0; i < gcount; i++) {
            cudaGetDeviceProperties(&prop, i);
            printf("### GPU DEVICE PROPERTIES.................................\n\n");
            printf("    Name: %s\n", prop.name);
            printf("    Compute capability: %d.%d\n", prop.major, prop.minor);
            printf("    Clock rate: %d\n", prop.clockRate);
            printf("    Total global memory: %ld\n", prop.totalGlobalMem);
            printf("    Total constant memory: %d\n", prop.totalConstMem);
            printf("    Multiprocessor count: %d\n", prop.multiProcessorCount);
            printf("    Shared mem per block: %d\n", prop.sharedMemPerBlock);
            printf("    Registers per block: %d\n", prop.regsPerBlock);
            printf("    Threads in warp: %d\n", prop.warpSize);
            printf("    Max threads per block: %d\n", prop.maxThreadsPerBlock);
            printf("    Max thread dimensions: %d,%d,%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("    Max grid dimensions: %d,%d,%d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("...........................................................\n\n");
        }
    }

    printf(" ------------------------------------------------------------\n");
    printf(" SOPHIA_gpu v.1.0 \n");
    printf(" Developed by E.S. Kim, Y.B. Jo, S.H. Park\n");
    printf(" 2017. 02. 20 \n");
    printf(" Optimized by Y.W. Sim, CoCoLink Inc.\n");
    printf(" 2018, 2019(C) \n");
    printf(" Restructured & Innovated by Eung Soo Kim, Hee Sang Yoo, Young Beom Jo, Hae Yoon Choi, Su-San Park, Jin Woo Kim, Yelyn Ahn, Tae Soo Choi\n");
    printf(" ESLAB, SEOUL NATIONAL UNIVERSITY, SOUTH KOREA.\n");
    printf(" 2019. 08. 08 \n");
    printf("------------------------------------------------------------\n\n");

    // Count the number of particles in the domain
    char INPUT_FILE_NAME[128];
    strcpy(INPUT_FILE_NAME, "./input/input.txt");
    num_part = gpu_count_particle_numbers2(INPUT_FILE_NAME);

    // Allocate and initialize host memory for particles
    HP1 = (part1 *)malloc(num_part * sizeof(part1));
    memset(HP1, 0, sizeof(part1) * num_part);

    // Read initial particle data from file
    read_input(HP1);

    // Domain and cell decomposition (spatial partitioning for neighbor search)
    Real cell_reduction_factor = 1.1;
    search_incr_factor = 1.0;
    search_kappa = kappa;
    find_minmax(vii, vif, HP1);

    Real h0 = h_max;
    dcell = cell_reduction_factor * kappa * h0 / ncell_init;

    NI = (int)((x_max - x_min) / dcell) + 1;
    NJ = (int)((y_max - y_min) / dcell) + 1;
    NK = (int)((z_max - z_min) / dcell) + 1;

    int tNx = (int)((x_max - x_min) / (h0 / h_coeff)) + 1;
    int tNz = (int)((z_max - z_min) / (h0 / h_coeff)) + 1;

    // Open boundary setup (for inflow/outflow simulation)
    if (open_boundary > 0) {
        space = h0 / h_coeff * 0.5;
        Nsx = (int)((x_max - x_min) / space) + 1;
        Nsz = (int)((z_max - z_min) / space) + 1;
        buffer_size = Nsx * Nsz;

        printf("\n\n----------------------------------------------------\n");
        printf("Open Boundary Space = %f \n", space);
        printf("Buffer memory size: %d    ratio: %.2f \n\n", buffer_size, (Real)buffer_size / (tNx * tNz));
        printf("Frequency of inlet particle generation: %d steps\n", (int)((h0 / h_coeff) / (Inlet_Velocity * dt)));
        printf("----------------------------------------------------\n");
    } else {
        buffer_size = 0;
    }

    // Particle counts with/without buffer zone and APS extension
    num_part2 = num_part + buffer_size;
    num_part3 = num_part2;

    if (aps_solv) {
        if (dim == 2) {
            num_part2 = num_part + num_part * 4;
            num_part3 = num_part2 + buffer_size;
        }
        if (dim == 3)
            num_part3 = 4 * num_part + buffer_size;
    }

    // Start the main solver (single GPU)
    int tid = 0;
    ISPH_Calc((void *)&tid);

    // Free allocated host memory
    free(HP1);
    return 0;
}
