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

//-------------------------------------------------------------------------------------------------
// ##. Global Variable Declarations
//-------------------------------------------------------------------------------------------------

// (Memory Allocation & Initialization)
// Host
void* malloc_and_zero(size_t size) {
    void* ptr = malloc(size);
    memset(ptr, 0, size);
    return ptr;
}
// Device
void cuda_malloc_and_zero(void** devPtr, size_t size) {
    cudaMalloc(devPtr, size);
    cudaMemset(*devPtr, 0, size);
}

// (solver options)
// Host
int_t *vii = (int_t*)malloc_and_zero(sizeof(int_t) * vii_size);
Real *vif = (Real*)malloc_and_zero(sizeof(Real) * vif_size);
// Device
__constant__ int_t k_vii[vii_size];
__constant__ Real k_vif[vif_size];

// (output options)
int num_plot_data;                // Number of plot variables
char plot_data[20][20];           // Names of variables to plot

// (table properties)
// Host
Real host_Tab_T[table_size];
Real host_Tab_h[table_size];
Real host_Tab_k[table_size];
Real host_Tab_cp[table_size];
Real host_Tab_vis[table_size];
int host_table_index[10];
int host_table_size[10];
// Device
__constant__ Real k_Tab_T[table_size];    // Temperature table
__constant__ Real k_Tab_h[table_size];    // Enthalpy table
__constant__ Real k_Tab_k[table_size];    // Conductivity table
__constant__ Real k_Tab_cp[table_size];   // Specific heat table
__constant__ Real k_Tab_vis[table_size];  // Viscosity table
__constant__ int k_table_index[10];       // Start index for each table
__constant__ int k_table_size[10];        // Number of entries for each table

// (OBC properties)
Real space;           // Virtual grid spacing for open boundary inlet
int Nsx = 0;          // Number of grid points along x for open boundary
int Nsz = 0;          // Number of grid points along z (or y) for open boundary
int buffer_size = 0;  // Additional memory for generated buffer particles

// (Particle Arrays)
part1* HP1;                       // All host particles
part1* DHP1[Max_GPU];             // Host-side arrays per GPU (for partitioning)

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
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    int tid = 0;
    // int tid = *(int*)arg;
    num_cells = clc_num_cells();
    count = floor(time/dt + 0.5);
    cudaSetDevice(tid);

    // --- Host 메모리 선언 및 할당 ---
    part1* file_P1 = (part1*)malloc_and_zero(sizeof(part1) * num_part3);
    part2* file_P2 = (part2*)malloc_and_zero(sizeof(part2) * num_part3);
    part3* file_P3 = (part3*)malloc_and_zero(sizeof(part3) * num_part3);

    Real *max_umag0 = (Real*)malloc_and_zero(sizeof(Real));
    Real *max_rho0  = (Real*)malloc_and_zero(sizeof(Real));
    Real *max_ftotal0 = (Real*)malloc_and_zero(sizeof(Real));
    Real *max_phi0  = (Real*)malloc_and_zero(sizeof(Real));
    Real *dt10 = (Real*)malloc_and_zero(sizeof(Real));
    Real *dt20 = (Real*)malloc_and_zero(sizeof(Real));
    Real *dt30 = (Real*)malloc_and_zero(sizeof(Real));
    Real *dt40 = (Real*)malloc_and_zero(sizeof(Real));
    Real *dt50 = (Real*)malloc_and_zero(sizeof(Real));

    // --- Device 메모리 선언 및 할당 ---
    int_t *g_idx, *p_idx, *g_idx_in, *p_idx_in, *g_str, *g_end;
    cuda_malloc_and_zero((void**)&g_idx,     sizeof(int_t) * num_part3);
    cuda_malloc_and_zero((void**)&p_idx,     sizeof(int_t) * num_part3);
    cuda_malloc_and_zero((void**)&g_idx_in,  sizeof(int_t) * num_part3);
    cuda_malloc_and_zero((void**)&p_idx_in,  sizeof(int_t) * num_part3);
    cuda_malloc_and_zero((void**)&g_str,     sizeof(int_t) * num_cells);
    cuda_malloc_and_zero((void**)&g_end,     sizeof(int_t) * num_cells);

    part1 *dev_P1, *dev_SP1;
    part2 *dev_P2, *dev_SP2;
    part3 *dev_SP3;
    cuda_malloc_and_zero((void**)&dev_P1,   sizeof(part1) * num_part3);
    cuda_malloc_and_zero((void**)&dev_SP1,  sizeof(part1) * num_part3);
    cuda_malloc_and_zero((void**)&dev_P2,   sizeof(part2) * num_part3);
    cuda_malloc_and_zero((void**)&dev_SP2,  sizeof(part2) * num_part3);
    cuda_malloc_and_zero((void**)&dev_SP3,  sizeof(part3) * num_part3);

    Real *max_rho, *max_umag, *max_ft, *max_phi;
    cuda_malloc_and_zero((void**)&max_rho,    sizeof(Real)*num_part3);
    cuda_malloc_and_zero((void**)&max_umag,   sizeof(Real)*num_part3);
    cuda_malloc_and_zero((void**)&max_ft,     sizeof(Real)*num_part3);
    cuda_malloc_and_zero((void**)&max_phi,    sizeof(Real)*num_part3);

    Real *d_max_umag0, *d_max_rho0, *d_max_ftotal0, *d_max_phi0;
    cuda_malloc_and_zero((void**)&d_max_umag0,   sizeof(Real));
    cuda_malloc_and_zero((void**)&d_max_rho0,    sizeof(Real));
    cuda_malloc_and_zero((void**)&d_max_ftotal0, sizeof(Real));
    cuda_malloc_and_zero((void**)&d_max_phi0,    sizeof(Real));

    Real *dt1, *dt2, *dt3, *dt4, *dt5;
    cuda_malloc_and_zero((void**)&dt1, sizeof(Real)*num_part3);
    cuda_malloc_and_zero((void**)&dt2, sizeof(Real)*num_part3);
    cuda_malloc_and_zero((void**)&dt3, sizeof(Real)*num_part3);
    cuda_malloc_and_zero((void**)&dt4, sizeof(Real)*num_part3);
    cuda_malloc_and_zero((void**)&dt5, sizeof(Real)*num_part3);

    Real *d_dt10, *d_dt20, *d_dt30, *d_dt40, *d_dt50;
    cuda_malloc_and_zero((void**)&d_dt10, sizeof(Real));
    cuda_malloc_and_zero((void**)&d_dt20, sizeof(Real));
    cuda_malloc_and_zero((void**)&d_dt30, sizeof(Real));
    cuda_malloc_and_zero((void**)&d_dt40, sizeof(Real));
    cuda_malloc_and_zero((void**)&d_dt50, sizeof(Real));

    void* dev_sort_storage = nullptr;
    void* dev_max_storage = nullptr;
    size_t sort_storage_bytes = 0;
    size_t max_storage_bytes = 0;

	cub::DeviceRadixSort::SortPairs(dev_sort_storage,sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part3);
	cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
	cudaMalloc((void**)&dev_sort_storage,sort_storage_bytes);
	cudaMalloc((void**)&dev_max_storage,max_storage_bytes);

    cudaMemcpyToSymbol(k_vii, vii, sizeof(int_t) * vii_size);
    cudaMemcpyToSymbol(k_vif, vif, sizeof(Real) * vif_size);
    cudaMemcpyToSymbol(k_Tab_T, host_Tab_T, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_h, host_Tab_h, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_k, host_Tab_k, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_cp, host_Tab_cp, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_vis, host_Tab_vis, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_table_index, host_table_index, sizeof(int) * 10);
    cudaMemcpyToSymbol(k_table_size, host_table_size, sizeof(int) * 10);

    DHP1[tid] = (part1*)malloc(num_part3 * sizeof(part1));
    memset(DHP1[tid], 0, sizeof(part1) * num_part3);
    for (int i = 0; i < num_part3; i++) DHP1[tid][i].i_type = 3;
    c_initial_inner_outer_particle_single(HP1, DHP1[tid], tid);
    cudaMemcpy(dev_P1, DHP1[tid], num_part3 * sizeof(part1), cudaMemcpyHostToDevice);

	if (tid == 0) {
		printf("-----------------------------------------------------------\n");
		printf("SOPHIA SPH Simulation: Input Summary\n");
		printf("-----------------------------------------------------------\n");
		printf("Total Particles     : %d\n", num_part);
		printf("Device Particles    : %d\n", num_part3);
		printf("Grid (NI,NJ,NK)     : %d, %d, %d\n", NI, NJ, NK);
		printf("Domain: X [%f, %f]\n", x_min, x_max);
		printf("Domain: Y [%f, %f]\n", y_min, y_max);
		printf("Domain: Z [%f, %f]\n", z_min, z_max);
		printf("Cell Size (dcell)   : %f\n", dcell);
		printf("Cells/GPU (X-dir)   : %d\n", calc_area);
		printf("-----------------------------------------------------------\n");
		printf("Simulation Start!\n");
		printf("-----------------------------------------------------------\n\n");
	}

	//-------------------------------------------------------------------------------------------------
	// ##. Main Loop
	//-------------------------------------------------------------------------------------------------

    int_t N = 2 * time_end / time_output;
    Real *Cdp    = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *Cdv    = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *Cd     = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *Clp    = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *Clv    = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *Cl     = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *P0     = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *P1     = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *P2     = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *P3     = (Real*)malloc_and_zero(sizeof(Real) * N);
    Real *time0  = (Real*)malloc_and_zero(sizeof(Real) * N);
    int_t *plotcount = (int_t*)malloc_and_zero(sizeof(int_t) * 1);

    clock_t start=clock();

    while(1){
    
        //-------------------------------------------------------------------------------------------------
        // Time-step Control
        //-------------------------------------------------------------------------------------------------
        if(tid==0){
                dim3 b,t;
                t.x=128;
                b.x=(num_part3-1)/t.x+1;
                int_t s=sizeof(int_t)*(t.x+1);

                kernel_copy_max_timestep<<<b,t>>>(dev_P1,dev_SP2,dev_SP3,dt1, dt2, dt3, dt4, dt5);
                cudaDeviceSynchronize();
                // Find Max Velocity & Force using CUB - TID=0
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_phi,d_max_phi0,num_part3);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_ft,d_max_ftotal0,num_part3);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
                cudaDeviceSynchronize();
                cudaMemcpy(max_phi0,d_max_phi0,sizeof(Real),cudaMemcpyDeviceToHost);
                cudaMemcpy(max_ftotal0,d_max_ftotal0,sizeof(Real),cudaMemcpyDeviceToHost);
                cudaMemcpy(max_umag0,d_max_umag0,sizeof(Real),cudaMemcpyDeviceToHost);

                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt1,d_dt10,num_part3);
                cudaDeviceSynchronize();
                cudaMemcpy(dt10,d_dt10,sizeof(Real),cudaMemcpyDeviceToHost);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt2,d_dt20,num_part3);
                cudaDeviceSynchronize();
                cudaMemcpy(dt20,d_dt20,sizeof(Real),cudaMemcpyDeviceToHost);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt3,d_dt30,num_part3);
                cudaDeviceSynchronize();
                cudaMemcpy(dt30,d_dt30,sizeof(Real),cudaMemcpyDeviceToHost);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt4,d_dt40,num_part3);
                cudaDeviceSynchronize();
                cudaMemcpy(dt40,d_dt40,sizeof(Real),cudaMemcpyDeviceToHost);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt5,d_dt50,num_part3);
                cudaDeviceSynchronize();
                cudaMemcpy(dt50,d_dt50,sizeof(Real),cudaMemcpyDeviceToHost);

                Real dt_delta = 0.1;
                Real dt_vel = 1.0/dt10[0];
                Real dt_ft = 0.1;
                Real dt_vis = 1.0/dt20[0];
                Real timestep[4] = {dt_delta,dt_vel,dt_ft,dt_vis};
                dt = min(timestep[0],timestep[1]);
                dt = min(dt,timestep[2]);
                dt = min(dt,timestep[3]);

                int_t integer=((time>=plotcount[0]*time_output)&&(time<(plotcount[0]+1)*time_output));

                if(integer) printf("dt_delta = %2.2e\ndt_vel = %2.2e\ndt_ft = %2.2e\ndt_vis = %2.2e\n",timestep[0],timestep[1],timestep[2],timestep[3]);
                if(integer) printf("dt = %e[s]\n",dt);

                // Set up ALE domain (update cell indices, alpha values)
                setALEdomain(b, t, dev_P1);

                // Perform neighbor search and particle sorting (NNPS)
                NNPS(
                    g_idx_in, g_idx, p_idx_in, p_idx, g_str, g_end,
                    dev_P1, dev_SP1, dev_P2, dev_SP2,
                    dev_sort_storage, &sort_storage_bytes,
                    b, t, s
                );

                // Enforce velocity boundary conditions (no-slip, penetration, etc.)
                velocityBC(b, t, g_str, g_end, dev_SP1);

                // Compute advection term (predictor step)
                advectionForce(b, t, g_str, g_end, dev_SP1, dev_SP2, dev_SP3);

                // Apply velocity projection
                projection(b, t, dev_SP1, dev_SP2, dev_SP3);

                // Prepare (gradient correction, variable updates) before pressure solve
                preparation(b, t, g_str, g_end, dev_SP1, dev_SP2, dev_SP3);

                // Set initial pressure boundary conditions (only at first step)
                if (count == 0)
                    pressureBC(b, t, g_str, g_end, dev_SP1, dev_SP2, dev_SP3);

                // Solve Pressure Poisson Equation (PPE)
                PPE(b, t, g_str, g_end, dev_SP1, dev_SP2, dev_SP3);

                // Enforce pressure boundary conditions (Neumann, Dirichlet, etc.)
                pressureBC(b, t, g_str, g_end, dev_SP1, dev_SP2, dev_SP3);

                //  Calculate pressure force and update acceleration
                pressureForce(b, t, g_str, g_end, dev_SP1, dev_SP2, dev_SP3);

                //  Update all time-dependent variables and particle positions
                timeUpdateProjection(b, t, dev_SP1, dev_P1, dev_SP2, dev_P2, dev_SP3);

                //  Apply particle shifting technique (PST) for improved regularity
                shifting(b, t, g_str, g_end, dev_P1, dev_SP2, dev_SP3);

                // Save particle data for visualization or further analysis
                saveOutput(
                    plotcount,
                    file_P1, file_P2, file_P3,
                    dev_P1, dev_SP2, dev_SP3,
                    time0, P0, P1, P2, P3
                );

                if(integer){
                    kernel_copy_max<<<b,t>>>(dev_P1,dev_SP2,dev_SP3,max_rho,max_ft,max_umag);
                    cudaDeviceSynchronize();
                
                // Find Max Velocity & Force using CUB - TID=0
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_rho,d_max_rho0,num_part3);
                cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_ft,d_max_ftotal0,num_part3);
                cudaDeviceSynchronize();
                cudaMemcpy(max_umag0,d_max_umag0,sizeof(Real),cudaMemcpyDeviceToHost);
                cudaMemcpy(max_rho0,d_max_rho0,sizeof(Real),cudaMemcpyDeviceToHost);
                cudaMemcpy(max_ftotal0,d_max_ftotal0,sizeof(Real),cudaMemcpyDeviceToHost);
                printf("dt = %e[s]\n",dt);
                printf("%d\t compress_max=%5.2f%\tu_max=%5.2f\tftotal_max=%5.2f\n\n",count,max_rho0[0],max_umag0[0],max_ftotal0[0]);
            }
            time+=dt;
            count++;
        }

        
        if(time>=time_end) break;

    }
	
    clock_t end=clock();
    clock_t elapsed = end-start;
    double calctime=(double)(elapsed/CLOCKS_PER_SEC);
    printf("total calculation time = %f\n",calctime);

	//-------------------------------------------------------------------------------------------------
	// ##. Save Restart File
	//-------------------------------------------------------------------------------------------------

    save_restart(file_P1,file_P2,file_P3);
    cudaMemcpy(file_P1,dev_SP1,num_part3*sizeof(part1),cudaMemcpyDeviceToHost);
    cudaMemcpy(file_P2,dev_SP2,num_part3*sizeof(part2),cudaMemcpyDeviceToHost);
    cudaMemcpy(file_P3,dev_SP3,num_part3*sizeof(part3),cudaMemcpyDeviceToHost);
    free(file_P2);
    free(file_P3);

	//-------------------------------------------------------------------------------------------------
	// ##. Memory Free
	//-------------------------------------------------------------------------------------------------
	free(file_P1);
    free(max_umag0); free(max_rho0); free(max_ftotal0); free(max_phi0);
    free(dt10); free(dt20); free(dt30); free(dt40); free(dt50);
    free(HP1);

    cudaFree(g_idx); cudaFree(p_idx); cudaFree(g_idx_in); cudaFree(p_idx_in); cudaFree(g_str); cudaFree(g_end);
    cudaFree(dev_P1); cudaFree(dev_SP1); cudaFree(dev_P2); cudaFree(dev_SP2); cudaFree(dev_SP3);
    cudaFree(max_rho); cudaFree(max_umag); cudaFree(max_ft); cudaFree(max_phi);
    cudaFree(d_max_umag0); cudaFree(d_max_rho0); cudaFree(d_max_ftotal0); cudaFree(d_max_phi0);
    cudaFree(dt1); cudaFree(dt2); cudaFree(dt3); cudaFree(dt4); cudaFree(dt5);
    cudaFree(d_dt10); cudaFree(d_dt20); cudaFree(d_dt30); cudaFree(d_dt40); cudaFree(d_dt50);
    cudaFree(dev_sort_storage); cudaFree(dev_max_storage);

    free(Cdp); free(Cdv); free(Cd);
    free(Clp); free(Clv); free(Cl);
    free(P0); free(P1); free(P2); free(P3); free(time0);
    free(plotcount);

    return 0;
}
