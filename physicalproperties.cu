// === Global Variable Definitions ===
// These variables are declared as 'extern' in SophiaSim.cuh.
// Here we provide the actual definitions and memory allocation.
// Only one definition is allowed across the entire program.

#include "physicalproperties.cuh"
#include <cuda_runtime.h>

Real host_Tab_T[table_size];
Real host_Tab_h[table_size];
Real host_Tab_k[table_size];
Real host_Tab_cp[table_size];
Real host_Tab_vis[table_size];

__constant__ Real k_Tab_T[table_size];
__constant__ Real k_Tab_h[table_size];
__constant__ Real k_Tab_k[table_size];
__constant__ Real k_Tab_cp[table_size];
__constant__ Real k_Tab_vis[table_size];

__constant__ int k_table_index[10];
__constant__ int k_table_size[10];

void initializePropertyTables()
{
    cudaMemcpyToSymbol(k_Tab_T, host_Tab_T, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_h, host_Tab_h, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_k, host_Tab_k, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_cp, host_Tab_cp, sizeof(Real) * table_size);
    cudaMemcpyToSymbol(k_Tab_vis, host_Tab_vis, sizeof(Real) * table_size);

    cudaMemcpyToSymbol(k_table_index, host_table_index, sizeof(int) * 10);
    cudaMemcpyToSymbol(k_table_size, host_table_size, sizeof(int) * 10);
}
