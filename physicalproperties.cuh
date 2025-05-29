#pragma once

#ifndef PHYSICALPROPERTIES_CUH
#define PHYSICALPROPERTIES_CUH

// NOTE: These are global variable declarations.
// They are defined in a single .cu file,
// and declared here using 'extern' to allow shared access across multiple source files.
// This prevents multiple definition errors during linking.

extern Real host_Tab_T[table_size];
extern Real host_Tab_h[table_size];
extern Real host_Tab_k[table_size];
extern Real host_Tab_cp[table_size];
extern Real host_Tab_vis[table_size];

extern int host_table_index[10];
extern int host_table_size[10];

extern __constant__ Real k_Tab_T[table_size];
extern __constant__ Real k_Tab_h[table_size];
extern __constant__ Real k_Tab_k[table_size];
extern __constant__ Real k_Tab_cp[table_size];
extern __constant__ Real k_Tab_vis[table_size];

extern __constant__ int k_table_index[10];
extern __constant__ int k_table_size[10];

void initializePropertyTables();

#endif
