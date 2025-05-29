#include <stdio.h>
#include <string>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <ctime>
#include <cub/cub.cuh>

#include "SophiaSim.cuh"

// === Global Variable Definitions ===
// These variables are declared as 'extern' in SophiaSim.cuh.
// Here we provide the actual definitions and memory allocation.
// Only one definition is allowed across the entire program.

int_t vii[vii_size];
Real vif[vif_size];

__constant__ int_t k_vii[vii_size];
__constant__ Real k_vif[vif_size];

part1* HP1;
part1* DHP1[Max_GPU];

Real space;
int Nsx = 0;
int Nsz = 0;
int buffer_size = 0;

int num_plot_data;
char plot_data[20][20];

#include "physicalproperties.cuh"
#include "function_init.cuh"
#include "functions_NNPS.cuh"
#include "functions_PROP.cuh"
#include "functions_KNL.cuh"
#include "functions_PREP.cuh"
#include "functions_MASS.cuh"
#include "functions_INTERACTION.cuh"
#include "functions_TIME.cuh"
#include "functions_TIME_ISPH.cuh"
#include "functions_OUTPUT.cuh"
#include "functions_PST.cuh"
#include "functions_ALE.cuh"
#include "functions_BC.cuh"
#include "functions_PPE.cuh"
#include "ISPH_Calc.cuh"

void SophiaSim::initialize(int argc, char** argv) {
    memset(vii, 0, sizeof(int_t) * vii_size);
    memset(vif, 0, sizeof(Real) * vif_size);

    if (argc < 2) {
        printf("Usage: %s <number_of_GPUs>\n", argv[0]);
        exit(-1);
    }
    ngpu = atoi(argv[1]);

    read_solv_input(vii, vif, "./input/solv.txt");
    read_table("./input/data.txt");
}

void SophiaSim::loadInput() {
    char INPUT_FILE_NAME[128] = "./input/input.txt";
    num_part = gpu_count_particle_numbers2(INPUT_FILE_NAME);

    HP1 = (part1*)malloc(num_part * sizeof(part1));
    memset(HP1, 0, sizeof(part1) * num_part);
    read_input(HP1);
}

void SophiaSim::setupDomain() {
    find_minmax(vii, vif, HP1);
    search_kappa = kappa;
    Real h0 = h_max;
    Real cell_reduction_factor = 1.1;
    search_incr_factor = 1.0;

    dcell = (cell_reduction_factor * kappa * h0) / static_cast<Real>(ncell_init);
    if (dcell <= 0.0) {
        printf("\u274c dcell must be positive. Current: %.12f\n", dcell);
        exit(EXIT_FAILURE);
    }

    NI = static_cast<int>((x_max - x_min) / dcell + 1.0);
    NJ = static_cast<int>((y_max - y_min) / dcell + 1.0);
    NK = static_cast<int>((z_max - z_min) / dcell + 1.0);

    int tNx = (int)((x_max - x_min) / (h0 / h_coeff)) + 1;
    int tNz = (int)((z_max - z_min) / (h0 / h_coeff)) + 1;

    calc_area = ceil((float)NI / ngpu);

    if (ngpu > 1) {
        num_p2p = (int)(num_part * 4 / NI * C_p2p);
        num_part2 = (int)((num_part / ngpu) * 1.2) + 2 * num_p2p;
    } else {
        if (open_boundary > 0) {
            Real space = h0 / h_coeff * 0.5;
            Nsx = (int)((x_max - x_min) / space) + 1;
            Nsz = (int)((y_max - y_min) / space) + 1;
            buffer_size = (Nsx * Nsz);
        }

        buffer_size = 0;
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

        if (open_boundary > 0) {
            printf("Buffer memory size: %d\n", buffer_size);
            printf("Inlet particle generation frequency: %d steps\n",
                   (int)((h0 / h_coeff) / (Inlet_Velocity * dt)));
        }
    }
}

void SophiaSim::runSimulation() {
    int tid = 0;
    printf("[DEBUG] tid = %d\n", tid);
    ISPH_Calc((void*)&tid);
}

void SophiaSim::cleanup() {
    if (HP1) free(HP1);
}