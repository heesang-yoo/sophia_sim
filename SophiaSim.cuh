#pragma once
#ifndef SOPHIA_SIM_CUH
#define SOPHIA_SIM_CUH

#include "common_includes.cuh"

// NOTE: These are global variable declarations.
// They are defined in a single .cu file (e.g., SophiaSim.cu),
// and declared here using 'extern' to allow shared access across multiple source files.
// This prevents multiple definition errors during linking.

extern int_t vii[vii_size];
extern Real vif[vif_size];

extern part1* HP1;
extern part1* DHP1[Max_GPU];

extern Real space;
extern int Nsx;
extern int Nsz;
extern int buffer_size;

extern int num_plot_data;
extern char plot_data[20][20];


class SophiaSim {
public:
    int ngpu;
    part1* HP1 = nullptr;

    void initialize(int argc, char** argv);
    void loadInput();
    void setupDomain();
    void runSimulation();
    void cleanup();
};

#endif
