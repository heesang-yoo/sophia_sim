#include "SophiaSim.cuh"

int main(int argc, char** argv) {
    SophiaSim sim;
    sim.initialize(argc, argv);
    sim.loadInput();
    sim.setupDomain();
    sim.runSimulation();
    sim.cleanup();
    return 0;
}