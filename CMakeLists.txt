cmake_minimum_required(VERSION 3.12)
project(SOPHIA_gpu LANGUAGES CXX CUDA)

# Set CUDA architecture and flags
set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -O3 -expt-relaxed-constexpr -w")

# Set output directory to the project root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Source directory (your .cu/.cuh files)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB_RECURSE SOURCES ${SOURCE_DIR}/*.cu)

# Executable target
add_executable(SOPHIA_gpu ${SOURCES})

# Add include directories (your code directory + external dependencies)
target_include_directories(SOPHIA_gpu PRIVATE
    ${SOURCE_DIR}
    /soft/cub-1.8.0
    /usr/local/cuda/samples/common/inc
)
