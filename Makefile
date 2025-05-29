# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -w -use_fast_math -arch=sm_60 -O3 -expt-relaxed-constexpr

# Libraries
LIBS = -lpthread

# Include directories
INCLUDE = -I/home/yhs/peridynamics/cub-1.8.0/ -I/usr/local/cuda/samples/common/inc

# Source files
SRCS = SOPHIA_gpu.cu SophiaSim.cu physicalproperties.cu

# Object files (optional, not used here but useful for extension)
OBJS = $(SRCS:.cu=.o)

# Target executable
TARGET = SOPHIA_gpu

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(INCLUDE) $(LIBS)

# Clean intermediate object files
clean_intermediate:
	rm -f $(OBJS)

# Clean everything
clean: clean_intermediate
	rm -f $(TARGET)

.PHONY: all clean clean_intermediate
