# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -w -use_fast_math -arch=sm_60 -O3 -expt-relaxed-constexpr

# Include directories
INCLUDE = -I/home/yhs/peridynamics/cub-1.8.0/ -I/usr/local/cuda/samples/common/inc

# Libraries
LIBS = -lpthread

# Target
TARGET = SOPHIA_gpu

# Source files
SRCS = SOPHIA_gpu.cu
OBJS = $(SRCS:.cu=.o)

# Default rule
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDE)

clean:
	rm -f $(TARGET) *.o
