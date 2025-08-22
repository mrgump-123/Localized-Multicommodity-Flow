# Makefile for the Multi-Commodity Flow Solver

# Compiler and flags
NVCC = nvcc
EXEC = mcf_solver

# GPU architecture. Change sm_80 to your GPU's compute capability.
# (e.g., sm_61 for Pascal, sm_75 for Turing, sm_86 for Ampere, sm_89 for Ada)
NVCCFLAGS = -O3 -std=c++17 -arch=sm_80

# <<< FIX IS HERE: Define the linker flags to include the required CUDA libraries >>>
LDFLAGS = -lcusparse -lcublas

# Source and object files
# We compile main.cpp with nvcc as well since it includes CUDA headers
SOURCES_CU = main.cpp MCFSolver.cu
OBJECTS = $(SOURCES_CU:.cpp=.o)
OBJECTS := $(OBJECTS:.cu=.o)

# Default target
all: $(EXEC)

# Linking step
# <<< FIX IS HERE: Add the $(LDFLAGS) to the link command >>>
$(EXEC): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $^ $(LDFLAGS)
	@echo "Build complete. Executable is '$(EXEC)'"

# Compilation rule for .cpp files (using nvcc)
%.o: %.cpp MCFSolver.h mcf_common.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Compilation rule for .cu files
%.o: %.cu MCFSolver.h kernels.cuh mcf_common.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean up build files
clean:
	rm -f $(EXEC) $(OBJECTS)
	@echo "Cleaned up build files."

.PHONY: all clean