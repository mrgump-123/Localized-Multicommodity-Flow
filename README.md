# CUDA-Accelerated Multi-Commodity Flow Solver

This repository contains the official implementation for the paper: **[A Localized Method for the Multi-commodity Flow Problem](https://arxiv.org/abs/2108.07549)**.

It provides a high-performance, GPU-accelerated solver for the Multi-Commodity Flow (MCF) problem, implemented in C++ and CUDA. It offers a flexible framework for comparing three different iterative optimization algorithms, all designed to leverage the massive parallelism of modern NVIDIA GPUs.

## Features

- **High Performance**: Utilizes CUDA, Thrust, and CUB libraries for efficient parallel computation on NVIDIA GPUs.
- **Multiple Solver Strategies**: Implements and allows for easy comparison between three distinct algorithms based on Price-Directive Relaxation (PDR):
    1.  **Quadratic Programming Projection (`qp`)**: An exact-step method that solves a QP subproblem exactly for each edge in every iteration (PDR via Exact Subproblem Optimization for MCF).
    2.  **Adaptive Gradient Descent (`gradient`)**: A first-order method with an adaptive per-edge learning rate to ensure stable convergence (PDR via Adaptive Gradient Descent).
    3.  **Momentum Gradient Descent (`momentum`)**: An enhanced gradient method that uses momentum to accelerate convergence (PDR via Gradient Descent with Momentum).
- **Simple and Portable Build System**: Uses a standard `Makefile` for easy compilation.

## Prerequisites

- **NVIDIA GPU**: A CUDA-enabled NVIDIA GPU is required.
- **CUDA Toolkit**: Version 11.0 or newer.
- **C++17 Compiler**: A C++17 compliant compiler (like `g++` 8 or newer).
- **Make**: A standard `make` utility for building the project.

## Tested Environment

The solver has been successfully built and tested on the following system configuration:

-   **Operating System**: `Debian GNU/Linux 11 (bullseye)`
-   **GPU**: `NVIDIA A100-SXM4 80GB`
-   **NVIDIA Driver Version**: `470.129.06`
-   **CUDA Version**: `11.7`

## Building the Solver

1.  **Clone the repository:**
    ```bash
    https://github.com/mrgump-123/Localized-Multicommodity-Flow.git
    cd your-repository-name
    ```

2.  **Adjust GPU Architecture (Optional but Recommended):**
    Open the `Makefile` and find the line `NVCCFLAGS = ... -arch=sm_80`. Change `sm_80` to the compute capability of your specific GPU. You can find your GPU's architecture [on the NVIDIA developer site](https://developer.nvidia.com/cuda-gpus).

3.  **Compile the project:**
    Simply run the `make` command.
    ```bash
    make
    ```
    This will create an executable named `mcf_solver` in the root directory.

4.  **Clean build files:**
    To remove the compiled object files and the executable, run:
    ```bash
    make clean
    ```

## Usage

The solver is run from the command line with the following syntax:

```
./mcf_solver <graph_file> <demand_file> [output_file] [solver_strategy]
```

### Arguments

-   `<graph_file>`: (Required) Path to the graph file.
-   `<demand_file>`: (Required) Path to the demand/commodities file.
-   `[output_file]`: (Optional) Path to save the non-zero flow results in a binary format. If not provided, results are not saved.
-   `[solver_strategy]`: (Optional) The algorithm to use. Choices are:
    -   `qp`: Quadratic Programming Projection
    -   `gradient`: Adaptive Gradient Descent
    -   `momentum`: Momentum Gradient Descent (**default**)

### Examples

-   **Run with the default momentum solver:**
    ```bash
    ./mcf_solver network_flow/Grid/Cgd1.txt network_flow/Grid/Dgd1.txt solution.bin
    ```

-   **Run with the Exact Subproblem (QP) solver and save the output:**
    ```bash
    ./mcf_solver network_flow/Grid/Cgd1.txt network_flow/Grid/Dgd1.txt solution.bin qp
    ```

-   **Run with the adaptive gradient solver and save the output:**
    ```bash
    ./mcf_solver network_flow/Grid/Cgd1.txt network_flow/Grid/Dgd1.txt solution.bin gradient
    ```

## File Formats

#### Graph File

A text file where each line represents a directed edge:
`source_node destination_node capacity`

-   Nodes can be 0-indexed or 1-indexed (the solver will auto-detect and convert to 0-indexed internally).
-   `capacity` is a floating-point number.

*Example (`graph.txt`):*
```
# src dst capacity
0 1 100.0
1 2 50.0
0 2 80.0
```

#### Demand File

A text file where each line represents a commodity:
`source_node destination_node demand_amount`

-   Nodes should use the same indexing as the graph file.
-   `demand_amount` is a floating-point number.

*Example (`demand.txt`):*
```
# src dst demand
0 2 70.0
```

## Dataset

The `network_flow` directory contains benchmark instances used for evaluating the solver. Files prefixed with `C` (e.g., `Cgd1.txt`) represent the graph structure and capacities, while files prefixed with `D` (e.g., `Dgd1.txt`) define the commodities and their demands.


### Data Source

These standard benchmark datasets were originally used in and compiled from the following papers:

-   Babonneau F, du Merle O, Vial J P. **Solving large-scale linear multicommodity flow problems with an active set strategy and proximal-ACCPM**[J]. Operations Research, 2006, 54(1): 184-197.

-   Larsson T, Yuan D. **An augmented lagrangian algorithm for large scale multicommodity routing**[J]. Computational Optimization and Applications, 2004, 27(2): 187-215.

## Project Structure

The codebase is organized into a clean, modular structure to separate concerns.

```
.
├── Makefile              # Build script for the project
├── README.md             # This documentation file
├── mcf_common.h          # Shared data structures, macros, and enums
├── kernels.cuh           # All CUDA __global__ kernel definitions
├── MCFSolver.h           # Header (declaration) for the main solver class
├── MCFSolver.cu          # Implementation of the MCFSolver class methods (CUDA code)
└── main.cpp              # Application entry point, file I/O, and CLI parsing (Host code)
```

## Acknowledgments

This project leverages several essential libraries from the CUDA ecosystem:
-   [**NVIDIA CUDA Toolkit**](https://developer.nvidia.com/cuda-toolkit)
-   [**Thrust Parallel Algorithms Library**](https://docs.nvidia.com/cuda/thrust/)
-   [**CUB Cooperative Primitives Library**](https://github.com/NVIDIA/cub)
-   [**cuBLAS**](https://docs.nvidia.com/cuda/cublas/index.html) and [**cuSPARSE**](https://docs.nvidia.com/cuda/cusparse/index.html) for GPU-accelerated linear algebra.