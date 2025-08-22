#ifndef MCF_COMMON_H
#define MCF_COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>
#include <fstream>
#include <sstream>
#include <chrono>
#include <set>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

// --- Constants and Macros ---
#define BLOCK_SIZE 256
#define TOLERANCE 1e-6f
#define MAX_ITERS 50000
#define ALPHA_INC 2.0f
#define ALPHA_DEC 0.5f


// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << " - " << cusparseGetErrorString(status) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << cublasGetStatusString(status) << std::endl; \
        exit(1); \
    } \
} while(0)

// --- Data Structures & Enums ---

struct GraphCSR {
    int n; int m;
    int *row_ptr; int *col_idx; int *edge_idx;
};

struct MCFProblem {
    int n; int m; int K;
    std::vector<int> h_graph_row_ptr;
    std::vector<int> h_graph_col_idx;
    std::vector<int> h_graph_edge_idx;
    std::vector<float> h_capacity;
    std::vector<int> h_origins;
    std::vector<int> h_destinations;
    std::vector<float> h_demands;
};

struct Edge { int src; int dst; float capacity; };
struct Commodity { int src; int dst; float demand; };

enum class SolverStrategy {
    QP_PROJECTION,
    GRADIENT_DESCENT,
    MOMENTUM_GRADIENT
};

#endif // MCF_COMMON_H
