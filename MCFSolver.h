#ifndef MCF_SOLVER_H
#define MCF_SOLVER_H

#include "mcf_common.h"

class MCFSolver {
public:
    MCFSolver(const MCFProblem& prob, SolverStrategy strat);
    ~MCFSolver();
    void solve(const std::vector<float>& initial_flow, std::vector<float>& final_flow, int max_iterations);

private:
    // --- Core solver loops for each strategy ---
    void solveQPProjection(int max_iterations);
    void solveGradientDescent(int max_iterations);
    void solveMomentumGradient(int max_iterations);

    // Memory management helpers
    void allocateCommonMemory();
    void allocateQPWorkspace();
    void allocateGradientWorkspace();
    void freeDeviceMemory();

    // Host-side setup helpers
    void copyProblemToDevice();
    void buildEdgeArrays();
    void buildReverseGraph();
    void setupCubSort();
    void computeNodeDegrees();

    const MCFProblem& problem;
    SolverStrategy strategy;

    // --- Device Memory Pointers ---
    GraphCSR d_graph{}, d_reverse_graph{};
    int *d_origins = nullptr, *d_destinations = nullptr, *d_edge_src = nullptr, *d_edge_dst = nullptr, *d_degrees = nullptr;
    float *d_capacity = nullptr, *d_demands = nullptr;
    float *d_flow = nullptr, *d_node_potential = nullptr, *d_congestion = nullptr;

    // QP Projection solver workspaces
    float *d_qp_c_pos = nullptr, *d_s0_t_buffer = nullptr;
    int   *d_s0_keys_out = nullptr, *d_segment_offsets = nullptr;
    void*  d_cub_temp_storage = nullptr;
    size_t cub_temp_storage_bytes = 0;

    // Gradient Descent solver workspaces
    //float *d_f_hat = nullptr
    float *d_delta_f = nullptr, *d_objective_components = nullptr, *d_learning_rate = nullptr;
    float *d_velocity = nullptr;

    // Library handles
    cusparseHandle_t cusparse_handle;
    cublasHandle_t cublas_handle;
};

#endif // MCF_SOLVER_H