#include "MCFSolver.h"
#include "kernels.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

// --- Host Utility Functions ---
float compute_square_sum(float* d_array, size_t n) {
    if (n == 0) return 0.0f;
    thrust::device_ptr<float> dev_ptr(d_array);
    return thrust::transform_reduce(
        thrust::device, dev_ptr, dev_ptr + n, square(), 0.0f, thrust::plus<float>());
}


// NEW: Function to compute the weighted square sum using the functor
float compute_weighted_square_sum(float* d_array, int* d_weights, size_t n, int K) {
    if (n == 0) return 0.0f;

    // Use a counting_iterator to pass the index (0, 1, 2, ...) to our functor
    return thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator<size_t>(0),          // Input range is just indices 0 to n-1
        thrust::make_counting_iterator<size_t>(n),
        weighted_square_from_index(d_array, d_weights, K),  // The functor that does the work
        0.0f,                                               // Initial value for the sum
        thrust::plus<float>()                               // The reduction operation (summation)
    );
}


// --- Constructor / Destructor ---

MCFSolver::MCFSolver(const MCFProblem& prob, SolverStrategy strat) : problem(prob), strategy(strat) {
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
    CUBLAS_CHECK(cublasCreate(&cublas_handle)); 
    
    allocateCommonMemory();

    switch (strategy) {
        case SolverStrategy::QP_PROJECTION:
            allocateQPWorkspace();
            break;
        case SolverStrategy::GRADIENT_DESCENT:
            allocateGradientWorkspace();
            break;
        case SolverStrategy::MOMENTUM_GRADIENT:
            allocateGradientWorkspace();
            CUDA_CHECK(cudaMalloc(&d_velocity, (size_t)problem.m * problem.K * sizeof(float)));
            break;
    }

    copyProblemToDevice();
    buildReverseGraph();
    computeNodeDegrees();
}

MCFSolver::~MCFSolver() {
    freeDeviceMemory();
    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);
}

// --- Memory Management ---

void MCFSolver::allocateCommonMemory() {
    size_t m = problem.m, n = problem.n, K = problem.K;
    size_t num_vars = m * K;

    CUDA_CHECK(cudaMalloc(&d_graph.row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_graph.col_idx, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_graph.edge_idx, m * sizeof(int)));
    d_graph.n = n; d_graph.m = m;

    CUDA_CHECK(cudaMalloc(&d_capacity, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_origins, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_destinations, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_demands, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edge_src, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_dst, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_degrees, n * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_flow, num_vars * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_node_potential, n * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_congestion, m * sizeof(float)));
}

void MCFSolver::allocateQPWorkspace() {
    size_t m = problem.m, K = problem.K;
    CUDA_CHECK(cudaMalloc(&d_qp_c_pos, (size_t)m * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s0_t_buffer, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s0_keys_out, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_segment_offsets, (m + 1) * sizeof(int)));
    setupCubSort();
}

void MCFSolver::allocateGradientWorkspace() {
    size_t m = problem.m, K = problem.K;
    CUDA_CHECK(cudaMalloc(&d_delta_f, (size_t)m * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_objective_components, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_learning_rate, m * sizeof(float)));
}

void MCFSolver::freeDeviceMemory() {
    cudaFree(d_graph.row_ptr); cudaFree(d_graph.col_idx); cudaFree(d_graph.edge_idx); cudaFree(d_degrees);
    cudaFree(d_reverse_graph.row_ptr); cudaFree(d_reverse_graph.col_idx); cudaFree(d_reverse_graph.edge_idx);
    cudaFree(d_capacity); cudaFree(d_origins); cudaFree(d_destinations); cudaFree(d_demands);
    cudaFree(d_edge_src); cudaFree(d_edge_dst);
    cudaFree(d_flow); cudaFree(d_node_potential); cudaFree(d_congestion);
    cudaFree(d_qp_c_pos); cudaFree(d_s0_t_buffer); cudaFree(d_s0_keys_out);
    cudaFree(d_segment_offsets); cudaFree(d_cub_temp_storage);
    cudaFree(d_delta_f); cudaFree(d_objective_components); cudaFree(d_learning_rate);
    cudaFree(d_velocity);
}

// --- Setup Helpers ---

void MCFSolver::copyProblemToDevice() {
    CUDA_CHECK(cudaMemcpy(d_graph.row_ptr, problem.h_graph_row_ptr.data(), (problem.n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_graph.col_idx, problem.h_graph_col_idx.data(), problem.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_graph.edge_idx, problem.h_graph_edge_idx.data(), problem.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_capacity, problem.h_capacity.data(), problem.m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_origins, problem.h_origins.data(), problem.K * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_destinations, problem.h_destinations.data(), problem.K * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_demands, problem.h_demands.data(), problem.K * sizeof(float), cudaMemcpyHostToDevice));
    buildEdgeArrays();
}

void MCFSolver::buildEdgeArrays() {
    std::vector<int> edge_src(problem.m);
    std::vector<int> edge_dst(problem.m);
    for (int i = 0; i < problem.n; i++) {
        for (int idx = problem.h_graph_row_ptr[i]; idx < problem.h_graph_row_ptr[i+1]; idx++) {
            int e = problem.h_graph_edge_idx[idx];
            edge_src[e] = i;
            edge_dst[e] = problem.h_graph_col_idx[idx];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_edge_src, edge_src.data(), problem.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_dst, edge_dst.data(), problem.m * sizeof(int), cudaMemcpyHostToDevice));
}

void MCFSolver::buildReverseGraph() {
    CUDA_CHECK(cudaMalloc(&d_reverse_graph.row_ptr, (problem.n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reverse_graph.col_idx, problem.m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reverse_graph.edge_idx, problem.m * sizeof(int)));
    d_reverse_graph.n = problem.n; d_reverse_graph.m = problem.m;

    std::vector<int> h_edge_src(problem.m);
    std::vector<int> h_edge_dst(problem.m);
    CUDA_CHECK(cudaMemcpy(h_edge_src.data(), d_edge_src, problem.m * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_edge_dst.data(), d_edge_dst, problem.m * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<std::vector<std::pair<int, int>>> reverse_adj(problem.n);
    for (int e = 0; e < problem.m; e++) {
        reverse_adj[h_edge_dst[e]].push_back({h_edge_src[e], e});
    }

    std::vector<int> rev_row_ptr(problem.n + 1, 0);
    std::vector<int> rev_col_idx(problem.m);
    std::vector<int> rev_edge_idx(problem.m);
    int counter = 0;
    for (int i = 0; i < problem.n; i++) {
        rev_row_ptr[i] = counter;
        for (const auto& edge : reverse_adj[i]) {
            rev_col_idx[counter] = edge.first;
            rev_edge_idx[counter] = edge.second;
            counter++;
        }
    }
    rev_row_ptr[problem.n] = counter;

    CUDA_CHECK(cudaMemcpy(d_reverse_graph.row_ptr, rev_row_ptr.data(), (problem.n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reverse_graph.col_idx, rev_col_idx.data(), problem.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reverse_graph.edge_idx, rev_edge_idx.data(), problem.m * sizeof(int), cudaMemcpyHostToDevice));
}

void MCFSolver::computeNodeDegrees() {
    dim3 block(BLOCK_SIZE);
    dim3 grid((problem.n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computeDegreesKernel<<<grid, block>>>(problem.n, d_graph.row_ptr, d_reverse_graph.row_ptr, d_degrees);
    CUDA_CHECK(cudaGetLastError());
}


void MCFSolver::setupCubSort() {
    size_t max_num_items = (size_t)problem.m * problem.K;
    int max_num_segments = problem.m;
    cub::DeviceSegmentedSort::SortKeys(
        nullptr, cub_temp_storage_bytes, (float*)nullptr, (float*)nullptr,
        max_num_items, max_num_segments, d_segment_offsets, d_segment_offsets + 1);
    CUDA_CHECK(cudaMalloc(&d_cub_temp_storage, cub_temp_storage_bytes));
    std::cout << "CUB temp storage for QP solver allocated: " 
              << cub_temp_storage_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
}

// --- Main Solve Dispatcher ---

void MCFSolver::solve(const std::vector<float>& initial_flow, std::vector<float>& final_flow, int max_iterations) {
    CUDA_CHECK(cudaMemcpy(d_flow, initial_flow.data(),
                        (size_t)problem.m * problem.K * sizeof(float), cudaMemcpyHostToDevice));
    
    switch (strategy) {
        case SolverStrategy::QP_PROJECTION:
            std::cout << "Starting solver with QP Projection strategy..." << std::endl;
            solveQPProjection(max_iterations);
            break;
        case SolverStrategy::GRADIENT_DESCENT:
            std::cout << "Starting solver with Gradient Descent strategy..." << std::endl;
            solveGradientDescent(max_iterations);
            break;
        case SolverStrategy::MOMENTUM_GRADIENT:
            std::cout << "Starting solver with Momentum Gradient strategy..." << std::endl;
            solveMomentumGradient(max_iterations);
            break;
    }

    CUDA_CHECK(cudaMemcpy(final_flow.data(), d_flow,
                        (size_t)problem.m * problem.K * sizeof(float), cudaMemcpyDeviceToHost));
}

// --- Solver Implementations ---

void MCFSolver::solveQPProjection(int max_iterations) {
    int iteration = 0;
    bool converged = false;

    size_t num_vars = (size_t)problem.m * problem.K;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size_nodes(((size_t)problem.n * problem.K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_size_edges((problem.m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_edges_opt(problem.m);
    dim3 grid_flow_vars((num_vars + BLOCK_SIZE - 1) / BLOCK_SIZE);
    float total_constraint_violation = 0.0;
    float prev_total_constraint_violation = std::numeric_limits<float>::max();
    while (!converged && iteration < max_iterations) {
        computeNodePotentialsOptimized<<<grid_size_nodes, block_size>>>(d_graph, d_reverse_graph, d_flow, problem.K, d_origins, d_destinations, d_demands, d_node_potential);
        computeCongestionCostOptimized<<<grid_edges_opt, block_size>>>(problem.m, problem.K, d_flow, d_capacity, d_congestion);


        // Calculate the weighted residual for node potentials
        float node_potential_residual = compute_weighted_square_sum(d_node_potential, d_degrees, (size_t)problem.n * problem.K, problem.K);
        // Calculate the standard residual for congestion
        float congestion_residual = compute_square_sum(d_congestion, problem.m);
        // Combine them for the convergence check
        total_constraint_violation = node_potential_residual + congestion_residual;
        

        float delta_violation = fabsf(prev_total_constraint_violation - total_constraint_violation);
        prev_total_constraint_violation = total_constraint_violation;
        if (iteration > 0 && delta_violation < TOLERANCE)
        {   
            converged = true;
            std::cout << "Converged due to small change in violation." << std::endl;         
        }


        calculate_c_pos_kernel<<<grid_flow_vars, block_size>>>(problem.m, problem.K, d_flow, d_node_potential, d_qp_c_pos, d_edge_src, d_edge_dst);
        
        auto keys_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(0), key_functor(problem.K));
        thrust::reduce_by_key(thrust::device, keys_begin, keys_begin + num_vars, thrust::device_ptr<float>(d_qp_c_pos), thrust::device_ptr<int>(d_s0_keys_out), thrust::device_ptr<float>(d_s0_t_buffer));

        thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(problem.m + 1), thrust::device_ptr<int>(d_segment_offsets), thrust::placeholders::_1 * problem.K);

        cub::DeviceSegmentedSort::SortKeys(d_cub_temp_storage, cub_temp_storage_bytes, d_qp_c_pos, d_qp_c_pos, num_vars, problem.m, d_segment_offsets, d_segment_offsets + 1);

        solveQPForActiveEdges_calc_t<<<grid_size_edges, block_size>>>(problem.m, problem.K, d_capacity, d_qp_c_pos, d_s0_t_buffer);

        recalc_and_update_flow_kernel<<<grid_flow_vars, block_size>>>(problem.m, problem.K, d_node_potential, d_s0_t_buffer, d_flow, d_edge_src, d_edge_dst);
        
        if (iteration % 10 == 0) std::cout << "Iteration " << iteration << ", total constraint violation: " << total_constraint_violation << std::endl;
        iteration++;
    }

    if (converged) std::cout << "Converged in " << iteration << " iterations" << ", total constraint violation: " << total_constraint_violation << std::endl;
    else std::cout << "Stopped after reaching max " << max_iterations  << ", total constraint violation: " << total_constraint_violation << std::endl;
}

void MCFSolver::solveGradientDescent(int max_iterations) {
    int iteration = 0;
    bool converged = false;

    dim3 block(BLOCK_SIZE);
    size_t num_vars = (size_t)problem.m * problem.K;
    dim3 grid_nodes(((size_t)problem.n * problem.K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_edges_opt(problem.m);
    dim3 grid_flow_vars((num_vars + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_edges((problem.m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    const float max_beta = 0.5f;
    const float min_beta = 1.0f / (float(problem.K) + 1e-6);
    
    thrust::device_ptr<float> d_lr_ptr(d_learning_rate);
    thrust::fill(thrust::device, d_lr_ptr, d_lr_ptr + problem.m, 0.25f);

    float total_constraint_violation = 0.0;
    float prev_total_constraint_violation = std::numeric_limits<float>::max();

    
    while (!converged && iteration < max_iterations) {
        computeNodePotentialsOptimized<<<grid_nodes, block>>>(d_graph, d_reverse_graph, d_flow, problem.K, d_origins, d_destinations, d_demands, d_node_potential);
        computeCongestionCostOptimized<<<grid_edges_opt, block>>>(problem.m, problem.K, d_flow, d_capacity, d_congestion);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate the weighted residual for node potentials
        float node_potential_residual = compute_weighted_square_sum(d_node_potential, d_degrees, (size_t)problem.n * problem.K, problem.K);
        // Calculate the standard residual for congestion
        float congestion_residual = compute_square_sum(d_congestion, problem.m);
        // Combine them for the convergence check
        total_constraint_violation = node_potential_residual + congestion_residual;

        float delta_violation = fabsf(prev_total_constraint_violation - total_constraint_violation);
        prev_total_constraint_violation = total_constraint_violation;
        if (iteration > 0 && delta_violation < TOLERANCE)
        {   
            converged = true;
            std::cout << "Converged due to small change in violation." << std::endl;         
        }
                

        update_delta_f_kernel<<<grid_flow_vars, block>>>(problem.m, problem.K, d_flow, d_node_potential, d_congestion, d_delta_f, d_edge_src, d_edge_dst, d_learning_rate);
        calculateObjectiveDiffKernel<<<grid_edges_opt, block>>>(problem.m, problem.K, d_flow, d_delta_f, d_node_potential, d_congestion, d_capacity, d_edge_src, d_edge_dst, d_objective_components);

        update_f_kernel<<<grid_flow_vars, block>>>(problem.m, problem.K, d_flow, d_delta_f, d_objective_components);
        
        CUDA_CHECK(cudaDeviceSynchronize());

        if (iteration > 0 && iteration % 10 == 0) {
            const float scale_factor = ALPHA_INC;
            CUBLAS_CHECK(cublasSscal(cublas_handle, problem.m, &scale_factor, d_learning_rate, 1));
        }
        updateLearningRateKernel<<<grid_edges, block>>>(problem.m, d_objective_components, d_learning_rate, min_beta, max_beta);
        
        if (iteration % 10 == 0) std::cout << "Iteration " << iteration << ": total constraint violation: " << total_constraint_violation << std::endl;
        iteration++;
    }

    if (converged) std::cout << "Converged in " << iteration << " iterations" << ", total constraint violation: " << total_constraint_violation << std::endl;
    else std::cout << "Stopped after reaching max " << max_iterations  << ", total constraint violation: " << total_constraint_violation << std::endl;
}

void MCFSolver::solveMomentumGradient(int max_iterations) {
    int iteration = 0;
    bool converged = false;

    dim3 block(BLOCK_SIZE);
    size_t num_vars = (size_t)problem.m * problem.K;
    dim3 grid_nodes(((size_t)problem.n * problem.K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_edges_opt(problem.m);
    dim3 grid_flow_vars((num_vars + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_edges((problem.m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    const float max_beta = 0.2f;
    const float min_beta = 1.0f / (float(problem.K) + 1e-6);
    const float momentum_coeff = 0.9f;
    
    thrust::device_ptr<float> d_lr_ptr(d_learning_rate);
    thrust::fill(thrust::device, d_lr_ptr, d_lr_ptr + problem.m, 0.1f);
    CUDA_CHECK(cudaMemset(d_velocity, 0, num_vars * sizeof(float)));

    float total_constraint_violation = 0.0;
    float prev_total_constraint_violation = std::numeric_limits<float>::max();
    while (!converged && iteration < max_iterations) {
        computeNodePotentialsOptimized<<<grid_nodes, block>>>(d_graph, d_reverse_graph, d_flow, problem.K, d_origins, d_destinations, d_demands, d_node_potential);
        computeCongestionCostOptimized<<<grid_edges_opt, block>>>(problem.m, problem.K, d_flow, d_capacity, d_congestion);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate the weighted residual for node potentials
        float node_potential_residual = compute_weighted_square_sum(d_node_potential, d_degrees, (size_t)problem.n * problem.K, problem.K);
        // Calculate the standard residual for congestion
        float congestion_residual = compute_square_sum(d_congestion, problem.m);
        // Combine them for the convergence check
        total_constraint_violation = node_potential_residual + congestion_residual;
        
        float delta_violation = fabsf(prev_total_constraint_violation - total_constraint_violation);
        prev_total_constraint_violation = total_constraint_violation;
        if (iteration > 0 && delta_violation < TOLERANCE)
        {   
            converged = true;
            std::cout << "Converged due to small change in violation." << std::endl;         
        }


        update_delta_f_kernel<<<grid_flow_vars, block>>>(problem.m, problem.K, d_flow, d_node_potential, d_congestion, d_delta_f, d_edge_src, d_edge_dst, d_learning_rate);
        calculateObjectiveDiffKernel<<<grid_edges_opt, block>>>(problem.m, problem.K, d_flow, d_delta_f, d_node_potential, d_congestion, d_capacity, d_edge_src, d_edge_dst, d_objective_components);

        update_flow_with_momentum_kernel<<<grid_flow_vars, block>>>(num_vars, d_flow, d_velocity, d_delta_f, momentum_coeff, d_objective_components, problem.K);
        
        CUDA_CHECK(cudaDeviceSynchronize());

        if (iteration > 0 && iteration % 10 == 0) {
            const float scale_factor = ALPHA_INC;
            CUBLAS_CHECK(cublasSscal(cublas_handle, problem.m, &scale_factor, d_learning_rate, 1));
        }
        updateLearningRateKernel<<<grid_edges, block>>>(problem.m, d_objective_components, d_learning_rate, min_beta, max_beta);
        
        if (iteration % 10 == 0) std::cout << "Iteration " << iteration << ": total constraint violation = " << total_constraint_violation << std::endl;
        iteration++;
    }

    if (converged) std::cout << "Converged in " << iteration << " iterations" << ", total constraint violation: " << total_constraint_violation << std::endl;
    else std::cout << "Stopped after reaching max " << max_iterations  << ", total constraint violation: " << total_constraint_violation << std::endl;
}