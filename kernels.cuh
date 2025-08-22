#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "mcf_common.h"
#include <cub/cub.cuh>

// --- Common Kernels (Used by multiple solvers) ---

__global__ void computeDegreesKernel(const int n, const int* graph_row_ptr, const int* reverse_graph_row_ptr, int* degrees) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int out_degree = graph_row_ptr[i+1] - graph_row_ptr[i];
        int in_degree = reverse_graph_row_ptr[i+1] - reverse_graph_row_ptr[i];
        degrees[i] = out_degree + in_degree;
    }
}

__global__ void computeNodePotentialsOptimized(
    const GraphCSR graph, const GraphCSR reverse_graph, const float *flow,
    const int K, const int *origins, const int *destinations, const float *demands,
    float *node_potential)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < graph.n * K; idx += total_threads) {
        int i = idx / K;
        int k = idx % K;

        float h_ik = 0.0f;
        for (int e_idx = reverse_graph.row_ptr[i]; e_idx < reverse_graph.row_ptr[i+1]; e_idx++) {
            h_ik += flow[(size_t)reverse_graph.edge_idx[e_idx] * K + k];
        }

        for (int e_idx = graph.row_ptr[i]; e_idx < graph.row_ptr[i+1]; e_idx++) {
            h_ik -= flow[(size_t)graph.edge_idx[e_idx] * K + k];
        }

        if (i == origins[k]) h_ik += demands[k];
        else if (i == destinations[k]) h_ik -= demands[k];

        int degree = (graph.row_ptr[i+1] - graph.row_ptr[i]) + (reverse_graph.row_ptr[i+1] - reverse_graph.row_ptr[i]);
        node_potential[idx] = (degree > 0) ? h_ik / degree : 0.0f;
    }
}

__global__ void computeCongestionCostOptimized(
    const int m, const int K, const float *flow, const float *capacity, float *congestion)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int e = blockIdx.x;
    if (e >= m) return;

    float thread_sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        thread_sum += flow[(size_t)e * K + k];
    }
    float total_flow = BlockReduce(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0) {
        congestion[e] = fmaxf(0.0f, total_flow - capacity[e]);
    }
}

__global__ void check_complementary_slackness_kernel(
    const int m, const int K, const float *flow, const float *node_potential, const float *congestion,
    int *d_violation_flag, const int *edge_src, const int *edge_dst)
{
    __shared__ int block_violation_flag;
    if (threadIdx.x == 0) block_violation_flag = 0;
    __syncthreads();

    if (*d_violation_flag == 1) return;

    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (size_t)m * K) {
        int e = tid / K;
        int k = tid % K;

        int i = edge_src[e];
        int j = edge_dst[e];

        float f_ijk = flow[tid];
        float h_ik = node_potential[(size_t)i * K + k];
        float h_jk = node_potential[(size_t)j * K + k];
        float psi_ij = congestion[e];
        float diff = h_ik - h_jk - psi_ij;
        
        bool is_violated = (f_ijk > 0 && fabsf(diff) > TOLERANCE) ||
                           (fabsf(f_ijk) < 1e-6f && diff > TOLERANCE);

        if (is_violated) atomicCAS(&block_violation_flag, 0, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0 && block_violation_flag == 1) atomicCAS(d_violation_flag, 0, 1);
}

// --- QP Projection Solver Kernels ---

__global__ void calculate_c_pos_kernel(
    const int m, const int K, const float* flow, const float* node_potential,
    float* g_c_pos, const int* edge_src, const int* edge_dst)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (size_t)m * K) {
        int e = tid / K;
        int k = tid % K;

        int i = edge_src[e];
        int j = edge_dst[e];

        float h_ik = node_potential[(size_t)i * K + k];
        float h_jk = node_potential[(size_t)j * K + k];
        float f_ijk_l = flow[tid];
        float c_k = 0.5f * ((h_ik + f_ijk_l) - (h_jk - f_ijk_l));
        
        g_c_pos[tid] = fmaxf(0.0f, c_k);
    }
}


__global__ void solveQPForActiveEdges_calc_t(
    const int m, const int K, const float* capacity,
    float* g_c_pos_sorted, float* t_values_out)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < m) {
        float u_ij = capacity[e];
        float s_0 = t_values_out[e];
        float t = 0.0f;
        

        if (s_0 > u_ij) {
            size_t offset_k = (size_t)e * K;
            float* c_pos_slice = &g_c_pos_sorted[offset_k];
            
            //To avoid precision loss when adding many small float values, we use double for all intermediate calculations.
            //Note that tiny round-off pushes a candidate t across the boundary and the loop never finds a valid interval.       
            double dsum = 0.0;
            
            for (int q = K - 1; q >= 0; q--) {
                dsum += static_cast<double>(c_pos_slice[q]);
                float sum = static_cast<float>(dsum);
                if (sum < u_ij) continue;
                
                float B_q = (float)(K - q) / 2.0f;
                      
                t = (sum - u_ij) / (B_q + 1.0f);        
                
                float L_q = (q > 0) ? (2.0f * c_pos_slice[q - 1]) : 0.0f;
                float R_q = 2.0f * c_pos_slice[q];

                if (L_q  <= t && t < R_q ){
                    break;
                } 
                if (c_pos_slice[q] < 1e-8) break;
            }
        }
        t_values_out[e] = t;
    }
}






__global__ void recalc_and_update_flow_kernel(
    const int m, const int K, const float* node_potential, const float* t_values,
    float* flow, const int* edge_src, const int* edge_dst)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (size_t)m * K) {
        int e = tid / K;
        int k = tid % K;

        float t = t_values[e];
        int i = edge_src[e];
        int j = edge_dst[e];
        
        float f_ijk_l = flow[tid];
        float h_ik = node_potential[(size_t)i * K + k];
        float h_jk = node_potential[(size_t)j * K + k];
        float c_k = 0.5f * ((h_ik + f_ijk_l) - (h_jk - f_ijk_l));

        float c_k_pos = fmaxf(0.0f, c_k);
        flow[tid] = fmaxf(0.0f, c_k_pos - 0.5f * t);
    }
}

// --- Gradient Descent Solver Kernels ---

__global__ void update_delta_f_kernel(
    const int m, const int K, const float* flow, const float* node_potential, const float* edge_congestion,
    float* d_delta_f, const int* edge_src, const int* edge_dst, const float* d_learning_rate)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (size_t)m * K) {
        int e = tid / K;
        int k = tid % K;

        int i = edge_src[e];
        int j = edge_dst[e];

        float h_ik = node_potential[(size_t)i * K + k];
        float h_jk = node_potential[(size_t)j * K + k];
        float f_ijk_l = flow[tid];
        float beta_e = d_learning_rate[e];

        float c_k = beta_e * (h_ik - h_jk - edge_congestion[e]);
        d_delta_f[tid] = fmaxf(-1.0f * f_ijk_l, c_k);
    }
}

__global__ void calculateObjectiveDiffKernel(
    const int m, const int K, const float* flow, const float* f_hat, const float* node_potential,
    const float* congestion, const float* capacity, const int* edge_src, const int* edge_dst,
    float* objective_per_edge)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_part1, temp_storage_flow;

    int e = blockIdx.x;
    if (e >= m) return;

    int i = edge_src[e];
    int j = edge_dst[e];

    float thread_part1_sum = 0.0f, thread_total_flow = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        size_t flow_idx = (size_t)e * K + k;
        float f_ijk = flow[flow_idx];
        float f_hat_ijk = f_hat[flow_idx];
        float h_ik = node_potential[(size_t)i * K + k];
        float h_jk = node_potential[(size_t)j * K + k];

        thread_part1_sum += ((h_jk - h_ik) * f_hat_ijk + f_hat_ijk * f_hat_ijk) * 2.0f;
        thread_total_flow += f_ijk + f_hat_ijk;
    }

    float total_part1_sum = BlockReduce(temp_storage_part1).Sum(thread_part1_sum);
    float total_flow_on_edge = BlockReduce(temp_storage_flow).Sum(thread_total_flow);

    if (threadIdx.x == 0) {
        float capacity_term = fmaxf(0.0f, total_flow_on_edge - capacity[e]);
        float part2_squared = capacity_term * capacity_term - congestion[e] * congestion[e];
        objective_per_edge[e] = total_part1_sum + part2_squared;
    }
}

__global__ void update_f_kernel(
    const int m, const int K, float* flow, const float* d_delta_f, const float* objective_components)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (size_t)m * K) {
        if (objective_components[tid / K] > 0.0f) return;
        flow[tid] += d_delta_f[tid];
    }
}

__global__ void updateLearningRateKernel(
    const int m, const float* objective_components, float* learning_rate, const float min_lr, const float max_lr)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < m) {
        //if (objective_components[e] > 0.0f) learning_rate[e] *= 0.5f;
        if (objective_components[e] > 0.0f) learning_rate[e] *= ALPHA_DEC;
        learning_rate[e] = fmaxf(min_lr, fminf(learning_rate[e], max_lr));
    }
}

// --- Momentum Gradient Solver Kernel ---

__global__ void update_flow_with_momentum_kernel(
    const size_t num_vars, float* flow, float* velocity, const float* d_delta_f,
    const float momentum_coeff, const float* objective_components, const int K)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vars) {
        int e = tid / K;
        if (objective_components[e] > 0.0f) {
            //velocity[tid] = 0.0f;
            return;
        }
        
        float new_velocity = momentum_coeff * velocity[tid] + d_delta_f[tid];
        flow[tid] += new_velocity;
        velocity[tid] = new_velocity;
        flow[tid] = fmaxf(0.0f, flow[tid]);
    }
}

// --- Thrust/CUB Functors ---
struct square {
    __host__ __device__ float operator()(const float& x) const { return x * x; }
};


// It calculates value[i] * value[i] * weight[i / K]
struct weighted_square_from_index {
    const float* d_values;
    const int*   d_weights;
    const int    K;

    // Constructor
    weighted_square_from_index(const float* values, const int* weights, int num_commodities)
        : d_values(values), d_weights(weights), K(num_commodities) {}

    // The operator() receives the index i and performs the calculation
    __device__ float operator()(size_t i) const {
        // Find the corresponding node index for the potential at index i
        size_t node_idx = i / K;
        
        // Get the potential and its corresponding degree
        float potential = d_values[i];
        float degree = static_cast<float>(d_weights[node_idx]);

        // Return the weighted square
        return potential * potential * degree;
    }
};

struct key_functor : public thrust::unary_function<int, int> {
    const int _K;
    __host__ __device__ key_functor(int K) : _K(K) {}
    __host__ __device__ int operator()(const int& i) const { return i / _K; }
};

#endif // KERNELS_CUH