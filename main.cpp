#include "MCFSolver.h" // Includes mcf_common.h indirectly

#pragma pack(push, 1)
struct Record {
    int src; int dst; int k; float flow;
};
#pragma pack(pop)

// --- File I/O ---
bool readGraphFile(const std::string& filename, std::vector<Edge>& edges, int& num_nodes, bool& is_one_indexed);
bool readDemandFile(const std::string& filename, std::vector<Commodity>& commodities, bool is_one_indexed);
void convertToCSR(const std::vector<Edge>& edges, int num_nodes, MCFProblem& problem);


int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <demand_file> [output_file] [solver]" << std::endl;
        std::cerr << "\nSolvers: qp, gradient, momentum (default: momentum)" << std::endl;
        return 1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA-enabled GPU found." << std::endl;
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;

    std::vector<Edge> edges;
    std::vector<Commodity> commodities;
    int num_nodes;
    bool is_one_indexed = false;

    if (!readGraphFile(argv[1], edges, num_nodes, is_one_indexed)) return 1;
    if (!readDemandFile(argv[2], commodities, is_one_indexed)) return 1;

    MCFProblem problem;
    problem.K = commodities.size();
    convertToCSR(edges, num_nodes, problem);

    problem.h_origins.resize(problem.K);
    problem.h_destinations.resize(problem.K);
    problem.h_demands.resize(problem.K);
    for (int k = 0; k < problem.K; k++) {
        problem.h_origins[k] = commodities[k].src;
        problem.h_destinations[k] = commodities[k].dst;
        problem.h_demands[k] = commodities[k].demand;
    }

    std::string solver_name = "momentum";
    if (argc > 4) {
        solver_name = argv[4];
    }
    SolverStrategy strategy;
    if (solver_name == "qp") strategy = SolverStrategy::QP_PROJECTION;
    else if (solver_name == "gradient") strategy = SolverStrategy::GRADIENT_DESCENT;
    else if (solver_name == "momentum") strategy = SolverStrategy::MOMENTUM_GRADIENT;
    else {
        std::cerr << "Error: Unknown solver '" << solver_name << "'. Use 'qp', 'gradient', or 'momentum'." << std::endl;
        return 1;
    }

    MCFSolver solver(problem, strategy);

    std::vector<float> initial_flow((size_t)problem.m * problem.K, 0.0f);
    std::vector<float> final_flow((size_t)problem.m * problem.K);

    auto start_time = std::chrono::high_resolution_clock::now();
    solver.solve(initial_flow, final_flow, MAX_ITERS);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nTotal solver time: " << elapsed.count() << " seconds" << std::endl;

    if (argc > 3) {
        std::string out_filename = argv[3];
        std::ofstream outfile(out_filename, std::ios::binary);
        if (outfile.is_open()) {
            std::cout << "\nWriting non-zero flows to binary file: " << out_filename << std::endl;
            for (int e = 0; e < problem.m; e++) {
                for (int k = 0; k < problem.K; k++) {
                    float flow_val = final_flow[(size_t)e * problem.K + k];
                    if (flow_val > 1e-6) {
                        Record rec = {edges[e].src, edges[e].dst, k, flow_val};
                        outfile.write(reinterpret_cast<const char*>(&rec), sizeof(Record));
                    }
                }
            }
            outfile.close();
            std::cout << "Solution saved." << std::endl;
        } else {
            std::cerr << "Error: Could not open output file " << out_filename << std::endl;
        }
    }

    return 0;
}


// --- File I/O Implementations ---

bool readGraphFile(const std::string& filename, std::vector<Edge>& edges, int& num_nodes, bool& is_one_indexed) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open graph file: " << filename << std::endl;
        return false;
    }
    std::set<int> nodes;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Edge edge;
        if (iss >> edge.src >> edge.dst >> edge.capacity) {
            edges.push_back(edge);
            nodes.insert(edge.src);
            nodes.insert(edge.dst);
        }
    }
    if (nodes.empty()) { num_nodes = 0; return true; }
    
    bool has_zero = nodes.count(0) > 0;
    int min_node = *nodes.begin();
    int max_node = *nodes.rbegin();
    
    if (min_node == 1 && !has_zero) {
        is_one_indexed = true;
        std::cout << "Graph appears 1-indexed. Converting to 0-indexed." << std::endl;
        for (auto& edge : edges) { edge.src--; edge.dst--; }
        num_nodes = max_node;
    } else {
        is_one_indexed = false;
        num_nodes = max_node + 1;
    }
    std::cout << "Read " << edges.size() << " edges with " << num_nodes << " nodes." << std::endl;
    return true;
}

bool readDemandFile(const std::string& filename, std::vector<Commodity>& commodities, bool is_one_indexed) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open demand file: " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Commodity comm;
        if (iss >> comm.src >> comm.dst >> comm.demand) {
            if (is_one_indexed) { comm.src--; comm.dst--; }
            commodities.push_back(comm);
        }
    }
    std::cout << "Read " << commodities.size() << " commodities." << std::endl;
    return true;
}

void convertToCSR(const std::vector<Edge>& edges, int num_nodes, MCFProblem& problem) {
    problem.n = num_nodes;
    problem.m = edges.size();

    std::vector<std::vector<std::pair<int, int>>> adj_list(num_nodes);
    for (size_t i = 0; i < edges.size(); i++) {
        adj_list[edges[i].src].push_back({edges[i].dst, (int)i});
    }

    problem.h_graph_row_ptr.resize(num_nodes + 1);
    problem.h_graph_col_idx.resize(edges.size());
    problem.h_graph_edge_idx.resize(edges.size());
    problem.h_capacity.resize(edges.size());

    problem.h_graph_row_ptr[0] = 0;
    int csr_edge_counter = 0;
    for (int i = 0; i < num_nodes; i++) {
        std::sort(adj_list[i].begin(), adj_list[i].end());
        for (const auto& pair : adj_list[i]) {
            problem.h_graph_col_idx[csr_edge_counter] = pair.first;
            problem.h_graph_edge_idx[csr_edge_counter] = pair.second;
            problem.h_capacity[pair.second] = edges[pair.second].capacity;
            csr_edge_counter++;
        }
        problem.h_graph_row_ptr[i + 1] = csr_edge_counter;
    }
}
