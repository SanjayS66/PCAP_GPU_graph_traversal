/**
 * @file uniform_benchmark.cu
 * @brief Unified Comparative Testing and Benchmarking Harness for Implemented Algorithms
 * 
 * Benchmarks and validates all currently implemented CPU and GPU graph traversal strategies:
 * 
 * BFS:
 *   1. CPU Sequential (Queue-based)
 *   2. GPU Strategy 1: Thread-per-frontier-vertex
 *   [Comments & hooks left for: OpenMP CPU BFS, GPU Warp-per-vertex, GPU Edge-based]
 * 
 * SSSP (Bellman-Ford):
 *   1. CPU Sequential Baseline
 *   2. CPU Multi-threaded OpenMP Baseline (Dynamic Scheduling + AtomicCAS)
 *   3. GPU Strategy 1: Thread-per-vertex
 *   4. GPU Strategy 2: Warp-per-vertex (32 threads per vertex)
 *   5. GPU Strategy 3: Thread-per-edge (fine-grained edge parallelism)
 *   [Comments & hooks left for: Delta-stepping style / Active frontier queue, Multi-GPU MPI]
 * 
 * Calculates:
 *   - Transfer vs Compute vs D2H breakdown (via CUDA Events)
 *   - Speedup vs CPU Sequential
 *   - Speedup vs CPU OpenMP
 *   - TEPS (Traversed Edges Per Second / Mega-TEPS)
 *   - Automated verification against CPU baseline (distance / level checking)
 *   - Clean formatted summary table and CSV export option
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <cuda_runtime.h>

#include "csr.h"
#include "bfs.h"
#include "sssp.h"
#include "timer.h"

// ANSI Color codes for formatted terminal output
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN    "\033[36m"

enum RunStatus {
    STATUS_SUCCESS = 0,
    STATUS_ERROR   = 1
};

struct BenchmarkResult {
    std::string algorithm;      // "BFS" or "SSSP"
    std::string strategy;       // e.g. "CPU Sequential", "GPU Thread-per-vertex", etc.
    RunStatus status;
    float h2d_ms;
    float compute_ms;
    float d2h_ms;
    float total_ms;
    int rounds;
    double speedup_compute;     // vs CPU Serial
    double speedup_total;       // vs CPU Serial
    double speedup_vs_omp;      // vs CPU OpenMP
    double mteps;               // Mega-TEPS (Traversed Edges Per Second / 1e6)
    int mismatches;             // -1 if baseline, 0 = PASS, >0 = FAIL
};

// Count reachable vertices and traversed edges from BFS / SSSP distance output
static void compute_graph_traversal_stats(const CSRGraph *g, const int *bfs_dist, const float *sssp_dist,
                                          int *out_reached_v, long long *out_traversed_e) {
    int reached_v = 0;
    long long traversed_e = 0;

    for (int u = 0; u < g->V; u++) {
        bool visited = false;
        if (bfs_dist && bfs_dist[u] >= 0) visited = true;
        if (sssp_dist && sssp_dist[u] < SSSP_INF) visited = true;

        if (visited) {
            reached_v++;
            int deg = g->row_offset[u + 1] - g->row_offset[u];
            traversed_e += deg;
        }
    }
    *out_reached_v = reached_v;
    *out_traversed_e = traversed_e;
}

// Compare BFS integer distance arrays
static int verify_bfs(const int *baseline, const int *test, int V) {
    int mismatches = 0;
    for (int i = 0; i < V; i++) {
        if (baseline[i] != test[i]) {
            if (mismatches < 5) {
                std::cerr << "  " << ANSI_RED << "[BFS Mismatch]" << ANSI_RESET
                          << " vertex " << i << ": baseline=" << baseline[i]
                          << " vs test=" << test[i] << "\n";
            }
            mismatches++;
        }
    }
    return mismatches;
}

// Compare SSSP float distance arrays
static int verify_sssp(const float *baseline, const float *test, int V, float tol = 1e-3f) {
    int mismatches = 0;
    for (int i = 0; i < V; i++) {
        float b = baseline[i];
        float t = test[i];
        if (b >= SSSP_INF / 2.0f && t >= SSSP_INF / 2.0f) {
            continue; // Both unreachable
        }
        float diff = std::fabs(b - t);
        if (diff > tol) {
            if (mismatches < 5) {
                std::cerr << "  " << ANSI_RED << "[SSSP Mismatch]" << ANSI_RESET
                          << " vertex " << i << ": baseline=" << b
                          << " vs test=" << t << " (diff=" << diff << ")\n";
            }
            mismatches++;
        }
    }
    return mismatches;
}

// Print benchmark results in a clean aligned table
static void print_results_table(const std::vector<BenchmarkResult> &results, int num_threads) {
    std::cout << "\n" << ANSI_BOLD << ANSI_CYAN
              << "=============================================================================================================\n"
              << "                                   COMPARATIVE BENCHMARK EXECUTION SUMMARY                                   \n"
              << "============================================================================================================="
              << ANSI_RESET << "\n";

    std::cout << ANSI_BOLD
              << std::left
              << std::setw(6)  << "ALGO"
              << std::setw(26) << "STRATEGY"
              << std::setw(13) << "STATUS"
              << std::setw(12) << "COMPUTE(ms)"
              << std::setw(12) << "TOTAL(ms)"
              << std::setw(11) << "SPEEDUP"
              << std::setw(11) << "MTEPS"
              << std::setw(8)  << "ROUNDS"
              << std::setw(14) << "VERIFICATION"
              << ANSI_RESET << "\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------\n";

    for (const auto &r : results) {
        std::cout << std::left
                  << std::setw(6)  << r.algorithm
                  << std::setw(26) << r.strategy;

        if (r.status == STATUS_SUCCESS) {
            std::cout << std::setw(13) << (std::string(ANSI_GREEN) + "COMPLETE" + ANSI_RESET);
            std::cout << std::right
                      << std::fixed << std::setprecision(4)
                      << std::setw(12) << r.compute_ms
                      << std::setw(12) << r.total_ms
                      << std::setprecision(2)
                      << std::setw(10) << r.speedup_compute << "x"
                      << std::setprecision(2)
                      << std::setw(11) << r.mteps
                      << std::setw(8)  << r.rounds << " ";

            if (r.mismatches == -1) {
                std::cout << std::left << std::setw(14) << "BASELINE";
            } else if (r.mismatches == 0) {
                std::cout << std::left << std::setw(14) << (std::string(ANSI_GREEN) + "PASS" + ANSI_RESET);
            } else {
                std::cout << std::left << std::setw(14) << (std::string(ANSI_RED) + "FAIL (" + std::to_string(r.mismatches) + ")" + ANSI_RESET);
            }
        } else {
            std::cout << std::setw(13) << (std::string(ANSI_RED) + "ERROR" + ANSI_RESET);
            std::cout << std::right
                      << std::setw(12) << "-"
                      << std::setw(12) << "-"
                      << std::setw(11) << "-"
                      << std::setw(11) << "-"
                      << std::setw(8)  << "-" << " "
                      << std::left << std::setw(14) << (std::string(ANSI_RED) + "FAILED" + ANSI_RESET);
        }
        std::cout << "\n";
    }
    std::cout << "=============================================================================================================\n";
    std::cout << "Notes: OpenMP threads = " << num_threads
              << " | Speedup = T(serial_compute) / T(target_compute) | MTEPS = 10^6 Traversed Edges / sec\n\n";
}

// Export results to CSV for plotting and analysis
static void export_csv(const std::string &path, const std::string &graph_file, int V, int E,
                       const std::vector<BenchmarkResult> &results, int num_threads) {
    bool file_exists = false;
    {
        std::ifstream f(path.c_str());
        file_exists = f.good();
    }

    std::ofstream out(path.c_str(), std::ios::app);
    if (!out.is_open()) {
        std::cerr << ANSI_RED << "Error: could not open CSV file: " << path << ANSI_RESET << "\n";
        return;
    }

    if (!file_exists) {
        out << "graph_file,V,E,threads,algorithm,strategy,status,h2d_ms,compute_ms,d2h_ms,total_ms,rounds,speedup,speedup_vs_omp,mteps,mismatches\n";
    }

    for (const auto &r : results) {
        std::string status_str = (r.status == STATUS_SUCCESS) ? "SUCCESS" : "ERROR";
        out << graph_file << ","
            << V << ","
            << E << ","
            << num_threads << ","
            << r.algorithm << ","
            << "\"" << r.strategy << "\","
            << status_str << ","
            << r.h2d_ms << ","
            << r.compute_ms << ","
            << r.d2h_ms << ","
            << r.total_ms << ","
            << r.rounds << ","
            << r.speedup_compute << ","
            << r.speedup_vs_omp << ","
            << r.mteps << ","
            << r.mismatches << "\n";
    }
    out.close();
    std::cout << ANSI_GREEN << "✓ Exported benchmark metrics to CSV: " << path << ANSI_RESET << "\n\n";
}

// Print load balancing comparison across implemented GPU strategies
static void print_load_balancing_comparison(const std::vector<BenchmarkResult> &results, const std::string &algo) {
    const BenchmarkResult *t_v = nullptr;
    const BenchmarkResult *warp_v = nullptr;
    const BenchmarkResult *edge = nullptr;

    for (const auto &r : results) {
        if (r.algorithm != algo) continue;
        if (r.strategy.find("Thread-per-vertex") != std::string::npos) t_v = &r;
        if (r.strategy.find("Warp-per-vertex") != std::string::npos) warp_v = &r;
        if (r.strategy.find("Thread-per-edge") != std::string::npos ||
            r.strategy.find("Edge-based") != std::string::npos) edge = &r;
    }

    std::cout << ANSI_BOLD << "--- " << algo << " GPU Load-Balancing Strategy Analysis ---" << ANSI_RESET << "\n";

    if (t_v && t_v->status == STATUS_SUCCESS && edge && edge->status == STATUS_SUCCESS) {
        float ratio = t_v->compute_ms / edge->compute_ms;
        if (ratio > 1.05f) {
            std::cout << "  • " << ANSI_GREEN << "Thread-per-edge is faster than Thread-per-vertex by "
                      << std::fixed << std::setprecision(2) << ratio << "x" << ANSI_RESET
                      << " (effectively mitigates degree-skew load imbalance).\n";
        } else if (ratio < 0.95f) {
            std::cout << "  • " << ANSI_CYAN << "Thread-per-vertex is faster than Thread-per-edge by "
                      << std::fixed << std::setprecision(2) << (1.0f / ratio) << "x" << ANSI_RESET
                      << " (low-degree/uniform structure avoids edge-mapping auxiliary overhead).\n";
        } else {
            std::cout << "  • Thread-per-vertex and Thread-per-edge show comparable performance.\n";
        }
    }

    if (t_v && t_v->status == STATUS_SUCCESS && warp_v && warp_v->status == STATUS_SUCCESS) {
        float ratio = t_v->compute_ms / warp_v->compute_ms;
        if (ratio > 1.05f) {
            std::cout << "  • " << ANSI_GREEN << "Warp-per-vertex is faster than Thread-per-vertex by "
                      << std::fixed << std::setprecision(2) << ratio << "x" << ANSI_RESET
                      << " (cooperative 32-thread warps eliminate tail latency on hubs).\n";
        } else if (ratio < 0.95f) {
            std::cout << "  • " << ANSI_CYAN << "Thread-per-vertex is faster than Warp-per-vertex by "
                      << std::fixed << std::setprecision(2) << (1.0f / ratio) << "x" << ANSI_RESET
                      << " (idle warp lanes on low-degree vertices reduce efficiency).\n";
        }
    }

    if (algo == "BFS" && (!warp_v || !edge)) {
        std::cout << "  • " << ANSI_YELLOW << "BFS currently has Thread-per-vertex implemented. "
                  << "Warp-per-vertex and Edge-based mappings can be plugged into the marked code sections below." << ANSI_RESET << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <graph_file> [options]\n\n"
                  << "Options:\n"
                  << "  --source <int>         Starting source vertex (default: 0)\n"
                  << "  --undirected           Treat graph as undirected (edges doubled in CSR)\n"
                  << "  --algo <all|bfs|sssp>  Algorithm suite to run (default: all)\n"
                  << "  --threads <int>        OpenMP thread count (default: max available)\n"
                  << "  --csv <filepath>       Export results to CSV file\n"
                  << "  --help                 Show this help message\n";
        return 1;
    }

    std::string graph_path = argv[1];
    int source = 0;
    int directed = 1;
    std::string algo_mode = "all";
    int num_threads = omp_get_max_threads();
    std::string csv_path = "";

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--undirected") {
            directed = 0;
        } else if (arg == "--source" && i + 1 < argc) {
            source = std::atoi(argv[++i]);
        } else if (arg == "--algo" && i + 1 < argc) {
            algo_mode = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--help") {
            const char *dummy[1] = { argv[0] };
            main(1, (char**)dummy);
            return 0;
        }
    }

    omp_set_num_threads(num_threads);

    std::cout << ANSI_BOLD << ANSI_MAGENTA << "========================================================\n"
              << "       PCAP GPU Graph Traversal: Uniform Benchmark      \n"
              << "========================================================" << ANSI_RESET << "\n";

    // 1. Read input edge list
    std::cout << "[1/4] Reading graph from: " << graph_path << "...\n";
    EdgeListGraph g = read_graph_from_file(graph_path.c_str());
    std::cout << "      Edges read: V=" << g.V << ", E=" << g.E << "\n";

    // 2. Build CSR
    std::cout << "[2/4] Constructing CSR (" << (directed ? "Directed" : "Undirected") << ")...\n";
    CSRGraph csr = build_csr(&g, directed);
    if (!validate_csr(&csr)) {
        std::cerr << ANSI_RED << "Fatal: CSR validation failed!\n" << ANSI_RESET;
        free(g.edges);
        free_csr(&csr);
        return 1;
    }
    double avg_degree = (double)csr.E / (double)csr.V;
    std::cout << "      CSR Validated OK: V=" << csr.V << ", E=" << csr.E
              << ", Avg Degree=" << std::fixed << std::setprecision(2) << avg_degree << "\n";

    if (source < 0 || source >= csr.V) {
        std::cerr << ANSI_RED << "Fatal: Source vertex " << source << " out of range [0, " << csr.V << ")\n" << ANSI_RESET;
        free(g.edges);
        free_csr(&csr);
        return 1;
    }
    std::cout << "      Source vertex: " << source << " | OpenMP threads: " << num_threads << "\n";

    std::vector<BenchmarkResult> results;

    // =========================================================================
    // BFS BENCHMARK SUITE (Implemented: CPU Sequential, GPU Thread-per-vertex)
    // =========================================================================
    if (algo_mode == "all" || algo_mode == "bfs") {
        std::cout << "\n" << ANSI_BOLD << "[3/4] Running BFS Benchmarking Suite..." << ANSI_RESET << "\n";

        std::vector<int> bfs_cpu_dist(csr.V, -1);
        std::vector<int> bfs_gv_dist(csr.V, -1);

        // 1. BFS CPU Sequential (Baseline)
        std::cout << "  -> Running BFS CPU Sequential...";
        std::fflush(stdout);
        Timer t_bfs_cpu;
        timer_start(&t_bfs_cpu);
        bfs_cpu(&csr, source, bfs_cpu_dist.data());
        timer_stop(&t_bfs_cpu);
        double ms_bfs_cpu = timer_elapsed_ms(&t_bfs_cpu);
        std::cout << " done (" << ms_bfs_cpu << " ms)\n";

        int bfs_reached_v = 0;
        long long bfs_traversed_e = 0;
        compute_graph_traversal_stats(&csr, bfs_cpu_dist.data(), nullptr, &bfs_reached_v, &bfs_traversed_e);
        double bfs_mteps_cpu = (ms_bfs_cpu > 0) ? ((double)bfs_traversed_e / (ms_bfs_cpu * 1000.0)) : 0.0;

        BenchmarkResult r_bfs_cpu;
        r_bfs_cpu.algorithm = "BFS";
        r_bfs_cpu.strategy = "CPU Sequential";
        r_bfs_cpu.status = STATUS_SUCCESS;
        r_bfs_cpu.h2d_ms = 0;
        r_bfs_cpu.compute_ms = ms_bfs_cpu;
        r_bfs_cpu.d2h_ms = 0;
        r_bfs_cpu.total_ms = ms_bfs_cpu;
        r_bfs_cpu.rounds = 0;
        r_bfs_cpu.speedup_compute = 1.0;
        r_bfs_cpu.speedup_total = 1.0;
        r_bfs_cpu.speedup_vs_omp = 1.0;
        r_bfs_cpu.mteps = bfs_mteps_cpu;
        r_bfs_cpu.mismatches = -1; // baseline
        results.push_back(r_bfs_cpu);

        std::vector<int> bfs_omp_dist(csr.V, -1);
        std::cout << "  -> Running BFS CPU OpenMP (" << num_threads << " threads)...";
        std::fflush(stdout);
        Timer t_bfs_omp;
        timer_start(&t_bfs_omp);
        int ok_bfs_omp = bfs_openmp(&csr, source, bfs_omp_dist.data());
        timer_stop(&t_bfs_omp);
        double ms_bfs_omp = timer_elapsed_ms(&t_bfs_omp);
        std::cout << " done (" << ms_bfs_omp << " ms)\n";

        BenchmarkResult r_bfs_omp;
        r_bfs_omp.algorithm = "BFS";
        r_bfs_omp.strategy = "CPU OpenMP";
        r_bfs_omp.status = ok_bfs_omp ? STATUS_SUCCESS : STATUS_ERROR;
        r_bfs_omp.h2d_ms = 0;
        r_bfs_omp.compute_ms = ms_bfs_omp;
        r_bfs_omp.d2h_ms = 0;
        r_bfs_omp.total_ms = ms_bfs_omp;
        r_bfs_omp.rounds = 0;
        r_bfs_omp.speedup_compute = (ms_bfs_omp > 0) ? (ms_bfs_cpu / ms_bfs_omp) : 0.0;
        r_bfs_omp.speedup_total = r_bfs_omp.speedup_compute;
        r_bfs_omp.speedup_vs_omp = 1.0;
        r_bfs_omp.mteps = (ms_bfs_omp > 0) ? ((double)bfs_traversed_e / (ms_bfs_omp * 1000.0)) : 0.0;
        r_bfs_omp.mismatches = verify_bfs(bfs_cpu_dist.data(), bfs_omp_dist.data(), csr.V);
        results.push_back(r_bfs_omp);

        // 2. BFS GPU Strategy 1: Thread-per-frontier-vertex
        std::cout << "  -> Running BFS GPU (Thread-per-vertex)...";
        std::fflush(stdout);
        GpuTiming tim_bfs_gv;
        memset(&tim_bfs_gv, 0, sizeof(tim_bfs_gv));
        int ok_bfs_gv = bfs_gpu_t_per_v(&csr, source, bfs_gv_dist.data(), &tim_bfs_gv);

        BenchmarkResult r_bfs_gv;
        r_bfs_gv.algorithm = "BFS";
        r_bfs_gv.strategy = "GPU Thread-per-vertex";
        if (ok_bfs_gv != 1) {
            r_bfs_gv.status = STATUS_ERROR;
            std::cout << " " << ANSI_RED << "FAILED\n" << ANSI_RESET;
        } else {
            r_bfs_gv.status = STATUS_SUCCESS;
            r_bfs_gv.h2d_ms = tim_bfs_gv.h2d_ms;
            r_bfs_gv.compute_ms = tim_bfs_gv.compute_ms;
            r_bfs_gv.d2h_ms = tim_bfs_gv.d2h_ms;
            r_bfs_gv.total_ms = tim_bfs_gv.total_ms;
            r_bfs_gv.rounds = tim_bfs_gv.rounds;
            r_bfs_gv.speedup_compute = (tim_bfs_gv.compute_ms > 0) ? (ms_bfs_cpu / tim_bfs_gv.compute_ms) : 0.0;
            r_bfs_gv.speedup_total = (tim_bfs_gv.total_ms > 0) ? (ms_bfs_cpu / tim_bfs_gv.total_ms) : 0.0;
            r_bfs_gv.speedup_vs_omp = 0.0;
            r_bfs_gv.mteps = (tim_bfs_gv.compute_ms > 0) ? ((double)bfs_traversed_e / (tim_bfs_gv.compute_ms * 1000.0)) : 0.0;
            r_bfs_gv.mismatches = verify_bfs(bfs_cpu_dist.data(), bfs_gv_dist.data(), csr.V);
            std::cout << " done (" << tim_bfs_gv.total_ms << " ms, " << tim_bfs_gv.rounds << " levels)\n";
        }
        results.push_back(r_bfs_gv);

        std::vector<int> bfs_gwarp_dist(csr.V, -1);
        std::cout << "  -> Running BFS GPU (Warp-per-vertex)...";
        std::fflush(stdout);
        GpuTiming tim_bfs_warp;
        memset(&tim_bfs_warp, 0, sizeof(tim_bfs_warp));
        int ok_bfs_warp = bfs_gpu_warp_per_v(&csr, source, bfs_gwarp_dist.data(), &tim_bfs_warp);

        BenchmarkResult r_bfs_warp;
        r_bfs_warp.algorithm = "BFS";
        r_bfs_warp.strategy = "GPU Warp-per-vertex";
        r_bfs_warp.status = ok_bfs_warp ? STATUS_SUCCESS : STATUS_ERROR;
        r_bfs_warp.h2d_ms = tim_bfs_warp.h2d_ms;
        r_bfs_warp.compute_ms = tim_bfs_warp.compute_ms;
        r_bfs_warp.d2h_ms = tim_bfs_warp.d2h_ms;
        r_bfs_warp.total_ms = tim_bfs_warp.total_ms;
        r_bfs_warp.rounds = tim_bfs_warp.rounds;
        r_bfs_warp.speedup_compute = (tim_bfs_warp.compute_ms > 0) ? (ms_bfs_cpu / tim_bfs_warp.compute_ms) : 0.0;
        r_bfs_warp.speedup_total = (tim_bfs_warp.total_ms > 0) ? (ms_bfs_cpu / tim_bfs_warp.total_ms) : 0.0;
        r_bfs_warp.speedup_vs_omp = 0.0;
        r_bfs_warp.mteps = (tim_bfs_warp.compute_ms > 0) ? ((double)bfs_traversed_e / (tim_bfs_warp.compute_ms * 1000.0)) : 0.0;
        r_bfs_warp.mismatches = verify_bfs(bfs_cpu_dist.data(), bfs_gwarp_dist.data(), csr.V);
        results.push_back(r_bfs_warp);

        std::vector<int> bfs_gedge_dist(csr.V, -1);
        std::cout << "  -> Running BFS GPU (Edge-based frontier)...";
        std::fflush(stdout);
        GpuTiming tim_bfs_edge;
        memset(&tim_bfs_edge, 0, sizeof(tim_bfs_edge));
        int ok_bfs_edge = bfs_gpu_edge_based(&csr, source, bfs_gedge_dist.data(), &tim_bfs_edge);

        BenchmarkResult r_bfs_edge;
        r_bfs_edge.algorithm = "BFS";
        r_bfs_edge.strategy = "GPU Edge-based";
        r_bfs_edge.status = ok_bfs_edge ? STATUS_SUCCESS : STATUS_ERROR;
        r_bfs_edge.h2d_ms = tim_bfs_edge.h2d_ms;
        r_bfs_edge.compute_ms = tim_bfs_edge.compute_ms;
        r_bfs_edge.d2h_ms = tim_bfs_edge.d2h_ms;
        r_bfs_edge.total_ms = tim_bfs_edge.total_ms;
        r_bfs_edge.rounds = tim_bfs_edge.rounds;
        r_bfs_edge.speedup_compute = (tim_bfs_edge.compute_ms > 0) ? (ms_bfs_cpu / tim_bfs_edge.compute_ms) : 0.0;
        r_bfs_edge.speedup_total = (tim_bfs_edge.total_ms > 0) ? (ms_bfs_cpu / tim_bfs_edge.total_ms) : 0.0;
        r_bfs_edge.speedup_vs_omp = 0.0;
        r_bfs_edge.mteps = (tim_bfs_edge.compute_ms > 0) ? ((double)bfs_traversed_e / (tim_bfs_edge.compute_ms * 1000.0)) : 0.0;
        r_bfs_edge.mismatches = verify_bfs(bfs_cpu_dist.data(), bfs_gedge_dist.data(), csr.V);
        results.push_back(r_bfs_edge);

        print_load_balancing_comparison(results, "BFS");
    }

    // =========================================================================
    // SSSP BENCHMARK SUITE (Implemented: CPU Serial, OpenMP, 3 GPU Strategies)
    // =========================================================================
    if (algo_mode == "all" || algo_mode == "sssp") {
        std::cout << "\n" << ANSI_BOLD << "[4/4] Running SSSP Benchmarking Suite..." << ANSI_RESET << "\n";

        std::vector<float> sssp_cpu_dist(csr.V, SSSP_INF);
        std::vector<float> sssp_omp_dist(csr.V, SSSP_INF);
        std::vector<float> sssp_gv_dist(csr.V, SSSP_INF);
        std::vector<float> sssp_gw_dist(csr.V, SSSP_INF);
        std::vector<float> sssp_ge_dist(csr.V, SSSP_INF);

        // 1. SSSP CPU Sequential Bellman-Ford
        std::cout << "  -> Running SSSP CPU Sequential...";
        std::fflush(stdout);
        Timer t_sssp_cpu;
        timer_start(&t_sssp_cpu);
        int ok_sssp_cpu = sssp_bellman_ford_cpu(&csr, source, sssp_cpu_dist.data());
        timer_stop(&t_sssp_cpu);
        double ms_sssp_cpu = timer_elapsed_ms(&t_sssp_cpu);
        std::cout << " done (" << ms_sssp_cpu << " ms)\n";

        int sssp_reached_v = 0;
        long long sssp_traversed_e = 0;
        compute_graph_traversal_stats(&csr, nullptr, sssp_cpu_dist.data(), &sssp_reached_v, &sssp_traversed_e);
        double sssp_mteps_cpu = (ms_sssp_cpu > 0) ? ((double)sssp_traversed_e / (ms_sssp_cpu * 1000.0)) : 0.0;

        BenchmarkResult r_sssp_cpu;
        r_sssp_cpu.algorithm = "SSSP";
        r_sssp_cpu.strategy = "CPU Sequential";
        r_sssp_cpu.status = ok_sssp_cpu ? STATUS_SUCCESS : STATUS_ERROR;
        r_sssp_cpu.h2d_ms = 0;
        r_sssp_cpu.compute_ms = ms_sssp_cpu;
        r_sssp_cpu.d2h_ms = 0;
        r_sssp_cpu.total_ms = ms_sssp_cpu;
        r_sssp_cpu.rounds = 0;
        r_sssp_cpu.speedup_compute = 1.0;
        r_sssp_cpu.speedup_total = 1.0;
        r_sssp_cpu.speedup_vs_omp = 1.0;
        r_sssp_cpu.mteps = sssp_mteps_cpu;
        r_sssp_cpu.mismatches = -1; // baseline
        results.push_back(r_sssp_cpu);

        // 2. SSSP CPU OpenMP Bellman-Ford
        std::cout << "  -> Running SSSP CPU OpenMP (" << num_threads << " threads)...";
        std::fflush(stdout);
        Timer t_sssp_omp;
        timer_start(&t_sssp_omp);
        int ok_sssp_omp = sssp_bellman_ford_openmp(&csr, source, sssp_omp_dist.data());
        timer_stop(&t_sssp_omp);
        double ms_sssp_omp = timer_elapsed_ms(&t_sssp_omp);
        std::cout << " done (" << ms_sssp_omp << " ms)\n";

        BenchmarkResult r_sssp_omp;
        r_sssp_omp.algorithm = "SSSP";
        r_sssp_omp.strategy = "CPU OpenMP";
        r_sssp_omp.status = ok_sssp_omp ? STATUS_SUCCESS : STATUS_ERROR;
        r_sssp_omp.h2d_ms = 0;
        r_sssp_omp.compute_ms = ms_sssp_omp;
        r_sssp_omp.d2h_ms = 0;
        r_sssp_omp.total_ms = ms_sssp_omp;
        r_sssp_omp.rounds = 0;
        r_sssp_omp.speedup_compute = (ms_sssp_omp > 0) ? (ms_sssp_cpu / ms_sssp_omp) : 0.0;
        r_sssp_omp.speedup_total = r_sssp_omp.speedup_compute;
        r_sssp_omp.speedup_vs_omp = 1.0;
        r_sssp_omp.mteps = (ms_sssp_omp > 0) ? ((double)sssp_traversed_e / (ms_sssp_omp * 1000.0)) : 0.0;
        r_sssp_omp.mismatches = verify_sssp(sssp_cpu_dist.data(), sssp_omp_dist.data(), csr.V);
        results.push_back(r_sssp_omp);

        // 3. SSSP GPU Strategy 1: Thread-per-vertex
        std::cout << "  -> Running SSSP GPU (Thread-per-vertex)...";
        std::fflush(stdout);
        GpuTiming tim_sssp_gv;
        memset(&tim_sssp_gv, 0, sizeof(tim_sssp_gv));
        int ok_sssp_gv = sssp_bellman_ford_gpu(&csr, source, sssp_gv_dist.data(), &tim_sssp_gv);
        std::cout << " done (" << tim_sssp_gv.total_ms << " ms, " << tim_sssp_gv.rounds << " rounds)\n";

        BenchmarkResult r_sssp_gv;
        r_sssp_gv.algorithm = "SSSP";
        r_sssp_gv.strategy = "GPU Thread-per-vertex";
        r_sssp_gv.status = ok_sssp_gv ? STATUS_SUCCESS : STATUS_ERROR;
        r_sssp_gv.h2d_ms = tim_sssp_gv.h2d_ms;
        r_sssp_gv.compute_ms = tim_sssp_gv.compute_ms;
        r_sssp_gv.d2h_ms = tim_sssp_gv.d2h_ms;
        r_sssp_gv.total_ms = tim_sssp_gv.total_ms;
        r_sssp_gv.rounds = tim_sssp_gv.rounds;
        r_sssp_gv.speedup_compute = (tim_sssp_gv.compute_ms > 0) ? (ms_sssp_cpu / tim_sssp_gv.compute_ms) : 0.0;
        r_sssp_gv.speedup_total = (tim_sssp_gv.total_ms > 0) ? (ms_sssp_cpu / tim_sssp_gv.total_ms) : 0.0;
        r_sssp_gv.speedup_vs_omp = (tim_sssp_gv.compute_ms > 0) ? (ms_sssp_omp / tim_sssp_gv.compute_ms) : 0.0;
        r_sssp_gv.mteps = (tim_sssp_gv.compute_ms > 0) ? ((double)sssp_traversed_e / (tim_sssp_gv.compute_ms * 1000.0)) : 0.0;
        r_sssp_gv.mismatches = verify_sssp(sssp_cpu_dist.data(), sssp_gv_dist.data(), csr.V);
        results.push_back(r_sssp_gv);

        // 4. SSSP GPU Strategy 2: Warp-per-vertex
        std::cout << "  -> Running SSSP GPU (Warp-per-vertex)...";
        std::fflush(stdout);
        GpuTiming tim_sssp_gw;
        memset(&tim_sssp_gw, 0, sizeof(tim_sssp_gw));
        int ok_sssp_gw = sssp_bellman_ford_gpu_warp_per_v(&csr, source, sssp_gw_dist.data(), &tim_sssp_gw);
        std::cout << " done (" << tim_sssp_gw.total_ms << " ms, " << tim_sssp_gw.rounds << " rounds)\n";

        BenchmarkResult r_sssp_gw;
        r_sssp_gw.algorithm = "SSSP";
        r_sssp_gw.strategy = "GPU Warp-per-vertex";
        r_sssp_gw.status = ok_sssp_gw ? STATUS_SUCCESS : STATUS_ERROR;
        r_sssp_gw.h2d_ms = tim_sssp_gw.h2d_ms;
        r_sssp_gw.compute_ms = tim_sssp_gw.compute_ms;
        r_sssp_gw.d2h_ms = tim_sssp_gw.d2h_ms;
        r_sssp_gw.total_ms = tim_sssp_gw.total_ms;
        r_sssp_gw.rounds = tim_sssp_gw.rounds;
        r_sssp_gw.speedup_compute = (tim_sssp_gw.compute_ms > 0) ? (ms_sssp_cpu / tim_sssp_gw.compute_ms) : 0.0;
        r_sssp_gw.speedup_total = (tim_sssp_gw.total_ms > 0) ? (ms_sssp_cpu / tim_sssp_gw.total_ms) : 0.0;
        r_sssp_gw.speedup_vs_omp = (tim_sssp_gw.compute_ms > 0) ? (ms_sssp_omp / tim_sssp_gw.compute_ms) : 0.0;
        r_sssp_gw.mteps = (tim_sssp_gw.compute_ms > 0) ? ((double)sssp_traversed_e / (tim_sssp_gw.compute_ms * 1000.0)) : 0.0;
        r_sssp_gw.mismatches = verify_sssp(sssp_cpu_dist.data(), sssp_gw_dist.data(), csr.V);
        results.push_back(r_sssp_gw);

        // 5. SSSP GPU Strategy 3: Thread-per-edge
        std::cout << "  -> Running SSSP GPU (Thread-per-edge)...";
        std::fflush(stdout);
        GpuTiming tim_sssp_ge;
        memset(&tim_sssp_ge, 0, sizeof(tim_sssp_ge));
        int ok_sssp_ge = sssp_bellman_ford_gpu_t_per_e(&csr, source, sssp_ge_dist.data(), &tim_sssp_ge);
        std::cout << " done (" << tim_sssp_ge.total_ms << " ms, " << tim_sssp_ge.rounds << " rounds)\n";

        BenchmarkResult r_sssp_ge;
        r_sssp_ge.algorithm = "SSSP";
        r_sssp_ge.strategy = "GPU Thread-per-edge";
        r_sssp_ge.status = ok_sssp_ge ? STATUS_SUCCESS : STATUS_ERROR;
        r_sssp_ge.h2d_ms = tim_sssp_ge.h2d_ms;
        r_sssp_ge.compute_ms = tim_sssp_ge.compute_ms;
        r_sssp_ge.d2h_ms = tim_sssp_ge.d2h_ms;
        r_sssp_ge.total_ms = tim_sssp_ge.total_ms;
        r_sssp_ge.rounds = tim_sssp_ge.rounds;
        r_sssp_ge.speedup_compute = (tim_sssp_ge.compute_ms > 0) ? (ms_sssp_cpu / tim_sssp_ge.compute_ms) : 0.0;
        r_sssp_ge.speedup_total = (tim_sssp_ge.total_ms > 0) ? (ms_sssp_cpu / tim_sssp_ge.total_ms) : 0.0;
        r_sssp_ge.speedup_vs_omp = (tim_sssp_ge.compute_ms > 0) ? (ms_sssp_omp / tim_sssp_ge.compute_ms) : 0.0;
        r_sssp_ge.mteps = (tim_sssp_ge.compute_ms > 0) ? ((double)sssp_traversed_e / (tim_sssp_ge.compute_ms * 1000.0)) : 0.0;
        r_sssp_ge.mismatches = verify_sssp(sssp_cpu_dist.data(), sssp_ge_dist.data(), csr.V);
        results.push_back(r_sssp_ge);

        /*
         * =====================================================================
         * FUTURE EXTENSION: SSSP Active-Frontier / Delta-Stepping Style
         * =====================================================================
         * When delta-stepping / active worklist SSSP is implemented:
         * 
         * std::vector<float> sssp_delta_dist(csr.V, SSSP_INF);
         * GpuTiming tim_delta;
         * int ok_delta = sssp_delta_stepping_gpu(&csr, source, delta, sssp_delta_dist.data(), &tim_delta);
         * ...
         * =====================================================================
         */

        print_load_balancing_comparison(results, "SSSP");
    }

    // Print Aligned Table
    print_results_table(results, num_threads);

    // Export to CSV if requested
    if (!csv_path.empty()) {
        export_csv(csv_path, graph_path, csr.V, csr.E, results, num_threads);
    }

    // Cleanup host memory
    free(g.edges);
    free_csr(&csr);

    return 0;
}
