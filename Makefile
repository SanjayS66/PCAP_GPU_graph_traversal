
# PCAP Implementation Makefile
# GPU-Accelerated Graph Traversal: BFS and SSSP with Load-Balancing Strategies

CC        ?= gcc
CXX       ?= g++
NVCC      ?= nvcc

CFLAGS    := -Wall -Wextra -O3 -Iinclude -fopenmp
CXXFLAGS  := -Wall -Wextra -O3 -Iinclude -fopenmp
NVCCFLAGS := -O3 -Iinclude -Xcompiler -fopenmp -Xcompiler -Wall

BUILD_DIR := build
SRC_DIR   := src
TEST_DIR  := test

# Core objects for implemented algorithms
CSR_OBJS := $(BUILD_DIR)/csr.o
CUDA_UTIL_OBJS := $(BUILD_DIR)/cuda_utils.o
TIMER_OBJS := $(BUILD_DIR)/timer.o

BFS_OBJS := $(BUILD_DIR)/bfs_cpu.o $(BUILD_DIR)/bfs_openmp.o $(BUILD_DIR)/bfs_gpu.o \
            $(BUILD_DIR)/bfs_gpu_warp_per_v.o $(BUILD_DIR)/bfs_gpu_edge_based.o
SSSP_OBJS := $(BUILD_DIR)/sssp_cpu.o $(BUILD_DIR)/sssp_openmp.o \
             $(BUILD_DIR)/sssp_gpu_t_per_v.o $(BUILD_DIR)/sssp_gpu_warp_per_v.o \
             $(BUILD_DIR)/sssp_gpu_t_per_e.o

ALL_CORE_OBJS := $(CSR_OBJS) $(CUDA_UTIL_OBJS) $(TIMER_OBJS) $(BFS_OBJS) $(SSSP_OBJS)

# Executables
BENCHMARK_BIN := $(BUILD_DIR)/uniform_benchmark
CSR_TEST_BIN  := $(BUILD_DIR)/csr_convert
BFS_TEST_BIN  := $(BUILD_DIR)/test_bfs_gpu
SSSP_TEST_BIN := $(BUILD_DIR)/test_sssp

.PHONY: all clean test run help

all: $(BUILD_DIR) $(BENCHMARK_BIN) $(CSR_TEST_BIN) $(BFS_TEST_BIN) $(SSSP_TEST_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Object file compilations
$(BUILD_DIR)/csr.o: $(SRC_DIR)/csr.c include/csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/timer.o: $(SRC_DIR)/timer.cpp include/timer.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/bfs_cpu.o: $(SRC_DIR)/bfs_cpu.c include/bfs.h include/csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/sssp_cpu.o: $(SRC_DIR)/sssp_cpu.c include/sssp.h include/csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/sssp_openmp.o: $(SRC_DIR)/sssp_openmp.c include/sssp.h include/csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/cuda_utils.o: $(SRC_DIR)/cuda_utils.cu include/cuda_utils.h include/csr.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/bfs_openmp.o: $(SRC_DIR)/bfs_openmp.c include/bfs.h include/csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bfs_gpu.o: $(SRC_DIR)/bfs_gpu.cu include/bfs.h include/csr.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/bfs_gpu_warp_per_v.o: $(SRC_DIR)/bfs_gpu_warp_per_v.cu include/bfs.h include/csr.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/bfs_gpu_edge_based.o: $(SRC_DIR)/bfs_gpu_edge_based.cu include/bfs.h include/csr.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/sssp_gpu_t_per_v.o: $(SRC_DIR)/sssp_gpu_t_per_v.cu include/sssp.h include/cuda_utils.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/sssp_gpu_warp_per_v.o: $(SRC_DIR)/sssp_gpu_warp_per_v.cu include/sssp.h include/cuda_utils.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/sssp_gpu_t_per_e.o: $(SRC_DIR)/sssp_gpu_t_per_e.cu include/sssp.h include/cuda_utils.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Unified benchmark executable
$(BENCHMARK_BIN): $(ALL_CORE_OBJS) $(TEST_DIR)/uniform_benchmark.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Ancillary test binaries
$(CSR_TEST_BIN): $(CSR_OBJS) $(TEST_DIR)/test_csr.c
	$(CC) $(CFLAGS) $^ -o $@

$(BFS_TEST_BIN): $(CSR_OBJS) $(BUILD_DIR)/bfs_cpu.o $(BUILD_DIR)/bfs_gpu.o $(TEST_DIR)/test_bfs_gpu.c
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(SSSP_TEST_BIN): $(CSR_OBJS) $(CUDA_UTIL_OBJS) $(TIMER_OBJS) $(BUILD_DIR)/sssp_cpu.o $(BUILD_DIR)/sssp_gpu_t_per_v.o $(BUILD_DIR)/sssp_gpu_t_per_e.o $(TEST_DIR)/test_sssp.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Run benchmark on sample graph
run: $(BENCHMARK_BIN)
	./$(BENCHMARK_BIN) graphs/sample_graph.txt --source 0 --undirected

test: $(BENCHMARK_BIN)
	@echo "Running uniform benchmark test suite on graphs/sample_graph.txt..."
	./$(BENCHMARK_BIN) graphs/sample_graph.txt --source 0 --undirected

clean:
	rm -rf $(BUILD_DIR)/*.o $(BENCHMARK_BIN) $(CSR_TEST_BIN) $(BFS_TEST_BIN) $(SSSP_TEST_BIN)

help:
	@echo "Available targets:"
	@echo "  make              - Build all targets and the unified benchmark suite"
	@echo "  make run          - Run benchmark on graphs/sample_graph.txt"
	@echo "  make test         - Run validation on sample graph"
	@echo "  make clean        - Remove all built object files and binaries"
