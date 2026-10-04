#!/bin/bash
# automated profiling script for PCAP report

mkdir -p results/profiling

# 1. GPU Profiling with Nsight Compute (ncu)
# We use the gplus graph because ncu slows down execution drastically, but it's small enough to finish quickly.
# It will profile all SSSP kernels (thread-per-vertex, warp-per-vertex, edge-based)
echo "Running NVIDIA Nsight Compute (ncu)... this may take a few minutes."
# --set full captures all metrics (occupancy, bandwidth, warp state)
ncu --set full -f -o results/profiling/ncu_sssp_profile ./build/uniform_benchmark graphs/gplus_combined.edgelist --algo sssp --threads 1 --source 0 --run 1 --repeat 1 > results/profiling/ncu_summary.txt

echo "ncu summary saved to results/profiling/ncu_summary.txt"

# 2. CPU Profiling with gprof
echo "Recompiling with -pg flag for gprof CPU profiling..."
make clean > /dev/null
# Pass -pg to all compilers. NVCC requires -Xcompiler -pg
make CFLAGS="-Wall -Wextra -O3 -Iinclude -fopenmp -pg" \
     CXXFLAGS="-Wall -Wextra -O3 -Iinclude -fopenmp -pg" \
     NVCCFLAGS="-O3 -Iinclude -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -pg" > /dev/null

echo "Running benchmark to generate gmon.out..."
./build/uniform_benchmark graphs/gplus_combined.edgelist --algo bfs --threads 1 --source 0 --run 1 --repeat 1 > /dev/null

echo "Extracting gprof report..."
gprof ./build/uniform_benchmark gmon.out > results/profiling/gprof_cpu_sequential.txt

# Clean up and recompile without -pg for normal use
make clean > /dev/null
make > /dev/null
rm -f gmon.out

echo "Done! Profiling reports are in the results/profiling/ directory."
