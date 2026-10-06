# PCAP Project — Lab-Machine GPU Checklist

**Project:** GPU-Accelerated Graph Traversal (BFS & SSSP, CUDA)
**For:** whoever has the lab GPU session
**Rule:** run everything below on the **lab machine**, not Colab — so all final numbers come from one consistent GPU and Nsight's hardware counters work (`ncu` is blocked on Colab).

Do these in order. The two that decide the report are **#1 (adaptive kernel)** and **#2 (Nsight)** — if GPU time is short, do those two first.

---

## 0. Pre-flight (once, at the start)

```bash
cd PCAP_GPU_graph_traversal
git checkout main
git pull --no-rebase        # get everyone's latest
make clean && make          # confirm current code builds before you change anything
```

If `make` fails here, fix that before touching anything else.

---

## REQUIRED — needed to finish the project

### 1. Optimized adaptive SSSP kernel  ⭐ (decides the novelty claim)

**What it does:** instead of every thread checking each vertex's degree at runtime, we **partition vertices once on the host** into a low-degree list (≤32 neighbours → 1 thread each) and a high-degree list (>32 → 1 warp each), then launch a right-sized grid for each. This removes the per-vertex branching that made the old adaptive kernel only *match* warp instead of beating it.

**Step 1a — replace `src/sssp_gpu_adaptive.cu` with the version below.**

> ⚠️ **One thing to verify first:** open your existing `src/sssp_gpu_warp_per_v.cu` and copy its exact `#include` lines and the **argument order/types** of its CSR arrays (`row_offset`, `col_index`, `weights`, `dist`, `changed`). The code below uses the standard names from our repo — if your warp file names them differently (e.g. `d_offsets` vs `row_offset`), rename to match so it links against the same harness.

```cuda
// src/sssp_gpu_adaptive.cu
// Degree-aware adaptive SSSP using pre-partitioned low/high vertex lists.
#include <cuda_runtime.h>
#include <float.h>
#include "csr.h"          // match this to your other sssp_gpu_*.cu files

#define ADAPT_THRESHOLD 32
#define BLOCK 256
#define WARPS_PER_BLOCK (BLOCK / 32)

// atomic min on float (same helper used by the other SSSP kernels)
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}

// LOW-degree vertices: one thread per vertex (from low_list)
__global__ void sssp_adaptive_low_kernel(
        const int* __restrict__ row_offset,
        const int* __restrict__ col_index,
        const float* __restrict__ weights,
        float* __restrict__ dist,
        const int* __restrict__ low_list,
        int low_count,
        int* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= low_count) return;
    int u = low_list[idx];
    float du = dist[u];
    if (du == FLT_MAX) return;
    int start = row_offset[u], end = row_offset[u + 1];
    for (int e = start; e < end; ++e) {
        int v = col_index[e];
        float nd = du + weights[e];
        if (nd < dist[v]) {
            atomicMinFloat(&dist[v], nd);
            *changed = 1;
        }
    }
}

// HIGH-degree vertices: one warp (32 threads) per vertex (from high_list)
__global__ void sssp_adaptive_high_kernel(
        const int* __restrict__ row_offset,
        const int* __restrict__ col_index,
        const float* __restrict__ weights,
        float* __restrict__ dist,
        const int* __restrict__ high_list,
        int high_count,
        int* changed) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // /32
    int lane    = threadIdx.x & 31;                              // %32
    if (warp_id >= high_count) return;
    int u = high_list[warp_id];
    float du = dist[u];
    if (du == FLT_MAX) return;
    int start = row_offset[u], end = row_offset[u + 1];
    for (int e = start + lane; e < end; e += 32) {   // 32 lanes stride the edges
        int v = col_index[e];
        float nd = du + weights[e];
        if (nd < dist[v]) {
            atomicMinFloat(&dist[v], nd);
            *changed = 1;
        }
    }
}
```

**Step 1b — in the host launcher (the function that currently calls your adaptive kernel), partition once before the main loop, then launch both kernels each iteration:**

```cuda
// ---- build the two lists ONCE on the host, after the CSR is ready ----
std::vector<int> h_low, h_high;
h_low.reserve(num_vertices);
h_high.reserve(num_vertices);
for (int u = 0; u < num_vertices; ++u) {
    int deg = h_row_offset[u + 1] - h_row_offset[u];
    if (deg <= ADAPT_THRESHOLD) h_low.push_back(u);
    else                        h_high.push_back(u);
}
int low_count  = (int)h_low.size();
int high_count = (int)h_high.size();

int *d_low = nullptr, *d_high = nullptr;
if (low_count)  { cudaMalloc(&d_low,  low_count  * sizeof(int));
                  cudaMemcpy(d_low,  h_low.data(),  low_count  * sizeof(int), cudaMemcpyHostToDevice); }
if (high_count) { cudaMalloc(&d_high, high_count * sizeof(int));
                  cudaMemcpy(d_high, h_high.data(), high_count * sizeof(int), cudaMemcpyHostToDevice); }

// ---- main Bellman-Ford iteration (time THIS part with CUDA events) ----
int h_changed;
do {
    h_changed = 0;
    cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

    if (low_count) {
        int grid = (low_count + BLOCK - 1) / BLOCK;
        sssp_adaptive_low_kernel<<<grid, BLOCK>>>(
            d_row_offset, d_col_index, d_weights, d_dist, d_low, low_count, d_changed);
    }
    if (high_count) {
        int grid = (high_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;  // 8 warps/block
        sssp_adaptive_high_kernel<<<grid, BLOCK>>>(
            d_row_offset, d_col_index, d_weights, d_dist, d_high, high_count, d_changed);
    }

    cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
} while (h_changed);

cudaFree(d_low); cudaFree(d_high);
```

> Notes: use `FLT_MAX` as the "infinity" init for `dist[]` (and `INF` in the comparison) — keep it the **same sentinel** your other SSSP kernels use, or verification will mismatch. The host partition loop runs once; it's CSR work, not part of the timed compute, so it doesn't inflate the kernel time.

**Step 1c — rebuild and run on both power-law graphs:**

```bash
make clean && make
./bin/sssp_benchmark graphs/gplus_combined.edgelist        # Google+
./bin/sssp_benchmark graphs/soc-LiveJournal1.edgelist      # LiveJournal
# (use whatever your actual benchmark binary + run command is)
```

**Record for each graph:**

| Graph | Adaptive correct? (PASS/FAIL) | Adaptive speedup | Warp-per-vertex speedup | Adaptive beats warp? |
|---|---|---|---|---|
| Google+ | | | | |
| LiveJournal | | | | |

→ If adaptive **beats** warp on both: we claim it as a win in the report.
→ If it only **matches/loses**: we report it honestly as "competitive but not superior" (the report is already written to say this — no rewrite needed).
**Either outcome is fine — just write down the numbers.**

---

### 2. gprof + Nsight profiling  ⭐ (the one analysis the report is still missing)

```bash
bash scripts/profile_kernels.sh
```

This produces:
- **gprof** → the CPU bottleneck report (which function dominates the sequential/OpenMP run).
- **Nsight / `ncu`** → per-kernel GPU report: **occupancy**, **achieved memory bandwidth**, and **warp execution efficiency** for each strategy.

**Why it matters:** the Nsight warp-efficiency number is what *explains* why warp-per-vertex wins on power-law graphs (even lane utilisation vs the idle lanes of thread-per-vertex on hubs). It fills the empty **Profiling** subsection of the report.

If `profile_kernels.sh` doesn't exist or errors, run `ncu` directly on one kernel, e.g.:
```bash
ncu --set full -o ncu_warp_livejournal \
    ./bin/sssp_benchmark graphs/soc-LiveJournal1.edgelist
```
and grab `sm__warps_active` (occupancy), `dram__throughput` (bandwidth), and `smsp__thread_inst_executed_per_inst_executed` (warp efficiency) from the report.

**Record:** occupancy %, achieved bandwidth (GB/s), and warp efficiency % for **thread-per-vertex vs warp-per-vertex vs edge-based** on LiveJournal.

---

### 3. More benchmark samples on high-variance configs

Thread-per-vertex had a **176 ms std** earlier — too noisy for clean error bars.

```bash
# run the benchmark a few more times so the CSV has more samples
for i in 1 2 3 4 5; do
    ./bin/sssp_benchmark graphs/soc-LiveJournal1.edgelist >> results/sssp_results.csv
    ./bin/bfs_benchmark  graphs/soc-LiveJournal1.edgelist >> results/bfs_results.csv
done
```
(match your real binary names / CSV paths). Focus the extra runs on thread-per-vertex and anything else with a large std.

---

### 4. Confirm the full build

```bash
make clean && make
```
Confirm it compiles **every** kernel, including the new adaptive one, with **no errors or warnings**. If the Makefile doesn't already list `sssp_gpu_adaptive.o`, add it to the object list.

---

## VALUABLE — strengthens the report and paper

### 5. roadNet timing benchmark (completes the three-regime story)

You already have roadNet's **imbalance** numbers but not its **speedups**.

```bash
./bin/bfs_benchmark  graphs/roadNet-CA.edgelist
./bin/sssp_benchmark graphs/roadNet-CA.edgelist
```

**Expected (and worth confirming):** on a genuinely low-degree graph, warp-per-vertex **wastes lanes** (most of the 32 sit idle), while edge-based and even thread-per-vertex do fine. That completes the structural story: **uniform → power-law → road network.**

### 6. Final full re-run + regenerate all plots

Once the adaptive kernel is finalised, do one clean sweep so every figure comes from the same run:

```bash
python scripts/run_benchmarks.py          # all graphs, all strategies
python scripts/visualize_results.py       # regenerate every figure
```
Check the efficiency curves start at 1.0 and the speedup bars include the adaptive kernel.

---

## OPTIONAL / STRETCH — only if everything above is done

7. **Block-size sweep** (128 / 256 / 512) — a quick "different configuration" study. Recompile with each `BLOCK` value (or make it a runtime arg) and record speedup vs block size on LiveJournal.
8. **BFS adaptive kernel** — mirror the low/high split for BFS (currently SSSP-only) to round out the comparison.
9. **Multi-GPU MPI / delta-stepping SSSP / direction-optimizing BFS** — genuine future-work items. Skip unless there's spare time; these are listed in the report's Future Work.

---

## NOT CUDA — but the file lives on the lab machine (don't forget)

**Google+ load-imbalance number** — fills the one `[TBD]` cell in the report's imbalance table:

```bash
python scripts/calc_imbalance.py graphs/gplus_combined.edgelist
```

Pure CPU/Python, one instant command. Record the three numbers it prints: **max per-thread work** for thread-per-vertex, for warp, and for edge-based (should look like LiveJournal's `22,889 → 716 → 1` pattern, scaled to Google+).

---

## What to send back to the team

After the session, paste these into the group:
1. The adaptive results table from Task 1 (PASS/FAIL + does it beat warp).
2. The Nsight numbers from Task 2 (occupancy / bandwidth / warp-efficiency per strategy).
3. The roadNet speedups from Task 5.
4. The Google+ imbalance numbers.
5. The updated `results/*.csv` and regenerated figures (Task 6), committed and pushed:

```bash
git add results/ figures/ src/sssp_gpu_adaptive.cu Makefile
git commit -m "Add optimized adaptive SSSP kernel + profiling, roadNet timings, final benchmark run"
git pull --no-rebase
git push
```

That's everything the report and paper still need from the GPU.
