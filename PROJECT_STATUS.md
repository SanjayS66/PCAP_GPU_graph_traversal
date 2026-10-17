# Project Status — PCAP GPU Graph Traversal

*First audit 2026-10-08. Updated 2026-10-17 after Sanjay's final GPU run (`43a0ef3`). Every item was checked against the repo files, not from memory.*

**Legend:** ✅ done · ⚠️ partial / caveat · ❌ missing

## ▶ Resume here

**Where we are (2026-10-17):** the **core project is done**, apart from one profiling re-run. Sanjay's final run is pushed:
- 660 benchmark rows, all **`[Verify] PASS`**, with 5 runs per config at 1/2/4/8 threads on all 3 graphs
- all 12 plots, hardware specs, imbalance numbers (including Google+) and gprof

**Next steps:**
1. **Sanjay (~20 min, optional but recommended):** `git pull`, then `bash scripts/profile_kernels.sh` from the up-to-date repo, then push. His Nsight run used the old script on the *directed* LiveJournal graph. The warp-efficiency column has been corrected (see below), but the graph still doesn't match the benchmarks. If this doesn't happen, the report must say the profile was taken on the directed graph.
2. **Report, deck and demo**, using the numbers below. Make a zip for this with `git archive --format=zip -o ../PCAP_for_report.zip HEAD`; it leaves out the datasets.

**Open team decisions:** unit edge weights (report note vs re-run) and the unmerged BFS thread-per-edge branch. See "Decisions for the team" below.

## Verdict

**✅ Core project complete**, apart from a ⚠️ profiling re-run. The report, deck and demo are not started.

| Area | Status |
|---|---|
| A. Source code | ✅ |
| B. Build | ✅ verified on Sanjay's machine (all results produced by a fresh build) |
| C. Correctness | ✅ 660/660 rows PASS (unit-weight caveat) |
| D. Datasets | ✅ all 3; roadNet now without duplicate edges |
| E. Harness + metrics | ✅ 5 samples per config, full thread sweep |
| F. Results | ✅ benchmarks, plots, imbalance, gprof · ⚠️ Nsight on the directed graph |
| G. Final report | ❌ not started |
| H. Deliverables | ❌ deck / demo plan; ⚠️ 1 unmerged branch |

---

## To-do list

Put your name next to a task when you take it.

### GPU work (Sanjay, `43a0ef3`, 2026-10-17)

- [x] Full build + smoke test
- [x] Hardware specs → `results/profiling/hardware.txt`
- [x] Final clean benchmark run: 5 repeats × 1/2/4/8 threads × 3 graphs, **660 rows, all PASS**
- [x] roadNet re-run `--directed` (E = 5,533,214, no duplicate edges), with a full thread sweep
- [x] All 12 plots regenerated, including roadNet
- [x] Google+ imbalance: **19,082 → 597 → 1**
- [x] gprof CPU profile
- [ ] ⚠️ **Re-run Nsight on the undirected graph.** The committed profile used the old pipeline (`parse_raw.py`) on the directed LiveJournal CSR. — owner: Sanjay

### Fixed on the laptop (2026-10-17)

- [x] **`ncu_metrics_summary.txt` corrected.** The old parser printed ncu's warp-efficiency ratio (0–32 active lanes) as if it were a percentage, so every value was about 3× too low. `scripts/parse_raw.py` is fixed and the summary was regenerated from the raw data. The file now also notes the directed-graph caveat.

### Done on the laptop earlier (2026-10-15)

- [x] `.gitignore` keeps `results/profiling/*.txt`, so profiling summaries get pushed.
- [x] `profile_kernels.sh` fixed (empty ncu dump with `-o`, missing `--undirected`, gprof link flag, `sudo` advice).
- [x] `parse_ncu_metrics.py` warp-efficiency conversion fixed.
- [x] `run_benchmarks.py`: `--threads` sweep, `--replace`, `[Verify] PASS/FAIL`, required `--graphs`.
- [x] `visualize_results.py`: all graphs by default; no fake efficiency point without a 1-thread baseline.
- [x] `calc_imbalance.py --directed` for already-symmetric inputs (roadNet).
- [x] Makefile `run`/`test` use a real graph.

### Decisions for the team

- [ ] **Unit weights.** All edge weights are `1.0`, so SSSP is equivalent to BFS. Default plan: state this in the report's methodology. The alternative is random weights plus a full re-run (runbook, Optional B).
- [ ] **`origin/feature/bfs-parallel-strategies`.** 1 unmerged commit adds BFS thread-per-edge, but it deletes `test/test_bfs_gpu.c`, which the Makefile uses. Merge with a Makefile fix (runbook, Optional A) or drop it.
- [ ] **Duplicate edges (report note).** LiveJournal and Google+ are symmetrised with `--undirected` without removing duplicates, so mutual friendships appear twice in the CSR. Results are still correct; state this in the methodology. roadNet runs `--directed`, so it has no duplicates.

### Report, deck and demo

- [ ] **Write the report.** — owner: ___
- [ ] **Make the presentation deck** and **decide the demo plan** (live terminal benchmark vs a dashboard with hardcoded numbers). — owner: ___

### Nice to have (paper extras)

- [ ] Block-size sweep (block size is currently hard-coded at 256)
- [ ] BFS adaptive kernel
- [ ] Multi-GPU / delta-stepping (only placeholder comments exist in the harness)

---

## Key results (for the report)

### Hardware (`results/profiling/hardware.txt`)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 (laptop), 4 GB, compute capability 8.6 |
| CUDA | 12.3 (V12.3.107), driver 595.99.02 |
| CPU | Intel Core i5-12450H: 8 cores / 12 threads, up to 4.4 GHz |
| RAM | 15 GB |
| OS / compiler | Ubuntu 22.04 (Linux 6.8), gcc 11.4.0 |

### Benchmarks

Compute time in ms (mean of 5 runs) from `results/csv/results_summary.csv`. OpenMP uses 8 threads. **Bold** = fastest in that row.

| | CPU seq | CPU OMP | GPU thread/V | GPU warp | GPU edge | GPU adaptive |
|---|---|---|---|---|---|---|
| SSSP Google+ | 73.5 | 36.5 | 193.6 | **13.0** | 20.5 | 17.1 |
| SSSP LiveJournal | 1411.5 | 542.8 | 1251.3 | **278.2** | 279.1 | 487.9 |
| SSSP roadNet | 3642.2 | 747.6 | 239.9 | 1293.4 | 388.8 | **235.4** |
| BFS Google+ | 18.9 | 3.9 | 10.4 | **2.4** | 11.6 | — |
| BFS LiveJournal | 568.4 | **66.5** | 98.7 | 71.3 | 109.2 | — |
| BFS roadNet | **48.5** | 207.7 | 49.2 | 696.7 | 161.8 | — |

GPU times are kernel compute only. Host↔device transfer times are in the CSV (`h2d_ms`, `d2h_ms`).

**Takeaways:**
- **Structural flip.** On the power-law graphs, warp-per-vertex wins SSSP: **5.7× faster** than sequential on Google+ and **5.1×** on LiveJournal, while thread-per-vertex is barely faster than the CPU, or slower. On the low-degree road network the order reverses: warp-per-vertex is the slowest GPU strategy, and thread-per-vertex and adaptive win at **~15.5×**.
- **Adaptive is the most consistent strategy, not the fastest on power-law graphs.** It's the best on roadNet (235.4 vs 239.9 ms for thread-per-vertex), but 1.3× slower than warp on Google+ and 1.75× slower on LiveJournal. It's never the worst GPU strategy on any graph.
- **The GPU doesn't always win for BFS.** It wins on Google+ (2.4 vs OpenMP 3.9 ms), OpenMP edges it out on LiveJournal (66.5 vs 71.3 ms), and on roadNet sequential is fastest (48.5 ms), essentially tied with GPU thread-per-vertex.

### Load imbalance

Max per-thread work: the edges handled by the busiest thread (`scripts/calc_imbalance.py`):

| Graph | Thread/V | Warp | Edge |
|---|---|---|---|
| LiveJournal | 22,889 | 716 | 1 |
| Google+ | 19,082 | 597 | 1 |
| roadNet-CA | 12 | 1 | 1 |

### Nsight Compute (SSSP kernels, LiveJournal)

From `results/profiling/ncu_metrics_summary.txt`. ⚠️ Taken on the **directed** LiveJournal CSR (see to-do).

| Kernel | Occupancy | Warp efficiency | DRAM bandwidth |
|---|---|---|---|
| Thread-per-vertex | 75.4% | 23.2% | 46.3% |
| Warp-per-vertex | 69.4% | 64.9% | 50.4% |
| Thread-per-edge | 86.2% | 96.3% | 64.3% |
| Adaptive (low-degree) | 81.9% | 45.5% | 61.4% |
| Adaptive (high-degree) | 73.2% | 85.8% | 51.0% |

Warp efficiency rises from 23% (thread-per-vertex) to 65% (warp) to 96% (edge), the same order as the imbalance metric. Thread-per-vertex wastes about three quarters of its warp lanes on degree skew.

### gprof (CPU sequential SSSP run)

`sssp_bellman_ford_cpu` takes **~70%** of CPU time, `build_csr` ~10%, and `read_graph_from_file` ~8%. The CPU baseline is dominated by the relaxation loop itself, which is the part the GPU kernels parallelise.

---

## Detailed audit

### A. Source code — ✅
- CSR builder: `src/csr.c`, `include/csr.h`
- CPU sequential: `src/bfs_cpu.c`, `src/sssp_cpu.c`
- OpenMP: `src/bfs_openmp.c`, `src/sssp_openmp.c`
- BFS GPU: `src/bfs_gpu.cu` (thread-per-vertex), `src/bfs_gpu_warp_per_v.cu`, `src/bfs_gpu_edge_based.cu`
- SSSP GPU: `src/sssp_gpu_t_per_v.cu`, `src/sssp_gpu_warp_per_v.cu`, `src/sssp_gpu_t_per_e.cu`
- Adaptive SSSP: `src/sssp_gpu_adaptive.cu`. ✅ The host builds `low_list` and `high_list` once, and two separately sized kernels run each Bellman-Ford iteration. There's no per-thread degree branching.

### B. Build — ✅
- The Makefile includes every kernel, including `sssp_gpu_adaptive.o`.
- Built with CUDA 12.3 on Sanjay's machine; every result in `results/` came from that build.

### C. Correctness — ✅
- The harness (`test/uniform_benchmark.cu`, `verify_bfs` / `verify_sssp`) checks every strategy against the CPU baseline.
- `python scripts/run_benchmarks.py --analyze-only` → `[Verify] PASS: all 660 rows succeeded with 0 mismatches`.
- ⚠️ All edge weights are 1.0 (see decisions).

### D. Datasets — ✅
- LiveJournal: V = 4,847,571, E = 137,987,546 (undirected CSR)
- Google+: V = 102,100, E = 24,227,002 (undirected CSR)
- roadNet-CA: V = 1,971,281, E = 5,533,214 (file already symmetric, run `--directed`)

### E. Harness + metrics — ✅
- 5 samples per config, at 1/2/4/8 OpenMP threads, for all 3 graphs (132 summary rows).
- H2D / compute / D2H timed separately with CUDA events.
- Efficiency curves start at 1.0 on all 3 graphs (checked on the roadNet plot).

### F. Results — ✅ / ⚠️
- ✅ BFS and SSSP results for all strategies on all 3 graphs
- ✅ Imbalance for all 3 graphs
- ✅ gprof
- ⚠️ Nsight: directed graph; warp efficiency corrected after the fact

### G. Final report — ❌
Not started. Planned after the core project.

### H. Deliverables
- ❌ Presentation deck
- ❌ Demo plan
- ⚠️ 1 unmerged commit on `origin/feature/bfs-parallel-strategies`
