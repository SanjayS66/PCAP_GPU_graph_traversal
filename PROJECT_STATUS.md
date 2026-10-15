# Project Status — PCAP GPU Graph Traversal

*Audit of `main` @ `2aebe71`, 2026-10-08; updated 2026-10-15 after the laptop-side fixes. Every item was checked against the repo files, not from memory.*

**Legend:** ✅ done · ⚠️ partial / caveat · ❌ missing

## Verdict

**⚠️ Not submission-ready yet.** The code, correctness checks and benchmark data are solid. What's missing is the **report**, the **deck** and the **Nsight/gprof profiling output**.

| Area | Status |
|---|---|
| A. Source code | ✅ |
| B. Build | ⚠️ not re-verified (needs `nvcc`) |
| C. Correctness | ✅ (unit-weight caveat) |
| D. Datasets | ✅ / ⚠️ |
| E. Harness + metrics | ✅ except sample counts |
| F. Results | ⚠️ profiling + Google+ imbalance missing |
| G. Final report | ❌ not found |
| H. Deliverables | ❌ deck / demo plan; ⚠️ 1 unmerged branch |

---

## To-do list

Put your name next to a task when you take it.

### ➡️ Lab GPU session: follow [LAB_RUNBOOK.md](LAB_RUNBOOK.md)

One session (~1–1.5 h) covers every remaining core-project task. Each step has a pass/fail check.

- [ ] Confirm the full build: `make clean && make` — owner: ___
- [ ] Record hardware specs for the report
- [ ] Final clean benchmark run: 5 repeats × 1/2/4/8 threads on all 3 graphs. This fixes the low sample counts (Google+ was n=1) and the missing roadNet thread sweep.
- [ ] Google+ imbalance numbers (fills the `[TBD]` row)
- [ ] Nsight Compute + gprof profiles, committed as text

### Done on the laptop (2026-10-15)

- [x] `.gitignore` now keeps `results/profiling/*.txt`, so profiling summaries get pushed (`.ncu-rep` binaries stay ignored).
- [x] **`profile_kernels.sh` fixed.** Three bugs would have produced no usable data:
  - With `-o`, ncu prints no metrics to stdout, so the parser got an empty dump. The script now exports the report with `ncu --import`.
  - It profiled LiveJournal *without* `--undirected`, i.e. a different graph than the benchmarks used.
  - gprof used `-Xcompiler -pg`, which doesn't link the gprof runtime, so no `gmon.out` was written. It now uses nvcc's own `-pg`.
  - Its `sudo sysctl` permission advice also doesn't work (that setting is a kernel-module parameter); it now says to re-run as root.
- [x] **`parse_ncu_metrics.py` fixed.** It read warp efficiency as a 0–1 fraction, but ncu's ratio is active lanes per instruction (0–32), so 28 lanes would have printed as "28%" instead of 88%. Tested on a synthetic dump.
- [x] **`run_benchmarks.py`:**
  - `--threads 1 2 4 8` runs a sweep in one command.
  - `--replace` re-runs a graph cleanly.
  - It prints `[Verify] PASS/FAIL` across all rows.
  - `--graphs` is now required, replacing the broken default.
- [x] **`visualize_results.py`:** plots every graph by default, and skips the efficiency plot (with a message) when there's no 1-thread baseline instead of drawing a wrong point.
- [x] **`calc_imbalance.py --directed`:** roadNet's true numbers are **12 → 1 → 1**. They were 24 before, because the already-symmetric file was counted twice.
- [x] Makefile `run`/`test` now use a real graph (roadNet by default; override with `GRAPH=...`).

### Decisions for the team

- [ ] **Unit weights.** All edge weights are `1.0`, so SSSP is equivalent to BFS. Default plan: state this in the report's methodology. The alternative is random weights plus a full re-run (runbook, Optional B).
- [ ] **`origin/feature/bfs-parallel-strategies`.** 1 unmerged commit adds BFS thread-per-edge, but it deletes `test/test_bfs_gpu.c`, which the Makefile uses. Merge with a Makefile fix (runbook, Optional A) or drop it.
- [ ] **Duplicate edges (report note).** LiveJournal and Google+ are symmetrised with `--undirected` without removing duplicates, so mutual friendships appear twice in the CSR. Results are still correct; state this in the methodology. roadNet is now run `--directed`, so it has no duplicates.

### Report, deck and demo (after the core project)

- [ ] **Write the report.** It's not in the repo. If it lives in Google Docs, link it here. — owner: ___
- [ ] **Make the presentation deck** and **decide the demo plan** (live terminal benchmark vs a dashboard with hardcoded numbers). — owner: ___

### Nice to have (paper extras)

- [ ] Block-size sweep (block size is currently hard-coded at 256)
- [ ] BFS adaptive kernel
- [ ] Multi-GPU / delta-stepping (only placeholder comments exist in the harness)

---

## Key results (for the report)

Compute time in ms, from `results/csv/results_summary.csv`. CPU OpenMP uses 8 threads. **Bold** = fastest in that row.

> ⚠️ These numbers are from *before* the final lab run. Replace them with the new `results_summary.csv` afterwards. The roadNet numbers especially will change, since it's now run without duplicate edges.

| | CPU seq | CPU OMP | GPU thread/V | GPU warp | GPU edge | GPU adaptive |
|---|---|---|---|---|---|---|
| SSSP Google+ | 78.2 | 52.8 | 224.0 | **14.1** | 26.2 | 18.6 |
| SSSP LiveJournal | 1550 | 572 | 1523 | **354** | 356 | 550 |
| SSSP roadNet | 3633 | 704 | **302** | 1386 | 713 | 367 |
| BFS Google+ | 22.6 | 5.0 | 12.8 | **2.6** | 12.4 | — |
| BFS LiveJournal | 597 | **70.1** | 102 | 72.9 | 111 | — |
| BFS roadNet | **51.6** | 190 | 52.7 | 729 | 345 | — |

**Takeaways:**
- **Structural flip:** warp-per-vertex wins on power-law graphs (Google+, LiveJournal). Thread-per-vertex wins on the low-degree road network.
- **Adaptive does *not* beat warp on power-law graphs.** It's about 1.3× slower on Google+ and 1.55× slower on LiveJournal. On roadNet it beats warp by 3.8× but loses to thread-per-vertex. So it's the most consistent strategy across all three graphs, but not the fastest on any.
- **GPU doesn't always win for BFS:** on LiveJournal, OpenMP edges out the GPU, and on roadNet sequential CPU is fastest.

**Load imbalance:** max per-thread work, the edges handled by the busiest thread (`scripts/calc_imbalance.py`):

| Graph | Thread/V | Warp | Edge |
|---|---|---|---|
| LiveJournal | 22,889 | 716 | 1 |
| roadNet-CA | 12 | 1 | 1 |
| Google+ | **TBD** | **TBD** | **TBD** |

---

## Detailed audit

### A. Source code — ✅
- CSR builder: `src/csr.c`, `include/csr.h`
- CPU sequential: `src/bfs_cpu.c`, `src/sssp_cpu.c`
- OpenMP: `src/bfs_openmp.c`, `src/sssp_openmp.c`
- BFS GPU: `src/bfs_gpu.cu` (thread-per-vertex), `src/bfs_gpu_warp_per_v.cu`, `src/bfs_gpu_edge_based.cu`
- SSSP GPU: `src/sssp_gpu_t_per_v.cu`, `src/sssp_gpu_warp_per_v.cu`, `src/sssp_gpu_t_per_e.cu`
- Adaptive SSSP: `src/sssp_gpu_adaptive.cu`. ✅ The host builds `low_list` and `high_list` once (lines 111–118), and two separately sized kernels run each Bellman-Ford iteration. There's no per-thread degree branching.

### B. Build — ⚠️
- ✅ The Makefile includes every kernel, including `sssp_gpu_adaptive.o`.
- ⚠️ `make` wasn't re-run during this audit (no CUDA toolchain on the audit machine). All CPU sources compile cleanly with `-Wall -Wextra`. The adaptive results in the CSV show the full build worked on the lab machine.
- Binaries go to `build/` (not `bin/`).

### C. Correctness — ✅
- The harness (`test/uniform_benchmark.cu`, `verify_bfs` / `verify_sssp`) checks every strategy against the CPU baseline.
- All 151 CSV rows are `SUCCESS`, and every non-baseline row has `mismatches = 0`.
- Adaptive passes on Google+ (4/4 runs), LiveJournal (10/10) and roadNet (2/2).
- ⚠️ All edge weights are 1.0 (see the to-do list).

### D. Datasets
- ✅ LiveJournal: `graphs/soc-LiveJournal1.edgelist`, V = 4,847,571
- ✅ roadNet-CA: `graphs/roadNet-CA.edgelist`, V = 1,971,281 (⚠️ double edges)
- ⚠️ Google+: only on the lab machine. The CSV shows it loaded (V = 102,100, E = 24.2M).

### E. Harness + metrics
- ✅ `scripts/run_benchmarks.py` writes a raw CSV plus a summary CSV with mean ± std.
- ❌ Not enough samples per config (see the to-do list).
- ✅ H2D / compute / D2H are timed separately with CUDA events.
- ✅ `calc_imbalance.py` reports max per-thread work as the headline metric (re-run during this audit).
- ✅ `visualize_results.py` runs cleanly for all three graphs.
- ✅ The efficiency curve starts at 1.0 (checked on the LiveJournal plot).

### F. Results
- ✅ BFS and SSSP speedups for Google+, LiveJournal and roadNet
- ✅ LiveJournal imbalance; ❌ Google+ imbalance
- ✅ Adaptive result recorded (it matches or loses to warp on power-law graphs)
- ❌ Nsight Compute metrics (occupancy, DRAM bandwidth, warp efficiency)
- ❌ gprof CPU hotspot report

### G. Final report — ❌
The report isn't in the repo. The only document is `PCAP_Synopsis_Draft.pdf`. All sections are unverified: abstract, intro, objectives, literature review, methodology, setup + hardware specs, results, performance analysis, profiling, conclusion, IEEE references, figures.

### H. Deliverables
- ❌ Presentation deck
- ❌ Demo plan
- ✅ `main` is clean and in sync with `origin/main`
- ⚠️ 1 unmerged commit on `origin/feature/bfs-parallel-strategies`
