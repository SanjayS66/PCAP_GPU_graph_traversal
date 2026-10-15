# Lab GPU Runbook

Everything left for the core project that needs a CUDA GPU. Run the steps in order from the repo root on the lab machine. Total time is about **1–1.5 hours**, mostly waiting.

Each step ends with a **✔ Check**. Don't move on until it passes.

---

## 0. Before leaving your laptop: push the script fixes

```bash
git add .gitignore Makefile scripts/ LAB_RUNBOOK.md PROJECT_STATUS.md
git commit -m "Fix profiling pipeline, add thread sweeps and --replace to benchmark runner, lab runbook"
git push
```

## 1. Get the code and check the datasets (~2 min)

```bash
git pull
ls -lh graphs/
```

✔ **Check:** all three files exist:
- `graphs/soc-LiveJournal1.edgelist`
- `graphs/gplus_combined.edgelist`
- `graphs/roadNet-CA.edgelist`

If `gplus_combined.edgelist` is missing but the `gplus/` folder (with the `*.edges` files) is there, run `python3 scripts/convert_gplus.py`.

## 2. Record hardware specs, for the report's experimental setup (~1 min)

```bash
mkdir -p results/profiling
{ echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  echo; echo "== CUDA =="; nvcc --version | tail -2
  echo; echo "== CPU =="; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz'
  echo; echo "== RAM =="; free -h | head -2
  echo; echo "== OS =="; uname -sr; gcc --version | head -1
} | tee results/profiling/hardware.txt
```

✔ **Check:** the GPU name and CPU model are both printed.

## 3. Clean build (~2 min)

```bash
make clean && make
```

✔ **Check:** there's no `error` in the output, and `ls build/` shows `uniform_benchmark`, `csr_convert`, `test_bfs_gpu` and `test_sssp`.

**Smoke test:** this runs on roadNet (~1 min):

```bash
make run
```

✔ **Check:** every row in the summary table shows `PASS` or `BASELINE`.

## 4. Final clean benchmark run (~25–35 min)

This replaces the old rows for these graphs with a fresh run: 5 repeats at 1, 2, 4 and 8 OpenMP threads, all on the current code.

```bash
# Power-law graphs: run with --undirected (the default)
python3 scripts/run_benchmarks.py \
    --graphs graphs/soc-LiveJournal1.edgelist graphs/gplus_combined.edgelist \
    --threads 1 2 4 8 --repeats 5 --replace

# roadNet: the file already lists both directions, so use --directed
python3 scripts/run_benchmarks.py \
    --graphs graphs/roadNet-CA.edgelist --directed \
    --threads 1 2 4 8 --repeats 5 --replace
```

Notes:
- Type the graph paths exactly as above. `--replace` matches rows by path.
- roadNet is now run `--directed` so each road edge is stored once. Previously every edge was stored twice, so expect roadNet times to drop.

✔ **Check:** the end of the second command prints
`[Verify] PASS: all N rows succeeded with 0 mismatches vs CPU baseline.`
If it prints `FAIL`, note the listed rows and don't commit the CSV.

## 5. Regenerate all plots (~1 min)

```bash
python3 scripts/visualize_results.py
```

✔ **Check:** `results/graphs/` has 4 PNGs for each of the 3 graphs (speedup BFS, speedup SSSP, OpenMP efficiency, OpenMP scaling). The efficiency curves start at 1.0.

## 6. Load-imbalance numbers (~3 min)

```bash
python3 scripts/calc_imbalance.py graphs/gplus_combined.edgelist     | tee results/profiling/imbalance_gplus.txt
python3 scripts/calc_imbalance.py graphs/soc-LiveJournal1.edgelist   | tee results/profiling/imbalance_livejournal.txt
python3 scripts/calc_imbalance.py --directed graphs/roadNet-CA.edgelist | tee results/profiling/imbalance_roadnet.txt
```

✔ **Check:**
- LiveJournal prints `22,889 → 716 → 1`.
- roadNet prints `12 → 1 → 1`.
- Write down the **Google+** numbers. They fill the `[TBD]` row in the report.

## 7. Nsight Compute + gprof profiling (~15–30 min)

```bash
bash scripts/profile_kernels.sh
```

If it fails with `ERR_NVGPUCTRPERM`, GPU counters are admin-only on this machine. Re-run it as root:

```bash
sudo env "PATH=$PATH" bash scripts/profile_kernels.sh
sudo chown -R "$USER" build results
```

✔ **Check:**
- `cat results/profiling/ncu_metrics_summary.txt` shows a table with occupancy, warp efficiency and DRAM bandwidth for thread-per-vertex, warp, edge, adaptive-low and adaptive-high. If it says `No kernel metrics found`, run `ncu --import results/profiling/ncu_livejournal.ncu-rep --page details | head -40` and send that output to the team.
- `head -30 results/profiling/gprof_cpu_sequential.txt` shows a flat profile. The top entries should include `sssp_bellman_ford_cpu` and the CSR builder.

## 8. Commit and push (~1 min)

```bash
git add results/ .gitignore
git status            # should list results/csv/*.csv, results/graphs/*.png, results/profiling/*.txt
git commit -m "Final benchmark run (5 reps, 1-8 threads), roadNet directed, Nsight + gprof profiles, imbalance numbers, hardware specs"
git pull --no-rebase
git push
```

The `.ncu-rep` binary stays uncommitted on purpose because it's large. Copy it to a USB stick if you want to open it in the Nsight UI later.

✔ **Check:** `git status` is clean, and `git log origin/main -1` shows your commit.

---

## Optional: if time is left

**A. Merge the BFS thread-per-edge branch.** It deletes `test/test_bfs_gpu.c`, which the Makefile still builds, so fix the Makefile in the same change:

```bash
git merge origin/feature/bfs-parallel-strategies
# then in Makefile: point BFS_TEST_BIN at test/test_bfs_all.cu and add build/bfs_gpu_t_per_e.o
make clean && make
```

Only commit if `make` succeeds and the test passes. Otherwise `git merge --abort`.

**B. Random edge weights.** All weights are currently `1.0`, which makes SSSP equivalent to BFS. The default plan is to say so in the report. To use real weights instead, you'd change both converters to write random weights, regenerate all three `.edgelist` files and redo step 4. That's about another 45 min, so only do it if the team agrees.

---

## What to send to the team afterwards

1. The `[Verify] PASS` line from step 4
2. `results/profiling/hardware.txt`
3. `results/profiling/ncu_metrics_summary.txt`
4. The Google+ imbalance numbers from step 6
5. A message confirming you pushed
