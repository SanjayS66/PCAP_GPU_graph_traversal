# GPU Runbook (for Sanjay)

Everything left for the core project that needs a CUDA GPU. Sanjay runs it on his own Linux machine with an **NVIDIA** GPU and **pushes the results himself**. Run the steps in order, in a terminal. Total time is about **1–1.5 hours**, mostly waiting; add ~20 min if the datasets need downloading.

Each step ends with a **✔ Check**. Don't move on until it passes.

**The flow:** update the repo → make sure the 3 datasets are there → build → benchmark → plots → imbalance numbers → profiling → commit and push `results/`.

> **Why a full re-run?** Your last commit (`98b5291`) redrew the plots from the *old* CSV. `results.csv` hasn't changed since early October: Google+ still has only 1 run per config, there's no profiling output, and roadNet has no thread sweep. This run fixes all of that.

---

## 0. Check the machine and update the repo (~5 min)

**a) Check the tools.** All of these must print a version or path:

```bash
nvidia-smi                 # NVIDIA GPU; memory column should say 4 GB+ (4096MiB+)
nvcc --version             # CUDA compiler (the driver alone isn't enough)
which ncu                  # Nsight Compute (only needed for step 7)
make --version | head -1
gcc --version | head -1
python3 --version
python3 -c "import pandas, matplotlib; print('plot libs OK')"
nproc                      # CPU cores: 8+ so the 8-thread OpenMP runs mean something
free -h                    # RAM: 8 GB+ total
df -h ~                    # disk: ~5 GB free if the datasets need downloading
```

- If the `pandas`/`matplotlib` line fails, run `pip3 install --user pandas matplotlib`.
- If `nvcc` isn't found but `/usr/local/cuda/bin` exists, run `export PATH=/usr/local/cuda/bin:$PATH` and try again.
- If `ncu` is missing, you can still do every step except the Nsight part of step 7.
- With under 4 GB of GPU memory, LiveJournal's edge-based kernel may run out of memory. With fewer than 8 cores, everything still works, but tell the team: the 8-thread OpenMP numbers won't be meaningful.

**b) Update your existing clone:**

```bash
cd <your>/PCAP_GPU_graph_traversal
git status
```

- If `git status` lists modified files **under `results/`** only, they're from earlier runs and will be regenerated, so discard them: `git checkout -- results/`
- If you have files in `results/profiling/` from an earlier profiling attempt, copy them somewhere safe first (`cp -r results/profiling ~/old_profiling`). The old script had bugs, so they're probably incomplete anyway.
- If it lists changes to **code** (`src/`, `test/`, `scripts/`, `Makefile`), stop and sort those out first. Don't discard them.

```bash
git checkout main
git pull
```

No clone on this machine? Run `git clone https://github.com/SanjayS66/PCAP_GPU_graph_traversal.git` and `cd` into it.

✔ **Check:** `git log --oneline -1` shows the latest commit on GitHub, and `ls` shows `LAB_RUNBOOK.md`.

**Every command below runs from inside the `PCAP_GPU_graph_traversal` folder.**

## 1. Make sure the datasets are there (~0–20 min)

The `graphs/` folder isn't on GitHub, because LiveJournal alone is 1.3 GB. If your machine already has the files from earlier runs, just check them:

```bash
ls -lh graphs/
head -1 graphs/*.edgelist
```

✔ **Check:** all three files exist and their first lines match:

| File | First line |
|---|---|
| `graphs/soc-LiveJournal1.edgelist` | `4847571 68993773` |
| `graphs/roadNet-CA.edgelist` | `1971281 5533214` |
| `graphs/gplus_combined.edgelist` | `102100 12113501` |

**Missing any?** Download and convert only those:

```bash
mkdir -p graphs

# LiveJournal (~250 MB download, ~3-5 min to convert)
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
gunzip soc-LiveJournal1.txt.gz
python3 scripts/convert_snap.py soc-LiveJournal1.txt graphs/soc-LiveJournal1.edgelist

# roadNet (~18 MB)
wget https://snap.stanford.edu/data/roadNet-CA.txt.gz
gunzip roadNet-CA.txt.gz
python3 scripts/convert_snap.py roadNet-CA.txt graphs/roadNet-CA.edgelist

# Google+ (~400 MB; unpacks into a gplus/ folder, which the script reads)
wget https://snap.stanford.edu/data/gplus.tar.gz
tar -xzf gplus.tar.gz
python3 scripts/convert_gplus.py
```

If `wget` isn't installed, use `curl -LO <url>` instead. The raw downloads (`*.txt`, `*.gz`, `gplus/`) are gitignored, so they won't get committed by accident.

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

This replaces the old rows for these graphs with a fresh run: 5 repeats at 1, 2, 4 and 8 OpenMP threads, all on the current code and this machine.

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
- roadNet is now run `--directed`, so each road edge is stored once. Previously every edge was stored twice, so expect roadNet times to drop.

✔ **Check:** the end of the second command prints
`[Verify] PASS: all N rows succeeded with 0 mismatches vs CPU baseline.`
If it prints `FAIL`, note the listed rows, don't push, and tell the team.

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
- The **Google+** numbers are new. They fill the `[TBD]` row in the report.

## 7. Nsight Compute + gprof profiling (~15–30 min)

```bash
bash scripts/profile_kernels.sh
```

If it fails with `ERR_NVGPUCTRPERM`, GPU counters are admin-only on this machine. Re-run it as root:

```bash
sudo env "PATH=$PATH" bash scripts/profile_kernels.sh
sudo chown -R "$USER" build results
```

No `sudo`? Then Nsight can't run here. The script still continues to gprof, so `gprof_cpu_sequential.txt` gets made anyway. Carry on with step 8 and tell the team Nsight is still missing.

✔ **Check:**
- `cat results/profiling/ncu_metrics_summary.txt` shows a table with occupancy, warp efficiency and DRAM bandwidth for thread-per-vertex, warp, edge, adaptive-low and adaptive-high. If it says `No kernel metrics found`, run `ncu --import results/profiling/ncu_livejournal.ncu-rep --page details | head -40` and send that output to the team.
- `head -30 results/profiling/gprof_cpu_sequential.txt` shows a flat profile. The top entries should include `sssp_bellman_ford_cpu` and the CSR builder.

## 8. Commit and push (~2 min)

```bash
python3 scripts/run_benchmarks.py --analyze-only   # final check: must print [Verify] PASS
git add results/
git status
```

✔ **Check:** `git status` lists only files under `results/`:
- `results/csv/results.csv` and `results_summary.csv`
- 12 PNGs in `results/graphs/`
- `results/profiling/*.txt`: hardware, 3 imbalance files, Nsight summary/dump/log, gprof

The `.ncu-rep` Nsight binary is **not** listed; it's ignored on purpose because it's large. Keep it locally if you want to open it in the Nsight UI.

```bash
git commit -m "Final benchmark run (5 reps, 1-8 threads), roadNet directed, Nsight + gprof profiles, imbalance numbers, hardware specs"
git pull --no-rebase
git push
```

✔ **Check:** `git status` says `up to date with 'origin/main'`, and the commit shows on GitHub.

**Optional cleanup:** to free ~1.5 GB, delete the raw downloads but keep `graphs/` for any future re-runs:
`rm -f soc-LiveJournal1.txt roadNet-CA.txt *.txt.gz gplus.tar.gz && rm -rf gplus/`

---

## Optional: if time is left

**A. Merge the BFS thread-per-edge branch** (`origin/feature/bfs-parallel-strategies`, your commit `60cfc1b`). It deletes `test/test_bfs_gpu.c`, which the Makefile still builds, so fix the Makefile in the same change:

```bash
git merge origin/feature/bfs-parallel-strategies
# then in Makefile: point BFS_TEST_BIN at test/test_bfs_all.cu and add build/bfs_gpu_t_per_e.o
make clean && make
```

Only commit and push if `make` succeeds and the test passes; otherwise `git merge --abort`. If it does go in, BFS gets a 4th strategy, and the benchmark harness would need wiring for it before step 4 is re-run. Check with the team before starting this.

**B. Random edge weights.** All weights are currently `1.0`, which makes SSSP equivalent to BFS. The default plan is to say so in the report. To use real weights, you'd change both converters to write random weights, regenerate all three `.edgelist` files and redo steps 4–8. That's about another 45 min, so only do it if the team agrees.

---

## After pushing, post in the team chat

1. The `[Verify] PASS` line from step 4
2. Your GPU and CPU (from `results/profiling/hardware.txt`)
3. The Google+ imbalance numbers from step 6
4. Whether Nsight ran (or why not)
5. "Pushed" with the commit hash

Diya will then update `PROJECT_STATUS.md` with the new numbers and start the report and deck from them.
