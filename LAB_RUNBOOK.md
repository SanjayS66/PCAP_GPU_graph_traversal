# Lab GPU Runbook

Everything left for the core project that needs a CUDA GPU (an **NVIDIA** GPU on Linux). Run the steps in order, in a terminal on the lab machine. Total time is about **1.5–2 hours**, mostly waiting (including ~20 min to download the datasets).

Each step ends with a **✔ Check**. Don't move on until it passes.

**The flow:** clone the public repo (no login or SSH key needed) → download the datasets → build and run → zip `results/` and email it to yourself → wipe the lab computer → push from your laptop (step 10). Nothing is ever pushed from the lab computer.

---

## 0. Check the machine and get the code (~5 min)

**a) Check the tools.** All of these must print a version or path:

```bash
nvidia-smi                 # shows an NVIDIA GPU
nvcc --version             # CUDA compiler
which ncu                  # Nsight Compute (only needed for step 7)
make --version | head -1
gcc --version | head -1
python3 --version
python3 -c "import pandas, matplotlib; print('plot libs OK')"
df -h ~                    # need ~5 GB free
```

- If the `pandas`/`matplotlib` line fails, run `pip3 install --user pandas matplotlib`.
- If `nvcc` isn't found but `/usr/local/cuda/bin` exists, run `export PATH=/usr/local/cuda/bin:$PATH` and try again.
- If `ncu` is missing, you can still do every step except the Nsight part of step 7.

**b) Get the code.** The repo is public, so HTTPS clone needs **no GitHub login and no SSH key**:

```bash
cd ~
git clone https://github.com/SanjayS66/PCAP_GPU_graph_traversal.git
cd PCAP_GPU_graph_traversal
```

If `git` isn't installed, open https://github.com/SanjayS66/PCAP_GPU_graph_traversal in the browser, click **Code → Download ZIP** (no login needed), unzip it, and `cd` into the folder.

✔ **Check:** `ls` shows `Makefile`, `src/`, `scripts/` and `LAB_RUNBOOK.md`.

**Every command below runs from inside this `PCAP_GPU_graph_traversal` folder.**

## 1. Download and convert the datasets (~15–20 min)

The `graphs/` folder is **not on GitHub**. It's gitignored because GitHub rejects files over 100 MB, and LiveJournal alone is 1.3 GB. Rebuild it from SNAP with the repo's own conversion scripts.

Skip any graph whose `.edgelist` already exists in `graphs/`.

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

head -1 graphs/*.edgelist
```

If `wget` isn't installed, use `curl -LO <url>` instead.

**Slow internet?** Bring `soc-LiveJournal1.txt.gz` and `roadNet-CA.txt.gz` (266 MB together; they're in the repo folder on Diya's laptop) on a USB stick and skip those two `wget` lines. Google+ always has to be downloaded.

✔ **Check:** `head -1` prints these exact values:

| File | First line |
|---|---|
| `graphs/soc-LiveJournal1.edgelist` | `4847571 68993773` |
| `graphs/roadNet-CA.edgelist` | `1971281 5533214` |
| `graphs/gplus_combined.edgelist` | `102100 12113501` |

The raw downloads (`*.txt`, `*.gz`, `gplus/`) are gitignored, so they won't get committed by accident.

**Note:** `convert_gplus.py` doesn't read the input files in sorted order, so on Linux it may number vertices differently than the original run did. The source vertex 0 can then be a different user and Google+ times may shift slightly. That's fine, because step 4 replaces all the old rows.

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

**No `sudo` on the lab account?** Then Nsight can't run here; ask the lab in-charge, or do the Nsight part on Google Colab later. The script still continues to gprof, so `gprof_cpu_sequential.txt` gets made anyway. Carry on with step 8.

✔ **Check:**
- `cat results/profiling/ncu_metrics_summary.txt` shows a table with occupancy, warp efficiency and DRAM bandwidth for thread-per-vertex, warp, edge, adaptive-low and adaptive-high. If it says `No kernel metrics found`, run `ncu --import results/profiling/ncu_livejournal.ncu-rep --page details | head -40` and send that output to the team.
- `head -30 results/profiling/gprof_cpu_sequential.txt` shows a flat profile. The top entries should include `sssp_bellman_ford_cpu` and the CSR builder.

## 8. Pack the results and email them to yourself (~3 min)

You can't push from the lab computer (no GitHub login there), so the results travel by email and get pushed from Diya's laptop. Only `results/` needs to come back; it's a few MB.

```bash
zip -r lab_results.zip results/ -x '*.ncu-rep'
ls -lh lab_results.zip        # should be a few MB, under Gmail's 25 MB limit
```

If `zip` isn't installed: `tar -czf lab_results.tar.gz --exclude='*.ncu-rep' results/`

Open Gmail in the browser and email the file to yourself. A USB stick or a Google Drive upload works too.

The zip holds **only** these files. Nothing else from the lab machine is needed: no code, no datasets, no `build/`.

| Folder | Files | What they are |
|---|---|---|
| `results/csv/` | `results.csv`, `results_summary.csv` | All benchmark data |
| `results/graphs/` | 12 `.png` files | Speedup, efficiency and scaling plots for each graph |
| `results/profiling/` | `hardware.txt` | GPU/CPU specs for the report |
| | `imbalance_gplus.txt`, `imbalance_livejournal.txt`, `imbalance_roadnet.txt` | Load-imbalance numbers |
| | `ncu_metrics_summary.txt`, `ncu_livejournal.txt`, `ncu_run_log.txt` | Nsight results (missing if step 7's Nsight part couldn't run) |
| | `gprof_cpu_sequential.txt` | CPU profile |

Check the zip's contents with `unzip -l lab_results.zip` (or `tar -tzf lab_results.tar.gz`).

The `.ncu-rep` Nsight binary is left out on purpose because it's large. Copy it to a USB stick if you want to open it in the Nsight UI later.

✔ **Check:** the email arrived with the attachment. Open it on your phone or laptop before going on to step 9.

## 9. Clean up the lab computer (~2 min)

Only do this after the email has arrived.

```bash
cd ~
rm -rf ~/PCAP_GPU_graph_traversal           # repo + graphs + downloads + the zip, ~4 GB
rm -f ~/Downloads/PCAP_GPU_graph_traversal*  # only if you used "Download ZIP" in step 0
git config --global --list                  # if your name/email shows up:
git config --global --unset user.name
git config --global --unset user.email
rm -f ~/.git-credentials                    # only exists if a GitHub password was typed
```

Then sign out of Gmail, remove your account from the browser if it offered to remember it, and close all browser windows.

---

## 10. Back on the laptop: unpack, verify, push (~5 min)

Download `lab_results.zip` from Gmail into your `Downloads` folder, then in Git Bash:

```bash
cd "/d/Documents/COLLEGE/SEM5/PCAP LAB/PROJECT/PCAP_GPU_graph_traversal"
git pull
python -m zipfile -e ~/Downloads/lab_results.zip .    # extracts into results/, replacing old files
python scripts/run_benchmarks.py --analyze-only
```

If you got the `.tar.gz` instead, use `tar -xzf ~/Downloads/lab_results.tar.gz` in place of the zipfile line.

✔ **Check:** the last command prints `[Verify] PASS: all N rows succeeded with 0 mismatches`. If it says `FAIL`, don't push; look at the listed rows first.

```bash
git add results/
git status    # should list results/csv/*.csv, results/graphs/*.png, results/profiling/*.txt
git commit -m "Final benchmark run (5 reps, 1-8 threads), roadNet directed, Nsight + gprof profiles, imbalance numbers, hardware specs"
git push
```

Then update [PROJECT_STATUS.md](PROJECT_STATUS.md):
- Tick the lab items.
- Replace the "Key results" table with the numbers from the new `results/csv/results_summary.csv`.
- Fill in the Google+ imbalance row from `results/profiling/imbalance_gplus.txt`.

(Claude can do all of step 10 if you give it the path to the zip.)

---

## Optional: if time is left

**A. Merge the BFS thread-per-edge branch.** It deletes `test/test_bfs_gpu.c`, which the Makefile still builds, so fix the Makefile in the same change:

```bash
git merge origin/feature/bfs-parallel-strategies
# then in Makefile: point BFS_TEST_BIN at test/test_bfs_all.cu and add build/bfs_gpu_t_per_e.o
make clean && make
```

Only keep it if `make` succeeds and the test passes; otherwise `git merge --abort`. Because the lab can't push, this needs someone with push access on a GPU machine. Otherwise leave it for the team to decide.

**B. Random edge weights.** All weights are currently `1.0`, which makes SSSP equivalent to BFS. The default plan is to say so in the report. To use real weights instead, you'd change both converters to write random weights, regenerate all three `.edgelist` files and redo step 4. That's about another 45 min, so only do it if the team agrees.

---

## What to send to the team afterwards

1. The `[Verify] PASS` line from step 4
2. `results/profiling/hardware.txt`
3. `results/profiling/ncu_metrics_summary.txt`
4. The Google+ imbalance numbers from step 6
5. A message confirming the results are pushed (step 10)
