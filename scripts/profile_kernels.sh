#!/bin/bash
# scripts/profile_kernels.sh
#
# Task 2: Nsight Compute + gprof profiling for the PCAP GPU report.
#
# Targets: LiveJournal (power-law, the graph that matters most for the report).
# Profiles all four SSSP strategies: thread-per-vertex, warp-per-vertex,
#   thread-per-edge, and the new pre-partitioned adaptive kernel.
#
# Outputs:
#   results/profiling/ncu_livejournal.ncu-rep   (binary, open with Nsight UI)
#   results/profiling/ncu_livejournal.txt        (per-kernel metric dump, via ncu --import)
#   results/profiling/ncu_run_log.txt            (benchmark stdout+stderr during profiling)
#   results/profiling/ncu_metrics_summary.txt    (clean 3-metric table)
#   results/profiling/gprof_cpu_sequential.txt   (CPU hotspot report)

# NOTE: intentionally NO set -e here — ncu may return non-zero even on
# partial success (e.g. if some kernels are skipped). We check manually.

BINARY="./build/uniform_benchmark"
GRAPH="graphs/soc-LiveJournal1.edgelist"
# Must match how run_benchmarks.py ran this graph, or we profile a different CSR
GRAPH_FLAGS="--undirected"
OUT_DIR="results/profiling"
NCU_REPORT="${OUT_DIR}/ncu_livejournal"
NCU_TEXT="${OUT_DIR}/ncu_livejournal.txt"
NCU_LOG="${OUT_DIR}/ncu_run_log.txt"
METRICS_OUT="${OUT_DIR}/ncu_metrics_summary.txt"
GPROF_OUT="${OUT_DIR}/gprof_cpu_sequential.txt"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -f "${BINARY}" ]; then
    echo "[Error] Binary ${BINARY} not found. Run 'make' first."
    exit 1
fi
if [ ! -f "${GRAPH}" ]; then
    echo "[Error] Graph ${GRAPH} not found."
    exit 1
fi
if ! command -v ncu &>/dev/null; then
    echo "[Error] 'ncu' (Nsight Compute) not found on PATH."
    exit 1
fi

# NOTE: GPU performance counters are admin-only by default on Linux.
# NVreg_RestrictProfilingToAdminUsers is a kernel-module parameter, NOT a
# sysctl, so it cannot be flipped at runtime. If ncu reports ERR_NVGPUCTRPERM,
# re-run the whole script as root, keeping your PATH so nvcc/ncu are found:
#   sudo env "PATH=$PATH" bash scripts/profile_kernels.sh
#   sudo chown -R "$USER" build results     # give the outputs back to you

# ---------------------------------------------------------------------------
# Step 1: GPU profiling with Nsight Compute on LiveJournal
#
# Uses targeted --metrics (3 counters) instead of --set full:
#   sm__warps_active.avg.pct_of_peak_sustained_active  → Achieved Occupancy
#   dram__throughput.avg.pct_of_peak_sustained_elapsed → DRAM BW utilisation
#   smsp__thread_inst_executed_per_inst_executed.ratio → Warp efficiency
#   l1tex__t_bytes.sum.per_second                      → L1 throughput
#
# See the NOTE above if counter access is denied (ERR_NVGPUCTRPERM).
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 1: Nsight Compute (ncu) — LiveJournal SSSP, all strategies"
echo "================================================================"
echo ""

NCU_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active"
NCU_METRICS+=",dram__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS+=",smsp__thread_inst_executed_per_inst_executed.ratio"
NCU_METRICS+=",l1tex__t_bytes.sum.per_second"

echo "[ncu] Running: ncu --metrics ... on ${GRAPH} ${GRAPH_FLAGS}"
echo "[ncu] Program output will be streamed to both terminal and ${NCU_LOG}"
echo ""

# tee so we see progress on screen AND save to file
# stderr also goes to stdout (for PROF== lines) then into tee
ncu \
    --metrics "${NCU_METRICS}" \
    --force-overwrite \
    -o "${NCU_REPORT}" \
    "${BINARY}" \
        "${GRAPH}" \
        ${GRAPH_FLAGS} \
        --algo sssp \
        --threads 1 \
        --source 0 \
        --run 1 \
        --repeat 1 \
    2>&1 | tee "${NCU_LOG}"

NCU_EXIT=${PIPESTATUS[0]}

# With -o, ncu writes metrics only into the binary report (nothing on stdout),
# so dump the per-kernel metric tables to text for parse_ncu_metrics.py.
if [ -f "${NCU_REPORT}.ncu-rep" ]; then
    ncu --import "${NCU_REPORT}.ncu-rep" --page details > "${NCU_TEXT}"
fi

echo ""
if [ "${NCU_EXIT}" -ne 0 ]; then
    echo "[Warning] ncu exited with code ${NCU_EXIT} — check ${NCU_LOG} for details."
    echo "          If you see ERR_NVGPUCTRPERM, re-run as root:"
    echo "            sudo env \"PATH=\$PATH\" bash scripts/profile_kernels.sh"
    echo "            sudo chown -R \"\$USER\" build results"
else
    echo "[ncu] Done."
fi
echo "[ncu] Binary report : ${NCU_REPORT}.ncu-rep"
echo "[ncu] Text dump     : ${NCU_TEXT}"

# ---------------------------------------------------------------------------
# Step 2: Extract 3 key metrics from the ncu text dump
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 2: Extracting key metrics → ${METRICS_OUT}"
echo "================================================================"
echo ""

if [ -f "${NCU_TEXT}" ] && [ -s "${NCU_TEXT}" ]; then
    python3 scripts/parse_ncu_metrics.py "${NCU_TEXT}" | tee "${METRICS_OUT}"
else
    echo "[Warning] ${NCU_TEXT} is empty or missing — skipping metric extraction."
fi

# ---------------------------------------------------------------------------
# Step 3: CPU profiling with gprof (sequential baseline)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 3: gprof CPU profiling (sequential SSSP on LiveJournal)"
echo "================================================================"
echo ""

echo "[gprof] Recompiling with -pg ..."
make clean > /dev/null 2>&1
# nvcc's own -pg (--profile) instruments host code AND links the gprof runtime;
# -Xcompiler -pg alone skips the link step, so no gmon.out would be written.
# Build errors are left visible on stderr.
make CFLAGS="-Wall -Wextra -O3 -Iinclude -fopenmp -pg" \
     CXXFLAGS="-Wall -Wextra -O3 -Iinclude -fopenmp -pg" \
     NVCCFLAGS="-O3 -Iinclude -Xcompiler -fopenmp -Xcompiler -Wall -pg" \
     > /dev/null

echo "[gprof] Running benchmark to generate gmon.out ..."
"${BINARY}" "${GRAPH}" ${GRAPH_FLAGS} --algo sssp --threads 1 --source 0 --run 1 --repeat 1 > /dev/null 2>&1

if [ -f gmon.out ]; then
    echo "[gprof] Extracting report → ${GPROF_OUT}"
    gprof "${BINARY}" gmon.out > "${GPROF_OUT}"
    rm -f gmon.out
    echo "[gprof] Done."
else
    echo "[Warning] gmon.out not found — gprof step skipped."
fi

# Recompile without -pg for normal use
echo "[build] Recompiling without -pg ..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1
echo "[build] Done."

echo ""
echo "================================================================"
echo " All profiling done!"
echo "  Nsight report : ${NCU_REPORT}.ncu-rep"
echo "  Nsight text   : ${NCU_TEXT}"
echo "  3-metric table: ${METRICS_OUT}"
echo "  gprof report  : ${GPROF_OUT}"
echo "================================================================"
