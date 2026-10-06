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
#   results/profiling/ncu_livejournal.txt        (full text dump, stdout+stderr)
#   results/profiling/ncu_metrics_summary.txt    (clean 3-metric table)
#   results/profiling/gprof_cpu_sequential.txt   (CPU hotspot report)

# NOTE: intentionally NO set -e here — ncu may return non-zero even on
# partial success (e.g. if some kernels are skipped). We check manually.

BINARY="./build/uniform_benchmark"
GRAPH="graphs/soc-LiveJournal1.edgelist"
OUT_DIR="results/profiling"
NCU_REPORT="${OUT_DIR}/ncu_livejournal"
NCU_TEXT="${OUT_DIR}/ncu_livejournal.txt"
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

# NOTE: GPU performance counter access must be unlocked before running this
# script. Run this ONCE in your terminal (lasts until reboot):
#   sudo sysctl -w dev.nvidia.NVreg_RestrictProfilingToAdminUsers=0
# After that, ncu works without sudo from any process including this script.

# ---------------------------------------------------------------------------
# Step 1: GPU profiling with Nsight Compute on LiveJournal
#
# Uses targeted --metrics (3 counters) instead of --set full:
#   sm__warps_active.avg.pct_of_peak_sustained_active  → Achieved Occupancy
#   dram__throughput.avg.pct_of_peak_sustained_elapsed → DRAM BW utilisation
#   smsp__thread_inst_executed_per_inst_executed.ratio → Warp efficiency
#   l1tex__t_bytes.sum.per_second                      → L1 throughput
#
# sudo is required for hardware performance counter access.
# Run "sudo -v" in your terminal first to cache credentials.
# OR run: sudo sysctl -w dev.nvidia.NVreg_RestrictProfilingToAdminUsers=0
# to unlock counters for all users until next reboot.
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

echo "[ncu] Running: sudo ncu --metrics ... on ${GRAPH}"
echo "[ncu] Output will be streamed to both terminal and ${NCU_TEXT}"
echo ""

# tee so we see progress on screen AND save to file
# stderr also goes to stdout (for PROF== lines) then into tee
ncu \
    --metrics "${NCU_METRICS}" \
    --force-overwrite \
    -o "${NCU_REPORT}" \
    "${BINARY}" \
        "${GRAPH}" \
        --algo sssp \
        --threads 1 \
        --source 0 \
        --run 1 \
        --repeat 1 \
    2>&1 | tee "${NCU_TEXT}"

NCU_EXIT=${PIPESTATUS[0]}

echo ""
if [ "${NCU_EXIT}" -ne 0 ]; then
    echo "[Warning] ncu exited with code ${NCU_EXIT} — check ${NCU_TEXT} for details."
    echo "          If you see ERR_NVGPUCTRPERM, run:"
    echo "            sudo sysctl -w dev.nvidia.NVreg_RestrictProfilingToAdminUsers=0"
    echo "          then re-run this script."
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
make CFLAGS="-Wall -Wextra -O3 -Iinclude -fopenmp -pg" \
     CXXFLAGS="-Wall -Wextra -O3 -Iinclude -fopenmp -pg" \
     NVCCFLAGS="-O3 -Iinclude -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -pg" \
     > /dev/null 2>&1

echo "[gprof] Running benchmark to generate gmon.out ..."
"${BINARY}" "${GRAPH}" --algo sssp --threads 1 --source 0 --run 1 --repeat 1 > /dev/null 2>&1

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
