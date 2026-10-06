#!/usr/bin/env python3
"""
scripts/parse_ncu_metrics.py

Reads the text dump produced by:
    sudo ncu --metrics sm__warps_active...,dram__throughput...,
             smsp__thread_inst_executed_per_inst_executed.ratio,
             l1tex__t_bytes.sum.per_second ... > ncu_livejournal.txt

and extracts the three metrics the PCAP report needs for each SSSP kernel:
    1. Achieved Occupancy       (sm__warps_active %)
    2. DRAM Bandwidth %         (dram__throughput %)
    3. Warp Efficiency          (smsp__thread_inst_executed_per_inst_executed.ratio)
       = fraction of threads active per instruction (1.0 = all 32 lanes active)

Usage:
    python3 scripts/parse_ncu_metrics.py results/profiling/ncu_livejournal.txt
    python3 scripts/parse_ncu_metrics.py results/profiling/ncu_livejournal.txt > results/profiling/ncu_metrics_summary.txt
"""

import sys
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Kernel name → friendly strategy label
# ---------------------------------------------------------------------------
KERNEL_LABELS = {
    "relax_kernel":              "Thread-per-vertex",
    "relax_kernel_warp_per_v":   "Warp-per-vertex",
    "relax_kernel_t_per_e":      "Thread-per-edge",
    "sssp_adaptive_low_kernel":  "Adaptive (Low-deg)",
    "sssp_adaptive_high_kernel": "Adaptive (High-deg)",
}

# Metric patterns to look for in the ncu text output.
# ncu with --metrics prints lines like:
#   sm__warps_active.avg.pct_of_peak_sustained_active           %    69.56
METRIC_PATTERNS = {
    "occ":  re.compile(r"sm__warps_active.*?pct.*?peak.*?\s+([\d,]+\.?\d*)\s*$"),
    "bw":   re.compile(r"dram__throughput.*?pct.*?peak.*?\s+([\d,]+\.?\d*)\s*$"),
    "eff":  re.compile(r"smsp__thread_inst_executed_per_inst_executed.*?ratio.*?\s+([\d,]+\.?\d*)\s*$"),
    "l1bw": re.compile(r"l1tex__t_bytes.*?per_second.*?\s+([\d,]+\.?\d*)\s*$"),
}

def parse_kernel_name(line):
    """Return the base kernel name from an ncu section header, or None."""
    for k in KERNEL_LABELS:
        if k + "(" in line:
            return k
    return None

def parse_ncu_text(path):
    """
    Walk through the ncu text dump (targeted --metrics output).
    For each kernel instance collect the 3 key metrics.
    Returns: dict  kernel_name -> list of metric dicts
    """
    results = defaultdict(list)

    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[Error] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    current_kernel = None
    current_metrics = {}

    def flush():
        nonlocal current_kernel, current_metrics
        if current_kernel and any(v is not None for v in current_metrics.values()):
            results[current_kernel].append(dict(current_metrics))
        current_kernel = None
        current_metrics = {}

    for line in lines:
        stripped = line.strip()

        # New kernel section
        kname = parse_kernel_name(stripped)
        if kname is not None:
            flush()
            current_kernel = kname
            current_metrics = {k: None for k in METRIC_PATTERNS}
            continue

        if current_kernel is None:
            continue

        # Try each metric pattern
        for mkey, pat in METRIC_PATTERNS.items():
            m = pat.search(stripped)
            if m and current_metrics[mkey] is None:
                current_metrics[mkey] = float(m.group(1).replace(",", ""))
                break

    flush()
    return results

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def fmt(v, suffix=""):
    if v is None:
        return "N/A"
    return f"{v:.2f}{suffix}"

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ncu_text_dump.txt>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    raw = parse_ncu_text(path)

    if not raw:
        print("[Warning] No kernel metrics found in the ncu dump.")
        print("          Make sure it's a valid 'ncu --metrics ...' text output.")
        sys.exit(0)

    # Average over all profiled rounds per kernel
    summary = {}
    for kname, runs in raw.items():
        summary[kname] = {
            "label":   KERNEL_LABELS.get(kname, kname),
            "samples": len(runs),
            "occ":  mean([r["occ"]  for r in runs]),
            "bw":   mean([r["bw"]   for r in runs]),
            "eff":  mean([r["eff"]  for r in runs]),
            "l1bw": mean([r["l1bw"] for r in runs]),
        }

    # Print table
    order = [
        "relax_kernel",
        "relax_kernel_warp_per_v",
        "relax_kernel_t_per_e",
        "sssp_adaptive_low_kernel",
        "sssp_adaptive_high_kernel",
    ]

    COL = [26, 14, 20, 14, 10]
    headers = ["Strategy", "Occupancy (%)", "Warp Efficiency", "DRAM BW (%)", "Samples"]

    sep = "+" + "+".join("-" * w for w in COL) + "+"
    def row(*cells):
        return "|" + "|".join(f" {str(c):<{w-2}} " for c, w in zip(cells, COL)) + "|"

    print()
    print("  Nsight Compute — SSSP Kernel Profiling Summary")
    print("  Graph: soc-LiveJournal1  |  Averaged over all profiled rounds")
    print()
    print(sep)
    print(row(*headers))
    print(sep)

    for kname in order:
        if kname not in summary:
            continue
        s = summary[kname]
        # Warp efficiency: ratio in ncu is threads_executed / (32 * instructions)
        # Display as percentage (multiply by 100 if <= 1.0, else it's already %)
        eff = s["eff"]
        if eff is not None:
            eff_pct = eff * 100 if eff <= 1.0 else eff
            eff_str = f"{eff_pct:.1f}%"
        else:
            eff_str = "N/A"

        print(row(
            s["label"],
            fmt(s["occ"], "%"),
            eff_str,
            fmt(s["bw"], "%"),
            str(s["samples"]),
        ))

    print(sep)
    print()
    print("  Metric definitions:")
    print("    Occupancy    : sm__warps_active % of peak — fraction of SM warp slots in use")
    print("    Warp Eff.    : smsp__thread_inst_executed_per_inst_executed — fraction of the")
    print("                   32 lanes active per instruction (100% = all lanes busy)")
    print("    DRAM BW      : dram__throughput % of peak — memory bus saturation")
    print()

    # Interpretation
    t_occ = summary.get("relax_kernel",            {}).get("occ")
    w_occ = summary.get("relax_kernel_warp_per_v", {}).get("occ")
    t_eff = summary.get("relax_kernel",            {}).get("eff")
    w_eff = summary.get("relax_kernel_warp_per_v", {}).get("eff")

    print("  Report interpretation:")
    if t_eff is not None and w_eff is not None:
        te = t_eff * 100 if t_eff <= 1.0 else t_eff
        we = w_eff * 100 if w_eff <= 1.0 else w_eff
        if we > te:
            print(f"    ✓ Warp-per-vertex has HIGHER warp efficiency ({we:.1f}% vs {te:.1f}%)")
            print(f"      → hub vertices keep all 32 lanes busy; thread-per-vertex has idle lanes")
        else:
            print(f"    ~ Warp efficiency: thread-per-v={te:.1f}%, warp-per-v={we:.1f}%")
    if t_occ is not None and w_occ is not None:
        print(f"    ✓ Occupancy: thread-per-v={t_occ:.1f}%, warp-per-v={w_occ:.1f}%")
    print()

if __name__ == "__main__":
    main()
