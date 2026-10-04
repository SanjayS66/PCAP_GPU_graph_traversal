#!/usr/bin/env python3
"""
Automated Benchmarking and CSV Analysis Runner for PCAP GPU Graph Traversal
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
import statistics

def build_benchmark_if_needed(binary_path):
    if not os.path.exists(binary_path):
        print(f"[Build] Binary {binary_path} not found. Running 'make'...")
        res = subprocess.run(["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"[Error] Build failed:\n{res.stderr}", file=sys.stderr)
            sys.exit(1)
        print("[Build] Build succeeded.")

def run_single_benchmark(binary_path, graph_path, source, undirected, algo, threads, csv_path, repeats, start_run_id):
    cmd = [
        binary_path,
        graph_path,
        "--source", str(source),
        "--algo", algo,
        "--threads", str(threads),
        "--csv", csv_path,
        "--repeat", str(repeats),
        "--run", str(start_run_id)
    ]
    if undirected:
        cmd.append("--undirected")

    print(f"\n[Run] Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[Warning] Benchmark returned non-zero exit code: {res.returncode}", file=sys.stderr)
    return res.returncode == 0

def parse_results_csv(csv_path):
    if not os.path.exists(csv_path):
        return []
    records = []
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rec = {
                    "graph_file": row["graph_file"],
                    "V": int(row["V"]),
                    "E": int(row["E"]),
                    "threads": int(row["threads"]),
                    "run": int(row.get("run", 1)),
                    "algorithm": row["algorithm"],
                    "strategy": row["strategy"],
                    "status": row["status"],
                    "h2d_ms": float(row["h2d_ms"]),
                    "compute_ms": float(row["compute_ms"]),
                    "d2h_ms": float(row["d2h_ms"]),
                    "total_ms": float(row["total_ms"]),
                    "rounds": int(row["rounds"]),
                    "speedup": float(row["speedup"]),
                    "speedup_vs_omp": float(row["speedup_vs_omp"]),
                    "mteps": float(row["mteps"]),
                    "mismatches": int(row["mismatches"]),
                }
                records.append(rec)
            except (ValueError, KeyError) as e:
                continue
    return records

def generate_summary(records, summary_csv_path):
    """
    Groups records by (graph_file, algorithm, strategy, threads)
    and computes mean ± std for compute_ms, total_ms, speedup, and mteps.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["graph_file"], r["V"], r["E"], r["algorithm"], r["strategy"], r["threads"])
        groups[key].append(r)

    summary_rows = []
    for (graph, V, E, algo, strat, threads), rows in sorted(groups.items()):
        computes = [r["compute_ms"] for r in rows]
        totals = [r["total_ms"] for r in rows]
        speedups = [r["speedup"] for r in rows]
        mteps_vals = [r["mteps"] for r in rows]

        mean_comp = statistics.mean(computes)
        std_comp = statistics.stdev(computes) if len(computes) > 1 else 0.0

        mean_tot = statistics.mean(totals)
        std_tot = statistics.stdev(totals) if len(totals) > 1 else 0.0

        mean_sp = statistics.mean(speedups)
        std_sp = statistics.stdev(speedups) if len(speedups) > 1 else 0.0

        mean_mteps = statistics.mean(mteps_vals)
        std_mteps = statistics.stdev(mteps_vals) if len(mteps_vals) > 1 else 0.0

        summary_rows.append({
            "graph_file": graph,
            "V": V,
            "E": E,
            "algorithm": algo,
            "strategy": strat,
            "threads": threads,
            "samples": len(rows),
            "compute_mean_ms": round(mean_comp, 4),
            "compute_std_ms": round(std_comp, 4),
            "total_mean_ms": round(mean_tot, 4),
            "total_std_ms": round(std_tot, 4),
            "speedup_mean": round(mean_sp, 2),
            "speedup_std": round(std_sp, 2),
            "mteps_mean": round(mean_mteps, 2),
            "mteps_std": round(std_mteps, 2),
        })

    if summary_csv_path and summary_rows:
        fieldnames = [
            "graph_file", "V", "E", "algorithm", "strategy", "threads", "samples",
            "compute_mean_ms", "compute_std_ms", "total_mean_ms", "total_std_ms",
            "speedup_mean", "speedup_std", "mteps_mean", "mteps_std"
        ]
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[Summary] Exported aggregated summary to {summary_csv_path}")

    return summary_rows

def print_summary_table(summary_rows):
    if not summary_rows:
        return
    print("\n" + "=" * 110)
    print("                     AGGREGATED BENCHMARK SUMMARY (MEAN ± STD SPREAD)")
    print("=" * 110)
    print(f"{'ALGO':<6}{'STRATEGY':<25}{'THREADS':<9}{'SAMPLES':<9}{'COMPUTE(ms)':<22}{'TOTAL(ms)':<22}{'SPEEDUP':<10}{'MTEPS'}")
    print("-" * 110)
    for r in summary_rows:
        comp_str = f"{r['compute_mean_ms']:.3f} ± {r['compute_std_ms']:.3f}"
        tot_str = f"{r['total_mean_ms']:.3f} ± {r['total_std_ms']:.3f}"
        sp_str = f"{r['speedup_mean']:.2f}x"
        mteps_str = f"{r['mteps_mean']:.2f}"
        print(f"{r['algorithm']:<6}{r['strategy']:<25}{r['threads']:<9}{r['samples']:<9}{comp_str:<22}{tot_str:<22}{sp_str:<10}{mteps_str}")
    print("=" * 110 + "\n")

def main():
    parser = argparse.ArgumentParser(description="PCAP Benchmark Runner and CSV Metrics Generator")
    parser.add_argument("--binary", default="./build/uniform_benchmark", help="Path to uniform_benchmark binary")
    parser.add_argument("--graphs", nargs="+", default=["graphs/sample_graph.txt"], help="Graphs to benchmark")
    parser.add_argument("--source", type=int, default=0, help="Source vertex ID")
    parser.add_argument("--directed", action="store_true", help="Treat graph as directed (default: undirected)")
    parser.add_argument("--algo", choices=["all", "bfs", "sssp"], default="all", help="Algorithm suite")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs per graph (default: 1)")
    parser.add_argument("--threads", type=int, default=8, help="Thread count for OpenMP CPU fallback (default: 8)")
    parser.add_argument("--csv", default="results/csv/results.csv", help="Raw results CSV output")
    parser.add_argument("--summary-csv", default="results/csv/results_summary.csv", help="Aggregated summary CSV output")
    parser.add_argument("--analyze-only", action="store_true", help="Skip running benchmarks; analyze existing CSV only")

    args = parser.parse_args()

    # Ensure directories exist
    for path in [args.csv, args.summary_csv]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if not args.analyze_only:
        build_benchmark_if_needed(args.binary)
        undirected = not args.directed

        current_run_id = 1
        # Check existing CSV to continue run_id contiguousness
        existing = parse_results_csv(args.csv)
        if existing:
            current_run_id = max(r["run"] for r in existing) + 1

        for g in args.graphs:
            if not os.path.exists(g):
                print(f"[Warning] Graph {g} not found. Skipping...", file=sys.stderr)
                continue

            print(f"\n=======================================================")
            print(f" Starting Benchmark ({args.repeats} runs) for {g}")
            print(f"=======================================================")
            run_single_benchmark(args.binary, g, args.source, undirected, args.algo, args.threads, args.csv, args.repeats, current_run_id)
            current_run_id += args.repeats

    # Parse and generate summaries
    records = parse_results_csv(args.csv)
    if not records:
        print("[Notice] No benchmark records found in CSV to summarize.")
        return

    summary = generate_summary(records, args.summary_csv)
    print_summary_table(summary)

if __name__ == "__main__":
    main()
