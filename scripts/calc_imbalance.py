#!/usr/bin/env python3
import sys
import os
import math
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Compute Static Load Imbalance Metrics for GPU Strategies")
    parser.add_argument("graph_file", help="Path to the .edgelist or .txt graph file")
    args = parser.parse_args()

    if not os.path.exists(args.graph_file):
        print(f"Error: {args.graph_file} not found.")
        sys.exit(1)

    print(f"Reading graph: {args.graph_file}...")
    degrees = defaultdict(int)
    num_edges = 0
    max_vid = -1

    with open(args.graph_file, 'r') as f:
        first_line = True
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if first_line and len(parts) == 2:
                # It's the V E header
                first_line = False
                continue

            first_line = False
            if len(parts) >= 2:
                u = int(parts[0])
                v = int(parts[1])
                degrees[u] += 1
                degrees[v] += 1  # Assuming undirected for the workload calculation
                num_edges += 2
                if u > max_vid: max_vid = u
                if v > max_vid: max_vid = v

    V = max_vid + 1
    if V == 0:
        print("Empty graph.")
        return

    # Convert degrees to a list
    deg_list = [degrees[i] for i in range(V)]

    max_degree = max(deg_list)
    mean_degree = num_edges / V

    print("\n=======================================================")
    print(f" GRAPH STATISTICS: {os.path.basename(args.graph_file)}")
    print("=======================================================")
    print(f" Vertices (V) : {V:,}")
    print(f" Edges (E)    : {num_edges:,}")
    print(f" Max Degree   : {max_degree:,}")
    print(f" Mean Degree  : {mean_degree:.2f}")

    print("\n=======================================================")
    print(" GPU LOAD-BALANCING METRICS")
    print("=======================================================")

    # --- Headline metric: MAX per-thread serial work (the critical path; predicts runtime) ---
    tpv_max_work  = max_degree                     # one thread does the whole vertex's edges
    warp_max_work = math.ceil(max_degree / 32.0)   # hub split across 32 warp lanes
    edge_max_work = 1                              # one edge per thread
    warp_reduction = tpv_max_work / warp_max_work if warp_max_work > 0 else 1.0

    print(" MAX PER-THREAD WORK  (critical path -- lower is better)")
    print("-------------------------------------------------------")
    print(f" [1] Thread-per-vertex : {tpv_max_work:,} edges   (worst thread handles the full {max_degree:,}-degree hub)")
    print(f" [2] Warp-per-vertex   : {warp_max_work:,} edges   ({warp_reduction:.1f}x less -- 32 lanes share the hub)")
    print(f" [3] Thread-per-edge   : {edge_max_work} edge      (perfectly flat)")

    print("\n MAX / MEAN RATIO  (classic imbalance; valid for vertex- and edge-mapping)")
    print("-------------------------------------------------------")
    tpv_imbalance = max_degree / mean_degree if mean_degree > 0 else 1.0
    print(f" [1] Thread-per-vertex : {tpv_imbalance:.2f}x")
    print(f" [3] Thread-per-edge   : 1.00x (perfect balance)")
    print("  (Warp ratio omitted: dividing by mean-over-launched-threads is distorted by idle lanes.)")

    print("=======================================================\n")

if __name__ == "__main__":
    main()