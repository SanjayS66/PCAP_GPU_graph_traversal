#!/usr/bin/env python3
import sys
import os
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
    print(" LOAD IMBALANCE METRICS (Max / Mean per-thread edge-work)")
    print("=======================================================")

    # 1. Thread-per-vertex
    # Each thread processes 1 vertex. Max work = max_degree. Mean work = mean_degree.
    tpv_imbalance = max_degree / mean_degree if mean_degree > 0 else 1.0
    print(f" [1] GPU Thread-per-vertex : {tpv_imbalance:.2f}x")
    print(f"     -> One thread processes the {max_degree}-degree influencer, while the average thread processes {mean_degree:.1f} edges.")

    # 2. Warp-per-vertex
    # Each warp (32 threads) processes 1 vertex. 
    # Thread work for vertex v is ceil(degree / 32).
    import math
    max_warp_thread_work = max(math.ceil(d / 32.0) for d in deg_list)
    # Total threads launched = V * 32. Total work = E.
    mean_warp_thread_work = num_edges / (V * 32.0)
    wpv_imbalance = max_warp_thread_work / mean_warp_thread_work if mean_warp_thread_work > 0 else 1.0
    
    print(f" [2] GPU Warp-per-vertex   : {wpv_imbalance:.2f}x")
    print(f"     -> Max thread work drops to {max_warp_thread_work} edges. 32 threads cooperatively process influencers.")

    # 3. Thread-per-edge
    # Each thread processes exactly 1 edge. Max work = 1, Mean work = 1.
    print(f" [3] GPU Thread-per-edge   : 1.00x (Perfect Balance)")
    print(f"     -> Work is perfectly distributed regardless of degree skew.")
    
    print("=======================================================\n")

if __name__ == "__main__":
    main()
