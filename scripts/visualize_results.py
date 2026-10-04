#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_speedup_comparison(summary_csv, target_graph, outdir):
    if not os.path.exists(summary_csv):
        print(f"File {summary_csv} not found.")
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        print(f"No data in {summary_csv}.")
        return
        
    # Filter for the target graph
    df = df[df['graph_file'].str.contains(target_graph)]
    if df.empty:
        print(f"No data found for graph containing '{target_graph}'.")
        return

    # Add thread counts to OpenMP labels to distinguish them
    df['strategy_label'] = df.apply(
        lambda r: f"{r['strategy']} ({r['threads']} threads)" if "OpenMP" in r['strategy'] else r['strategy'], 
        axis=1
    )

    for algo in ['BFS', 'SSSP']:
        algo_df = df[df['algorithm'] == algo].copy()
        if algo_df.empty:
            continue
            
        # Drop duplicate strategy labels (keeping the latest run) so we don't draw multiple text labels on top of each other
        algo_df = algo_df.drop_duplicates(subset=['strategy_label'], keep='last')
        
        algo_df = algo_df.sort_values('speedup_mean', ascending=True)
        
        plt.figure(figsize=(10, 6))
        
        # Use log scale if GPU speedups are massively larger than CPU speedups
        max_speedup = algo_df['speedup_mean'].max()
        if max_speedup > 100:
            plt.xscale('log')
            plt.xlabel('Speedup (Log Scale, relative to CPU Sequential)')
        else:
            plt.xlabel('Speedup (relative to CPU Sequential)')

        bars = plt.barh(algo_df['strategy_label'], algo_df['speedup_mean'], color='coral')
        
        plt.title(f'{algo} Speedup Comparison\nGraph: {target_graph}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            # If log scale, place text slightly after the bar ends so it doesn't overlap weirdly
            x_pos = width * 1.1 if max_speedup > 100 else width + (max_speedup * 0.02)
            plt.text(x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}x', 
                     ha='left', va='center', fontweight='bold', color='black')
            
        if max_speedup > 100:
            plt.xlim(right=max_speedup * 2)
        else:
            plt.xlim(right=max_speedup * 1.15)
            
        plt.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        out_file = os.path.join(outdir, f"{algo}_speedup_comparison.png")
        plt.savefig(out_file, dpi=300)
        print(f"Saved plot to {out_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Benchmark Results")
    parser.add_argument("--summary", default="results/csv/results_summary.csv", help="Path to results_summary.csv")
    parser.add_argument("--graph", default="graph_large_dense.edgelist", help="Graph file to visualize")
    parser.add_argument("--outdir", default="results/graphs", help="Directory to save output graphs")
    args = parser.parse_args()

    summary_path = args.summary if os.path.exists(args.summary) else "results/csv/results_summary.csv"

    print("Generating visualizations...")
    plot_speedup_comparison(summary_path, args.graph, args.outdir)
    print("Done!")

if __name__ == "__main__":
    main()
