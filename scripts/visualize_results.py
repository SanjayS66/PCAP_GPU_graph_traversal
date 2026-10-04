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

def plot_parallel_efficiency(summary_csv, target_graph, outdir):
    if not os.path.exists(summary_csv):
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        return
        
    df = df[df['graph_file'].str.contains(target_graph)]
    
    omp_df = df[df['strategy'].str.contains('OpenMP')].copy()
    if omp_df.empty:
        return
        
    # Keep only the latest run for each (algo, strategy, threads) combination
    omp_df = omp_df.drop_duplicates(subset=['algorithm', 'strategy', 'threads'], keep='last')
    
    omp_df['efficiency'] = omp_df['speedup_mean'] / omp_df['threads']
    
    plt.figure(figsize=(10, 6))
    
    colors = {'BFS': 'blue', 'SSSP': 'green'}
    markers = {'BFS': 'o', 'SSSP': 's'}
    
    for algo in ['BFS', 'SSSP']:
        algo_df = omp_df[omp_df['algorithm'] == algo].sort_values('threads')
        if not algo_df.empty:
            plt.plot(algo_df['threads'].values, algo_df['efficiency'].values, marker=markers[algo], 
                     color=colors[algo], label=algo, linewidth=2, markersize=8)
            
    plt.title(f'OpenMP Parallel Efficiency\nGraph: {target_graph}')
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency (Speedup / Threads)')
    plt.xticks(omp_df['threads'].unique())
    # Add a horizontal line at y=1 (Ideal Efficiency)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal Efficiency (1.0)')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.ylim(bottom=0.0)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, f"OpenMP_Efficiency_{target_graph.split('.')[0]}.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved efficiency plot to {out_file}")
    plt.close()

def plot_speedup_scaling(summary_csv, target_graph, outdir):
    if not os.path.exists(summary_csv):
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        return
        
    df = df[df['graph_file'].str.contains(target_graph)]
    
    omp_df = df[df['strategy'].str.contains('OpenMP')].copy()
    if omp_df.empty:
        return
        
    omp_df = omp_df.drop_duplicates(subset=['algorithm', 'strategy', 'threads'], keep='last')
    
    plt.figure(figsize=(10, 6))
    
    colors = {'BFS': 'blue', 'SSSP': 'green'}
    markers = {'BFS': 'o', 'SSSP': 's'}
    
    for algo in ['BFS', 'SSSP']:
        algo_df = omp_df[omp_df['algorithm'] == algo].sort_values('threads')
        if not algo_df.empty:
            plt.plot(algo_df['threads'].values, algo_df['speedup_mean'].values, marker=markers[algo], 
                     color=colors[algo], label=f'{algo} Speedup', linewidth=2, markersize=8)
            
    plt.title(f'OpenMP Scaling (Speedup vs Threads)\nGraph: {target_graph}')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (relative to Sequential CPU)')
    
    # Draw the ideal linear scaling line
    max_threads = omp_df['threads'].max()
    plt.plot([1, max_threads], [1, max_threads], 'r--', label='Ideal Linear Scaling')
    
    plt.xticks(omp_df['threads'].unique())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.ylim(bottom=0.0)
    plt.xlim(left=0.5)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, f"OpenMP_Speedup_Scaling_{target_graph.split('.')[0]}.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved speedup scaling plot to {out_file}")
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
    plot_parallel_efficiency(summary_path, args.graph, args.outdir)
    plot_speedup_scaling(summary_path, args.graph, args.outdir)
    print("Done!")

if __name__ == "__main__":
    main()
