import sys
from collections import defaultdict

metrics = defaultdict(lambda: {"occ": [], "bw": [], "eff": []})
curr_kernel = None

with open("results/profiling/ncu_raw_metrics.txt", "r") as f:
    for line in f:
        if "relax_kernel(" in line: curr_kernel = "relax_kernel"
        elif "relax_kernel_warp_per_v(" in line: curr_kernel = "relax_kernel_warp_per_v"
        elif "relax_kernel_t_per_e(" in line: curr_kernel = "relax_kernel_t_per_e"
        elif "sssp_adaptive_low_kernel(" in line: curr_kernel = "sssp_adaptive_low_kernel"
        elif "sssp_adaptive_high_kernel(" in line: curr_kernel = "sssp_adaptive_high_kernel"
        
        if not curr_kernel: continue
        
        parts = line.strip().split()
        if not parts: continue
        
        metric_name = parts[0]
        try:
            val = float(parts[-1].replace(",", ""))
        except:
            continue
            
        if metric_name == "sm__warps_active.avg.pct_of_peak_sustained_active":
            metrics[curr_kernel]["occ"].append(val)
        elif metric_name == "dram__throughput.avg.pct_of_peak_sustained_elapsed":
            metrics[curr_kernel]["bw"].append(val)
        elif metric_name == "smsp__thread_inst_executed_per_inst_executed.ratio":
            metrics[curr_kernel]["eff"].append(val)

labels = {
    "relax_kernel": "Thread-per-vertex",
    "relax_kernel_warp_per_v": "Warp-per-vertex",
    "relax_kernel_t_per_e": "Thread-per-edge",
    "sssp_adaptive_low_kernel": "Adaptive (Low-deg)",
    "sssp_adaptive_high_kernel": "Adaptive (High-deg)",
}

print(f"{'Strategy':<25} | {'Occupancy (%)':<15} | {'Warp Efficiency (%)':<20} | {'DRAM BW (%)':<15} | {'Samples'}")
print("-" * 90)
for k in ["relax_kernel", "relax_kernel_warp_per_v", "relax_kernel_t_per_e", "sssp_adaptive_low_kernel", "sssp_adaptive_high_kernel"]:
    if k not in metrics: continue
    m = metrics[k]
    occ = sum(m["occ"])/len(m["occ"]) if m["occ"] else 0
    bw = sum(m["bw"])/len(m["bw"]) if m["bw"] else 0
    eff = sum(m["eff"])/len(m["eff"]) if m["eff"] else 0
    samples = len(m["occ"])
    eff_pct = eff * 100 if eff <= 1.0 else eff
    print(f"{labels[k]:<25} | {occ:>14.2f}% | {eff_pct:>19.2f}% | {bw:>14.2f}% | {samples}")
