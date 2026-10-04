#!/usr/bin/env python3
import os
import glob
import sys

def main():
    gplus_dir = "gplus"
    output_file = "graphs/gplus_combined.edgelist"
    
    if not os.path.exists(gplus_dir):
        print(f"Error: Directory '{gplus_dir}' not found.")
        sys.exit(1)
        
    edge_files = glob.glob(os.path.join(gplus_dir, "*.edges"))
    if not edge_files:
        print(f"Error: No .edges files found in {gplus_dir}")
        sys.exit(1)
        
    print(f"Found {len(edge_files)} edge files. Processing...")
    
    # We need to map the giant 21-digit string IDs to standard 32-bit integers (0, 1, 2, ...)
    # because our C++ code uses 'int' for vertex IDs.
    node_map = {}
    next_id = 0
    
    # We use a set to avoid writing duplicate edges
    unique_edges = set()
    
    for f in edge_files:
        with open(f, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                
                u_str, v_str = parts[0], parts[1]
                
                if u_str not in node_map:
                    node_map[u_str] = next_id
                    next_id += 1
                if v_str not in node_map:
                    node_map[v_str] = next_id
                    next_id += 1
                    
                u = node_map[u_str]
                v = node_map[v_str]
                
                # Keep smaller vertex first just for canonical uniqueness (optional for undirected)
                edge = (min(u, v), max(u, v))
                unique_edges.add(edge)
                
    os.makedirs("graphs", exist_ok=True)
    
    print(f"Writing {len(unique_edges)} unique edges to {output_file}...")
    with open(output_file, 'w') as out:
        out.write(f"{next_id} {len(unique_edges)}\n")
        for u, v in unique_edges:
            out.write(f"{u} {v} 1.0\n")
            
    print(f"Done! Graph has {next_id} vertices and {len(unique_edges)} edges.")

if __name__ == "__main__":
    main()
