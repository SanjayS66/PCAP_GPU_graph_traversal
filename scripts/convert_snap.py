#!/usr/bin/env python3
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert SNAP txt to .edgelist")
    parser.add_argument("input_file", help="Input SNAP txt file")
    parser.add_argument("output_file", help="Output .edgelist file")
    args = parser.parse_args()

    # Pass 1: Find max vertex ID and count edges
    print(f"Scanning {args.input_file} to find V and E...")
    max_vid = -1
    num_edges = 0
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u = int(parts[0])
                v = int(parts[1])
                if u > max_vid: max_vid = u
                if v > max_vid: max_vid = v
                num_edges += 1

    V = max_vid + 1
    E = num_edges
    print(f"Detected V={V}, E={E}")

    # Pass 2: Write output
    print(f"Writing converted graph to {args.output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    with open(args.output_file, 'w') as out:
        out.write(f"{V} {E}\n")
        with open(args.input_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    out.write(f"{parts[0]} {parts[1]} 1.0\n")
                    
    print("Done!")

if __name__ == "__main__":
    main()
