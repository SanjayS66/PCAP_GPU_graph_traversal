import random

V = 15000000  # try scaling this up
avg_degree = 4    # keep this low/uniform, unlike your current graph's ~670
E = V * avg_degree

with open("graph_largest_sparse.edgelist", "w") as f:
    f.write(f"{V} {E}\n")
    for _ in range(E):
        u = random.randint(0, V - 1)
        v = random.randint(0, V - 1)
        w = round(random.uniform(1.0, 10.0), 3)
        f.write(f"{u} {v} {w}\n")
