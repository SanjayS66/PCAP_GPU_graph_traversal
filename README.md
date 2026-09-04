# PCAP Implementation

Parallel Computing Assignment Project - CSR Graph Format with BFS and SSSP (CPU & GPU)

## Project Structure

```
PCAP_implementation/
│
├── .gitignore                              # Ignores .o files, binaries, editor/OS junk
├── PCAP_Synopsis_Draft.pdf                 # Project synopsis document
│
├── include/
│   ├── csr.h                               # CSR data structures (Edge, EdgeListGraph, CSRGraph) and function prototypes
│   ├── bfs.h                               # BFS header
│   └── sssp.h                              # SSSP header
│
├── src/
│   ├── csr.c                               # CSR implementation: read graph, build CSR, validate, dump
│   ├── csr.o                               # Compiled object file for csr.c
│   ├── bfs_cpu.c                           # CPU BFS implementation
│   ├── bfs_gpu.cu                          # GPU BFS CUDA kernel
│   ├── sssp_cpu.c                          # CPU SSSP implementation
│   └── sssp_gpu.cu                         # GPU SSSP CUDA kernel
│
├── test/
│   ├── test_csr.c                          # Test harness for CSR conversion
│   ├── test_csr.o                          # Compiled object file for test_csr.c
│   └── csr_convert                         # Compiled test binary
│
└── graphs/
    ├── sample_graph.txt                    # Small sample graph (5 vertices, 6 edges)
    └── graph1.edgelist                     # Large graph (2701 vertices, 1,808,853 edges)
```

## Graph Preprocessing and CSR Notation

Real-world graphs contain millions of vertices and edges, making dense adjacency matrices wasteful and raw edge lists slow for neighbour lookups. The Compressed Sparse Row (CSR) format solves both: it stores only existing edges contiguously in memory, giving fast, cache-friendly access — essential for coalesced memory access on the GPU.

CSR represents the graph using three parallel arrays:

- **`row_offset`** (size V+1): For each vertex `i`, stores the starting index into the other two arrays. Neighbours of vertex `i` occupy `[row_offset[i], row_offset[i+1])`. The extra entry at index V acts as a sentinel so the formula works for the last vertex.
- **`col_index`** (size E): A flat array holding the destination vertex of every edge, with each vertex's neighbours stored contiguously.
- **`weights`** (size E): Parallel to `col_index`, storing the weight of the edge leading to each corresponding neighbour.



### CSR Construction (`src/csr.c`)


**1. `read_graph_from_file`**

> **⚠️ Input file format:** Each graph is a plain-text edge list. The first line contains `<num_vertices> <num_edges>`, followed by `<num_edges>` lines of `<u> <v> <w>` (source, destination, weight).

```c
EdgeListGraph read_graph_from_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: could not open input file '%s'\n", path);
        exit(1);
    }

    EdgeListGraph g;
    if (fscanf(fp, "%d %d", &g.V, &g.E) != 2) {
        fprintf(stderr, "Error: failed to read header (<num_vertices> <num_edges>) "
                        "from '%s'\n", path);
        fclose(fp);
        exit(1);
    }

    if (g.V < 0 || g.E < 0) {
        fprintf(stderr, "Error: header has negative vertex/edge count: %d %d\n",
                g.V, g.E);
        fclose(fp);
        exit(1);
    }

    g.edges = (Edge *)malloc(sizeof(Edge) * (size_t)g.E);
    if (g.E > 0 && !g.edges) {
        fprintf(stderr, "Error: out of memory allocating %d edges\n", g.E);
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < g.E; i++) {
        int u, v;
        float w;
        if (fscanf(fp, "%d %d %f", &u, &v, &w) != 3) {
            fprintf(stderr, "Error: expected %d edge lines but failed to read "
                            "edge %d (only read %d successfully). Check the file "
                            "has '<u> <v> <w>' per line and matches the header count.\n",
                    g.E, i, i);
            fclose(fp);
            free(g.edges);
            exit(1);
        }
        if (u < 0 || u >= g.V || v < 0 || v >= g.V) {
            fprintf(stderr, "Error: edge %d has out-of-range endpoint(s): (%d, %d); "
                            "expected both in [0, %d)\n", i, u, v, g.V);
            fclose(fp);
            free(g.edges);
            exit(1);
        }
        g.edges[i].u = u;
        g.edges[i].v = v;
        g.edges[i].w = w;
    }

    fclose(fp);
    return g;
}
```

Reads the plain-text edge list from disk into an `EdgeListGraph`. It first parses the header line `<num_vertices> <num_edges>` into `V` and `E`, allocates an array of `E` `Edge` structs, then reads exactly `E` lines of `<u> <v> <w>` into that array. Along the way it validates that the header values aren't negative, that every edge line actually contains three parseable values, and that every vertex index `u`/`v` falls within `[0, V)`. The output of this function is just a raw, unprocessed list of edges — no CSR structure exists yet.

---

**2. `build_csr`**

```c
CSRGraph build_csr(const EdgeListGraph *g, int directed) {
    CSRGraph csr;
    csr.V = g->V;
    csr.E = directed ? g->E : g->E * 2;

    csr.row_offset = (int *)calloc((size_t)csr.V + 1, sizeof(int));
    csr.col_index  = (int *)malloc(sizeof(int) * (size_t)csr.E);
    csr.weights    = (float *)malloc(sizeof(float) * (size_t)csr.E);
    if (!csr.row_offset || (csr.E > 0 && (!csr.col_index || !csr.weights))) {
        fprintf(stderr, "Error: out of memory building CSR for V=%d E=%d\n",
                csr.V, csr.E);
        exit(1);
    }

    // Pass 1: count out-degree of every vertex
    for (int i = 0; i < g->E; i++) {
        csr.row_offset[g->edges[i].u]++;
        if (!directed) {
            csr.row_offset[g->edges[i].v]++;
        }
    }

    // Prefix sum to convert degrees into offsets
    int running = 0;
    for (int i = 0; i < csr.V; i++) {
        int degree = csr.row_offset[i];
        csr.row_offset[i] = running;
        running += degree;
    }
    csr.row_offset[csr.V] = running;

    int *cursor = (int *)malloc(sizeof(int) * (size_t)csr.V);
    if (!cursor) {
        fprintf(stderr, "Error: out of memory allocating cursor array\n");
        exit(1);
    }
    memcpy(cursor, csr.row_offset, sizeof(int) * (size_t)csr.V);

    // Pass 2: fill col_index and weights using cursor
    for (int i = 0; i < g->E; i++) {
        int u = g->edges[i].u, v = g->edges[i].v;
        float w = g->edges[i].w;

        int pos = cursor[u]++;
        csr.col_index[pos] = v;
        csr.weights[pos] = w;

        if (!directed) {
            int pos_rev = cursor[v]++;
            csr.col_index[pos_rev] = u;
            csr.weights[pos_rev] = w;
        }
    }

    free(cursor);
    return csr;
}
```

Converts the raw edge list into the three CSR arrays (`row_offset`, `col_index`, `weights`). This is the core function and the one most worth understanding carefully:

- **Directed vs undirected, and why E can look "doubled":** if the graph is undirected, `csr.E` is set to `2 × g->E`, because each input edge `(u, v)` must be stored as two directed entries — `u → v` and `v → u` — so that both endpoints can traverse to each other. This is *not* a bug or duplication error; it's intentional, since the CSR format itself only knows about directed adjacency, and undirectedness is simulated by inserting the edge both ways.

- **`row_offset` is reused for two different purposes across the function** — this is the main source of earlier confusion. In the first pass over the edges, `row_offset[i]` is temporarily used purely as a *degree counter*: it's incremented once for every edge where vertex `i` is an endpoint. Only after this counting pass finishes does `row_offset` get transformed, via a prefix sum, into actual *starting offsets* — each `row_offset[i]` is overwritten with the running total of all degrees before it. So the same array holds two conceptually different things at two different points in the function: first "how many edges does vertex i have," then "at what index does vertex i's block begin." `row_offset[V]` is explicitly set to the total edge count as a sentinel, which is what allows the degree of *any* vertex, including the last one, to always be computed safely as `row_offset[i+1] - row_offset[i]`.

- **`cursor` is a write-position tracker, not a lookup array.** After the prefix sum, `row_offset` correctly stores where each vertex's block *starts*, but it must be left untouched afterward since callers rely on it. So a separate array, `cursor`, is created as a copy of `row_offset`. As edges are placed into `col_index`/`weights` during the second pass, `cursor[u]` is used as "the next free slot for vertex u" — the edge is written to `col_index[cursor[u]]`, and only *after* writing does `cursor[u]` get incremented. This means `cursor[u]` is never read to "look up" a position; it's a running pointer that advances by exactly one for every edge placed for vertex `u`, until all of that vertex's edges have been placed contiguously.

---

**3. `validate_csr`**

```c
int validate_csr(const CSRGraph *g) {
    if (g->row_offset[0] != 0) {
        fprintf(stderr, "Validation error: row_offset[0] must be 0, got %d\n",
                g->row_offset[0]);
        return 0;
    }
    if (g->row_offset[g->V] != g->E) {
        fprintf(stderr, "Validation error: row_offset[V] must equal E (%d), got %d\n",
                g->E, g->row_offset[g->V]);
        return 0;
    }
    for (int i = 0; i < g->V; i++) {
        if (g->row_offset[i] > g->row_offset[i + 1]) {
            fprintf(stderr, "Validation error: row_offset not monotonic at index %d\n", i);
            return 0;
        }
    }
    for (int i = 0; i < g->E; i++) {
        if (g->col_index[i] < 0 || g->col_index[i] >= g->V) {
            fprintf(stderr, "Validation error: col_index[%d] = %d out of range [0, %d)\n",
                    i, g->col_index[i], g->V);
            return 0;
        }
    }
    return 1;
}
```

Performs four structural checks to verify that a `CSRGraph` is internally consistent and safe to traverse by BFS/SSSP:

1. **`row_offset[0] == 0`** — the first vertex's neighbours must start at index 0; any other value would indicate a corrupted prefix sum or incorrect degree counting.
2. **`row_offset[V] == E`** — the sentinel entry past the last vertex must equal the total number of edges (or `2*E` for undirected graphs), confirming every edge was placed and no slots were left unfilled.
3. **Monotonicity** — each `row_offset[i] <= row_offset[i+1]`; a violation means some vertex's degree was computed as negative or the prefix sum was corrupted.
4. **`col_index` range** — every entry in `col_index` must be in `[0, V)`, catching any out-of-range vertex ID that would cause an out-of-bounds memory access during traversal.

Returns `1` (valid) if all checks pass, or `0` (invalid) with a descriptive `stderr` message pinpointing the exact failure.

---

**4. `dump_csr`**

```c
void dump_csr(const CSRGraph *g) {
    printf("\nrow_offset (%d): ", g->V + 1);
    for (int i = 0; i <= g->V; i++) printf("%d ", g->row_offset[i]);
    printf("\ncol_index  (%d): ", g->E);
    for (int i = 0; i < g->E; i++) printf("%d ", g->col_index[i]);
    printf("\nweights    (%d): ", g->E);
    for (int i = 0; i < g->E; i++) printf("%.3f ", g->weights[i]);
    printf("\n");
}
```

Debug utility that prints the contents of all three CSR arrays to `stdout`, making it easy to visually inspect a small graph's CSR representation. It prints each array on its own line with a label and element count: `row_offset` (V+1 entries), `col_index` (E entries), and `weights` (E entries, formatted to 3 decimal places). This is especially useful for verifying that `build_csr` produced the expected layout — for example, confirming that the neighbours of vertex `i` occupy positions `row_offset[i]` through `row_offset[i+1]-1` in `col_index`.
