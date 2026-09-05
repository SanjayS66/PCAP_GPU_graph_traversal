#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int read_dist(const char *path, int *V, float **out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    if (fscanf(fp, "%d", V) != 1) { fprintf(stderr, "bad header in %s\n", path); fclose(fp); return 0; }
    *out = (float *)malloc(sizeof(float) * (*V));
    for (int i = 0; i < *V; i++) {
        if (fscanf(fp, "%f", &(*out)[i]) != 1) {
            fprintf(stderr, "%s: expected %d values, stopped at %d\n", path, *V, i);
            fclose(fp); return 0;
        }
    }
    fclose(fp);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s <fileA> <fileB>\n", argv[0]); return 1; }

    int Va, Vb; float *a, *b;
    if (!read_dist(argv[1], &Va, &a)) return 1;
    if (!read_dist(argv[2], &Vb, &b)) return 1;

    if (Va != Vb) { printf("MISMATCH: different vertex counts (%d vs %d)\n", Va, Vb); return 1; }

    const float abs_tol = 1e-3f, rel_tol = 1e-3f;
    for (int i = 0; i < Va; i++) {
        float diff = fabsf(a[i] - b[i]);
        float largest = fabsf(a[i]) > fabsf(b[i]) ? fabsf(a[i]) : fabsf(b[i]);
        if (diff > abs_tol && diff > largest * rel_tol) {
            printf("MISMATCH at vertex %d: %.6f vs %.6f (diff %.6f)\n", i, a[i], b[i], diff);
            free(a); free(b); return 1;
        }
    }
    printf("MATCH: all %d distances agree within tolerance\n", Va);
    free(a); free(b);
    return 0;
}
