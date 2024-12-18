#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiply_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    int N = 1024;  // Matrix size
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    clock_t start = clock();
    matrix_multiply_cpu(A, B, C, N);
    clock_t end = clock();

    printf("CPU Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(A);
    free(B);
    free(C);
    return 0;
}
