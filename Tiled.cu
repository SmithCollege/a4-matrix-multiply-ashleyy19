#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void tiled_matrix_multiply(float* A, float* B, float* C, int N) {
    __shared__ float Asub[16][16];
    __shared__ float Bsub[16][16];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float value = 0.0;

    for (int t = 0; t < (N + 16 - 1) / 16; t++) {
        // Load tiles into shared memory
        if (row < N && t * 16 + tx < N)
            Asub[ty][tx] = A[row * N + t * 16 + tx];
        else
            Asub[ty][tx] = 0.0;

        if (col < N && t * 16 + ty < N)
            Bsub[ty][tx] = B[(t * 16 + ty) * N + col];
        else
            Bsub[ty][tx] = 0.0;

        __syncthreads(); // Synchronize threads to ensure tiles are fully loaded

        // Perform multiplication for the current tile
        for (int k = 0; k < 16; k++) {
            value += Asub[ty][k] * Bsub[k][tx];
        }

        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    // Matrix dimensions
    int N = 1024; // Matrix size (N x N)
    size_t bytes = N * N * sizeof(float);

    // Host memory allocation
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16); // Tile size is 16x16
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Start the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel
    tiled_matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tiled GPU Matrix Multiply Time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (h_C[i * N + j] != N) {
                printf("Verification failed at (%d, %d): %f\n", i, j, h_C[i * N + j]);
                goto cleanup;
            }
        }
    }
    printf("Result verification passed.\n");

cleanup:
    // Free host and device memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
