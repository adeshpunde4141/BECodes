
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 16

// CUDA Kernel to perform matrix multiplication
__global__ void matrixMul(int *A, int *B, int *C, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check for valid matrix indices within bounds
  if (row < width && col < width) {
    int sum = 0;
    for (int k = 0; k < width; ++k) {
      sum += A[row * width + k] * B[k * width + col];
    }
    C[row * width + col] = sum;
  }
}

void fillMatrix(int *matrix, int width) {
  for (int i = 0; i < width * width; i++) {
    matrix[i] = rand() % 10;
  }
}

void printMatrix(int *matrix, int width) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      printf("%4d ", matrix[i * width + j]);
    }
    printf("\n");
  }
}

int main() {
  int size = N * N * sizeof(int);

  // Allocate memory on host
  int *h_A = (int*)malloc(size);
  int *h_B = (int*)malloc(size);
  int *h_C = (int*)malloc(size);

  // Initialize matrices on host
  fillMatrix(h_A, N);
  fillMatrix(h_B, N);

  // Allocate memory on device
  int *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 dimBlock(16, 16);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.x - 1) / dimBlock.x);

  //Launch kernel on GPU
  matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  //Print results
  printf("Matrix A:\n");
  printMatrix(h_A, N);
  printf("\nMatrix B:\n");
  printMatrix(h_B, N);
  printf("\nMatrix C (A x B):\n");
  printMatrix(h_C, N);

  //Free memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
