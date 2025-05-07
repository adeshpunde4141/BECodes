
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1000000

// CUDA Kernel to perform vector addition
__global__ void vectorAdd(int* A, int* B, int* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// Fill array with random integers
void fillArray(int *arr, int n){
  for (int i = 0; i < n; i++) {
    arr[i] = rand() % 100;
  }
}

int main() {
  int size = N * sizeof(int);

  // Allocate memory on host
  int *h_A = (int*)malloc(size);
  int *h_B = (int*)malloc(size);
  int *h_C = (int*)malloc(size);

  // Initialize arrays on host
  fillArray(h_A, N);
  fillArray(h_B, N);

  // Allocate memory on device
  int *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Launch kernel on GPU
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Print the first 10 elements of the result
  printf("Vector Addition Result (first 10 element):\n");
  for (int i = 0; i < 10; i++) {
    printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
  }

  // Free memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
