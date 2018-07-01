#include <stdio.h>

__global__
void saxpy(int n, float a, float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  int N = 1 << 20;
  float* x = (float*)malloc(N * sizeof(float));
  float* y = (float*)malloc(N * sizeof(float));

  float* dev_x, *dev_y;
  cudaMalloc((void**)&dev_x, N * sizeof(float));
  cudaMalloc((void**)&dev_y, N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0;
  }

  cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  saxpy<<<(N + 255)/256, 256>>>(N, 2.0f, dev_x, dev_y);

  cudaMemcpy(y, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  float max_err = 0.0f;
  for (int i = 0; i < N; ++i) {
    max_err = max(max_err, abs(y[i] - 4.0f));
  }
  printf("Max Error: %f\n", max_err);

  cudaFree(dev_x);
  cudaFree(dev_y);

  delete[] x;
  delete[] y;

  return 0;
}
