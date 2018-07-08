#ifndef VUDU_H
#define VUDU_H

#include <stddef.h>

typedef enum cudaError {
  cudaSuccess,
  // TODO: more error codes
  cudaErrorUnknown
} cudaError_t;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
};

// Memory allocation
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);

// Memory transfer
cudaError_t cudaMemcpy(void* dst,
                       const void* src,
                       size_t count,
                       enum cudaMemcpyKind kind);

#endif // VUDU_H
