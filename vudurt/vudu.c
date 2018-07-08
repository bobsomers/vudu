#include "vudu.h"

#include <stdio.h>

static void __attribute__((constructor)) vuduCreate() {
  // TODO: init vulkan, set up static data structures
  printf("vuduCreate()!\n");
}

static void __attribute__((destructor)) vuduDestroy() {
  // TODO: shutdown vulkan
  printf("vuduDestroy()!\n");
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
  (void)devPtr;
  (void)size;
  return cudaErrorUnknown;
}

cudaError_t cudaFree(void* devPtr) {
  (void)devPtr;
  return cudaErrorUnknown;
}

cudaError_t cudaMemcpy(void* dst,
                       const void* src,
                       size_t count,
                       enum cudaMemcpyKind kind) {
  (void)dst;
  (void)src;
  (void)count;
  (void)kind;
  return cudaErrorUnknown;
}
