#include <stdio.h>

#include "vudu.h"

int main() {
  printf("In main()!\n");

  void* dev_ptr;
  cudaError_t err = cudaMalloc(&dev_ptr, 1024);
  err = cudaFree(dev_ptr);

  return 0;
}
