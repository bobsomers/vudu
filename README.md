# Vudu

An implementation of CUDA on top of the Vulkan Compute API.

## Components

 * vudurt.so: Shared library that provides cuda-compatible symbols, such as
   `cudaMalloc()`, `cudaFree()`, `cudaMemcpy()`, etc.
 * vuducc: Compiler which takes in `.cu` source files and does the following
   steps:
    * Generate PTX assembly for device code by invoking `nvcc`.
    * Split host code out and rewrite kernel invocation syntax `foo<<<...>>>()`.
    * Transpile PTX assembly into SPIR-V compute code.
    * Generate kernel invocations which run the SPIR-V kernel using Vulkan.
