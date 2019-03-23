#include "vudu.h"

#include <stdio.h>

#include <vulkan/vulkan.h>

static VkInstance vudu_instance;
static int vudu_instance_valid = 0;
static VkPhysicalDevice vudu_device;
static int vudu_device_valid = 0;

static int vudu__CreateInstance() {
  VkApplicationInfo ai;
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pNext = NULL;
  ai.pApplicationName = "vudurt";
  ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.pEngineName = "vudu";
  ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo ci;
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pNext = NULL;
  ci.flags = 0;
  ci.pApplicationInfo = &ai;
  ci.enabledLayerCount = 0;
  ci.ppEnabledLayerNames = NULL;
  ci.enabledExtensionCount = 0;
  ci.ppEnabledExtensionNames = NULL;

  VkResult result = vkCreateInstance(&ci, NULL, &vudu_instance);
  if (result != VK_SUCCESS) {
    return 0;
  }
  return 1;
}

static void __attribute__((constructor)) vuduCreate() {
  vudu_instance_valid = vudu__CreateInstance();

  printf("Created Vulkan instance!\n");
}

static void __attribute__((destructor)) vuduDestroy() {
  if (vudu_instance != NULL) {
    printf("Destroying Vulkan instance!\n");
    vkDestroyInstance(vudu_instance, NULL);
  }
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
