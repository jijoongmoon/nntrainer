// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_loader.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Load required OpenCL functions
 *
 */

#include "opencl_loader.h"

#include <dynamic_library_loader.h>
#include <nntrainer_log.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h> // _ReturnAddress ([NNTR_CL_ALLOC_TRACE] IAT hook)
#include <psapi.h>  // GetProcessMemoryInfo ([NNTR_CL_FIRSTLAUNCH_WS] probe)
#pragma comment(lib, "psapi.lib")
#include <cstring> // strcmp/strrchr (IAT hook)
#endif

namespace nntrainer::opencl {

#define LoadFunction(function)                                                 \
  function = reinterpret_cast<PFN_##function>(                                 \
    DynamicLibraryLoader::loadSymbol(libopencl, #function));

/**
 * @brief Declaration of loading function for OpenCL APIs
 *
 * @param libopencl
 */
void LoadOpenCLFunctions(void *libopencl);

static bool open_cl_initialized = false;

static bool opencl_init_failed = false;

/**
 * @brief Loading OpenCL libraries and required function
 *
 * @return true if successfull or false otherwise
 */
bool LoadOpenCL() {
  // check if already loaded
  if (open_cl_initialized) {
    return true;
  }
  // if OpenCL is not available
  if (opencl_init_failed) {
    return false;
  }

  void *libopencl = nullptr;

#if defined(_WIN32)
  static const char *kClLibName = "OpenCL.dll";
#else
  static const char *kClLibName = "libOpenCL.so";
#endif

  libopencl =
    DynamicLibraryLoader::loadLibrary(kClLibName, RTLD_NOW | RTLD_LOCAL);
  if (libopencl) {
    LoadOpenCLFunctions(libopencl);
    open_cl_initialized = true;
    return true;
  }

#if !defined(_WIN32)
  // Android Qualcomm/Adreno: vendor's libOpenCL.so isn't always reachable via
  // the default linker namespace from a shell-launched executable; try the
  // well-known vendor paths explicitly so the GPU path works without the
  // caller having to set LD_LIBRARY_PATH=/system/vendor/lib64 (which on some
  // devices drags in libandroid_runtime.so with unresolved symbols).
  static const char *kAndroidVendorPaths[] = {
    "/vendor/lib64/libOpenCL.so",
    "/system/vendor/lib64/libOpenCL.so",
    "/vendor/lib/libOpenCL.so",
    "/system/vendor/lib/libOpenCL.so",
  };
  for (const char *p : kAndroidVendorPaths) {
    libopencl = DynamicLibraryLoader::loadLibrary(p, RTLD_NOW | RTLD_LOCAL);
    if (libopencl) {
      LoadOpenCLFunctions(libopencl);
      open_cl_initialized = true;
      return true;
    }
  }
#endif

  // record error
  std::string error(DynamicLibraryLoader::getLastError());
  ml_loge("Cannot open OpenCL library on this device - %s", error.c_str());
  opencl_init_failed = true;
  return false;
}

/**
 * @brief Retrieves string representation of OpenCL status code
 *
 * @return OpenCL status code as string
 */
const char *OpenCLErrorCodeToString(const cl_int code) {
#define SWITCH_CASE_RETURN(ENUM)                                               \
  case ENUM:                                                                   \
    return #ENUM

  switch (code) {
    SWITCH_CASE_RETURN(CL_SUCCESS);
    SWITCH_CASE_RETURN(CL_DEVICE_NOT_FOUND);
    SWITCH_CASE_RETURN(CL_DEVICE_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_COMPILER_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    SWITCH_CASE_RETURN(CL_OUT_OF_RESOURCES);
    SWITCH_CASE_RETURN(CL_OUT_OF_HOST_MEMORY);
    SWITCH_CASE_RETURN(CL_PROFILING_INFO_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_MEM_COPY_OVERLAP);
    SWITCH_CASE_RETURN(CL_IMAGE_FORMAT_MISMATCH);
    SWITCH_CASE_RETURN(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    SWITCH_CASE_RETURN(CL_BUILD_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
    SWITCH_CASE_RETURN(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    SWITCH_CASE_RETURN(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#ifdef CL_VERSION_1_2
    SWITCH_CASE_RETURN(CL_COMPILE_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_LINKER_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_LINK_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_DEVICE_PARTITION_FAILED);
    SWITCH_CASE_RETURN(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
    SWITCH_CASE_RETURN(CL_INVALID_VALUE);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_TYPE);
    SWITCH_CASE_RETURN(CL_INVALID_PLATFORM);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE);
    SWITCH_CASE_RETURN(CL_INVALID_CONTEXT);
    SWITCH_CASE_RETURN(CL_INVALID_QUEUE_PROPERTIES);
    SWITCH_CASE_RETURN(CL_INVALID_COMMAND_QUEUE);
    SWITCH_CASE_RETURN(CL_INVALID_HOST_PTR);
    SWITCH_CASE_RETURN(CL_INVALID_MEM_OBJECT);
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_SAMPLER);
    SWITCH_CASE_RETURN(CL_INVALID_BINARY);
    SWITCH_CASE_RETURN(CL_INVALID_BUILD_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_PROGRAM);
    SWITCH_CASE_RETURN(CL_INVALID_PROGRAM_EXECUTABLE);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_NAME);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_DEFINITION);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_INDEX);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_VALUE);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_ARGS);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_DIMENSION);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_GROUP_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_ITEM_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_GLOBAL_OFFSET);
    SWITCH_CASE_RETURN(CL_INVALID_EVENT_WAIT_LIST);
    SWITCH_CASE_RETURN(CL_INVALID_EVENT);
    SWITCH_CASE_RETURN(CL_INVALID_OPERATION);
    SWITCH_CASE_RETURN(CL_INVALID_GL_OBJECT);
    SWITCH_CASE_RETURN(CL_INVALID_BUFFER_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_MIP_LEVEL);
    SWITCH_CASE_RETURN(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
    SWITCH_CASE_RETURN(CL_INVALID_PROPERTY);
#endif
#ifdef CL_VERSION_1_2
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_DESCRIPTOR);
    SWITCH_CASE_RETURN(CL_INVALID_COMPILER_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_LINKER_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
    SWITCH_CASE_RETURN(CL_INVALID_PIPE_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_QUEUE);
#endif
#ifdef CL_VERSION_2_2
    SWITCH_CASE_RETURN(CL_INVALID_SPEC_ID);
    SWITCH_CASE_RETURN(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif
  default:
    return "(unknown)";
  }
#undef SWITCH_CASE_RETURN
}

// [NNTR_CL_FIRSTLAUNCH_WS] Per-kernel first-launch working-set delta probe
// (Windows). The +1.4GB of resident all-zero 24/32MB regions on gauss4/Xe3
// appears as ~40 driver allocations exactly when the first prefill launches
// each distinct kernel for the first time (zero-region time series
// 2026-07-14); CL_KERNEL_SPILL_MEM_SIZE_INTEL reports 0 for every kernel on
// this driver, so attribute the cost empirically instead: on the FIRST
// clEnqueueNDRangeKernel of each kernel NAME, drain the queue, snapshot the
// process WS, launch, drain again, and print the delta. Purely diagnostic,
// env-gated, and a pass-through trampoline otherwise.
#if defined(_WIN32)
namespace {
PFN_clEnqueueNDRangeKernel flws_real_enqueue = nullptr;

size_t flws_ws_kb() {
  PROCESS_MEMORY_COUNTERS pmc{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return 0;
  return pmc.WorkingSetSize >> 10;
}

// [NNTR_CL_ALLOC_TRACE] log every clSVMAlloc/clCreateBuffer >= 4MB with the
// returned VA (SVM) or handle, size, and a monotonic ms timestamp, so large
// anonymous regions found by the external fingerprint walker can be matched
// to their allocation site class. Same trampoline pattern as [klws].
PFN_clSVMAlloc alct_real_svmalloc = nullptr;
PFN_clCreateBuffer alct_real_createbuffer = nullptr;

long long alct_ms() {
  static LARGE_INTEGER f = {}, t0 = {};
  LARGE_INTEGER t;
  if (f.QuadPart == 0) {
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
  }
  QueryPerformanceCounter(&t);
  return (t.QuadPart - t0.QuadPart) * 1000 / f.QuadPart;
}

void *CL_API_CALL alct_svmalloc_hook(cl_context c, cl_svm_mem_flags flags,
                                     size_t size, cl_uint align) {
  void *p = alct_real_svmalloc(c, flags, size, align);
  if (size >= (4u << 20)) {
    std::fprintf(stderr, "[alct] %6lld ms svm  %p size=%.1fMB flags=0x%llx\n",
                 alct_ms(), p, size / 1048576.0,
                 (unsigned long long)flags);
    std::fflush(stderr);
  }
  return p;
}

cl_mem CL_API_CALL alct_createbuffer_hook(cl_context c, cl_mem_flags flags,
                                          size_t size, void *host_ptr,
                                          cl_int *err) {
  cl_mem m = alct_real_createbuffer(c, flags, size, host_ptr, err);
  if (size >= (4u << 20)) {
    std::fprintf(stderr, "[alct] %6lld ms buf  %p size=%.1fMB flags=0x%llx\n",
                 alct_ms(), (void *)m, size / 1048576.0,
                 (unsigned long long)flags);
    std::fflush(stderr);
  }
  return m;
}

// ---- [NNTR_CL_ALLOC_TRACE] IAT-level clCreateBuffer hook -------------------
// The pointer trampoline above only intercepts calls routed through the
// loader globals; several TUs import-link OpenCL.dll directly (it is in
// nntrainer.dll's import table), so their clCreateBuffer traffic bypassed the
// trace entirely — round-12 W3 saw 423 v8c backing creations produce ZERO
// [alct] buf lines. Patch the OpenCL.dll!clCreateBuffer IAT slot of every
// loaded module instead, and log the RETURN ADDRESS (module+offset) so the
// call site class is identifiable.
using PFN_clCreateBuffer_t = cl_mem(CL_API_CALL *)(cl_context, cl_mem_flags,
                                                   size_t, void *, cl_int *);
PFN_clCreateBuffer_t alct_iat_real_createbuffer = nullptr;

cl_mem CL_API_CALL alct_iat_createbuffer_hook(cl_context c, cl_mem_flags flags,
                                              size_t size, void *host_ptr,
                                              cl_int *err) {
  void *ret = _ReturnAddress();
  cl_mem m = alct_iat_real_createbuffer(c, flags, size, host_ptr, err);
  if (size >= (4u << 20)) {
    char modname[64] = "?";
    long long off = 0;
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)ret, &hm) &&
        hm) {
      char path[MAX_PATH] = {0};
      GetModuleFileNameA(hm, path, sizeof(path) - 1);
      const char *base = std::strrchr(path, '\\');
      std::snprintf(modname, sizeof(modname), "%s", base ? base + 1 : path);
      off = (long long)((char *)ret - (char *)hm);
    }
    std::fprintf(stderr,
                 "[alct] %6lld ms IATbuf %p size=%zu (%.1fMB) flags=0x%llx "
                 "ret=%s+0x%llx\n",
                 alct_ms(), (void *)m, size, size / 1048576.0,
                 (unsigned long long)flags, modname, off);
    std::fflush(stderr);
  }
  return m;
}

/** Patch mod's IAT entry for OpenCL.dll!clCreateBuffer. Returns true if a
 *  slot was replaced. */
bool alct_patch_module_iat(HMODULE mod) {
  auto *base = reinterpret_cast<uint8_t *>(mod);
  auto *dos = reinterpret_cast<IMAGE_DOS_HEADER *>(base);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE)
    return false;
  auto *nt = reinterpret_cast<IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE)
    return false;
  const auto &dir =
    nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (dir.VirtualAddress == 0 || dir.Size == 0)
    return false;
  auto *imp =
    reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress);
  bool patched = false;
  for (; imp->Name != 0; ++imp) {
    const char *dll = reinterpret_cast<const char *>(base + imp->Name);
    if (_stricmp(dll, "OpenCL.dll") != 0)
      continue;
    auto *oft =
      reinterpret_cast<IMAGE_THUNK_DATA *>(base + imp->OriginalFirstThunk);
    auto *ft = reinterpret_cast<IMAGE_THUNK_DATA *>(base + imp->FirstThunk);
    for (; oft->u1.AddressOfData != 0; ++oft, ++ft) {
      if (oft->u1.Ordinal & IMAGE_ORDINAL_FLAG)
        continue;
      auto *ibn = reinterpret_cast<IMAGE_IMPORT_BY_NAME *>(
        base + oft->u1.AddressOfData);
      if (std::strcmp(reinterpret_cast<const char *>(ibn->Name),
                      "clCreateBuffer") != 0)
        continue;
      DWORD old = 0;
      if (!VirtualProtect(&ft->u1.Function, sizeof(void *), PAGE_READWRITE,
                          &old))
        continue;
      ft->u1.Function =
        reinterpret_cast<ULONG_PTR>(&alct_iat_createbuffer_hook);
      VirtualProtect(&ft->u1.Function, sizeof(void *), old, &old);
      patched = true;
    }
  }
  return patched;
}

void alct_patch_all_modules() {
  HMODULE ocl = GetModuleHandleA("OpenCL.dll");
  if (!ocl)
    return;
  alct_iat_real_createbuffer = reinterpret_cast<PFN_clCreateBuffer_t>(
    GetProcAddress(ocl, "clCreateBuffer"));
  if (!alct_iat_real_createbuffer)
    return;
  HMODULE mods[512];
  DWORD needed = 0;
  if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed))
    return;
  int n = 0;
  const int count = (int)(needed / sizeof(HMODULE));
  for (int i = 0; i < count && i < 512; ++i)
    if (mods[i] != ocl && alct_patch_module_iat(mods[i]))
      ++n;
  std::fprintf(stderr, "[alct] IAT clCreateBuffer patched in %d modules\n", n);
  std::fflush(stderr);
}
// ---------------------------------------------------------------------------

cl_int CL_API_CALL flws_enqueue_hook(cl_command_queue q, cl_kernel k,
                                     cl_uint dim, const size_t *goff,
                                     const size_t *gws, const size_t *lws,
                                     cl_uint nwl, const cl_event *wl,
                                     cl_event *ev) {
  static std::mutex mtx;
  static std::set<std::string> seen;
  char nm[128] = {0};
  if (clGetKernelInfo)
    clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm, nullptr);
  bool first = false;
  {
    std::lock_guard<std::mutex> lk(mtx);
    first = seen.insert(nm).second;
  }
  if (!first || !clFinish)
    return flws_real_enqueue(q, k, dim, goff, gws, lws, nwl, wl, ev);
  clFinish(q);
  const long long ws0 = (long long)flws_ws_kb();
  const cl_int rc =
    flws_real_enqueue(q, k, dim, goff, gws, lws, nwl, wl, ev);
  clFinish(q);
  const long long ws1 = (long long)flws_ws_kb();
  static int seq = 0;
  std::fprintf(stderr, "[klws] #%03d %-44s dWS=%+lld KB (%.1f MB)\n", ++seq,
               nm, ws1 - ws0, (ws1 - ws0) / 1024.0);
  std::fflush(stderr);
  return rc;
}
} // namespace
#endif // _WIN32

/**
 * @brief Utility to load the required OpenCL APIs
 *
 * @param libopencl
 */
void LoadOpenCLFunctions(void *libopencl) {
  LoadFunction(clGetPlatformIDs);
  LoadFunction(clGetDeviceIDs);
  LoadFunction(clGetDeviceInfo);
  LoadFunction(clCreateContext);
  LoadFunction(clCreateCommandQueue);
  LoadFunction(clCreateBuffer);
  LoadFunction(clCreateSubBuffer);
  LoadFunction(clCreateImage);
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clEnqueueMapBuffer);
  LoadFunction(clEnqueueUnmapMemObject);
  LoadFunction(clEnqueueWriteBufferRect);
  LoadFunction(clEnqueueReadBufferRect);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clCreateProgramWithBinary);
  LoadFunction(clBuildProgram);
  LoadFunction(clGetProgramInfo);
  LoadFunction(clGetProgramBuildInfo);
  LoadFunction(clRetainProgram);
  LoadFunction(clCreateKernel);
  LoadFunction(clSetKernelArg);
  LoadFunction(clEnqueueNDRangeKernel);
  LoadFunction(clGetEventProfilingInfo);
  LoadFunction(clRetainContext);
  LoadFunction(clReleaseContext);
  LoadFunction(clRetainCommandQueue);
  LoadFunction(clReleaseCommandQueue);
  LoadFunction(clReleaseMemObject);
  LoadFunction(clFlush);
  LoadFunction(clFinish);
  LoadFunction(clSVMAlloc);
  LoadFunction(clSVMFree);
  LoadFunction(clEnqueueSVMMap);
  LoadFunction(clEnqueueSVMUnmap);
  LoadFunction(clSetKernelArgSVMPointer);
  LoadFunction(clWaitForEvents);
  LoadFunction(clGetKernelInfo);
  LoadFunction(clReleaseEvent);
  LoadFunction(clGetExtensionFunctionAddressForPlatform);
  LoadFunction(clCreateCommandQueueWithProperties);

#if defined(_WIN32)
  // [NNTR_CL_FIRSTLAUNCH_WS] install the first-launch WS probe trampoline.
  if (std::getenv("NNTR_CL_FIRSTLAUNCH_WS") && clEnqueueNDRangeKernel) {
    flws_real_enqueue = clEnqueueNDRangeKernel;
    clEnqueueNDRangeKernel = flws_enqueue_hook;
    std::fprintf(stderr, "[klws] first-launch WS probe installed\n");
    std::fflush(stderr);
  }
  // [NNTR_CL_ALLOC_TRACE] install the large-allocation trace trampolines.
  if (std::getenv("NNTR_CL_ALLOC_TRACE")) {
    if (clSVMAlloc) {
      alct_real_svmalloc = clSVMAlloc;
      clSVMAlloc = alct_svmalloc_hook;
    }
    if (clCreateBuffer) {
      alct_real_createbuffer = clCreateBuffer;
      clCreateBuffer = alct_createbuffer_hook;
    }
    // Import-linked callers bypass the pointers above — patch every loaded
    // module's OpenCL.dll!clCreateBuffer IAT slot (idempotent per process:
    // the first LoadOpenCL instance that runs patches all modules; later
    // instances re-patch the already-hooked slots to the same function).
    alct_patch_all_modules();
  }
#endif
}

PFN_clGetPlatformIDs clGetPlatformIDs;
PFN_clGetDeviceIDs clGetDeviceIDs;
PFN_clGetDeviceInfo clGetDeviceInfo;
PFN_clCreateContext clCreateContext;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateBuffer clCreateBuffer;
PFN_clCreateSubBuffer clCreateSubBuffer;
PFN_clCreateImage clCreateImage;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect;
PFN_clCreateProgramWithSource clCreateProgramWithSource;
PFN_clCreateProgramWithBinary clCreateProgramWithBinary;
PFN_clBuildProgram clBuildProgram;
PFN_clGetProgramInfo clGetProgramInfo;
PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
PFN_clRetainProgram clRetainProgram;
PFN_clCreateKernel clCreateKernel;
PFN_clSetKernelArg clSetKernelArg;
PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
PFN_clRetainContext clRetainContext;
PFN_clReleaseContext clReleaseContext;
PFN_clRetainCommandQueue clRetainCommandQueue;
PFN_clReleaseCommandQueue clReleaseCommandQueue;
PFN_clReleaseMemObject clReleaseMemObject;
PFN_clFlush clFlush;
PFN_clFinish clFinish;
PFN_clSVMAlloc clSVMAlloc;
PFN_clSVMFree clSVMFree;
PFN_clEnqueueSVMMap clEnqueueSVMMap;
PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
PFN_clWaitForEvents clWaitForEvents;
PFN_clGetKernelInfo clGetKernelInfo;
PFN_clReleaseEvent clReleaseEvent;
PFN_clGetExtensionFunctionAddressForPlatform
  clGetExtensionFunctionAddressForPlatform;
PFN_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties;
} // namespace nntrainer::opencl
