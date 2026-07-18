/* SPDX-License-Identifier: Apache-2.0
 *
 * sflare_exit_keeper.c — spawn-and-forget GPU keeper for Windows/Intel iGPU.
 *
 * Keeps the GPU actively submitting trivial kernels for argv[1] ms
 * (default 3000), then exits. Purpose: a process that owned large GPU-mapped
 * (SVM) state pays a 0.8-2.4s kernel-side teardown AFTER its last
 * instruction (Windows defers the GPU allocation teardown to process death,
 * and the teardown runs slowly once the GPU parks). Keeping the device busy
 * from a SEPARATE process through that rundown window makes the teardown run
 * at awake speed: measured 2.4s -> ~0.4s on Xe3. The parent spawns this
 * detached right before returning from main; it costs ~3s of trivial GPU
 * work and its own exit tail is negligible (no large allocations).
 *
 * Build (MSVC): cl sflare_exit_keeper.c /I<CL headers> /link OpenCL.lib
 *
 * usage: sflare_exit_keeper [keep_ms]                 (burst mode)
 *        sflare_exit_keeper --watch <pid> [linger_ms] [gap_ms]
 *   --watch: throttled while <pid> lives (one trivial kernel every gap_ms,
 *   default 100 — negligible contention), then CONTINUOUS for linger_ms
 *   (default 3000) after it dies, covering its teardown window.
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#endif

static cl_platform_id pick_intel_platform(void) {
  cl_platform_id plats[8];
  cl_uint n = 0;
  clGetPlatformIDs(8, plats, &n);
  for (cl_uint i = 0; i < n; ++i) {
    char name[256] = {0};
    clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
    if (strstr(name, "Intel") != NULL)
      return plats[i];
  }
  return n ? plats[0] : NULL;
}

int main(int argc, char **argv) {
  int keep_ms = 3000;
  long watch_pid = 0;
  int gap_ms = 100;
  if (argc > 2 && strcmp(argv[1], "--watch") == 0) {
    watch_pid = atol(argv[2]);
    if (argc > 3)
      keep_ms = atoi(argv[3]);
    if (argc > 4)
      gap_ms = atoi(argv[4]);
  } else if (argc > 1) {
    keep_ms = atoi(argv[1]);
  }
  if (keep_ms <= 0 || keep_ms > 30000)
    keep_ms = 3000;
  if (gap_ms < 10 || gap_ms > 2000)
    gap_ms = 100;

  cl_platform_id plat = pick_intel_platform();
  if (!plat)
    return 1;
  cl_device_id dev = NULL;
  if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS)
    return 1;
  cl_int err = CL_SUCCESS;
  cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
  const char *src = "__kernel void k(__global int *o){ o[get_global_id(0)] += 1; }";
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
  clBuildProgram(prog, 1, &dev, "", NULL, NULL);
  cl_kernel k = clCreateKernel(prog, "k", &err);
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096, NULL, &err);
  clSetKernelArg(k, 0, sizeof(buf), &buf);

#if defined(_WIN32)
  size_t gws = 1024;
  if (watch_pid > 0) {
    /* throttled phase: barely-there pings while the parent is alive */
    HANDLE hp = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)watch_pid);
    if (hp) {
      for (;;) {
        clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFinish(q);
        /* Sleep doubles as the parent-death poll interval */
        if (WaitForSingleObject(hp, (DWORD)gap_ms) != WAIT_TIMEOUT)
          break; /* parent gone (or wait error) -> go continuous */
      }
      CloseHandle(hp);
    }
  }
  /* continuous phase: cover the (parent's) teardown window */
  ULONGLONG deadline = GetTickCount64() + (ULONGLONG)keep_ms;
  while (GetTickCount64() < deadline) {
    clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clFinish(q); /* no gaps for the GPU to park */
  }
#endif

  clReleaseMemObject(buf);
  clReleaseKernel(k);
  clReleaseProgram(prog);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);
  return 0;
}
