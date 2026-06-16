// Feasibility probe for cl_qcom_recordable_queues on Adreno 840.
// Verifies: (1) a CL_QUEUE_RECORDABLE_QCOM queue can be created, (2) a kernel
// sequence can be recorded + replayed, (3) a kernel arg can be updated per
// replay (cl_array_arg_qcom) -- the mechanism needed for the per-token decode
// active-row/seq offsets. Build: NDK clang, link device libOpenCL.so.
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef cl_recording_qcom (*pfnNewRec)(cl_command_queue, cl_int *);
typedef cl_int (*pfnEndRec)(cl_recording_qcom);
typedef cl_int (*pfnEnqRec)(cl_command_queue, cl_recording_qcom, size_t,
                            const cl_array_arg_qcom *, size_t,
                            const cl_offset_qcom *, size_t,
                            const cl_workgroup_qcom *, size_t,
                            const cl_workgroup_qcom *, cl_uint,
                            const cl_event *, cl_event *);

#define CK(x)                                                                  \
  do {                                                                         \
    cl_int _e = (x);                                                           \
    if (_e != CL_SUCCESS) {                                                    \
      printf("FAIL %s -> %d\n", #x, _e);                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char *SRC =
  "__kernel void addk(__global int* buf, int add){ buf[get_global_id(0)] += add; }\n";

int main() {
  cl_platform_id plat;
  CK(clGetPlatformIDs(1, &plat, NULL));
  cl_device_id dev;
  CK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));

  char ext[4096] = {0};
  clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, NULL);
  printf("recordable_queues advertised: %s\n",
         strstr(ext, "cl_qcom_recordable") ? "YES" : "NO");

  pfnNewRec NewRec =
    (pfnNewRec)clGetExtensionFunctionAddressForPlatform(plat, "clNewRecordingQCOM");
  pfnEndRec EndRec =
    (pfnEndRec)clGetExtensionFunctionAddressForPlatform(plat, "clEndRecordingQCOM");
  pfnEnqRec EnqRec =
    (pfnEnqRec)clGetExtensionFunctionAddressForPlatform(plat, "clEnqueueRecordingQCOM");
  printf("fn ptrs: NewRec=%p EndRec=%p EnqRec=%p\n", (void *)NewRec,
         (void *)EndRec, (void *)EnqRec);
  if (!NewRec || !EndRec || !EnqRec) {
    printf("FAIL: extension function addresses not found\n");
    return 1;
  }

  cl_int err;
  cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  CK(err);

  cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_RECORDABLE_QCOM, 0};
  cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
  if (err != CL_SUCCESS) {
    printf("FAIL: recordable queue create -> %d\n", err);
    return 1;
  }
  printf("recordable queue created OK\n");
  // A normal (non-recordable) queue for host I/O: a recordable queue rejects
  // plain enqueues like clEnqueueReadBuffer (CL_INVALID_OPERATION).
  cl_command_queue qio = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
  CK(err);

  cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
  CK(err);
  CK(clBuildProgram(prog, 1, &dev, NULL, NULL, NULL));
  cl_kernel k = clCreateKernel(prog, "addk", &err);
  CK(err);

  int host[4] = {0, 0, 0, 0};
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(host), host, &err);
  CK(err);
  int add = 10;
  CK(clSetKernelArg(k, 0, sizeof(cl_mem), &buf));
  CK(clSetKernelArg(k, 1, sizeof(int), &add));

  // Record one dispatch.
  cl_recording_qcom rec = NewRec(q, &err);
  if (err != CL_SUCCESS || !rec) {
    printf("FAIL: clNewRecordingQCOM -> %d\n", err);
    return 1;
  }
  size_t gws = 4;
  CK(clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, NULL));
  CK(EndRec(rec));
  printf("recorded 1 dispatch OK\n");

  // Replay #1: no arg update (add stays 10).
  CK(EnqRec(q, rec, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL));
  CK(clFinish(q));
  CK(clEnqueueReadBuffer(qio, buf, CL_TRUE, 0, sizeof(host), host, 0, NULL, NULL));
  printf("after replay#1 (add=10): %d %d %d %d  [expect 10 10 10 10]\n", host[0],
         host[1], host[2], host[3]);

  // Replay #2: update arg 1 (add -> 100) via cl_array_arg_qcom.
  int add2 = 100;
  cl_array_arg_qcom upd = {0 /*dispatch_index*/, 1 /*arg_index*/, sizeof(int),
                           &add2};
  CK(EnqRec(q, rec, 1, &upd, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL));
  CK(clFinish(q));
  CK(clEnqueueReadBuffer(qio, buf, CL_TRUE, 0, sizeof(host), host, 0, NULL, NULL));
  printf("after replay#2 (add=100): %d %d %d %d  [expect 110 110 110 110]\n",
         host[0], host[1], host[2], host[3]);

  int ok = (host[0] == 110 && host[1] == 110 && host[2] == 110 && host[3] == 110);
  printf("RESULT: %s\n", ok ? "PASS (record+replay+arg-update work)" : "FAIL");
  return ok ? 0 : 1;
}
