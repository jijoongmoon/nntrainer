#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Y = A^T * X, A read column-wise (consecutive outputs are consecutive
// addresses).
//
// KSPLIT work-items share each output and each walk a strided slice of the
// reduction, then combine through local memory. One work-item per output could
// only offer dim1 work-items to hide the weight stream's latency with, which
// left the 2688x2688 case an order of magnitude below the bandwidth the device
// actually delivers.
//
// KSPLIT and MAXLWSX are contracted with the host launcher (GEMV_KSPLIT /
// GEMV_MAX_LWS_X in blas_kernels_templates.h): axis 1 of the NDRange is exactly
// KSPLIT and axis 0's local size never exceeds MAXLWSX, so the scratch below
// always covers the group.
#define KSPLIT 8
#define MAXLWSX 32

__kernel void sgemv_cl_fp16(const __global half *A, const __global half *X,
                            __global half *Y, unsigned int N,
                            unsigned int lda) {
  const unsigned int i = get_global_id(0);
  const unsigned int s = get_local_id(1);
  const unsigned int lx = get_local_id(0);
  const unsigned int lsx = get_local_size(0);

  float y0 = 0.0f;
  for (unsigned int j = s; j < N; j += KSPLIT)
    y0 += A[i + j * lda] * X[j];

  __local float part[KSPLIT * MAXLWSX];
  part[s * lsx + lx] = y0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (s == 0) {
    float sum = 0.0f;
    for (unsigned int t = 0; t < KSPLIT; ++t)
      sum += part[t * lsx + lx];
    Y[i] = sum;
  }
}
