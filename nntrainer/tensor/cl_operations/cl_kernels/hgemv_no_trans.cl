#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Y = A * X, A read row-wise. Same KSPLIT cooperative reduction as hgemv.cl --
// see that file for the KSPLIT / MAXLWSX contract with the host launcher.
#define KSPLIT 8
#define MAXLWSX 32

__kernel void sgemv_cl_noTrans_fp16(const __global half *A,
                                    const __global half *X, __global half *Y,
                                    unsigned int N, unsigned int lda) {
  const unsigned int i = get_global_id(0);
  const unsigned int s = get_local_id(1);
  const unsigned int lx = get_local_id(0);
  const unsigned int lsx = get_local_size(0);

  float y0 = 0.0f;
  for (unsigned int j = s; j < N; j += KSPLIT)
    y0 += A[j + i * lda] * X[j];

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
