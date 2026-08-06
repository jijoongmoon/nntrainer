// Y = A * X, FP32 twin of hgemv_no_trans.cl -- same KSPLIT cooperative
// reduction and the same contract with the host launcher (see hgemv.cl).
#define KSPLIT 8
#define MAXLWSX 32

__kernel void sgemv_cl_noTrans(const __global float *A, const __global float *X,
                               __global float *Y, unsigned int N,
                               unsigned int lda) {
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
