// Register-blocked C = A(MxK) * B(KxN), row-major, no transpose.
// FP32 twin of hgemm_no_trans.cl -- same tiling, same launch geometry, same
// accumulation order (float, k ascending over the whole of K, edges
// zero-filled). See that file for why the outputs are blocked into registers.
#define TSM 64            // output tile rows (M) per work-group
#define TSN 64            // output tile cols (N) per work-group
#define TSK 16            // K-slab staged in local memory per iteration
#define WPTM 4            // output rows per work-item
#define WPTN 4            // output cols per work-item
#define RTSM (TSM / WPTM) // local size in M
#define RTSN (TSN / WPTN) // local size in N
#define WGS (RTSM * RTSN)
#define LPTA (TSM * TSK / WGS) // A elements staged per work-item per slab
#define LPTB (TSK * TSN / WGS) // B elements staged per work-item per slab

__kernel void sgemm_cl_noTrans(__global const float *A, __global const float *B,
                               __global float *C, const int M, const int N,
                               const int K) {
  const int tidn = get_local_id(0); // 0 .. RTSN-1
  const int tidm = get_local_id(1); // 0 .. RTSM-1
  const int offsetM = TSM * get_group_id(1);
  const int offsetN = TSN * get_group_id(0);
  const int lid = tidm * RTSN + tidn;

  // Asub is staged k-major so the inner loop walks a contiguous run per k.
  __local float Asub[TSK][TSM];
  __local float Bsub[TSK][TSN];

  float acc[WPTM][WPTN];
  for (int wm = 0; wm < WPTM; ++wm)
    for (int wn = 0; wn < WPTN; ++wn)
      acc[wm][wn] = 0.0f;

  const int numTiles = (K + TSK - 1) / TSK;
  for (int t = 0; t < numTiles; ++t) {
    const int kbase = t * TSK;

    for (int i = 0; i < LPTA; ++i) {
      const int idx = lid + i * WGS;
      const int m = idx / TSK;
      const int k = idx % TSK;
      const int gm = offsetM + m;
      const int gk = kbase + k;
      Asub[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
    }
    for (int i = 0; i < LPTB; ++i) {
      const int idx = lid + i * WGS;
      const int k = idx / TSN;
      const int n = idx % TSN;
      const int gk = kbase + k;
      const int gn = offsetN + n;
      Bsub[k][n] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TSK; ++k) {
      float areg[WPTM];
      float breg[WPTN];
      for (int wm = 0; wm < WPTM; ++wm)
        areg[wm] = Asub[k][tidm * WPTM + wm];
      for (int wn = 0; wn < WPTN; ++wn)
        breg[wn] = Bsub[k][tidn * WPTN + wn];
      for (int wm = 0; wm < WPTM; ++wm)
        for (int wn = 0; wn < WPTN; ++wn)
          acc[wm][wn] += areg[wm] * breg[wn];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int wm = 0; wm < WPTM; ++wm) {
    const int gm = offsetM + tidm * WPTM + wm;
    if (gm >= M)
      continue;
    for (int wn = 0; wn < WPTN; ++wn) {
      const int gn = offsetN + tidn * WPTN + wn;
      if (gn < N)
        C[gm * N + gn] = acc[wm][wn];
    }
  }
}
