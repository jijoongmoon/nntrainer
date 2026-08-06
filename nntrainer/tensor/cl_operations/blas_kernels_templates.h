// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 *
 * @file	blas_kernels_templates.hpp
 * @date	07 July 2025
 * @brief	Common blas OpenCL kernels (common templates used by
 * blas_kernels_fp16.cpp and blas_kernels.cpp)
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @author	Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_TEMPLATES_H__
#define __BLAS_KERNELS_TEMPLATES_H__

#include <cstdio>
#include <stdexcept>
#include <string>

#include <blas_kernels.h>

namespace nntrainer {

/**
 * @brief Name a hard OpenCL failure inside a dense BLAS primitive, then throw.
 *
 * Every step of these routines (staging write, argument bind, dispatch,
 * read-back) used to `return;` on failure. The caller's contract is "the
 * output is written", and there is no fallback behind these calls -- so a
 * bare return leaves the output plane holding whatever was there before and
 * the process keeps running on it. That is the "rc=0 but the text is garbage"
 * failure mode: nothing in the log, exit status 0, wrong numbers. Name the op,
 * the step and the shape on stderr and fail loudly instead.
 *
 * @param op     primitive + dtype, e.g. "sgemm_cl<fp16>"
 * @param step   which OpenCL step refused
 * @param d0d1d2 shape triple (M,N,K for gemm; dim1,dim2,lda for gemv)
 */
[[noreturn]] inline void clBlasFail(const char *op, const char *step,
                                    unsigned int d0, unsigned int d1,
                                    unsigned int d2) {
  char msg[256];
  std::snprintf(msg, sizeof(msg),
                "[cl-blas] %s refused at '%s' (%u x %u x %u): the OpenCL call "
                "failed and the output was left unwritten",
                op, step, d0, d1, d2);
  std::fprintf(stderr, "%s\n", msg);
  std::fflush(stderr);
  throw std::runtime_error(msg);
}

/**
 * @brief Fail with clBlasFail() unless @a cond holds.
 */
#define NNTR_CL_BLAS_REQUIRE(cond, op, step, d0, d1, d2)                       \
  do {                                                                         \
    if (!(cond))                                                               \
      clBlasFail((op), (step), (d0), (d1), (d2));                              \
  } while (0)

/**
 * @brief Number of work-items that cooperate on one GEMV output row.
 *
 * MUST equal KSPLIT in sgemv.cl / sgemv_no_trans.cl / hgemv.cl /
 * hgemv_no_trans.cl: the kernels index their reduction scratch by
 * get_local_id(1) and the host is what sizes that dimension.
 */
constexpr unsigned int GEMV_KSPLIT = 8;

/**
 * @brief Cap on the row-axis work-group size, matching MAXLWSX in those same
 * kernels -- their reduction scratch is a fixed KSPLIT x MAXLWSX array.
 */
constexpr unsigned int GEMV_MAX_LWS_X = 32;

/**
 * @brief Largest row-axis work-group size that exactly divides @a n.
 *
 * The dense GEMV kernels are one-output-per-row-of-work-items and carry no
 * bounds guard, so the global size must stay exactly @a n and the local size
 * must divide it. Picking an exact divisor keeps that contract; a shape with no
 * good divisor degrades to a single row per group rather than reading out of
 * range.
 */
inline int selectGemvLws(unsigned int n, unsigned int cap = GEMV_MAX_LWS_X) {
  for (unsigned int w = cap; w >= 2; w >>= 1) {
    if (n % w == 0)
      return (int)w;
  }
  return 1;
}

/**
 * @brief GEMV on OpenCL.
 *
 * @note Each of A / X / Y is bound either as an SVM pointer (when the caller
 * says the buffer lives in the SVM pool) or staged through the shared
 * ClBufferManager buffers. The distinction is not cosmetic: a coarse-grained
 * SVM pointer is a valid *source* for clEnqueueWriteBufferRect but NOT a valid
 * *destination* for clEnqueueReadBufferRect -- the read reports CL_SUCCESS and
 * never lands, so an SVM-resident output stayed at its previous contents. That
 * is what broke every dense (non-quantized) FC on the OpenCL lane, whose
 * activations all come from the SVM pool. Binding SVM directly also drops the
 * host round-trip entirely (no dim1*dim2 upload per call).
 */
template <typename T = float>
inline static void sgemv_cl_internal(ClContext::SharedPtrClKernel kernel,
                                     const T *matAdata, const T *vecXdata,
                                     T *vecYdata, unsigned int dim1,
                                     unsigned int dim2, unsigned int lda,
                                     const char *op, bool a_svm, bool x_svm,
                                     bool y_svm) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();
  auto &q = blas_cc->command_queue_inst_;

  const size_t dim1_size = sizeof(T) * dim1;
  const size_t dim2_size = sizeof(T) * dim2;
  const size_t dim1_dim2_size = sizeof(T) * dim1 * dim2;

  // A (dim1 x dim2)
  if (a_svm) {
    q.enqueueSVMUnmap(const_cast<T *>(matAdata));
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelSVMArguments(0, matAdata), op,
                         "bind A (svm)", dim1, dim2, lda);
  } else {
    NNTR_CL_BLAS_REQUIRE(clbuffInstance.getInBufferA()->WriteDataRegion(
                           q, dim1_dim2_size, matAdata),
                         op, "stage A", dim1, dim2, lda);
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(
                           0, clbuffInstance.getInBufferA(), sizeof(cl_mem)),
                         op, "bind A", dim1, dim2, lda);
  }

  // X (dim2)
  if (x_svm) {
    q.enqueueSVMUnmap(const_cast<T *>(vecXdata));
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelSVMArguments(1, vecXdata), op,
                         "bind X (svm)", dim1, dim2, lda);
  } else {
    NNTR_CL_BLAS_REQUIRE(
      clbuffInstance.getInBufferB()->WriteDataRegion(q, dim2_size, vecXdata),
      op, "stage X", dim1, dim2, lda);
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(
                           1, clbuffInstance.getInBufferB(), sizeof(cl_mem)),
                         op, "bind X", dim1, dim2, lda);
  }

  // Y (dim1). Write-only for the kernel (it stores every Y[i]), so the old
  // upload of Y's previous contents is dead work and is not reproduced here.
  if (y_svm) {
    q.enqueueSVMUnmap(vecYdata);
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelSVMArguments(2, vecYdata), op,
                         "bind Y (svm)", dim1, dim2, lda);
  } else {
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(
                           2, clbuffInstance.getOutBufferA(), sizeof(cl_mem)),
                         op, "bind Y", dim1, dim2, lda);
  }

  NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(3, &dim2, sizeof(int)), op,
                       "bind dim2", dim1, dim2, lda);
  NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(4, &lda, sizeof(int)), op,
                       "bind lda", dim1, dim2, lda);

  // Axis 0 walks the outputs, axis 1 splits the reduction GEMV_KSPLIT ways.
  //
  // The previous geometry was a flat {dim1,1,1} global over {1,1,1} groups:
  // one work-item per output, each alone in its work-group. That runs one lane
  // per hardware thread, leaves neighbouring outputs -- which the transposed
  // kernel reads as CONSECUTIVE addresses -- in separate groups so nothing
  // coalesces, and offers only dim1 work-items to hide memory latency with.
  // Measured on a 2688x2688 dense FP16 FC on the Intel lane: 7.5 ms/call,
  // ~1.9 GB/s, against a weight plane the quantized GEMV streams at ~66 GB/s.
  const int lws_x = selectGemvLws(dim1);
  const int work_groups_count[3] = {(int)dim1, (int)GEMV_KSPLIT, 1};
  const int work_group_size[3] = {lws_x, (int)GEMV_KSPLIT, 1};

  NNTR_CL_BLAS_REQUIRE(
    q.DispatchCommand(kernel, work_groups_count, work_group_size), op,
    "dispatch", dim1, dim2, lda);

  if (y_svm) {
    // Re-map the GPU-written result for the host consumer (coarse-grained SVM;
    // no-op under the resident mode gate inside enqueueSVMMap).
    q.enqueueSVMMap(vecYdata, dim1_size, false);
  } else {
    NNTR_CL_BLAS_REQUIRE(
      clbuffInstance.getOutBufferA()->ReadDataRegion(q, dim1_size, vecYdata),
      op, "read back Y", dim1, dim2, lda);
  }
}

template <typename T = float>
T dot_cl_internal(ClContext::SharedPtrClKernel kernel, const T *vecAdata,
                  const T *vecXdata, unsigned int dim1) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  T cl_ret = 0;

  do {
    size_t dim1_size = sizeof(T) * dim1;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecXdata);
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(1, clbuffInstance.getInBufferB(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(3, clbuffInstance.getOutBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {1, 1, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, sizeof(T), &cl_ret);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

/**
 * @brief Output tiling of one GEMM kernel, i.e. how it maps outputs to
 * work-items. @a tile_m / @a tile_n are the output tile a work-group owns and
 * @a wpt_m / @a wpt_n how many of those rows/cols each work-item computes, so
 * the work-group is (tile_n/wpt_n) x (tile_m/wpt_m) work-items. The default is
 * the one-output-per-work-item 16x16 tiling the transposing kernels use.
 */
struct GemmTiling {
  unsigned int tile_m = 16;
  unsigned int tile_n = 16;
  unsigned int wpt_m = 1;
  unsigned int wpt_n = 1;
};

/**
 * @brief GEMM on OpenCL.
 *
 * @note See sgemv_cl_internal() for why each of A / B / C is bound either as
 * an SVM pointer or through the shared staging buffers -- a coarse-grained SVM
 * destination silently swallows the clEnqueueReadBufferRect read-back.
 *
 * @note @a tiling must describe the kernel actually passed in: the tile sizes
 * are compiled into the kernel source, and only the launch geometry is chosen
 * here.
 */
template <typename T = float>
inline static void
sgemm_cl_internal(ClContext::SharedPtrClKernel kernel, bool TransA, bool TransB,
                  const T *A, const T *B, T *C, unsigned int M, unsigned int N,
                  unsigned int K, unsigned int lda, unsigned int ldb,
                  unsigned int ldc, const char *op, bool a_svm, bool b_svm,
                  bool c_svm, GemmTiling tiling = {}) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();
  auto &q = blas_cc->command_queue_inst_;

  // sizes will be same for transpose
  const size_t m_k_size = (size_t)M * K * sizeof(T);
  const size_t k_n_size = (size_t)K * N * sizeof(T);
  const size_t m_n_size = (size_t)M * N * sizeof(T);

  if (a_svm) {
    q.enqueueSVMUnmap(const_cast<T *>(A));
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelSVMArguments(0, A), op,
                         "bind A (svm)", M, N, K);
  } else {
    NNTR_CL_BLAS_REQUIRE(
      clbuffInstance.getInBufferA()->WriteDataRegion(q, m_k_size, A), op,
      "stage A", M, N, K);
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(
                           0, clbuffInstance.getInBufferA(), sizeof(cl_mem)),
                         op, "bind A", M, N, K);
  }

  if (b_svm) {
    q.enqueueSVMUnmap(const_cast<T *>(B));
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelSVMArguments(1, B), op,
                         "bind B (svm)", M, N, K);
  } else {
    NNTR_CL_BLAS_REQUIRE(
      clbuffInstance.getInBufferB()->WriteDataRegion(q, k_n_size, B), op,
      "stage B", M, N, K);
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(
                           1, clbuffInstance.getInBufferB(), sizeof(cl_mem)),
                         op, "bind B", M, N, K);
  }

  // C is write-only for every kernel in this family (each stores C[m*N+n] for
  // all valid m,n), so its previous contents are not uploaded.
  if (c_svm) {
    q.enqueueSVMUnmap(C);
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelSVMArguments(2, C), op,
                         "bind C (svm)", M, N, K);
  } else {
    NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(
                           2, clbuffInstance.getOutBufferA(), sizeof(cl_mem)),
                         op, "bind C", M, N, K);
  }

  NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(3, &M, sizeof(int)), op,
                       "bind M", M, N, K);
  NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(4, &N, sizeof(int)), op,
                       "bind N", M, N, K);
  NNTR_CL_BLAS_REQUIRE(kernel->SetKernelArguments(5, &K, sizeof(int)), op,
                       "bind K", M, N, K);

  // Round the output up to whole tiles, then hand each work-item wpt_m x
  // wpt_n of them (the kernel guards its stores, so the rounded-up edge is
  // computed and dropped).
  const unsigned int tiles_m = (M + tiling.tile_m - 1) / tiling.tile_m;
  const unsigned int tiles_n = (N + tiling.tile_n - 1) / tiling.tile_n;
  const int work_groups_count[3] = {
    (int)(tiles_n * (tiling.tile_n / tiling.wpt_n)),
    (int)(tiles_m * (tiling.tile_m / tiling.wpt_m)), 1};

  const int work_group_size[3] = {(int)(tiling.tile_n / tiling.wpt_n),
                                  (int)(tiling.tile_m / tiling.wpt_m), 1};

  NNTR_CL_BLAS_REQUIRE(
    q.DispatchCommand(kernel, work_groups_count, work_group_size), op,
    "dispatch", M, N, K);

  if (c_svm) {
    q.enqueueSVMMap(C, m_n_size, false);
  } else {
    NNTR_CL_BLAS_REQUIRE(
      clbuffInstance.getOutBufferA()->ReadDataRegion(q, m_n_size, C), op,
      "read back C", M, N, K);
  }
}

template <typename T = float>
inline static void
addition_cl_internal(ClContext::SharedPtrClKernel kernel, const T *input,
                     T *res, unsigned int size_input, unsigned int size_res,
                     bool use_svm = false) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim1_size = sizeof(T) * size_input;
  size_t dim2_size = sizeof(T) * size_res;

  if (use_svm) {
    // SVM-direct: input/res are GPU-resident pointers; accumulate in place
    // (res += input) with no host round-trip (residency path). Coarse-grained
    // SVM needs explicit coherence: release host mappings so the GPU sees the
    // current host-side contents (res was just host-written by the copy of the
    // first addend), then re-map res after the dispatch for the host read.
    blas_cc->command_queue_inst_.enqueueSVMUnmap(const_cast<T *>(input));
    blas_cc->command_queue_inst_.enqueueSVMUnmap(res);
    if (!kernel->SetKernelSVMArguments(0, input))
      return;
    if (!kernel->SetKernelSVMArguments(1, res))
      return;
  } else {
    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, input);
    if (!result) {
      return;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);
    if (!result) {
      return;
    }

    result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      return;
    }

    result = kernel->SetKernelArguments(1, clbuffInstance.getOutBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      return;
    }
  }

  result = kernel->SetKernelArguments(2, &size_input, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &size_res, sizeof(int));
  if (!result) {
    return;
  }

  // lws was {1,1,1} ("test-value") -> one work-item per work-group, i.e. ~1/64
  // of the SIMD wave used (measured: addition_cl_fp16 at ~1.5 GB/s). Use a
  // full work-group; round the global size up to a multiple of lws -- both
  // addition_cl / addition_cl_fp16 guard `if (idx < size_res)` so the
  // rounded-up tail work-items are no-ops.
  const int add_lws = 64;
  const int add_gws = (((int)size_res + add_lws - 1) / add_lws) * add_lws;
  const int work_groups_count[3] = {add_gws, 1, 1};
  const int work_group_size[3] = {add_lws, 1, 1};
  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  if (!use_svm) {
    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);

    if (!result) {
      return;
    }
  } else {
    // re-map the in-place result so the host sees the GPU-written values.
    // Kept BLOCKING: an async map here measured faster but corrupted output
    // (coherence race) — removing the pair entirely needs a GPU-resident
    // producer+consumer, not an async flip.
    blas_cc->command_queue_inst_.enqueueSVMMap(res, dim2_size, true);
  }
}

template <typename T = float>
inline static void rmsnorm_cl_internal(ClContext::SharedPtrClKernel kernel,
                                       const T *input, const T *gamma,
                                       T *result, const T epsilon,
                                       unsigned int height, unsigned int width,
                                       const bool use_svm = true) {
  unsigned dim_in = height * width;
  unsigned dim_gamma = width;
  unsigned size_in = dim_in * sizeof(T);
  unsigned size_gamma = dim_gamma * sizeof(T);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  if (use_svm) {
    if (!kernel->SetKernelSVMArguments(0, input)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(1, result)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(2, gamma)) {
      return;
    }
  } else {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getInBufferA()->WriteDataRegion(
          blas_cc->command_queue_inst_, size_in, input)) {
      return;
    }
    if (!clbuffInstance.getInBufferB()->WriteDataRegion(
          blas_cc->command_queue_inst_, size_gamma, gamma)) {
      return;
    }

    if (!kernel->SetKernelArguments(
          0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
    if (!kernel->SetKernelArguments(
          1, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
    if (!kernel->SetKernelArguments(
          2, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
  }

  if (!kernel->SetKernelArguments(3, &epsilon, sizeof(float))) {
    return;
  }
  if (!kernel->SetKernelArguments(4, &height, sizeof(int))) {
    return;
  }
  if (!kernel->SetKernelArguments(5, &width, sizeof(int))) {
    return;
  }
#ifdef __ANDROID__
  constexpr int SUBGROUP_SIZE = 64;
#else
  constexpr int SUBGROUP_SIZE = 32;
#endif
  const int work_groups_count[3] = {static_cast<int>(height) * SUBGROUP_SIZE, 1,
                                    1};

  const int work_group_size[3] = {SUBGROUP_SIZE, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel, work_groups_count,
                                                    work_group_size)) {
    return;
  }

  if (!use_svm) {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
          blas_cc->command_queue_inst_, size_in, result)) {
      return;
    }
  } else {
    blas_cc->command_queue_inst_.enqueueSVMMap(result, size_in, false);
  }
}

template <typename T = float>
inline static void sscal_cl_internal(ClContext::SharedPtrClKernel kernel, T *X,
                                     const unsigned int N, const float alpha) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t x_size = N * sizeof(T);

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, x_size, X);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, &alpha, sizeof(float));
  if (!result) {
    return;
  }

  const int work_groups_count[3] = {(int)N, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, x_size, X);
  if (!result) {
    return;
  }
}

template <typename T = float>
inline static void transpose_cl_axis_internal(
  ClContext::SharedPtrClKernel kernel, const T *in, T *res,
  unsigned int input_batch_size, unsigned int input_channels,
  unsigned int input_height, unsigned int input_width, unsigned int axis) {

  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim_size =
    sizeof(T) * input_batch_size * input_height * input_width * input_channels;

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim_size, in);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim_size, res);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, &input_batch_size, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &input_channels, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(4, &input_height, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(5, &input_width, sizeof(int));
  if (!result) {
    return;
  }

  int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
  if (axis == 2)
    work_groups_count[0] = (int)input_channels;

  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, dim_size, res);
  if (!result) {
    return;
  }
}

} // namespace nntrainer

#endif
