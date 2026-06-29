// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.cpp
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <clblast_interface.h>

namespace nntrainer {
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans, bool trans_m) {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < input.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const Tensor this_b = input.getBatchSlice(b, 1);
    Tensor m_b = m.getBatchSlice(b, 1);
    Tensor result_b = result.getBatchSlice(b, 1);

    dotCl(this_b, m_b, result_b, trans, trans_m);
  }
}

Tensor dotCl(Tensor const &input, Tensor const &m, bool trans, bool trans_m) {
  Tensor output("", input.getFormat(), input.getDataType());
  dotCl(input, m, output, trans, trans_m);

  return output;
}

void dotCl(Tensor const &input, Tensor const &m, Tensor &result, bool trans,
           bool trans_m) {
  unsigned int dim1, dim2, mdim1, mdim2;
  if (input.getFormat() == Tformat::NHWC) {
    dim1 = input.batch() * input.height() * input.width();
    dim2 = input.channel();
    mdim1 = m.batch() * m.height() * m.width();
    mdim2 = m.channel();
  } else {
    dim1 = input.batch() * input.channel() * input.height();
    dim2 = input.width();
    mdim1 = m.batch() * m.channel() * m.height();
    mdim2 = m.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(),
                           input.getTensorType()); //  NHWC Result Tensor
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(), input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  }

  lda = dim2;
  ldb = mdim2;
  ldc =
    (input.getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    const float *mdata = m.getData();
    float *rdata = result.getData();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      // *rdata = dot_cl(data, mdata, K) + (*rdata);
      *rdata = dot_cl(K, data, mdata) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      gemv_cl(0, trans, dim1, dim2, 1.0f, data, lda, mdata, 0.0f, rdata, 1);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      gemv_cl(0, !trans_m, mdim1, mdim2, 1.0f, mdata, ldb, data, 0.0f, rdata,
              1);
    }
    /// case others: use gemm
    else {
      if (input.getFormat() == Tformat::NHWC) {
        sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc);
      } else {
        gemm_cl(0, trans, trans_m, M, N, K, 1.0f, data, (trans) ? M : K, mdata,
                (trans_m) ? K : N, 1.0f, rdata, N);
      }
    }
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = input.getData<_FP16>();
    const _FP16 *mdata = m.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = dot_cl(data, mdata, K) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      trans ? sgemv_cl(data, mdata, rdata, trans, dim2, dim1, lda)
            : sgemv_cl(data, mdata, rdata, trans, dim1, dim2, lda);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      trans_m ? sgemv_cl(mdata, data, rdata, !trans_m, mdim1, mdim2, ldb)
              : sgemv_cl(mdata, data, rdata, !trans_m, mdim2, mdim1, ldb);
    }
    /// case others: use sgemm
    else {
      sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void multiplyCl(Tensor &input, float const &value) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData<float>();
    unsigned int len = input.size();

    scal_cl(len, value, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data = input.getData<_FP16>();
    unsigned int len = input.size();
    sscal_cl(data, len, value);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void add_i_cl(Tensor &result, Tensor const &input) {

  NNTR_THROW_IF(input.getData() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(result.getData() == nullptr, std::invalid_argument)
    << result.getName() << " is not allocated";

  // Bind device memory directly (SVM-direct, in-place accumulate) only when both
  // tensors are GPU-resident (SVM pool); otherwise fall back to the host
  // round-trip. Keeps the residual on the GPU when residency is enabled.
  const bool use_svm =
    result.getMemoryData() && result.getMemoryData()->isSVM() &&
    input.getMemoryData() && input.getMemoryData()->isSVM();

  // Broadcasting done for the case where batch size vary for both inputs
  // If batch size vary, batch size of input must be 1
  if ((result.getDim() == input.getDim()) ||
      (result.getDim() != input.getDim() && input.batch() == 1 &&
       result.channel() == input.channel() &&
       result.height() == input.height() && result.width() == input.width())) {

    if (result.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *Y = result.getData();
      const float *X = input.getData();

      // axpy with alpha=1 is just elementwise add. Use our own addition_cl
      // kernel so this path doesn't pull in CLBlast (the bigger BLAS dep
      // is gated behind -Denable-clblast; the v8c paper path doesn't need
      // it). FP16 already uses addition_cl below — make FP32 symmetric.
      unsigned int size_input = input.size();
      for (unsigned int i = 0; i < result.batch() / input.batch(); ++i) {
        addition_cl(X, Y, size_input, size_input, use_svm);
        Y += size_input;
      }
    } else if (result.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      unsigned int size_res = result.size();
      unsigned int size_input = input.size();
      _FP16 *data_res = result.getData<_FP16>();
      const _FP16 *data_input = input.getData<_FP16>();

      addition_cl(data_input, data_res, size_input, size_res, use_svm);

#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }
  }

  else {
    throw std::invalid_argument(
      "Error: Broadcasting not supported for these dimensions!");
  }
}

void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result) {

  unsigned int input_batch_size, input_height, input_width, input_channels;

  input_batch_size = in.batch();
  input_height = in.height();
  input_width = in.width();
  input_channels = in.channel();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = in.getData();
    float *rdata = result.getData();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }

  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = in.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void copyCl(const Tensor &input, Tensor &result) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    float *rdata = result.getData();

    unsigned int len = input.size();

    copy_cl(len, data, rdata);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, copyCl not supported for FP16");
#endif
  }
}

float nrm2Cl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = nrm2_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, nrm2Cl not supported for FP16");
#endif
  }

  return result;
}

float asumCl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = asum_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, asumCl not supported for FP16");
#endif
  }

  return result;
}

int amaxCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amax_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

int aminCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amin_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

} // namespace nntrainer

// =============================================================================
// v8c (paper 8/4/4) dispatch entry — env-gated, dotCl fallback.
// =============================================================================
#include "blas_kernels.h"
#include "cl_tensor_backing_pool.h"
#include "cl_tensor_view.h"
#include <atomic>
#include <network_graph.h> // resolveResidentEdge (cl_mem residency overlay)
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace nntrainer {
namespace {
struct V8cWeightEntry {
  std::unique_ptr<tv::TensorBacking> backing;
  cl_mem scale_buf = nullptr;       // [N] fp32 recip-scale (owned)
  cl_mem row_sum_w_int4 = nullptr;  // [N] int32 sum_k(int4 w_nk) (owned)
  unsigned int N = 0, K = 0;
  cl_mem weight_image = nullptr; // cached image2d view (also released via TensorBacking)
  cl_mem weight_buf = nullptr;   // raw backing buffer (buffer-path / Intel NEO)
};

// Buffer-path (NNTR_V8C_BUF): on Intel NEO the v8c GEMM uses the *_buf kernels
// whose args are declared __global uint4* — they must be bound to raw cl_mem
// BUFFERS, not image2d objects. [T8] single source of truth is
// blas_kernels.cpp::v8c_use_buffer_path() (caps-derived from vendor_id; the env
// flag still overrides); this file-local name forwards to it.
static bool v8c_buffer_path() { return v8c_use_buffer_path(); }

static bool v8c_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_FC_INT8_GPU") != nullptr ? 1 : 0;
  return cached != 0;
}

static std::mutex &v8c_cache_mtx() {
  static std::mutex m;
  return m;
}
static std::unordered_map<const void *, V8cWeightEntry> &v8c_weight_cache() {
  static std::unordered_map<const void *, V8cWeightEntry> c;
  return c;
}

// Number of per-fanout activation slots. RACE#1 (R3) fix: the GEMM reads the
// int8 activation through an image2d-from-buffer VIEW; the Adreno driver may
// not track the image<->parent-buffer alias, so a later fanout's quant WRITE
// to a *shared* act_i8 can race the prior fanout's still-in-flight GEMM image
// READ (a WAR hazard) once the queue-draining host maps are removed (the
// static cl_mem residency chain). gpu_native is race-free because each fanout
// role (qkv / wo / ffn-up+gate / ffn-down) owns a DISTINCT activation buffer +
// cached image, so an inter-fanout writer never aliases an in-flight reader.
// Mirror that with a small ring of slots, advanced only on a quant-cache MISS
// (a new input / new fanout). With exactly 4 distinct-input fanouts per
// transformer layer this gives the same ~1-layer reuse distance as
// gpu_native's 4 per-purpose buffers, while wq/wk/wv (which share one input =>
// cache hits) all reuse the same slot read-only.
static constexpr int V8C_ACT_SLOTS = 4;

// Grow-only scratch buffer pool, reused across all dotCl_v8c forward calls
// to avoid per-call clCreateBuffer/clReleaseMemObject churn (the dominant
// integration overhead, especially in M=1 decode where the same FC shapes
// recur thousands of times).
struct V8cScratch {
  // fp staging buffer for the quant input. SHARED: it is only ever read as a
  // plain buffer by the quant kernel (never via an image alias), so the
  // in-order SVM-pool queue already orders a later fanout's copy-write against
  // a prior fanout's quant-read of it -- no per-slot copy needed.
  cl_mem act_in = nullptr;
  size_t act_in_bytes = 0;
  // Per-fanout activation int8 + scale/zp/rowsum + the cached image2d view
  // over act_i8 (the only buffer the GEMM reads through an image -> the only
  // one needing distinct buffers per fanout). Ring-selected on quant-cache
  // miss.
  cl_mem act_i8[V8C_ACT_SLOTS] = {};
  size_t act_i8_bytes[V8C_ACT_SLOTS] = {};
  cl_mem act_scale[V8C_ACT_SLOTS] = {};
  size_t act_scale_bytes[V8C_ACT_SLOTS] = {};
  cl_mem act_rs[V8C_ACT_SLOTS] = {};
  size_t act_rs_bytes[V8C_ACT_SLOTS] = {};
  cl_mem act_zp[V8C_ACT_SLOTS] = {}; // [M] int32, asymmetric act zero-point
  size_t act_zp_bytes[V8C_ACT_SLOTS] = {};
  // Cached image2d-from-buffer view per slot, built once per (buffer, M_pad,
  // K) and reused across the fanout's GEMMs instead of per-call create/release
  // (which also leaked on the exception path). Rebuilt when the slot's buffer
  // is grown or M_pad/K change.
  cl_mem act_image[V8C_ACT_SLOTS] = {};
  cl_mem act_image_buf[V8C_ACT_SLOTS] = {};
  unsigned int act_image_M_pad[V8C_ACT_SLOTS] = {};
  unsigned int act_image_K[V8C_ACT_SLOTS] = {};
  int ring_pos = 0; /**< last slot handed out; advance-on-miss */
  cl_mem y_fp16 = nullptr;
  size_t y_fp16_bytes = 0;

  // Step 2b.0 shared-quant cache (paper §3.6 fused-quant motivation, host-side).
  // Qwen3 layer graph dispatches three consecutive dotCl_v8c calls with the
  // SAME input pointer (wq/wk/wv all read the same post-RMSNorm activation),
  // and similarly gate/up MLP FCs share their input. After the first call
  // populates act_i8/act_scale/act_zp/act_rs for that input, the next 2-of-3
  // calls can skip the host→device upload AND the quant kernel entirely.
  //
  // Cache key: (input data pointer, M, K, M_pad, dtype). Pointer identity is
  // sufficient within one forward pass since the layer graph executes
  // serially — the input buffer isn't aliased between dispatches.
  const void *last_quant_in_ptr = nullptr;
  unsigned int last_quant_M = 0;
  unsigned int last_quant_K = 0;
  unsigned int last_quant_M_pad = 0;
  int last_quant_dtype = -1;
  int last_quant_slot = 0; /**< slot whose int8 the cache hit refers to */
  unsigned long long last_quant_resident_generation = 0;
};

// Process-global Segment A resident-buffer generation counter. Producers
// (Segment A's RMSNorm helpers) bump this on every successful write to a
// resident TensorBacking, signalling to dotCl_v8c that any quant cache
// keyed on a backing pointer is now stale. Same forward pass / multiple
// FCs reusing the same backing all share the generation, so the cache
// still hits on wq→wk→wv within the pass.
static std::atomic<unsigned long long> g_resident_quant_generation{0};
static V8cScratch &v8c_scratch() {
  static V8cScratch s;
  return s;
}
// Ensure *buf has at least `bytes` capacity with the given flags; (re)alloc
// only when too small. Returns false on alloc failure.
static bool v8c_ensure_buf(cl_context ctx, cl_mem *buf, size_t *cap,
                           size_t bytes, cl_mem_flags flags) {
  if (*buf && *cap >= bytes)
    return true;
  if (*buf) {
    clReleaseMemObject(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  cl_int err = CL_SUCCESS;
  *buf = clCreateBuffer(ctx, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  return true;
}

// Get or build the cached v8c weight backing for a given Int4QTensor weight.
// Returns nullptr if shape unsupported (caller falls back).
static V8cWeightEntry *v8c_get_or_build_weight(const Tensor &weight,
                                               unsigned int K, unsigned int N) {
  if (K % 32 != 0 || N % 8 != 0)
    return nullptr;
  const void *key = weight.getData<uint8_t>();
  if (!key)
    return nullptr;
  std::lock_guard<std::mutex> lock(v8c_cache_mtx());
  auto &cache = v8c_weight_cache();
  auto it = cache.find(key);
  if (it != cache.end())
    return &it->second;
  const uint8_t *section_a = weight.getData<uint8_t>();
  const uint16_t *fp16_scales = weight.getScale<uint16_t>();
  if (!section_a || !fp16_scales)
    return nullptr;
  V8cWeightEntry e;
  cl_mem sb = nullptr;
  cl_mem rsw = nullptr;
  try {
    // The on-disk QINT4 weight is the KAI Section A nibble payload + a
    // per-output-channel fp16 scale (one fp16 per N). Permute the nibbles
    // straight to the v8c row-major + offset-encoded layout — no dequant→
    // requant round-trip, so no extra quantization noise and no fp32
    // intermediate buffer. The scales transfer 1:1 (fp16 → fp32). The
    // helper also precomputes per-channel Σ_k int4_w[n,k] for the
    // asymmetric-act zero-point correction the GEMM applies later.
    e.backing = make_v8c_weight_backing_from_kai_section_a(
      section_a, fp16_scales, N, K, &sb, &rsw);
  } catch (...) {
    return nullptr;
  }
  e.scale_buf = sb;
  e.row_sum_w_int4 = rsw;
  e.N = N;
  e.K = K;
  // Raw backing buffer for the NNTR_V8C_BUF path (Intel NEO). Always available
  // (zero-copy); the image2d view below is only used by the image-sampling
  // kernels (Adreno).
  e.weight_buf = e.backing->buffer();
  tv::ViewSpec ws;
  ws.kind = tv::ViewKind::IMAGE_2D;
  ws.image_channel_order = CL_RGBA;
  ws.image_channel_type = CL_UNSIGNED_INT32;
  ws.width = K / 32;
  ws.height = N;
  ws.row_pitch_bytes = K / 2;
  try {
    e.weight_image = e.backing->imageView(ws);
  } catch (...) {
    // The image2d view fails when N (height) exceeds the device image cap
    // (~16384) -- the untied int4 lm_head has N=vocab=262144. The row-major
    // weight_buf + scale_buf are still valid, so keep the entry with a null
    // image: dotCl_v8c routes the (M=1 decode) huge-N case to the buffer GEMV
    // (lmhead_int4_v8c_gemv_cl) instead of failing to the CPU KAI path. Every
    // other (image-sized) weight builds its image normally, so this only
    // affects the oversized lm_head.
    e.weight_image = nullptr;
    static int logged = 0;
    if (!logged++)
      std::fprintf(stderr,
                   "[v8c] image view unavailable for N=%u K=%u (>image cap); "
                   "keeping buffer path for the GEMV\n",
                   N, K);
  }
  auto inserted = cache.emplace(key, std::move(e));
  return &inserted.first->second;
}

// fp16 → fp32 (host-side decode used to convert kernel fp16 output)
static inline float v8c_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t o;
  if (e == 0) {
    if (m == 0)
      o = s;
    else {
      e = 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        e--;
      }
      m &= 0x3ffu;
      o = s | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 0x1f) {
    o = s | 0x7f800000u | (m << 13);
  } else {
    o = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &o, 4);
  return f;
}
} // anonymous namespace

// Eager v8c weight build (see header). Moves the lazy per-weight nibble
// permute + upload (~4.1ms x 182 weights = 753ms, measured NNTR_FC_TPROF)
// out of the first timed prefill: the FC layer calls this right after its
// weight is read at model load. No-op (false) off the v8c path. Must live
// OUTSIDE the anonymous namespace (public symbol; the helpers it calls are
// TU-internal).
bool dotCl_v8c_prebuild_weight(const Tensor &weight) {
  if (!v8c_env_enabled())
    return false;
  if (weight.getDataType() != ml::train::TensorDim::DataType::QINT4)
    return false;
  const unsigned int N = weight.width();
  const unsigned int K = weight.height();
  if (N == 0 || K == 0 || N % 8 != 0 || K % 32 != 0)
    return false;
  return v8c_get_or_build_weight(weight, K, N) != nullptr;
}

// fp16 GEMM output -> output tensor, written on the GPU (residency: no host
// readback). One source, two entry points: cvt_h2f converts fp16->fp32,
// copy_h2h copies fp16->fp16.
static const std::string v8c_out_residency_kernels = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void v8c_cvt_h2f(__global const half *in, __global float *out,
                          const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = (float)in[i];
}
__kernel void v8c_copy_h2h(__global const half *in, __global half *out,
                           const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = in[i];
}
__kernel void v8c_copy_f2f(__global const float *in, __global float *out,
                           const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = in[i];
}
__kernel void v8c_add_h2h(__global const half *in, __global half *out,
                          const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] += in[i];
}
// Dedicated PROBE copy kernel: identical body to v8c_copy_h2h but a DISTINCT
// kernel object, so probe captures never re-bind args on a kernel object the
// pipeline has in flight (re-binding a shared kernel object was measured to
// ALTER the generated tokens => enqueued-arg isolation is not airtight here).
__kernel void v8c_probe_copy(__global const half *in, __global half *out,
                             const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = in[i];
}
)CL";

// Pre-build the residency-kernel program at context init (see header).
void v8c_prewarm_programs(ClContext &cc) {
  cc.registerClKernel(v8c_out_residency_kernels, "v8c_copy_h2h");
}

// Write the fp16 GEMM result (y_fp16, device cl_mem, n = M*N valid elements)
// directly into the GPU-resident SVM output, converting to fp32 when needed.
// Coarse-grained SVM coherence: unmap the output before the kernel (GPU owns
// it), re-map after (host / next layer can read it).
static void v8c_write_output_resident(cl_mem y_fp16, Tensor &output,
                                      unsigned int n, bool out_fp16,
                                      void *out_clmem = nullptr) {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto kp = cc->registerClKernel(v8c_out_residency_kernels,
                                 out_fp16 ? "v8c_copy_h2h" : "v8c_cvt_h2f");
  if (!kp)
    return;
  // Static GPU_CLMEM residency: write the planner sub-buffer via THIS KERNEL
  // (cl_mem arg) instead of clEnqueueCopyBuffer -- the blit/copy engine is
  // not reliably ordered against compute kernels on this driver without a
  // drain (measured: a drained readback sees correct bytes, an undrained
  // kernel consumer sees stale), while kernel->kernel ordering is solid
  // (gpu_native's model). No SVM maps, no device_valid bits on this path.
  if (out_clmem != nullptr) {
    cl_mem oh = static_cast<cl_mem>(out_clmem);
    int ni2 = (int)n;
    if (!kp->SetKernelArguments(0, &y_fp16, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &oh, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &ni2, sizeof(int)))
      return;
    const int gws2[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
    const int lws2[3] = {64, 1, 1};
    cc->command_queue_inst_.DispatchCommand(kp, gws2, lws2);
    return;
  }
  void *out_svm = output.getData<uint8_t>();
  int ni = (int)n;
  // NNTR_DEVRES Step 1: clear the device-residency bit before the GPU rewrites
  // this output (the prior contents are about to be overwritten). Set it after
  // the write below. Gated by the master flag; off => bit untouched (byte-id).
  static const bool devres = std::getenv("NNTR_DEVRES") != nullptr;
  if (devres) {
    if (auto md = output.getMemoryData())
      md->setDeviceValid(false);
  }
  cc->command_queue_inst_.enqueueSVMUnmap(out_svm);
  if (!kp->SetKernelArguments(0, &y_fp16, sizeof(cl_mem)) ||
      !kp->SetKernelSVMArguments(1, out_svm) ||
      !kp->SetKernelArguments(2, &ni, sizeof(int)))
    return;
  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
  size_t out_bytes = (size_t)n * (out_fp16 ? sizeof(uint16_t) : sizeof(float));
  // async map: the FC output is always consumed by the next GPU op (attention,
  // geglu, next FC) — never read on the host — so the in-order queue orders
  // this map before the next op's unmap and the host need not block here.
  // Removes ~182 per-forward queue drains (the dominant FC sync band).
  // NNTR_FC_SVM_SYNC=1 (Xe3 coherence regression probe): make the FC-output SVM
  // map BLOCKING so the GPU writes are guaranteed visible to the next consumer
  // before it reads (the suspected coarse-grained-SVM stale-shadow on NEO 26.22).
  static const bool fc_svm_sync = std::getenv("NNTR_FC_SVM_SYNC") != nullptr;
  cc->command_queue_inst_.enqueueSVMMap(out_svm, out_bytes, true,
                                        /*async=*/!fc_svm_sync);
  // NNTR_DEVRES Step 1: the GPU now holds the fresh FC output in out_svm. Flag
  // it device-resident so a downstream GPU consumer sharing this MemoryData
  // (edge view) sees a HIT. No map is skipped yet (Step 4+); this only sets the
  // bit. Cleared again on the next producer write (above) or a host read (S7).
  if (devres) {
    if (auto md = output.getMemoryData())
      md->setDeviceValid(true, out_svm);
  }
}

// Copy an SVM-resident activation (n = M*K elements) into the device cl_mem
// quant scratch on the GPU -- replaces the host upload (clEnqueueWriteBuffer)
// when the input is GPU-resident, so no PCIe round-trip. Downstream (quantize ->
// image2d -> GEMM) is unchanged; only the source of sc.act_in changes.
// Coarse-grained SVM coherence: unmap the input before the copy (GPU owns it),
// re-map after.
static void v8c_copy_svm_to_clmem(const void *in_svm, cl_mem out,
                                  unsigned int n, bool fp16,
                                  bool device_owned = false) {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto kp = cc->registerClKernel(v8c_out_residency_kernels,
                                 fp16 ? "v8c_copy_h2h" : "v8c_copy_f2f");
  if (!kp)
    return;
  int ni = (int)n;
  // NNTR_DEVRES Step 4: when device_owned, the producer (e.g. geglu) already
  // left in_svm GPU-owned (skipped its trailing map), so skip the matching
  // unmap here — removing the map/unmap PAIR together. A one-sided skip would
  // read a host-mapped buffer on the GPU = asymmetric SVM state = crash.
  if (!device_owned)
    cc->command_queue_inst_.enqueueSVMUnmap(const_cast<void *>(in_svm));
  if (!kp->SetKernelSVMArguments(0, const_cast<void *>(in_svm)) ||
      !kp->SetKernelArguments(1, &out, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &ni, sizeof(int)))
    return;
  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
  // async map: GPU→GPU handoff (the input copy feeds the quant/GEMM kernels);
  // no host access before then, in-order queue preserves ordering. Skipped on
  // the device-owned path (the buffer stays GPU-owned for the resident edge).
  if (!device_owned)
    cc->command_queue_inst_.enqueueSVMMap(const_cast<void *>(in_svm),
                                          (size_t)n * (fp16 ? 2 : 4), true,
                                          /*async=*/true);
}

// Explicit host->cl_mem RAISE for a boundary tensor (see header).
bool clmem_raise_cl(const Tensor &t, unsigned int valid_bytes) {
  if (!t.isClMem())
    return false;
  void *sub = t.getClMem();
  if (sub == nullptr)
    return false;
  // The sub-buffer covers the WHOLE tensor; a nonzero-offset view cannot be
  // bridged from base. Live path is offset-0; fail loudly, never misread.
  if (t.getOffset() != 0)
    throw std::runtime_error("clmem_raise_cl: nonzero-offset view unsupported");
  const size_t bytes =
    valid_bytes ? (size_t)valid_bytes : t.bytes();
  if (bytes == 0)
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_raise_cl: no GPU context");
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  // Non-blocking upload from the host-written SVM shadow: the in-order queue
  // orders it before every later consumer; the source memory stays untouched
  // until the next forward (host writes only after the lm_head drain).
  if (clEnqueueWriteBuffer(q, static_cast<cl_mem>(sub), CL_FALSE, 0, bytes,
                           t.getData<uint8_t>(), 0, nullptr,
                           nullptr) != CL_SUCCESS)
    throw std::runtime_error("clmem_raise_cl: clEnqueueWriteBuffer failed");
  // NNTR_RAISE_VERIFY=1 (Xe3): confirm the SVM->cl_mem upload landed (the
  // cl_mem the next consumer reads == the SVM source the attention wrote).
  if (std::getenv("NNTR_RAISE_VERIFY")) {
    clFinish(q);
    const size_t cnt = std::min(bytes, (size_t)4096) / 2;
    std::vector<uint16_t> back(cnt);
    clEnqueueReadBuffer(q, static_cast<cl_mem>(sub), CL_TRUE, 0, cnt * 2,
                        back.data(), 0, nullptr, nullptr);
    const uint16_t *svmsrc =
      reinterpret_cast<const uint16_t *>(t.getData<uint8_t>());
    float maxd = 0;
    for (size_t i = 0; i < cnt; ++i)
      maxd = std::max(maxd, std::fabs(v8c_h2f(back[i]) - v8c_h2f(svmsrc[i])));
    std::fprintf(stderr,
                 "[RAISEVERIFY] %-26s cl_mem vs SVM maxdiff=%.4f bytes=%zu\n",
                 t.getName().c_str(), maxd, bytes);
    std::fflush(stderr);
  }
  return true;
}

// Explicit cl_mem->host LOWER for a boundary tensor (see header).
bool clmem_lower_cl(const Tensor &t, unsigned int valid_bytes) {
  if (!t.isClMem())
    return false;
  void *sub = t.getClMem();
  if (sub == nullptr)
    return false;
  // See clmem_raise_cl: offset-0 views only, loud failure otherwise.
  if (t.getOffset() != 0)
    throw std::runtime_error("clmem_lower_cl: nonzero-offset view unsupported");
  const size_t bytes = valid_bytes ? (size_t)valid_bytes : t.bytes();
  if (bytes == 0)
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_lower_cl: no GPU context");
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  // BLOCKING read on the in-order queue: waits for every prior command (the
  // whole forward), then lands the bytes in host memory (the SVM shadow used
  // as a plain host pointer). The host consumer reads ordinary memory next.
  cl_int rb_err = clEnqueueReadBuffer(q, static_cast<cl_mem>(sub), CL_TRUE, 0,
                                      bytes, t.getData<uint8_t>(), 0, nullptr,
                                      nullptr);
  if (rb_err != CL_SUCCESS)
    throw std::runtime_error(
      "clmem_lower_cl: clEnqueueReadBuffer failed err=" +
      std::to_string(rb_err) + " bytes=" + std::to_string(bytes) +
      " name=" + t.getName());
  return true;
}

// Dump the CL-event kernel/idle profile (no-op unless NNTR_OPENCL_PROFILING).
void clmem_dump_clprof(const char *tag) {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (cc)
    cc->command_queue_inst_.dumpProfile(tag);
}

// ---- NNTR_CLMEM_PROBE: non-invasive value probe (see header) ----
namespace {
struct ClmemProbeEntry {
  std::string tag;
  size_t bytes;
  cl_mem buf;
};
std::vector<ClmemProbeEntry> &clmem_probe_entries() {
  static std::vector<ClmemProbeEntry> v;
  return v;
}
void clmem_probe_dump() {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  clFinish(q);
  for (auto &e : clmem_probe_entries()) {
    std::vector<uint8_t> host(e.bytes);
    if (clEnqueueReadBuffer(q, e.buf, CL_TRUE, 0, e.bytes, host.data(), 0,
                            nullptr, nullptr) != CL_SUCCESS) {
      std::fprintf(stderr, "[probe] %s READ-FAIL\n", e.tag.c_str());
      clReleaseMemObject(e.buf);
      continue;
    }
    // FNV-1a 64 over the bytes + first 4 fp16 raw values for quick eyeballing.
    unsigned long long h = 1469598103934665603ull;
    for (uint8_t b : host) {
      h ^= b;
      h *= 1099511628211ull;
    }
    const uint16_t *v16 = reinterpret_cast<const uint16_t *>(host.data());
    std::fprintf(stderr,
                 "[probe] %-44s bytes=%-7zu fnv=%016llx v=%04x %04x %04x %04x\n",
                 e.tag.c_str(), e.bytes, h, v16[0], v16[1], v16[2], v16[3]);
    clReleaseMemObject(e.buf);
  }
  std::fflush(stderr);
  clmem_probe_entries().clear();
}
} // namespace

void clmem_probe_capture(const char *tag, const void *svm_ptr, void *clmem,
                         unsigned int bytes) {
  static const bool on = std::getenv("NNTR_CLMEM_PROBE") != nullptr;
  if (!on || bytes == 0)
    return;
  static const int maxn = [] {
    const char *e = std::getenv("NNTR_CLMEM_PROBE_MAX");
    return e ? std::atoi(e) : 128;
  }();
  auto &v = clmem_probe_entries();
  if ((int)v.size() >= maxn)
    return;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    return;
  cl_context ctx = cc->context_inst_.GetContext();
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  // NNTR_CLMEM_PROBE_DRAIN=1: clFinish BEFORE each capture. Heavily
  // schedule-invasive, but the only way to make the CopyBuffer snapshot
  // trustworthy: the blit engine is NOT ordered against compute kernels on
  // this driver (kernel-write -> CopyBuffer-read returns stale/zeros), so
  // undrained captures of freshly kernel-written buffers show phantom
  // zeros (the FFN act_in "zeros" artifact). With the drain, a zero
  // capture is a REAL zero.
  static const bool probe_drain = []() {
    const char *e = std::getenv("NNTR_CLMEM_PROBE_DRAIN");
    return e && e[0] == '1';
  }();
  if (probe_drain)
    clFinish(q);
  cl_int err = CL_SUCCESS;
  cl_mem dbg = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || dbg == nullptr)
    return;
  if (clmem != nullptr) {
    if (clEnqueueCopyBuffer(q, static_cast<cl_mem>(clmem), dbg, 0, 0, bytes, 0,
                            nullptr, nullptr) != CL_SUCCESS) {
      clReleaseMemObject(dbg);
      return;
    }
  } else if (svm_ptr != nullptr) {
    // SVM source: device-side copy kernel (fp16 element count), no host sync.
    // DEDICATED kernel object (v8c_probe_copy): re-binding a kernel object the
    // pipeline still has in flight (v8c_copy_h2h) measurably alters the output.
    auto kp = cc->registerClKernel(v8c_out_residency_kernels, "v8c_probe_copy");
    int n = (int)(bytes / 2);
    if (!kp || !kp->SetKernelSVMArguments(0, const_cast<void *>(svm_ptr)) ||
        !kp->SetKernelArguments(1, &dbg, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &n, sizeof(int))) {
      clReleaseMemObject(dbg);
      return;
    }
    const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
    const int lws[3] = {64, 1, 1};
    if (!cc->command_queue_inst_.DispatchCommand(kp, gws, lws)) {
      clReleaseMemObject(dbg);
      return;
    }
  } else {
    clReleaseMemObject(dbg);
    return;
  }
  v.push_back({std::string(tag), (size_t)bytes, dbg});
  // Dump ONLY at process exit: a mid-run dump (clFinish + blocking readbacks)
  // measurably ALTERS the generated tokens even on the pure SVM baseline --
  // i.e. the baseline itself is drain-placement sensitive (latent race). The
  // captures alone are verified non-invasive. At maxn we simply stop capturing.
  static const bool registered = [] {
    std::atexit([] { clmem_probe_dump(); });
    return true;
  }();
  (void)registered;
  // NNTR_CLMEM_PROBE_FINISH=<k>: inject ONE pure clFinish (no readbacks) right
  // after capture #k -- a semantically NEUTRAL drain. Bisecting k against the
  // token output locates WHERE the baseline's latent race sits (the op whose
  // correctness depends on drain placement).
  static const int finish_at = [] {
    const char *e = std::getenv("NNTR_CLMEM_PROBE_FINISH");
    return e ? std::atoi(e) : -1;
  }();
  if (finish_at >= 0 && (int)v.size() == finish_at) {
    clFinish(q);
    std::fprintf(stderr, "[probe] clFinish injected after #%d (%s)\n",
                 finish_at, tag);
    std::fflush(stderr);
  }
}

// FP16 elementwise residual copy (dst = src) / accumulate (dst += src) where
// dst/src each bind the plane their STATIC ResidencyClass picked at allocation:
// the planner cl_mem sub-buffer for GPU_CLMEM, the SVM pointer otherwise.
// Mixed cl_mem/SVM args are valid. Returns false (caller keeps its SVM/host
// path) only when NEITHER side is cl_mem; after the static-class commitment a
// failure throws -- a silent SVM fallback would recreate the corrupting hybrid.
// No clFinish: the in-order SVM-pool queue orders producer -> this op ->
// consumer (gpu_native's coherence model).
bool clmem_residual_op_cl(Tensor &dst, const Tensor &src, bool accumulate) {
  if (dst.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      src.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  if (dst.size() != src.size() || dst.size() == 0)
    return false;
  void *dst_cl = dst.isClMem() ? dst.getClMem() : nullptr;
  void *src_cl = src.isClMem() ? src.getClMem() : nullptr;
  if (dst_cl == nullptr && src_cl == nullptr)
    return false;

  // The cl_mem handle covers the WHOLE tensor; a step/batch view at a nonzero
  // offset cannot bind it (kernels address from base). Live path is batch==1 /
  // offset 0 -- fail loudly rather than silently corrupting via a base bind.
  if ((dst_cl != nullptr && dst.getOffset() != 0) ||
      (src_cl != nullptr && src.getOffset() != 0))
    throw std::runtime_error(
      "clmem_residual_op_cl: GPU_CLMEM tensor accessed at a nonzero offset "
      "(batch>1 step views are unsupported on the cl_mem plane)");

  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_residual_op_cl: no GPU context");
  const unsigned int n = (unsigned int)dst.size();
  const size_t bytes = (size_t)n * sizeof(uint16_t);

  // Pure cl_mem->cl_mem copy: a plain buffer copy beats a kernel dispatch.
  // BUT clEnqueueCopyBuffer is a NON-NDRange op that the recq recordable queue
  // does NOT capture -> it would be dropped from a recorded decode replay,
  // leaving the destination stale. Under recq (NNTR_RECQ_DESVM) fall through to
  // the v8c_copy_h2h KERNEL (captured, byte-identical) instead.
  static const bool _recq_no_copyfast =
    std::getenv("NNTR_RECQ_DESVM") != nullptr;
  if (!accumulate && dst_cl != nullptr && src_cl != nullptr &&
      !_recq_no_copyfast) {
    cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
    if (clEnqueueCopyBuffer(q, static_cast<cl_mem>(src_cl),
                            static_cast<cl_mem>(dst_cl), 0, 0, bytes, 0,
                            nullptr, nullptr) != CL_SUCCESS)
      throw std::runtime_error("clmem_residual_op_cl: clEnqueueCopyBuffer");
    clmem_probe_capture((dst.getName() + ":cp").c_str(), nullptr, dst_cl,
                        (unsigned int)bytes);
    return true;
  }

  auto kp = cc->registerClKernel(v8c_out_residency_kernels,
                                 accumulate ? "v8c_add_h2h" : "v8c_copy_h2h");
  if (!kp)
    throw std::runtime_error("clmem_residual_op_cl: kernel registration");

  // SVM-side args keep the established per-op map protocol (unmap before the
  // kernel, async map after); cl_mem args need none.
  void *src_svm = const_cast<void *>(
    static_cast<const void *>(src.getData<uint8_t>()));
  void *dst_svm = static_cast<void *>(dst.getData<uint8_t>());
  if (src_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMUnmap(src_svm);
  if (dst_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMUnmap(dst_svm);

  bool ok = true;
  if (src_cl != nullptr) {
    cl_mem h = static_cast<cl_mem>(src_cl);
    ok = ok && kp->SetKernelArguments(0, &h, sizeof(cl_mem));
  } else {
    ok = ok && kp->SetKernelSVMArguments(0, src_svm);
  }
  if (dst_cl != nullptr) {
    cl_mem h = static_cast<cl_mem>(dst_cl);
    ok = ok && kp->SetKernelArguments(1, &h, sizeof(cl_mem));
  } else {
    ok = ok && kp->SetKernelSVMArguments(1, dst_svm);
  }
  int ni = (int)n;
  ok = ok && kp->SetKernelArguments(2, &ni, sizeof(int));
  if (!ok)
    throw std::runtime_error("clmem_residual_op_cl: arg binding");

  // NNTR_RESID_VERIFY=1 (Xe3): snapshot src and dst before the op so we can
  // confirm the result == (accumulate ? src+dst : src) from the SAME buffers
  // the kernel reads. A large diff => the residual add/copy itself is wrong;
  // correct here but garbage output => an UPSTREAM op fed it a stale buffer.
  const bool resid_verify = std::getenv("NNTR_RESID_VERIFY") != nullptr;
  std::vector<uint16_t> rv_s, rv_d0;
  if (resid_verify) {
    cl_command_queue qq = cc->command_queue_inst_.GetCommandQueue();
    clFinish(qq);
    const size_t cnt = std::min((size_t)n, (size_t)2048);
    rv_s.resize(cnt);
    rv_d0.resize(cnt);
    if (src_cl)
      clEnqueueReadBuffer(qq, static_cast<cl_mem>(src_cl), CL_TRUE, 0, cnt * 2,
                          rv_s.data(), 0, nullptr, nullptr);
    else
      std::memcpy(rv_s.data(), src_svm, cnt * 2);
    if (dst_cl)
      clEnqueueReadBuffer(qq, static_cast<cl_mem>(dst_cl), CL_TRUE, 0, cnt * 2,
                          rv_d0.data(), 0, nullptr, nullptr);
    else
      std::memcpy(rv_d0.data(), dst_svm, cnt * 2);
  }

  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  if (!cc->command_queue_inst_.DispatchCommand(kp, gws, lws))
    throw std::runtime_error("clmem_residual_op_cl: dispatch");
  if (resid_verify) {
    cl_command_queue qq = cc->command_queue_inst_.GetCommandQueue();
    clFinish(qq);
    const size_t cnt = rv_s.size();
    std::vector<uint16_t> rv_d1(cnt);
    if (dst_cl)
      clEnqueueReadBuffer(qq, static_cast<cl_mem>(dst_cl), CL_TRUE, 0, cnt * 2,
                          rv_d1.data(), 0, nullptr, nullptr);
    else
      std::memcpy(rv_d1.data(), dst_svm, cnt * 2);
    float maxd = 0;
    for (size_t i = 0; i < cnt; ++i) {
      float exp = accumulate ? (v8c_h2f(rv_s[i]) + v8c_h2f(rv_d0[i]))
                             : v8c_h2f(rv_s[i]);
      maxd = std::max(maxd, std::fabs(v8c_h2f(rv_d1[i]) - exp));
    }
    std::fprintf(stderr, "[RESIDVERIFY] %-28s acc=%d maxdiff=%.4f\n",
                 dst.getName().c_str(), (int)accumulate, maxd);
    std::fflush(stderr);
  }

  if (src_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMMap(src_svm, bytes, true, /*async=*/true);
  if (dst_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMMap(dst_svm, bytes, true, /*async=*/true);
  // NNTR_CLMEM_PROBE: capture the residual dst after the op (copy/add) for
  // the fan-out bisect; cl_mem captures are reliable.
  clmem_probe_capture(
    (dst.getName() + (accumulate ? ":add" : ":cp")).c_str(),
    dst_cl == nullptr ? dst_svm : nullptr, dst_cl, (unsigned int)bytes);
  return true;
}

// NNTR_FC_TPROF=1: host wall time of the dotCl_v8c hot path, split at the
// input-staging boundary (decomposes the rmsnorm->v8c_copy_h2h GPU idle).
static bool fc_tprof_on() {
  static const bool on = std::getenv("NNTR_FC_TPROF") != nullptr;
  return on;
}
static double fc_tprof_now() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static double fc_tp_entry = 0, fc_tp_stage = 0, fc_tp_tail = 0;
static int fc_tp_n = 0;

bool dotCl_v8c(const Tensor &input, const Tensor &weight, Tensor &output) {
  const double _fc_t0 = fc_tprof_on() ? fc_tprof_now() : 0;
  if (!v8c_env_enabled())
    return false;
  if (weight.getDataType() != ml::train::TensorDim::DataType::QINT4)
    return false;
  // Derive M, K, N from tensor dims (no-transpose case only).
  unsigned int M, K, N;
  if (input.getFormat() == Tformat::NHWC) {
    M = input.batch() * input.height() * input.width();
    K = input.channel();
  } else {
    M = input.batch() * input.channel() * input.height();
    K = input.width();
  }
  N = weight.width();
  if (K != weight.height())
    return false;
  if (N % 8 != 0 || K % 32 != 0)
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP32 &&
      input.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;

  // NNTR_DEVRES Step 0 tracer: log whether this FC's input activation is
  // flagged device-resident on its shared MemoryData (read-only; no behavior
  // change). Lets S0/S1 verify the bit propagates across the producer->consumer
  // edge (HIT) before any host map is skipped. Default off => silent + inert.
  static const bool devres_trace = std::getenv("NNTR_DEVRES_TRACE") != nullptr;
  if (devres_trace) {
    auto md = input.getMemoryData();
    std::fprintf(stderr, "[devres] dotCl_v8c K=%u N=%u in.device_valid=%d\n", K,
                 N, (md && md->isDeviceValid()) ? 1 : 0);
  }
  // Round M up to the kernel's tile size (V8C_TM=4). Padded rows produce
  // throwaway output that we never read back to the caller. Skips the
  // "M not divisible by 4 → CPU fallback" cliff so v8c runs for any prefill
  // length (the 18-token Qwen3 chat-template case in particular).
  constexpr unsigned int V8C_TM = 4;
  // M_pad alignment. The v8c GEMM dispatches gws M-axis = M_pad / V8C_TM; the
  // tuned 4x16 work-group needs gws_y = M_pad/4 to be a multiple of 16, i.e.
  // M_pad a multiple of 64, or select2dLws (cl_tensor_view.cpp) fails its
  // divisibility gate and falls back to a NULL (driver-chosen) work-group.
  // On BOTH paths that fallback is a cliff:
  //   - Intel/buffer (NNTR_V8C_BUF): a non-power-of-2 M-workgroup count maps
  //     poorly to the EU array (M=842 prefill 175 -> 671 TPS at align 64).
  //   - Adreno/image: measured 2026-06-18 on gemma4 (M=999 -> M_pad=1000,
  //     gws_y=250, 250%16=10 != 0 -> NULL LWS). The driver's NULL choice is
  //     near-optimal for some N (gate/up N6144 = 5.5 TFLOP/s) but PATHOLOGICAL
  //     for others (full-Q N4096 = 0.41, per_layer_input N8960 = 0.36 -- 13x
  //     slower, ~28% of prefill). Forcing M_pad%64=0 restores the tuned 4x16
  //     to every FC shape: M=1024 prefill 1527 -> ~2280 TPS (+50%), coherent.
  // Padded rows are computed but never stored (M-valid store guard in
  // v8c_gemm_int8_int4), so output is bit-identical. So align to 64 by default
  // on BOTH paths. Override with NNTR_FC_MPAD_ALIGN (mult of V8C_TM). Only
  // applied for prefill-sized M (M >= align): decode (M=1) must never pad to 64
  // (that would be a 64x FC blow-up) -- guarded by eff_align below.
  static const unsigned int _mpad_align = []() {
    const char *e = std::getenv("NNTR_FC_MPAD_ALIGN");
    unsigned int v = e ? (unsigned int)std::atoi(e) : 64u;
    if (v < V8C_TM)
      v = V8C_TM;
    v = (v + V8C_TM - 1) / V8C_TM * V8C_TM; // keep a multiple of V8C_TM
    return v;
  }();
  const unsigned int eff_align = (M >= _mpad_align) ? _mpad_align : V8C_TM;
  const unsigned int M_pad = (M + eff_align - 1) / eff_align * eff_align;

  // [resident-act Step 0] Validate the producer->consumer edge map without
  // using it (zero runtime change). NNTR_RESIDENT_ACT_TRIP=1 logs whether this
  // FC's input view resolves to a producing edge name — the "forcing function"
  // that proves the finalize-built map works before any data path depends on it.
  if (std::getenv("NNTR_RESIDENT_ACT_TRIP") != nullptr) {
    static int trip = 0;
    if (trip < 12) {
      ++trip;
      const std::string in_name = input.getName();
      const std::string src = nntrainer::resolveResidentEdge(in_name);
      std::fprintf(stderr, "[RESACT-TRIP] FC in='%s' -> producer='%s' (%s)\n",
                   in_name.c_str(), src.c_str(),
                   src.empty() ? "MISS" : "HIT");
      std::fflush(stderr);
    }
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  V8cWeightEntry *w = v8c_get_or_build_weight(weight, K, N);
  if (!w)
    return false;

  // Imageless v8c weight (N > image2d height cap, e.g. the untied int4 lm_head
  // with N=vocab=262144): the image GEMM path cannot run, so dispatch the
  // dedicated fp-act int4 GEMV over the row-major weight buffer (best argmax
  // fidelity; no int8 act quant). Only decode (M=1) is supported -- the lm_head
  // FC runs only on the last position and prefill is skipped; any larger M with
  // no image legitimately falls back to the host path.
  if (w->weight_image == nullptr) {
#ifdef ENABLE_FP16
    if (M == 1 &&
        input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        (output.getDataType() == ml::train::TensorDim::DataType::FP16 ||
         output.getDataType() == ml::train::TensorDim::DataType::FP32)) {
      void *act = input.isClMem() ? input.getClMem()
                                  : static_cast<void *>(input.getData<_FP16>());
      const bool act_clmem = input.isClMem();
      const bool out_fp16 =
        output.getDataType() == ml::train::TensorDim::DataType::FP16;
      void *logits_host =
        out_fp16 ? static_cast<void *>(output.getData<_FP16>())
                 : static_cast<void *>(output.getData<float>());
      if (lmhead_int4_v8c_gemv_cl(w->weight_buf, w->scale_buf, act, act_clmem,
                                  logits_host, out_fp16, N, K))
        return true;
    }
#endif
    return false;
  }

  // Reused scratch buffers (grow-only pool). The weight backing + scale are
  // already cached per-weight; only the activation/output scratch scales with
  // (M_pad, K, N), so we grow these lazily and reuse them across forwards.
  cl_int err = CL_SUCCESS;
  const size_t act_elem =
    (input.getDataType() == ml::train::TensorDim::DataType::FP16)
      ? sizeof(uint16_t)
      : sizeof(float);
  std::lock_guard<std::mutex> slock(v8c_cache_mtx());
  V8cScratch &sc = v8c_scratch();
  // Shared staging + output buffers (slot-independent). The per-fanout act_i8
  // / scale / zp / rs slot buffers are grown AFTER the slot is selected
  // (below), so only the used slot grows to this call's K -- a slot that only
  // ever serves qkv (K=hidden) never pays for the ffn-down K (the larger one).
  if (!v8c_ensure_buf(ctx, &sc.act_in, &sc.act_in_bytes,
                      (size_t)M_pad * K * act_elem, CL_MEM_READ_ONLY) ||
      !v8c_ensure_buf(ctx, &sc.y_fp16, &sc.y_fp16_bytes,
                      sizeof(uint16_t) * (size_t)M_pad * N, CL_MEM_READ_WRITE))
    return false;

  // === v8c stage profiling (env-gated). Wall-clock with clFinish between
  // stages so each step's elapsed time is isolated. Skipped entirely when
  // NNTR_V8C_PROFILE is unset. Per-bin aggregates so prefill (large M) and
  // decode (M=1) regimes are separable. ===
  static bool prof_enabled = std::getenv("NNTR_V8C_PROFILE") != nullptr;
  struct V8cBin {
    long long write_ns = 0, quant_ns = 0, image_ns = 0, gemm_ns = 0,
              read_ns = 0;
    long long write_bytes = 0, read_bytes = 0;
    int calls = 0;
    long long m_sum = 0;
  };
  struct V8cProf {
    V8cBin bin[5]; // 0:M=1, 1:M=2-4, 2:M=5-32, 3:M=33-256, 4:M>256
    int total_calls = 0;
    static const char *bin_name(int b) {
      static const char *N[5] = {"M=1", "M=2-4", "M=5-32", "M=33-256", "M>256"};
      return N[b];
    }
    void dump(const char *tag) {
      std::FILE *f = std::fopen("/data/local/tmp/qwen3_qint4/v8c_prof.log", "a");
      if (!f) f = stderr;
      std::fprintf(f, "\n[V8C-PROF] %s after %d total calls:\n", tag,
                   total_calls);
      for (int b = 0; b < 5; b++) {
        const V8cBin &x = bin[b];
        if (!x.calls) continue;
        double total_ms = (x.write_ns + x.quant_ns + x.image_ns + x.gemm_ns +
                           x.read_ns) /
                          1e6;
        double avg_M = (double)x.m_sum / x.calls;
        std::fprintf(
          f,
          "  [%s] %d calls (avg M=%.1f) total=%.2f ms:\n"
          "    write_act    %7.2f ms (%.1f MB)  %5.1f%%\n"
          "    quant_kernel %7.2f ms             %5.1f%%\n"
          "    image_view   %7.2f ms             %5.1f%%\n"
          "    gemm_kernel  %7.2f ms             %5.1f%%\n"
          "    read_output  %7.2f ms (%.1f MB)  %5.1f%%\n",
          bin_name(b), x.calls, avg_M, total_ms, x.write_ns / 1e6,
          x.write_bytes / 1048576.0, 100.0 * x.write_ns / 1e6 / total_ms,
          x.quant_ns / 1e6, 100.0 * x.quant_ns / 1e6 / total_ms,
          x.image_ns / 1e6, 100.0 * x.image_ns / 1e6 / total_ms,
          x.gemm_ns / 1e6, 100.0 * x.gemm_ns / 1e6 / total_ms,
          x.read_ns / 1e6, x.read_bytes / 1048576.0,
          100.0 * x.read_ns / 1e6 / total_ms);
      }
      std::fflush(f);
      if (f != stderr) std::fclose(f);
    }
    ~V8cProf() {
      if (total_calls) dump("FINAL");
    }
  };
  static V8cProf prof;
  static int last_dumped_bin = -1;
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto NS = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
      .count();
  };
  std::chrono::steady_clock::time_point T0, T1;

  // Upload activation into the (reused) act_in buffer. Zero-fill the padded
  // rows so the act_quant kernel sees deterministic values (per-row amax → 0
  // → scale defaults to 1.0 → q=0 → row_sum=0; padded rows produce 0 output).
  int prof_bin = (M == 1)     ? 0
                 : (M <= 4)   ? 1
                 : (M <= 32)  ? 2
                 : (M <= 256) ? 3
                              : 4;

  // Step 2b.0 shared-quant cache check (paper §3.6 fused-quant insight,
  // host-side). If this dotCl_v8c was called with the same (input ptr,
  // M, K, M_pad, dtype) as the most recent call, sc.act_i8/scale/zp/rs are
  // already correctly populated — skip both the host→device write AND the
  // quant kernel. Hits fire on the wq→wk and wq→wv legs of Qwen3 QKV (and
  // gate→up of the MLP block), where the input activation is literally the
  // same tensor across multiple FC dispatches.
  //
  // Segment A.1 residency input mode (paper §3.2 cross-layer GPU residency).
  // When NNTR_RESIDENT_FC=1 AND the input Tensor has a TensorBacking with
  // FP16 encoding holding the activation in cl_mem (set by a preceding GPU
  // op such as Segment A's RMSNorm), bypass the host→device upload entirely
  // — clEnqueueCopyBuffer (GPU→GPU) from the backing into sc.act_in instead.
  // Cache key uses the backing's cl_mem as the source identifier so the
  // wq→wk→wv repeat continues to skip redundant quant. Caller-set Tensor
  // host data is ignored in this mode.
  //
  // Tensor::getBacking() returns null when the producer set the backing on
  // a different Tensor instance than the one this consumer received (the
  // typical nntrainer pattern: layer N's "output" Tensor and layer N+1's
  // "input" Tensor are distinct instances that share the underlying data
  // buffer via TensorPool). Fall back to a pool lookup keyed by the host
  // data pointer; shared-data tensors share the data pointer even when
  // their names/instances differ.
  static const bool resident_fc_enabled =
    std::getenv("NNTR_RESIDENT_FC") != nullptr;
  const tv::TensorBacking *in_backing = nullptr;
  std::shared_ptr<tv::TensorBacking> in_backing_from_pool;
  if (resident_fc_enabled) {
    in_backing = input.getBacking();
    if (!in_backing) {
      const void *in_data_ptr = input.getData<uint8_t>();
      char key_buf[64];
      std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", in_data_ptr);
      in_backing_from_pool =
        tv::TensorBackingPool::Global().get(std::string(key_buf));
      if (!in_backing_from_pool)
        in_backing_from_pool =
          tv::TensorBackingPool::Global().get(input.getName());
      if (in_backing_from_pool)
        in_backing = in_backing_from_pool.get();
    }
  }
  // [resident-act Step 1] cl_mem residency overlay: resolve this FC's input
  // through the graph edge map to its producer's output name, and consume the
  // cl_mem TensorBacking the producer published under `resact:`+producer-name.
  // This is the robust key (the producer→consumer edge), independent of the
  // brittle ptr:%p aliasing. Gated by NNTR_RESIDENT_ACT; misses fall through to
  // the existing SVM upload path (token-identical).
  static const bool resident_act_enabled =
    std::getenv("NNTR_RESIDENT_ACT") != nullptr;
  if (resident_act_enabled && !in_backing) {
    const std::string src = nntrainer::resolveResidentEdge(input.getName());
    if (!src.empty()) {
      in_backing_from_pool =
        tv::TensorBackingPool::Global().get("resact:" + src);
      if (in_backing_from_pool)
        in_backing = in_backing_from_pool.get();
    }
  }
  const bool resident_dtype_match =
    in_backing != nullptr &&
    ((in_backing->encoding() == tv::Encoding::FP16 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP16) ||
     (in_backing->encoding() == tv::Encoding::FP32 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP32));
  const bool use_resident_input = resident_dtype_match;
  // Planner-decided STATIC residency: this input tensor's ResidencyClass is
  // GPU_CLMEM, so by construction its producer (rms norm / geglu) wrote the
  // planner cl_mem sub-buffer (MemoryData.device_mem) -- uniformly, every
  // forward, no runtime device_valid flip. The FC reads it device-direct: a
  // cl_mem->cl_mem GPU copy into sc.act_in, NO SVM map (the measured prefill
  // blocker). The ptr-keyed quant cache is disabled on this edge (the
  // sub-buffer handle recurs across tokens with different contents).
  // Require the in-order SVM-pool queue (NNTR_GPU_SVM_POOL): this copy is
  // ordered after the producer's cl_mem write ONLY on the in-order queue (the
  // default is out-of-order and the copy could race ahead -> garbage).
  static const bool clmem_pool =
    std::getenv("NNTR_GPU_CLMEM_POOL") != nullptr &&
    std::getenv("NNTR_GPU_SVM_POOL") != nullptr;
  const bool device_clmem_in =
    clmem_pool && input.getMemoryData() &&
    input.getMemoryData()->isClMem() &&
    input.getMemoryData()->deviceMem() != nullptr;
  cl_mem clmem_in =
    device_clmem_in
      ? static_cast<cl_mem>(input.getMemoryData()->deviceMem())
      : nullptr;
  // Input GPU-residency: when the activation lives in the SVM pool (and no
  // cl_mem backing was found), copy it into the quant scratch device-side
  // instead of uploading it from the host -- removing the input round-trip.
  const bool in_svm = !use_resident_input && !device_clmem_in &&
                      input.getMemoryData() &&
                      input.getMemoryData()->isSVM();
  // NNTR_DEVRES Step 4: the SVM input is GPU-owned (its producer skipped the
  // trailing map) iff its MemoryData is device_valid. On that resident edge the
  // FC skips its matching unmap/map (v8c_copy_svm_to_clmem) AND must force a
  // re-quant — the SVM pointer recurs every token, so the (ptr,M,K) quant cache
  // would false-hit on stale int8 from a prior token (G5).
  static const bool devres_fc = std::getenv("NNTR_DEVRES") != nullptr;
  const bool device_in =
    devres_fc && in_svm && input.getMemoryData()->isDeviceValid();

  const int cur_dtype =
    (input.getDataType() == ml::train::TensorDim::DataType::FP16) ? 1 : 0;
  const void *cur_in_ptr =
    device_clmem_in ? static_cast<const void *>(clmem_in)
    : use_resident_input
      ? static_cast<const void *>(in_backing->buffer())
      : static_cast<const void *>(input.getData<uint8_t>());
  // Step 2b.0 quant cache. For host-uploaded inputs the (data_ptr,
  // shape, dtype) tuple uniquely identifies the activation, so a hit
  // means the same data is already int8-quantized in sc.act_i8 and
  // we can skip both the copy and the quant kernel.
  //
  // For backing-sourced inputs the same logic holds *within a single
  // forward pass* because the backing pointer is stable. Across passes
  // the same backing pointer points to different data (RMSNorm
  // overwrites it). That cross-pass staleness is invalidated below
  // by an external generation counter tied to RMSNorm writes:
  // resident_input_quant_generation is bumped whenever a Segment A
  // RMSNorm producer writes to a backing, and cached against the
  // generation at the time of the last quant. If the generation has
  // advanced since the last cache update, the cache is invalidated.
  const bool quant_cache_hit =
    !device_clmem_in && !device_in && sc.last_quant_in_ptr != nullptr &&
    sc.last_quant_in_ptr == cur_in_ptr && sc.last_quant_M == M &&
    sc.last_quant_K == K && sc.last_quant_M_pad == M_pad &&
    sc.last_quant_dtype == cur_dtype &&
    (!use_resident_input ||
     sc.last_quant_resident_generation == g_resident_quant_generation);

  // Fused-rmsq consumer (paper §3.6 Step 4). If a preceding RMSNormLayer
  // ran fused_rmsnorm_quant_resident_fp32 on this input, the pool holds
  // ready-to-use int8/scale/zp/rs buffers keyed by host data pointer.
  // Skip the (a) host upload + (c) quant kernel entirely and bind those
  // buffers as the GEMM inputs. Eliminates the rmsnorm→FC fp32 boundary
  // documented in [chain-robustification-dead].
  static const bool consume_fused_rmsq =
    std::getenv("NNTR_V8C_CONSUME_FUSED_RMSQ") != nullptr;
  // Filled either by the fused-rmsq consumer (external buffers) or, in the
  // normal path, by the per-fanout slot selected just below.
  cl_mem act_i8_arg = nullptr;
  cl_mem act_scale_arg = nullptr;
  cl_mem act_zp_arg = nullptr;
  cl_mem act_rs_arg = nullptr;
  bool fused_rmsq_hit = false;
  std::shared_ptr<tv::TensorBacking> fr_i8_bk, fr_sc_bk, fr_zp_bk, fr_rs_bk;
  if (consume_fused_rmsq &&
      input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const void *in_data_ptr = input.getData<uint8_t>();
    char k1[80], k2[80], k3[80], k4[80];
    std::snprintf(k1, sizeof(k1), "ptr:%p:fused_i8", in_data_ptr);
    std::snprintf(k2, sizeof(k2), "ptr:%p:fused_scale", in_data_ptr);
    std::snprintf(k3, sizeof(k3), "ptr:%p:fused_zp", in_data_ptr);
    std::snprintf(k4, sizeof(k4), "ptr:%p:fused_rs", in_data_ptr);
    auto &pool = tv::TensorBackingPool::Global();
    fr_i8_bk = pool.get(k1);
    fr_sc_bk = pool.get(k2);
    fr_zp_bk = pool.get(k3);
    fr_rs_bk = pool.get(k4);
    const size_t need_i8 = (size_t)M * K;
    const size_t need_meta = (size_t)M * 4;
    static int hits = 0, misses = 0;
    if (fr_i8_bk && fr_sc_bk && fr_zp_bk && fr_rs_bk &&
        fr_i8_bk->bytes() >= need_i8 && fr_sc_bk->bytes() >= need_meta &&
        fr_zp_bk->bytes() >= need_meta && fr_rs_bk->bytes() >= need_meta) {
      fused_rmsq_hit = true;
      hits++;
      act_i8_arg = fr_i8_bk->buffer();
      act_scale_arg = fr_sc_bk->buffer();
      act_zp_arg = fr_zp_bk->buffer();
      act_rs_arg = fr_rs_bk->buffer();
      if (std::getenv("NNTR_V8C_CONSUME_FUSED_RMSQ_TRIP") != nullptr &&
          (hits <= 6 || hits % 60 == 0)) {
        std::fprintf(stderr,
                     "[V8C-FUSED-RMSQ] HIT #%d: input.name=%s M=%u K=%u\n",
                     hits, input.getName().c_str(), M, K);
        std::fflush(stderr);
      }
      // Note: M_pad > M case would mean the fused kernel's output is too
      // short by (M_pad - M)*K bytes for the GEMM kernel which expects
      // padded rows. Disable consumer in that case until the fused kernel
      // is updated to write padded zero rows.
      if (M_pad > M) {
        // Fall back to the per-fanout slot path (selected below), which
        // quantizes the M_pad padded rows the fused kernel can't supply.
        fused_rmsq_hit = false;
      }
    } else {
      misses++;
      if (std::getenv("NNTR_V8C_CONSUME_FUSED_RMSQ_TRIP") != nullptr &&
          misses <= 6) {
        std::fprintf(stderr,
                     "[V8C-FUSED-RMSQ] MISS #%d: input.name=%s M=%u K=%u  "
                     "i8=%d sc=%d zp=%d rs=%d  i8_sz=%zu (need %zu)\n",
                     misses, input.getName().c_str(), M, K,
                     fr_i8_bk ? 1 : 0, fr_sc_bk ? 1 : 0,
                     fr_zp_bk ? 1 : 0, fr_rs_bk ? 1 : 0,
                     fr_i8_bk ? fr_i8_bk->bytes() : 0ul, need_i8);
        std::fflush(stderr);
      }
    }
  }
  // When the fused consumer wins, treat as a quant-cache hit so the
  // existing skip-quant branches below trigger. Also invalidate the
  // quant cache so a subsequent call with the same input pointer but
  // no fused entries (rare, but possible if the producer didn't run)
  // doesn't falsely hit sc.act_i8 (which may hold stale data from
  // before the fused path took over).
  const bool skip_upload_and_quant = fused_rmsq_hit || quant_cache_hit;
  if (fused_rmsq_hit) {
    sc.last_quant_in_ptr = nullptr;
  }

  // [Lever 1] NNTR_FC_QUANT_DIRECT: on the cl_mem residency edge, quantize the
  // producer's (rmsnorm) cl_mem output IN PLACE, skipping the cl_mem->sc.act_in
  // staging copy (the v8c_copy_h2h kernel) and the padded-row zero write. The
  // act-quant kernel reads exactly M real rows from clmem_in (gws=M*64, bounded
  // by `if (row>=M) return`), so there is no OOB on the M-row producer buffer.
  // Safe because (a) GEMM output rows are independent -- acc[i] depends only on
  // act row i (int8_int4_gemm_v8c.cl) -- so the now-unquantized padded rows
  // [M, M_pad) of act_i8/scale/zp/rs (stale, not zeroed) only corrupt padded
  // OUTPUT rows, and (b) v8c_write_output_resident copies just M*N valid
  // elements, discarding those padded rows. In-order SVM-pool queue keeps the
  // quant ordered after the rmsnorm write of the same cl_mem. Removes one
  // dispatch + its host-bound inter-kernel idle per FC input (decode is
  // dispatch-bound: clprof rmsnorm->v8c_copy_h2h was 37% of GPU idle).
  // Gated, default off => byte-identical baseline.
  //
  // MEASURED 2026-06-15 (decode clprof): DECODE-NEUTRAL (the M=1 copy is tiny).
  // BUT at PREFILL it is a real win: the skipped cl_mem->sc.act_in staging copy
  // is M*K per FC -- at M=1024 that is ~850MB of GPU->GPU CopyBuffer across the
  // 182 prefill FCs. Skipping it (act-quant reads clmem_in directly) measured
  // (Adreno 840, gemma2_lg QINT4, M=1024, NNTR_GPU_CLMEM_POOL): prefill
  // 859 -> 901 TPS (+5%, crossing 900 = gpu_native parity), token-IDENTICAL
  // (md5 a6710b4d unchanged). So default ON; NNTR_FC_QUANT_DIRECT=0 restores the
  // staging copy. Only engages on the cl_mem-input edge (device_clmem_in); the
  // SVM path is unaffected (no-op).
  static const bool fc_quant_direct = []() {
    const char *e = std::getenv("NNTR_FC_QUANT_DIRECT");
    return !e || e[0] != '0';
  }();
  const bool quant_direct_clmem = fc_quant_direct && device_clmem_in &&
                                  !skip_upload_and_quant && clmem_in != nullptr;

  // RACE#1 fix: select this call's per-fanout activation slot. On a quant-cache
  // HIT (wk/wv after wq -- same input) reuse the slot that already holds the
  // int8 (read-only, like gpu_native "quantize ONCE"); on a MISS (a new fanout
  // or a cl_mem-input edge with the cache disabled) advance the ring so this
  // fanout's quant WRITE lands in a buffer distinct from the prior fanout's
  // still-in-flight GEMM image READ. The fused-rmsq consumer (off by default)
  // binds its own external buffers and uses no slot.
  int act_slot = -1;
  if (!fused_rmsq_hit) {
    act_slot = quant_cache_hit
                 ? sc.last_quant_slot
                 : (sc.ring_pos = (sc.ring_pos + 1) % V8C_ACT_SLOTS);
    // Grow only the chosen slot to this call's (M_pad, K). Grow-only => a hit
    // (same M_pad,K as the miss that filled it) never reallocates, so the
    // cached int8/scale/zp/rs survive for the wk/wv reuse.
    if (!v8c_ensure_buf(ctx, &sc.act_i8[act_slot], &sc.act_i8_bytes[act_slot],
                        (size_t)M_pad * K, CL_MEM_READ_WRITE) ||
        !v8c_ensure_buf(ctx, &sc.act_scale[act_slot],
                        &sc.act_scale_bytes[act_slot], sizeof(float) * M_pad,
                        CL_MEM_READ_WRITE) ||
        !v8c_ensure_buf(ctx, &sc.act_rs[act_slot], &sc.act_rs_bytes[act_slot],
                        sizeof(int) * M_pad, CL_MEM_READ_WRITE) ||
        !v8c_ensure_buf(ctx, &sc.act_zp[act_slot], &sc.act_zp_bytes[act_slot],
                        sizeof(int) * M_pad, CL_MEM_READ_WRITE))
      return false;
    act_i8_arg = sc.act_i8[act_slot];
    act_scale_arg = sc.act_scale[act_slot];
    act_zp_arg = sc.act_zp[act_slot];
    act_rs_arg = sc.act_rs[act_slot];
  }

  // Submit the accumulated batch BEFORE this FC re-quants (DEFAULT ON,
  // NNTR_FC_FLUSH=0 disables): the quant's act-image WRITE and every GEMM
  // image READ of that slot (this FC + the cache-hit siblings wk/wv/up that
  // reuse it) then share ONE submission. The end-of-FC flush (mode 1)
  // corrupted outputs because it split the cache-hit siblings' image reads
  // from the producer's write across submissions (the image-from-buffer
  // texture-L1 staleness); flushing only at re-quant boundaries keeps every
  // image write->read pair batch-local. Validated token-identical (20/20
  // cross-build + 10/10 staging suite) at +6% TPS (547 -> 580 hot).
  // Default ON only on the Adreno image path: under NNTR_V8C_BUF (Intel
  // buffer path) the flush measurably ALTERS outputs there too, and the
  // deferred-submission stall it fixes is an Adreno driver behavior -- keep
  // Intel byte-identical to the pre-flush baseline (same gating precedent as
  // the program prewarm).
  static const int fc_flush_mode = []() {
    const char *e = std::getenv("NNTR_FC_FLUSH");
    if (e)
      return std::atoi(e);
    // 2026-06-12 re-baseline: default mode 1 (trailing flush after every
    // FC) -- the submit-split output perturbation it caused is a race
    // pattern, not a math change (drained-capture proof), and the
    // re-baselined reference outputs absorb it. +15%-class idle recovery.
    return v8c_use_buffer_path() ? 0 : 1; // [T8] Intel buffer ⇒ mode 0
  }();
  if (fc_flush_mode == 2 && !skip_upload_and_quant)
    clFlush(q);

  if (!skip_upload_and_quant) {
    if (prof_enabled) T0 = NOW();
    if (device_clmem_in) {
      // NNTR_GPU_CLMEM_POOL: GPU->GPU copy of the producer's cl_mem sub-buffer
      // (the normed activation, written device-direct by the converted rmsnorm)
      // into sc.act_in. No SVM map/unmap -- the in-order SVM-pool queue orders
      // this copy after the rmsnorm coop write of the same cl_mem.
      // [Lever 1] quant_direct_clmem skips this copy: the act-quant below reads
      // clmem_in directly.
      if (!quant_direct_clmem &&
          clEnqueueCopyBuffer(q, clmem_in, sc.act_in, 0, 0,
                              (size_t)M * K * act_elem, 0, nullptr,
                              nullptr) != CL_SUCCESS)
        return false;
    } else if (use_resident_input) {
      // GPU→GPU copy of the resident FP32/FP16 activation into sc.act_in.
      // Same shape as a host upload would produce, just without crossing
      // PCIe. Padded rows (M_pad > M) are zero-filled below.
      if (clEnqueueCopyBuffer(q, in_backing->buffer(), sc.act_in, 0, 0,
                              (size_t)M * K * act_elem, 0, nullptr,
                              nullptr) != CL_SUCCESS)
        return false;
    } else if (in_svm) {
      // GPU copy of the SVM-resident activation into sc.act_in -- no host
      // upload. Downstream quant/image/GEMM see the same sc.act_in as before.
      v8c_copy_svm_to_clmem(cur_in_ptr, sc.act_in,
                            (unsigned int)((size_t)M * K), cur_dtype == 1,
                            /*device_owned=*/device_in);
    } else {
      if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE, 0,
                               (size_t)M * K * act_elem, cur_in_ptr, 0,
                               nullptr, nullptr) != CL_SUCCESS)
        return false;
    }
    if (M_pad > M && !quant_direct_clmem) {
      const size_t pad_bytes = (size_t)(M_pad - M) * K * act_elem;
      std::vector<uint8_t> zeros(pad_bytes, 0);
      if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE,
                               (size_t)M * K * act_elem, pad_bytes,
                               zeros.data(), 0, nullptr, nullptr) != CL_SUCCESS)
        return false;
    }
    if (prof_enabled) {
      clFinish(q);
      T1 = NOW();
      prof.bin[prof_bin].write_ns += NS(T1, T0);
      prof.bin[prof_bin].write_bytes += (long long)M * K * act_elem;
    }
  }
  const double _fc_t1 = fc_tprof_on() ? fc_tprof_now() : 0;
  if (fc_tprof_on())
    fc_tp_entry += _fc_t1 - _fc_t0;

  try {
    // (c) fp→int8 asymmetric act quant + zero-point + row_sum over M_pad rows.
    //     Padded rows map to (scale=1, zp=0, q=0, row_sum=0), so they
    //     contribute zero in the GEMM and don't pollute valid rows.
    //     Step 2b.0: skipped entirely on cache hit (see above).
    // NNTR_CLMEM_PROBE: capture the staged FC input (act_in, post-copy) for
    // the gate/up FCs -- device-side copy only, fan-out corruption bisect.
    {
      static const bool probe_on = std::getenv("NNTR_CLMEM_PROBE") != nullptr;
      if (probe_on) {
        const std::string &on_ = output.getName();
        if (on_.find("ffn_gate") != std::string::npos ||
            on_.find("ffn_up") != std::string::npos ||
            on_.find("_wq") != std::string::npos ||
            on_.find("_wk") != std::string::npos ||
            on_.find("_wv") != std::string::npos)
          clmem_probe_capture((on_ + ":act_in").c_str(), nullptr, sc.act_in,
                              (unsigned int)((size_t)M * K * act_elem));
      }
    }
    if (!skip_upload_and_quant) {
      if (prof_enabled) T0 = NOW();
      // [Lever 1] quant the producer cl_mem (M real rows) directly when
      // quant_direct_clmem; otherwise the staged sc.act_in (M_pad rows incl.
      // the zero pad). Padded act_i8/scale/zp/rs rows [M, M_pad) are left stale
      // in the direct case -- they feed only discarded padded GEMM output rows.
      cl_mem quant_src = quant_direct_clmem ? clmem_in : sc.act_in;
      const unsigned int quant_rows = quant_direct_clmem ? M : M_pad;
      if (input.getDataType() == ml::train::TensorDim::DataType::FP16)
        quantize_act_v8c_fp16_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      else
        quantize_act_v8c_fp32_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      if (prof_enabled) {
        clFinish(q);
        T1 = NOW();
        prof.bin[prof_bin].quant_ns += NS(T1, T0);
      }
      // Update cache key only after a successful quant.
      sc.last_quant_in_ptr = cur_in_ptr;
      sc.last_quant_M = M;
      sc.last_quant_K = K;
      sc.last_quant_M_pad = M_pad;
      sc.last_quant_dtype = cur_dtype;
      sc.last_quant_resident_generation = g_resident_quant_generation.load();
      // Record WHICH slot now holds this input's int8 so a subsequent cache
      // hit (wk/wv) reads the right per-fanout buffer, not whatever the ring
      // last pointed at.
      sc.last_quant_slot = act_slot;
    }

    // === Per-call CPU vs GPU quant equality check ===
    // For NNTR_V8C_QUANT_CHECK=1, recompute KAI-style asymmetric int8 act
    // quant on CPU for the same input row, compare against GPU readback.
    // Prints first divergent index per row and aggregate counts.
    if (std::getenv("NNTR_V8C_QUANT_CHECK") && M == 1003 &&
        input.getDataType() == ml::train::TensorDim::DataType::FP32) {
      static int qcheck_id = 0;
      ++qcheck_id;
      if (qcheck_id <= 3) {
        std::vector<int8_t> gpu_q(M_pad * K);
        std::vector<float> gpu_scale(M_pad);
        std::vector<int32_t> gpu_zp(M_pad);
        std::vector<int32_t> gpu_rs(M_pad);
        clEnqueueReadBuffer(q, act_i8_arg, CL_TRUE, 0, (size_t)M_pad * K,
                            gpu_q.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, act_scale_arg, CL_TRUE, 0,
                            sizeof(float) * M_pad, gpu_scale.data(), 0,
                            nullptr, nullptr);
        clEnqueueReadBuffer(q, act_zp_arg, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_zp.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, act_rs_arg, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_rs.data(), 0, nullptr, nullptr);
        const float *in = input.getData<float>();
        // CPU reference: same algorithm as KAI qai8dxp_f32 (lines 120-186 of
        // kai_lhs_quant_pack_qai8dxp_f32.c). Check first 2 rows only.
        for (int row = 0; row < (int)std::min(M, 2u); ++row) {
          float fmin = 0.0f, fmax = 0.0f;
          for (unsigned int k = 0; k < K; ++k) {
            float v = in[row * K + k];
            if (v < fmin) fmin = v;
            if (v > fmax) fmax = v;
          }
          float rmin = fmin < 0.0f ? fmin : 0.0f;
          float rmax = fmax > 0.0f ? fmax : 0.0f;
          float qmin = -128.0f, qmax = 127.0f;
          float scale = rmin == rmax ? 1.0f : (qmax - qmin) / (rmax - rmin);
          float recip = scale ? 1.0f / scale : 0.0f;
          float dmin = rmin * scale, dmax = rmax * scale;
          float zp_from_min = qmin + dmin;
          float zp_from_max = qmax + dmax;
          float zpf = (zp_from_min + zp_from_max > 0.0f) ? (qmin - dmin)
                                                         : (qmax - dmax);
          zpf = std::max(zpf, qmin);
          zpf = std::min(zpf, qmax);
          int cpu_zp = (int)std::round(zpf);
          int cpu_rs = 0;
          int q_diffs = 0, first_diff_k = -1;
          int8_t cpu_q_first = 0, gpu_q_first = 0;
          for (unsigned int k = 0; k < K; ++k) {
            int v = (int)std::round(in[row * K + k] * scale) + cpu_zp;
            if (v < -128) v = -128;
            if (v > 127) v = 127;
            int8_t cpuq = (int8_t)v;
            int8_t gpuq = gpu_q[row * K + k];
            if (cpuq != gpuq) {
              if (first_diff_k < 0) {
                first_diff_k = k;
                cpu_q_first = cpuq;
                gpu_q_first = gpuq;
              }
              q_diffs++;
            }
            cpu_rs += cpuq;
          }
          std::fprintf(
            stderr,
            "[V8C-QCHECK id=%d row=%d] cpu_scale=%.6f gpu_scale=%.6f | "
            "cpu_zp=%d gpu_zp=%d | cpu_rs=%d gpu_rs=%d | rmin=%.4f rmax=%.4f | "
            "q_diffs=%d/1024 first_k=%d cpu_q=%d gpu_q=%d\n",
            qcheck_id, row, recip, gpu_scale[row], cpu_zp, gpu_zp[row],
            cpu_rs, gpu_rs[row], rmin, rmax, q_diffs, first_diff_k,
            (int)cpu_q_first, (int)gpu_q_first);
          std::fflush(stderr);
        }
      }
    }

    // v8c GEMM input binding. The buffer path (NNTR_V8C_BUF, Intel NEO) selects
    // the *_buf kernels whose args are __global uint4* — they MUST be bound to
    // raw cl_mem buffers (the int8 act scratch and the weight backing buffer),
    // NOT image2d objects. Only the Adreno image-sampling path builds an
    // image2d view over the act buffer. (Mirror of gpu_native qwen3_forward.cpp
    // use_v8c_buf ? *_buf : *_image selection — the previous code always passed
    // images, so the buffer kernel read wrong memory and produced garbage.)
    const bool use_buf = v8c_buffer_path();
    cl_mem act_image = nullptr;
    bool act_image_transient = false; // true => owned here, release after GEMM
    if (!use_buf) {
      // Build the image2d view over the int8 act buffer (zero-copy, tensor
      // virtualization). RACE#1 fix: for the per-fanout slot path the view is
      // CACHED on the slot and reused across the fanout's GEMMs, rebuilt only
      // when the slot's buffer is grown or (M_pad, K) change -- removing the
      // per-call clCreateImage/clReleaseMemObject churn AND the old exception-
      // path image leak. The fused-rmsq consumer (no slot, external buffer
      // that varies per input) keeps a transient per-call view.
      if (prof_enabled) T0 = NOW();
      cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
      cl_image_desc adesc{};
      adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
      adesc.image_width = K / 16;
      adesc.image_height = M_pad;
      adesc.image_row_pitch = K;
      adesc.buffer = act_i8_arg;
      if (act_slot >= 0) {
        if (sc.act_image[act_slot] == nullptr ||
            sc.act_image_buf[act_slot] != act_i8_arg ||
            sc.act_image_M_pad[act_slot] != M_pad ||
            sc.act_image_K[act_slot] != K) {
          if (sc.act_image[act_slot]) {
            clReleaseMemObject(sc.act_image[act_slot]);
            sc.act_image[act_slot] = nullptr;
          }
          cl_mem img =
            clCreateImage(ctx, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);
          if (err != CL_SUCCESS)
            throw std::runtime_error("act image view fail");
          sc.act_image[act_slot] = img;
          sc.act_image_buf[act_slot] = act_i8_arg;
          sc.act_image_M_pad[act_slot] = M_pad;
          sc.act_image_K[act_slot] = K;
        }
        act_image = sc.act_image[act_slot];
      } else {
        act_image =
          clCreateImage(ctx, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);
        if (err != CL_SUCCESS)
          throw std::runtime_error("act image view fail");
        act_image_transient = true;
      }
      if (prof_enabled) {
        T1 = NOW();
        prof.bin[prof_bin].image_ns += NS(T1, T0);
        T0 = T1;
      }
    }

    // (b) v8c GEMM — run on padded M_pad rows, but only read back the valid
    // M rows to the caller buffer.
    //
    // Direct output (kernel-store, no copy): when the FC output is a
    // GPU_CLMEM-resident FP16 tensor, point the GEMM's Y at its planner
    // sub-buffer with the M_valid store guard, eliminating the separate
    // v8c_copy_h2h writer kernel (46ms GPU + 182 enqueues per 1K prefill).
    // Same kernel->kernel ordering the copy writer relied on. Disabled when
    // a debug consumer needs sc.y_fp16 (probe/dualout/trace) and by
    // NNTR_V8C_DIRECT_OUT=0.
    const bool out_clmem =
      clmem_pool && output.getMemoryData() &&
      output.getMemoryData()->isClMem() &&
      output.getMemoryData()->deviceMem() != nullptr &&
      output.getDataType() == ml::train::TensorDim::DataType::FP16;
    static const bool direct_out_enabled = []() {
      const char *e = std::getenv("NNTR_V8C_DIRECT_OUT");
      return !(e && e[0] == '0');
    }();
    static const bool y_dbg_consumer =
      std::getenv("NNTR_CLMEM_PROBE") != nullptr ||
      std::getenv("NNTR_CLMEM_DUALOUT") != nullptr ||
      std::getenv("NNTR_CLMEM_OUTCHECK") != nullptr ||
      std::getenv("NNTR_CLMEM_OUTBAR") != nullptr ||
      std::getenv("NNTR_V8C_TRACE") != nullptr;
    const bool direct_out =
      direct_out_enabled && out_clmem && !y_dbg_consumer;
    cl_mem gemm_y_arg =
      direct_out ? static_cast<cl_mem>(output.getMemoryData()->deviceMem())
                 : sc.y_fp16;
    cl_mem gemm_act_arg = use_buf ? act_i8_arg : act_image;
    cl_mem gemm_wgt_arg = use_buf ? w->weight_buf : w->weight_image;
    gemm_int8_v8c_cl(gemm_act_arg, gemm_wgt_arg, act_scale_arg, w->scale_buf,
                     act_rs_arg, act_zp_arg, w->row_sum_w_int4, gemm_y_arg,
                     M_pad, N, K, direct_out ? M : M_pad);
    // NNTR_XE3_FC_SYNC: narrowed Xe3 coherence fix. The in-order queue does not
    // give kernel->kernel coarse-grained-SVM coherence on NEO 26.22; the global
    // hammer (NNTR_XE3_SYNC, clFinish after EVERY dispatch) fixes it but serializes
    // decode. The bisect showed a clFinish after the FC GEMM alone is sufficient
    // (it is the dominant SVM-producing op and lands between most consumers), so
    // draining only here keeps coherence while restoring decode pipelining.
    static const bool xe3_fc_sync = std::getenv("NNTR_XE3_FC_SYNC") != nullptr;
    if (xe3_fc_sync)
      clFinish(q);
    if (prof_enabled) {
      clFinish(q);
      T1 = NOW();
      prof.bin[prof_bin].gemm_ns += NS(T1, T0);
    }

    // === FP16 v8c GEMM output sum/min/max trace (NNTR_V8C_FP16_TRACE) ===
    // Debug: which FC (by N) produces a degenerate (all-zero / Inf) output.
    if (std::getenv("NNTR_V8C_FP16_TRACE")) {
      clFinish(q);
      const size_t rows_dbg = direct_out ? (size_t)M : (size_t)M_pad;
      const size_t cnt_dbg = rows_dbg * (size_t)N;
      std::vector<uint16_t> ph_dbg(cnt_dbg);
      clEnqueueReadBuffer(q, gemm_y_arg, CL_TRUE, 0,
                          sizeof(uint16_t) * cnt_dbg, ph_dbg.data(), 0, nullptr,
                          nullptr);
      double s_dbg = 0.0;
      float mn_dbg = 1e30f, mx_dbg = -1e30f;
      size_t nz_dbg = 0;
      for (size_t i = 0; i < (size_t)M * N; ++i) {
        float v = v8c_h2f(ph_dbg[i]);
        s_dbg += v;
        if (v != 0.0f) ++nz_dbg;
        if (v < mn_dbg) mn_dbg = v;
        if (v > mx_dbg) mx_dbg = v;
      }
      std::vector<float> wsc_dbg(N);
      clEnqueueReadBuffer(q, w->scale_buf, CL_TRUE, 0, sizeof(float) * N,
                          wsc_dbg.data(), 0, nullptr, nullptr);
      float wsc_mn = 1e30f, wsc_mx = -1e30f;
      double wsc_sum = 0;
      for (unsigned int i = 0; i < N; ++i) {
        if (wsc_dbg[i] < wsc_mn) wsc_mn = wsc_dbg[i];
        if (wsc_dbg[i] > wsc_mx) wsc_mx = wsc_dbg[i];
        wsc_sum += wsc_dbg[i];
      }
      // === Definitive per-FC CPU reference (int8 act × int4 w + bias-corr) ===
      // No cross-engine confound: checks the GPU output against the math-correct
      // value computed from the SAME quantized inputs the GPU used.
      float max_rel = -1.0f, gpu_at = 0, ref_at = 0;
      unsigned worst_m = 0, worst_n = 0;
      if (use_buf) {
        std::vector<int8_t> gq((size_t)M_pad * K);
        std::vector<float> gsa(M_pad);
        std::vector<int32_t> gzp(M_pad), grs(M_pad), grsw(N);
        clEnqueueReadBuffer(q, act_i8_arg, CL_TRUE, 0, (size_t)M_pad * K,
                            gq.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, act_scale_arg, CL_TRUE, 0, sizeof(float) * M_pad,
                            gsa.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, act_zp_arg, CL_TRUE, 0, sizeof(int) * M_pad,
                            gzp.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, act_rs_arg, CL_TRUE, 0, sizeof(int) * M_pad,
                            grs.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, w->row_sum_w_int4, CL_TRUE, 0, sizeof(int) * N,
                            grsw.data(), 0, nullptr, nullptr);
        const uint8_t *sa = weight.getData<uint8_t>();
        const size_t NR = 4, KRSR = 8, BPK = 2 * NR * KRSR, nbpsr = NR * (K / 2);
        auto dec = [&](unsigned n, unsigned k) -> int {
          size_t kbl = k / 32, kp = k % 32, sr = n / NR, nr = n % NR;
          const uint8_t *ba = sa + sr * nbpsr + kbl * BPK + nr * KRSR;
          const uint8_t *bb = ba + NR * KRSR;
          uint8_t nib;
          if (kp < 8) nib = (ba[kp] ^ 0x88) & 0xF;
          else if (kp < 16) nib = (bb[kp - 8] ^ 0x88) & 0xF;
          else if (kp < 24) nib = ((ba[kp - 16] ^ 0x88) >> 4) & 0xF;
          else nib = ((bb[kp - 24] ^ 0x88) >> 4) & 0xF;
          return (int)nib - 8;
        };
        for (unsigned m = 0; m < std::min(M, 2u); ++m) {
          for (unsigned n = 0; n < std::min(N, 16u); ++n) {
            long acc = 0;
            for (unsigned k = 0; k < K; ++k)
              acc += (long)gq[m * (size_t)K + k] * (dec(n, k) + 8);
            long corr = acc - 8L * grs[m] - (long)gzp[m] * grsw[n];
            float ref = (float)corr * gsa[m] * wsc_dbg[n];
            float gv = v8c_h2f(ph_dbg[m * (size_t)N + n]);
            float rel = std::fabs(gv - ref) / (std::fabs(ref) + 1e-3f);
            if (rel > max_rel) {
              max_rel = rel; gpu_at = gv; ref_at = ref; worst_m = m; worst_n = n;
            }
          }
        }
      }
      std::fprintf(stderr,
                   "[FP16FC] %-28s M=%u N=%u K=%u out[%.2f,%.2f] wsc=%.4g "
                   "RELERR=%.4f @(%u,%u) gpu=%.3f ref=%.3f\n",
                   weight.getName().c_str(), M, N, K, mn_dbg, mx_dbg,
                   wsc_sum / N, max_rel, worst_m, worst_n, gpu_at, ref_at);
      std::fflush(stderr);
    }

    // === GEMM-output check: same int8 act + same int4 w + same formula. ===
    // Diagnoses whether the v8c GEMM kernel itself computes a value
    // mathematically identical to the int8×int4 + bias-correction formula
    // CPU would compute given the SAME quantized inputs. Quant (verified
    // bit-exact via NNTR_V8C_QUANT_CHECK), permute (verified byte-exact via
    // test/v8c_permute_test) are independently checked, so any divergence
    // here is in the GEMM/correction itself.
    if (std::getenv("NNTR_V8C_GEMM_CHECK") && M == 1003 &&
        input.getDataType() == ml::train::TensorDim::DataType::FP32) {
      static int gcheck_id = 0;
      ++gcheck_id;
      if (gcheck_id <= 2) {
        clFinish(q);
        std::vector<int8_t> gpu_q(M_pad * K);
        std::vector<float> gpu_scale_act(M_pad);
        std::vector<int32_t> gpu_zp(M_pad);
        std::vector<int32_t> gpu_rs(M_pad);
        std::vector<int32_t> gpu_rsw(N);
        std::vector<float> gpu_scale_wgt(N);
        std::vector<uint16_t> gpu_y((size_t)M * N);
        clEnqueueReadBuffer(q, act_i8_arg, CL_TRUE, 0, (size_t)M_pad * K,
                            gpu_q.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, act_scale_arg, CL_TRUE, 0,
                            sizeof(float) * M_pad, gpu_scale_act.data(), 0,
                            nullptr, nullptr);
        clEnqueueReadBuffer(q, act_zp_arg, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_zp.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, act_rs_arg, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_rs.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, w->row_sum_w_int4, CL_TRUE, 0,
                            sizeof(int) * N, gpu_rsw.data(), 0, nullptr,
                            nullptr);
        clEnqueueReadBuffer(q, w->scale_buf, CL_TRUE, 0, sizeof(float) * N,
                            gpu_scale_wgt.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.y_fp16, CL_TRUE, 0,
                            sizeof(uint16_t) * gpu_y.size(), gpu_y.data(), 0,
                            nullptr, nullptr);
        // Decode v8c weight bytes for the same (n, k) used in GPU kernel.
        // section_a is KAI Section A, but we wrote v8c-permuted into
        // backing buffer. Easier to walk Section A → int4 directly.
        const uint8_t *section_a = weight.getData<uint8_t>();
        constexpr size_t KAI_NR2 = 4, KAI_KR_BY_SR2 = 8;
        constexpr size_t KAI_BYTES_PER_KBLK2 = 2 * KAI_NR2 * KAI_KR_BY_SR2;
        const size_t nibble_bytes_per_super_row = KAI_NR2 * (K / 2);
        auto decode_int4 = [&](unsigned int n, unsigned int k) -> int {
          const size_t kbl = k / 32;
          const size_t kp = k % 32;
          const size_t sr = n / KAI_NR2;
          const size_t nr = n % KAI_NR2;
          const uint8_t *sr_base =
            section_a + sr * nibble_bytes_per_super_row;
          const uint8_t *blk_a =
            sr_base + kbl * KAI_BYTES_PER_KBLK2 + nr * KAI_KR_BY_SR2;
          const uint8_t *blk_b = blk_a + KAI_NR2 * KAI_KR_BY_SR2;
          uint8_t nib = 0;
          if (kp < 8)
            nib = (blk_a[kp] ^ 0x88) & 0x0F;
          else if (kp < 16)
            nib = (blk_b[kp - 8] ^ 0x88) & 0x0F;
          else if (kp < 24)
            nib = ((blk_a[kp - 16] ^ 0x88) >> 4) & 0x0F;
          else
            nib = ((blk_b[kp - 24] ^ 0x88) >> 4) & 0x0F;
          return (int)nib - 8;
        };
        // CPU reference for a handful of (m, n) positions: replicate the
        // GPU bias-corrected dot product exactly.
        int diffs = 0;
        double max_abs_diff = 0;
        unsigned int worst_m = 0, worst_n = 0;
        float gpu_at_worst = 0, ref_at_worst = 0;
        for (unsigned int m = 0; m < std::min(M, 2u); ++m) {
          int rs = gpu_rs[m];
          int zp = gpu_zp[m];
          float s_act = gpu_scale_act[m];
          for (unsigned int n = 0; n < std::min(N, 32u); ++n) {
            int acc = 0;
            for (unsigned int k = 0; k < K; ++k) {
              int aq = gpu_q[m * K + k];
              int w_int4 = decode_int4(n, k);
              acc += aq * (w_int4 + 8);
            }
            int corrected = acc - 8 * rs - zp * gpu_rsw[n];
            float ref_v = (float)corrected * s_act * gpu_scale_wgt[n];
            float gpu_v = v8c_h2f(gpu_y[m * N + n]);
            float d = gpu_v - ref_v;
            if (std::fabs(d) > max_abs_diff) {
              max_abs_diff = std::fabs(d);
              worst_m = m;
              worst_n = n;
              gpu_at_worst = gpu_v;
              ref_at_worst = ref_v;
            }
            if (std::fabs(d) > 1e-3f) ++diffs;
          }
        }
        std::fprintf(stderr,
                     "[V8C-GCHECK id=%d M=%u N=%u K=%u] diffs(>1e-3)=%d/%u "
                     "max|diff|=%.5f at (%u,%u) gpu=%.5f ref=%.5f\n",
                     gcheck_id, M, N, K, diffs, 2 * std::min(N, 32u),
                     max_abs_diff, worst_m, worst_n, gpu_at_worst,
                     ref_at_worst);
        std::fflush(stderr);
      }
    }

    // === Per-call CPU vs GPU divergence trace ===
    // For NNTR_V8C_TRACE=1, compute the CPU "fp32 act × fp32 dequant w"
    // reference for the same (input, weight) and report relL2 vs the GPU
    // fp16 readback. This is the math-correct reference (no act quant), so
    // any extra error beyond ~ int4 quant noise points at v8c's symmetric
    // int8 act quant path.
    if (std::getenv("NNTR_V8C_TRACE") &&
        input.getDataType() == ml::train::TensorDim::DataType::FP32) {
      static int trace_id = 0;
      ++trace_id;
      std::vector<uint16_t> y_peek((size_t)M * N);
      clEnqueueReadBuffer(q, sc.y_fp16, CL_TRUE, 0,
                          sizeof(uint16_t) * y_peek.size(), y_peek.data(), 0,
                          nullptr, nullptr);
      std::vector<float> w_scale_h(N);
      clEnqueueReadBuffer(q, w->scale_buf, CL_TRUE, 0, sizeof(float) * N,
                          w_scale_h.data(), 0, nullptr, nullptr);
      const uint8_t *section_a = weight.getData<uint8_t>();
      const float *in_f = input.getData<float>();
      // Reference: v[m,n] = Σ_k act_fp[m,k] × (int4_w[n,k]) × scale_w[n]
      //   where int4_w is the actually-stored quantized weight value
      //   (decoded from KAI Section A nibble, range [-8..7]).
      constexpr size_t KAI_NR2 = 4, KAI_KR_BY_SR2 = 8;
      constexpr size_t KAI_BYTES_PER_KBLK2 = 2 * KAI_NR2 * KAI_KR_BY_SR2;
      const size_t nibble_bytes_per_super_row = KAI_NR2 * (K / 2);
      double sum_sq_diff = 0.0;
      double sum_sq_ref = 0.0;
      float max_abs_diff = 0.0f;
      unsigned int worst_m = 0, worst_n = 0;
      float gpu_at_worst = 0, ref_at_worst = 0;
      // Sample 8 (m, n) positions for tractability when M or N is large.
      const unsigned int sample_M = std::min(M, 4u);
      const unsigned int sample_N = std::min(N, 32u);
      for (unsigned int m = 0; m < sample_M; ++m) {
        for (unsigned int j = 0; j < sample_N; ++j) {
          // Decode int4 for (n=j, k) for all k
          float ref = 0.0f;
          for (unsigned int k = 0; k < K; ++k) {
            const size_t kbl = k / 32;
            const size_t kp = k % 32;
            const size_t sr = j / KAI_NR2;
            const size_t nr = j % KAI_NR2;
            const uint8_t *sr_base =
              section_a + sr * nibble_bytes_per_super_row;
            const uint8_t *blk_a = sr_base + kbl * KAI_BYTES_PER_KBLK2 +
                                   nr * KAI_KR_BY_SR2;
            const uint8_t *blk_b = blk_a + KAI_NR2 * KAI_KR_BY_SR2;
            uint8_t nib = 0;
            if (kp < 8) nib = (blk_a[kp] ^ 0x88) & 0x0F;
            else if (kp < 16) nib = (blk_b[kp - 8] ^ 0x88) & 0x0F;
            else if (kp < 24) nib = ((blk_a[kp - 16] ^ 0x88) >> 4) & 0x0F;
            else nib = ((blk_b[kp - 24] ^ 0x88) >> 4) & 0x0F;
            int int_w = (int)nib - 8;
            ref += in_f[m * K + k] * (float)int_w;
          }
          ref *= w_scale_h[j];
          const float gpu_v = v8c_h2f(y_peek[m * N + j]);
          const float d = gpu_v - ref;
          sum_sq_diff += d * d;
          sum_sq_ref += ref * ref;
          float ad = std::fabs(d);
          if (ad > max_abs_diff) {
            max_abs_diff = ad;
            worst_m = m;
            worst_n = j;
            gpu_at_worst = gpu_v;
            ref_at_worst = ref;
          }
        }
      }
      double relL2 =
        sum_sq_ref > 0.0 ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;
      std::fprintf(stderr,
                   "[V8C-TRACE] id=%d M=%u N=%u K=%u sampled=%ux%u "
                   "relL2=%.4f%% max|diff|=%.4f at (m=%u,n=%u) gpu=%.4f "
                   "ref=%.4f\n",
                   trace_id, M, N, K, sample_M, sample_N, relL2 * 100.0,
                   max_abs_diff, worst_m, worst_n, gpu_at_worst,
                   ref_at_worst);
      std::fflush(stderr);
    }

    // Read output fp16 (only the valid M rows; padded rows are discarded),
    // convert to output dtype.
    // Planner-decided STATIC residency: a GPU_CLMEM output (FP16 by
    // derivation) either was written DIRECTLY by the GEMM store guard
    // (direct_out above -- nothing left to do) or gets the fp16 result
    // written into its planner cl_mem sub-buffer by the kernel writer.
    // (out_clmem was hoisted above the GEMM dispatch for direct_out.)
    const bool out_resident =
      !out_clmem && output.getMemoryData() && output.getMemoryData()->isSVM() &&
      (output.getDataType() == ml::train::TensorDim::DataType::FP32 ||
       output.getDataType() == ml::train::TensorDim::DataType::FP16);
    // NNTR_CLMEM_PROBE: capture the raw fp16 GEMM result for the gate/up FCs
    // (sc.y_fp16 is cl_mem in BOTH modes -- directly comparable across runs).
    {
      static const bool probe_on = std::getenv("NNTR_CLMEM_PROBE") != nullptr;
      if (probe_on) {
        const std::string &on_ = output.getName();
        if (on_.find("ffn_gate") != std::string::npos ||
            on_.find("ffn_up") != std::string::npos ||
            on_.find("_wq") != std::string::npos ||
            on_.find("_wk") != std::string::npos ||
            on_.find("_wv") != std::string::npos)
          clmem_probe_capture((on_ + ":y").c_str(), nullptr, sc.y_fp16,
                              (unsigned int)(sizeof(uint16_t) * (size_t)M * N));
      }
    }
    if (out_clmem) {
      cl_mem out_sub =
        static_cast<cl_mem>(output.getMemoryData()->deviceMem());
      // KERNEL writer (not clEnqueueCopyBuffer): see the note inside
      // v8c_write_output_resident -- the copy engine is not reliably ordered
      // against the producing GEMM without a drain on this driver.
      // Skipped in direct_out mode (the GEMM already stored into out_sub).
      if (!direct_out)
        v8c_write_output_resident(sc.y_fp16, output,
                                  (unsigned int)((size_t)M * N), true,
                                  static_cast<void *>(out_sub));
      // NNTR_CLMEM_OUTBAR=1 (diagnostic): drain right after the cl_mem out
      // write -- if this heals the divergence, kernel-write -> kernel-read
      // ordering on sub-buffers is broken without a drain on this driver.
      static const bool outbar = std::getenv("NNTR_CLMEM_OUTBAR") != nullptr;
      if (outbar)
        clFinish(q);
      // NNTR_CLMEM_DUALOUT=1 (bisect): ALSO run the legacy SVM resident
      // writer, so both planes hold identical fresh bytes and a legacy
      // consumer (NNTR_CLMEM_MHA_OFF) sees exactly the baseline values.
      static const bool dualout =
        std::getenv("NNTR_CLMEM_DUALOUT") != nullptr;
      if (dualout && output.getMemoryData()->isSVM() &&
          output.getDataType() == ml::train::TensorDim::DataType::FP16)
        v8c_write_output_resident(sc.y_fp16, output,
                                  (unsigned int)((size_t)M * N), true);
      // NNTR_CLMEM_OUTCHECK=1 (diagnostic, requires DUALOUT): drain and
      // compare the two planes byte-for-byte right here. Schedule-invasive
      // (clFinish) -- debugging only.
      static const bool outcheck =
        std::getenv("NNTR_CLMEM_OUTCHECK") != nullptr;
      if (dualout && outcheck) {
        clFinish(q);
        const size_t n_b = sizeof(uint16_t) * (size_t)M * N;
        std::vector<uint8_t> dev(n_b);
        clEnqueueReadBuffer(q, out_sub, CL_TRUE, 0, n_b, dev.data(), 0,
                            nullptr, nullptr);
        const uint8_t *svm_p = output.getData<uint8_t>();
        size_t diff = 0;
        for (size_t i = 0; i < n_b; ++i)
          if (dev[i] != svm_p[i])
            ++diff;
        static int oc_n = 0;
        if (diff != 0 || ++oc_n <= 8)
          std::fprintf(stderr, "[outcheck] %-28s M=%u N=%u diff=%zu/%zu\n",
                       output.getName().c_str(), M, N, diff, n_b);
        std::fflush(stderr);
      }
    } else if (out_resident) {
      // Residency: write the fp16 GEMM result straight into the SVM output on
      // the GPU, no host readback. fp16 output is a plain copy (no conversion);
      // fp32 output is converted via cvt_h2f.
      v8c_write_output_resident(
        sc.y_fp16, output, (unsigned int)((size_t)M * N),
        output.getDataType() == ml::train::TensorDim::DataType::FP16);
    } else {
      // Host-bounce path: read output fp16 (only the valid M rows; padded rows
      // are discarded), convert to output dtype on the host.
      if (prof_enabled) T0 = NOW();
      std::vector<uint16_t> y_host((size_t)M * N);
      clEnqueueReadBuffer(q, sc.y_fp16, CL_TRUE, 0,
                          sizeof(uint16_t) * y_host.size(), y_host.data(), 0,
                          nullptr, nullptr);
      if (prof_enabled) {
        T1 = NOW();
        prof.bin[prof_bin].read_ns += NS(T1, T0);
        prof.bin[prof_bin].read_bytes +=
          (long long)sizeof(uint16_t) * (long long)M * N;
        prof.bin[prof_bin].calls++;
        prof.bin[prof_bin].m_sum += M;
        prof.total_calls++;
        // Dump on bin transition (prefill → decode shows up as M>256 → M=1).
        if (last_dumped_bin != prof_bin && last_dumped_bin >= 0) {
          char tag[64];
          std::snprintf(tag, sizeof(tag), "BIN-TRANSITION %s->%s",
                        V8cProf::bin_name(last_dumped_bin),
                        V8cProf::bin_name(prof_bin));
          prof.dump(tag);
        }
        last_dumped_bin = prof_bin;
        // Periodic dump every 500 calls so we don't depend on shutdown.
        if (prof.total_calls % 500 == 0) {
          char tag[32];
          std::snprintf(tag, sizeof(tag), "PERIODIC@%d", prof.total_calls);
          prof.dump(tag);
        }
      }
      if (output.getDataType() == ml::train::TensorDim::DataType::FP32) {
        float *out = output.getData<float>();
        for (size_t i = 0; i < y_host.size(); ++i) out[i] = v8c_h2f(y_host[i]);
      } else if (output.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
        std::memcpy(output.getData<uint8_t>(), y_host.data(),
                    sizeof(uint16_t) * y_host.size());
      } else {
        if (act_image_transient && act_image)
          clReleaseMemObject(act_image);
        throw std::runtime_error("unsupported output dtype");
      }
    }
    // Only the transient (fused-rmsq) view is owned here; the per-fanout
    // slot views are cached on V8cScratch and released on rebuild.
    if (act_image_transient && act_image)
      clReleaseMemObject(act_image);

    // FC_FLUSH mode 1 (2026-06-12 re-baseline DEFAULT): submit this FC's
    // enqueue chain now instead of at the next blocking call -- recovers the
    // norm->FC idle band (~+15% TPS class). The output perturbation that
    // kept this opt-in was proven by the drained-capture probe to be a race
    // PATTERN change, not a math change (all intermediate values
    // bit-identical); the re-baselined reference outputs absorb it.
    // NNTR_FC_FLUSH=0 disables all flushing, =2 restores the
    // re-quant-entry batch-local rule.
    if (fc_flush_mode == 1)
      clFlush(q);
    if (fc_tprof_on()) {
      const double t2 = fc_tprof_now();
      fc_tp_tail += t2 - (_fc_t1 > 0 ? _fc_t1 : t2);
      if (++fc_tp_n % 182 == 0) {
        std::fprintf(stderr,
                     "[FC-TPROF] n=%d entry+stage=%.2fms quant..flush=%.2fms\n",
                     fc_tp_n, fc_tp_entry, fc_tp_tail);
        std::fflush(stderr);
        fc_tp_entry = fc_tp_tail = 0;
      }
    }

    // === Step 1e bridge round-trip (paper §3.2). Attach the cached
    // v8c weight backing to the output tensor as a non-owning tracer.
    // CPU consumers ignore this field. This is purely a bridge
    // integrity hook today; Step 2's fused QKV kernel will replace it
    // with a real output backing pointing at the cl_mem the next
    // GPU layer will consume. NNTR_TENSOR_BRIDGE_TRIP=1 logs the
    // first round-trip on real device traffic to confirm wiring.
    output.setBacking(w->backing.get());
    static int logged_trip = 0;
    if (!logged_trip && std::getenv("NNTR_TENSOR_BRIDGE_TRIP") != nullptr) {
      logged_trip = 1;
      tv::TensorBacking *back = output.getBacking();
      std::fprintf(stderr,
                   "[Step1e] bridge round-trip: set=%p get=%p %s\n",
                   (void *)w->backing.get(), (void *)back,
                   back == w->backing.get() ? "OK" : "MISMATCH");
      std::fflush(stderr);
    }
  } catch (...) {
    return false;
  }
  return true;
}

// =============================================================================
// Step 2 — Fused Q + K + V projection + RoPE + layout transform.
//
// Paper reference: arXiv:2505.00232 §3.6, "We crafted a custom kernel to
// combine rotary embedding with the layout transformations of query (Q),
// key (K), and value (V) projections."
//
// THIS IS THE STEP 2a SKELETON. The host-side dispatch + env gate are wired
// against the final-form OpenCL kernel signature, but the kernel body is a
// stub (zero-fills outputs). Returns false in all cases except when
// NNTR_FUSED_QKV_GPU=1 AND every binding precondition holds — and even then,
// the actual compute is the stub. Step 2b will replace the kernel body with
// the real shared-quant-then-3-GEMM-then-RoPE math.
//
// The function returns `false` even on stub success: callers (qkv_layer.cpp,
// once wired) MUST fall back to the existing 3-FC + CPU RoPE path. Step 2d
// flips the return to `true` once the kernel body is correctness-validated.
// =============================================================================

namespace {

static bool fused_qkv_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_FUSED_QKV_GPU") != nullptr ? 1 : 0;
  return cached != 0;
}

} // anonymous namespace

bool fused_qkv_rope_layout_gpu(
  const Tensor &input, const Tensor &wq, const Tensor &wk, const Tensor &wv,
  const Tensor &cos_table, const Tensor &sin_table,
  unsigned int from_pos, unsigned int hq, unsigned int hkv, unsigned int dh,
  Tensor &q_out, Tensor &k_out, Tensor &v_out) {

  // Step 2a: gate off by default. No-op when the env flag isn't set so this
  // commit is invisible on every existing run (Qwen3-0.6B baseline, all CIs).
  if (!fused_qkv_env_enabled())
    return false;

  // === Precondition checks ===
  // (Mirroring dotCl_v8c's contract; Step 2b will tighten these once we know
  // exactly what the kernel body requires.)
  if (wq.getDataType() != ml::train::TensorDim::DataType::QINT4 ||
      wk.getDataType() != ml::train::TensorDim::DataType::QINT4 ||
      wv.getDataType() != ml::train::TensorDim::DataType::QINT4)
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false; // Step 2a is FP16-only; FP32 path TBD in 2b.
  if (cos_table.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      sin_table.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  if (hq == 0 || hkv == 0 || dh == 0 || hq % hkv != 0)
    return false;
  if (dh % 2 != 0)
    return false; // RoPE pair rotation needs even head_dim.

  // Step 2a stub: log once on first invocation that the gate fired, then
  // return false so the caller falls back. This proves the dispatch path
  // compiled and is reachable without changing model output.
  static int logged_stub = 0;
  if (!logged_stub) {
    logged_stub = 1;
    std::fprintf(
      stderr,
      "[Step2a] fused_qkv_rope_layout_gpu stub reached: "
      "S=%u hidden=%u hq=%u hkv=%u dh=%u from=%u\n",
      input.height(),
      (input.getFormat() == Tformat::NHWC) ? input.channel() : input.width(),
      hq, hkv, dh, from_pos);
    std::fflush(stderr);
  }

  // Silence unused-param warnings while the body is a stub.
  (void)wq; (void)wk; (void)wv;
  (void)cos_table; (void)sin_table;
  (void)q_out; (void)k_out; (void)v_out;

  // Stub: return false so the caller's existing 3-FC + CPU RoPE path runs.
  // Step 2d flips this once the kernel body produces correct output.
  return false;
}

// =============================================================================
// Segment A.2 — GPU RMSNorm with TensorBacking output residency.
//
// Paper §3.2 cross-layer residency. Produces a cl_mem holding FP16
// normalized activation that downstream FC layers consume directly via
// `dotCl_v8c`'s residency input mode (Segment A.1). No host materialize
// when env-gated.
//
// First wired consumer: Qwen3's `attention_norm` and `ffn_norm` (per
// transformer.cpp:340, 355). q_norm / k_norm use ReshapedRMSNormLayer
// (different class) and are out of Segment A scope.
// =============================================================================
namespace {

// Per-gamma persistent upload cache. Gamma weights don't change at
// inference; upload once per unique gamma name, reuse forever.
struct ResidentRmsState {
  std::unordered_map<std::string, cl_mem> gamma_bufs;
  std::mutex mtx;
};
static ResidentRmsState &resident_rms_state() {
  static ResidentRmsState s;
  return s;
}

static bool resident_rmsnorm_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_RESIDENT_RMSNORM") != nullptr ? 1 : 0;
  return cached != 0;
}

} // anonymous namespace

bool rmsnorm_resident_fp16(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int B, unsigned int C,
                           unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output) {
  if (!resident_rmsnorm_env_enabled())
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  if (B == 0 || C == 0 || H == 0 || W == 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t total_elems = (size_t)B * C * H * W;
  const size_t total_bytes = total_elems * sizeof(uint16_t);

  // 1) Get or create the output backing in the pool. Reused across calls
  //    with the same output_name (output is overwritten in place).
  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> out_bk = pool.get(output_name);
  if (!out_bk || out_bk->bytes() < total_bytes) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    out_bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP16, tv::Layout::ROW_MAJOR, total_bytes,
      /*owned=*/true);
    pool.set(output_name, out_bk);
  }

  // 2) Source cl_mem for the input — from upstream backing if present
  //    (zero host transfer), else upload from host on each call.
  cl_mem in_cl = nullptr;
  cl_mem in_upload_owned = nullptr; // freed at function exit if allocated
  if (const tv::TensorBacking *in_bk = input.getBacking();
      in_bk != nullptr && in_bk->encoding() == tv::Encoding::FP16 &&
      in_bk->bytes() >= total_bytes) {
    in_cl = in_bk->buffer();
  } else {
    cl_int err = CL_SUCCESS;
    in_upload_owned = clCreateBuffer(ctx, CL_MEM_READ_ONLY, total_bytes,
                                     nullptr, &err);
    if (err != CL_SUCCESS || !in_upload_owned)
      return false;
    if (clEnqueueWriteBuffer(q, in_upload_owned, CL_TRUE, 0, total_bytes,
                             input.getData<uint8_t>(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      clReleaseMemObject(in_upload_owned);
      return false;
    }
    in_cl = in_upload_owned;
  }

  // 3) Gamma upload cache — once per gamma name. Gamma doesn't change at
  //    inference. Keyed by gamma's name (stable per layer).
  cl_mem gamma_cl = nullptr;
  {
    auto &st = resident_rms_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    const std::string &gn = gamma.getName();
    auto it = st.gamma_bufs.find(gn);
    if (it == st.gamma_bufs.end()) {
      cl_int err = CL_SUCCESS;
      cl_mem gbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   (size_t)W * sizeof(uint16_t), nullptr,
                                   &err);
      if (err != CL_SUCCESS || !gbuf) {
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      if (clEnqueueWriteBuffer(q, gbuf, CL_TRUE, 0,
                               (size_t)W * sizeof(uint16_t),
                               gamma.getData<uint8_t>(), 0, nullptr,
                               nullptr) != CL_SUCCESS) {
        clReleaseMemObject(gbuf);
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      st.gamma_bufs[gn] = gbuf;
      gamma_cl = gbuf;
    } else {
      gamma_cl = it->second;
    }
  }

  // 4) Register the kernel (cached by ClContext on repeat registration).
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rmsnorm_fp16_kernel, "rmsnorm_cl_fp16");
  if (!kp) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  cl_mem out_buf = out_bk->buffer();
  cl_half eps_half;
  {
    const float ef = epsilon;
    // round-half-to-nearest float→half via the helper available in this TU.
    // The activation residual stream is FP16 so this matches the upstream
    // precision exactly.
    uint16_t bits = 0;
    // simple manual conversion for non-NaN positive epsilon (always small).
    union { float f; uint32_t u; } v;
    v.f = ef;
    uint32_t e = (v.u >> 23) & 0xFF;
    uint32_t m = v.u & 0x7FFFFF;
    if (e == 0) bits = 0;
    else if (e >= 143) bits = 0x7BFF; // clamp to fp16 max
    else if (e <= 112) bits = 0;       // underflow to 0
    else bits = ((e - 112) << 10) | (m >> 13);
    eps_half = bits;
  }

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &in_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &out_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &gamma_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &eps_half, sizeof(cl_half)) ||
      !kp->SetKernelArguments(arg++, &B, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &C, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &H, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &W, sizeof(int))) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  // work-groups per existing RMSNormLayerCl: {B*C, H, 1}, local {W, 1, 1}.
  const int wg_count[3] = {(int)(B * C), (int)H, 1};
  const int wg_size[3] = {(int)W, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wg_count, wg_size)) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  // 5) Publish the backing to the consumer side. Both setBacking() on the
  //    Tensor (if the consumer's Tensor instance is the same one we got
  //    here) AND pool.set() (already done at allocation) so a name-based
  //    lookup is also possible. Host data of `output` is left undefined.
  output.setBacking(out_bk.get());
  // Bump the global resident-quant generation: any downstream FC that
  // sees this backing pointer in its quant cache must re-quant; the
  // backing's data has just changed.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);

  // OUT-OF-ORDER QUEUE FIX: the ClContext queue uses
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (opencl_command_queue_
  // manager.cpp:56). Without explicit events or a barrier, subsequent
  // enqueues are NOT guaranteed to wait for this kernel even when they
  // read the same cl_mem. The pool's ptr-keyed entry is overwritten by
  // every RMSNorm in the chain (TensorPool reuses the same host buffer
  // for all per-layer norm outputs), and multiple FC consumers race
  // against multiple RMSNorm producers. Barrier serializes.
  clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  // Static one-shot log for first invocation so we can confirm wiring
  // when running a model. NNTR_RESIDENT_RMSNORM_TRIP=1 prints once.
  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_RESIDENT_RMSNORM_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[SegA-RMS] first invocation: out_name=%s B=%u C=%u H=%u "
                 "W=%u in_from_backing=%d\n",
                 output_name.c_str(), B, C, H, W,
                 in_upload_owned == nullptr ? 1 : 0);
    std::fflush(stderr);
  }

  if (in_upload_owned) {
    // The kernel reads input as a buffer; we can't release until the
    // kernel has finished. Block once to ensure the upload buffer is
    // safe to release. For backing-supplied inputs we don't allocate.
    clFinish(q);
    clReleaseMemObject(in_upload_owned);
  }

  return true;
}

// FP32 variant (Qwen3's actual residual-stream dtype). Same lifecycle/
// caching/backing semantics as the FP16 variant; just different kernel.
bool rmsnorm_resident_fp32(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output) {
  if (!resident_rmsnorm_env_enabled())
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP32)
    return false;
  if (H == 0 || W == 0)
    return false;
  // rmsnorm_cl kernel reads input as float4 — W must be a multiple of 4.
  if (W % 4 != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t total_elems = (size_t)H * W;
  const size_t total_bytes = total_elems * sizeof(float);

  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> out_bk = pool.get(output_name);
  if (!out_bk || out_bk->bytes() < total_bytes ||
      out_bk->encoding() != tv::Encoding::FP32) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    out_bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP32, tv::Layout::ROW_MAJOR, total_bytes,
      /*owned=*/true);
    pool.set(output_name, out_bk);
  }
  // Also register under the host-data-pointer key so consumers receiving a
  // different Tensor instance (with the same underlying data pointer) can
  // find this backing. See the dotCl_v8c residency-input lookup code.
  {
    const void *out_data_ptr = output.getData<uint8_t>();
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", out_data_ptr);
    pool.set(std::string(key_buf), out_bk);
  }

  cl_mem in_cl = nullptr;
  cl_mem in_upload_owned = nullptr;
  std::shared_ptr<tv::TensorBacking> in_bk_pool_strong;
  if (const tv::TensorBacking *in_bk = input.getBacking();
      in_bk != nullptr && in_bk->encoding() == tv::Encoding::FP32 &&
      in_bk->bytes() >= total_bytes) {
    in_cl = in_bk->buffer();
  } else {
    const void *in_data_ptr = input.getData<uint8_t>();
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", in_data_ptr);
    in_bk_pool_strong =
      tv::TensorBackingPool::Global().get(std::string(key_buf));
    if (in_bk_pool_strong &&
        in_bk_pool_strong->encoding() == tv::Encoding::FP32 &&
        in_bk_pool_strong->bytes() >= total_bytes) {
      in_cl = in_bk_pool_strong->buffer();
    }
  }
  if (in_cl == nullptr) {
    cl_int err = CL_SUCCESS;
    in_upload_owned = clCreateBuffer(ctx, CL_MEM_READ_ONLY, total_bytes,
                                     nullptr, &err);
    if (err != CL_SUCCESS || !in_upload_owned)
      return false;
    if (clEnqueueWriteBuffer(q, in_upload_owned, CL_TRUE, 0, total_bytes,
                             input.getData<uint8_t>(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      clReleaseMemObject(in_upload_owned);
      return false;
    }
    in_cl = in_upload_owned;
  }

  cl_mem gamma_cl = nullptr;
  {
    auto &st = resident_rms_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    const std::string gn = gamma.getName() + ":fp32";
    auto it = st.gamma_bufs.find(gn);
    if (it == st.gamma_bufs.end()) {
      cl_int err = CL_SUCCESS;
      cl_mem gbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   (size_t)W * sizeof(float), nullptr, &err);
      if (err != CL_SUCCESS || !gbuf) {
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      if (clEnqueueWriteBuffer(q, gbuf, CL_TRUE, 0, (size_t)W * sizeof(float),
                               gamma.getData<uint8_t>(), 0, nullptr,
                               nullptr) != CL_SUCCESS) {
        clReleaseMemObject(gbuf);
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      st.gamma_bufs[gn] = gbuf;
      gamma_cl = gbuf;
    } else {
      gamma_cl = it->second;
    }
  }

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rmsnorm_kernel, "rmsnorm_cl");
  if (!kp) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  cl_mem out_buf = out_bk->buffer();
  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &in_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &out_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &gamma_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &epsilon, sizeof(float)) ||
      !kp->SetKernelArguments(arg++, &H, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &W, sizeof(int))) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  // rmsnorm_cl uses get_group_id(0) → H groups, subgroup reduce inside.
  // DispatchCommand interprets the first array as the GLOBAL work-item
  // count (NDRange standard), so global = H * subgroup, local = subgroup.
  // Matches rmsnorm_cl_internal at blas_kernels_templates.h:428.
  // Diagnostic NNTR_SEGA_RMS_LOCAL=N overrides the local size. With
  // N=1, the kernel runs single-threaded per row (no subgroup reduce),
  // useful to isolate whether subgroup_reduce_add is the divergence
  // source from CPU NEON.
  int subgroup_size = 64; // Adreno default
  if (const char *e = std::getenv("NNTR_SEGA_RMS_LOCAL"))
    subgroup_size = std::atoi(e);
  const int wg_count[3] = {(int)H * subgroup_size, 1, 1};
  const int wg_size[3] = {subgroup_size, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wg_count, wg_size)) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  output.setBacking(out_bk.get());
  // Bump the global resident-quant generation: any downstream FC that
  // sees this backing pointer in its quant cache must re-quant; the
  // backing's data has just changed.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);

  // OUT-OF-ORDER QUEUE FIX: the ClContext queue uses
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (opencl_command_queue_
  // manager.cpp:56). Without explicit events or a barrier, subsequent
  // enqueues are NOT guaranteed to wait for this kernel even when they
  // read the same cl_mem. The pool's ptr-keyed entry is overwritten by
  // every RMSNorm in the chain (TensorPool reuses the same host buffer
  // for all per-layer norm outputs), and multiple FC consumers race
  // against multiple RMSNorm producers. Barrier serializes.
  clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_RESIDENT_RMSNORM_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[SegA-RMS-FP32] first invocation: out_name=%s H=%u W=%u "
                 "in_from_backing=%d\n",
                 output_name.c_str(), H, W,
                 in_upload_owned == nullptr ? 1 : 0);
    std::fflush(stderr);
  }

  if (in_upload_owned) {
    clFinish(q);
    clReleaseMemObject(in_upload_owned);
  }

  return true;
}

// CPU-norm + GPU-residency-handoff path. Uploads the output Tensor's
// host FP32 data into a TensorBacking under `output_name`. Bit-exact
// w.r.t. CPU computation because no GPU compute occurs here. Caller
// must have already populated output.getData<float>() via the existing
// CPU RMSNorm code.
bool publish_host_fp32_to_backing(const Tensor &output,
                                  const std::string &output_name) {
  if (output.getDataType() != ml::train::TensorDim::DataType::FP32)
    return false;
  const auto &dim = output.getDim();
  const size_t total_elems = (size_t)dim.batch() * dim.channel() *
                             dim.height() * dim.width();
  if (total_elems == 0)
    return false;
  const size_t total_bytes = total_elems * sizeof(float);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> bk = pool.get(output_name);
  if (!bk || bk->bytes() < total_bytes ||
      bk->encoding() != tv::Encoding::FP32) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP32, tv::Layout::ROW_MAJOR, total_bytes,
      /*owned=*/true);
    pool.set(output_name, bk);
  }
  // Also register under the host-data-pointer key so consumers receiving
  // a different Tensor instance (same underlying buffer) find this entry.
  {
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p",
                  static_cast<const void *>(output.getData<uint8_t>()));
    pool.set(std::string(key_buf), bk);
  }

  // Upload the CPU-computed RMSNorm output into the backing's cl_mem.
  if (clEnqueueWriteBuffer(q, bk->buffer(), CL_FALSE, 0, total_bytes,
                           output.getData<uint8_t>(), 0, nullptr,
                           nullptr) != CL_SUCCESS)
    return false;
  // Bump the generation so the v8c quant cache invalidates stale entries.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);
  // We don't need a barrier here — the FC's queued ops follow this write
  // in the same queue; OoO scheduler tracks the cl_mem write dependency
  // for the FC's read (and barrier would prevent the FC enqueue from
  // starting earlier than necessary anyway).

  return true;
}

// [resident-act] Publish a GPU-resident activation: GPU-copy the producer's
// SVM output (FP16/FP32) into a cl_mem TensorBacking keyed `resact:`+name (the
// producer's graph-output name), so a downstream CL layer that resolved this
// edge (resolveResidentEdge) consumes the cl_mem directly instead of the SVM
// buffer. No host bounce (reuses the GPU v8c_copy_svm_to_clmem). Step 1 of the
// cl_mem residency overlay. Returns false on failure (caller keeps SVM path).
// Create/reuse a cl_mem TensorBacking keyed `resact:`+name (no data written),
// bump the residency generation, and return its cl_mem. A producer can bind
// this buffer as its kernel's output to write the activation device-resident
// directly (no SVM intermediate); the downstream consumer resolves the edge and
// reads it. Returns nullptr on failure.
void *get_or_create_resident_backing(const std::string &name,
                                     unsigned int n_elems, bool fp16) {
  if (n_elems == 0)
    return nullptr;
  const size_t total_bytes = (size_t)n_elems * (fp16 ? 2u : 4u);
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return nullptr;
  cl_context ctx = blas_cc->context_inst_.GetContext();
  const tv::Encoding enc = fp16 ? tv::Encoding::FP16 : tv::Encoding::FP32;
  const std::string key = "resact:" + name;
  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> bk = pool.get(key);
  if (!bk || bk->bytes() < total_bytes || bk->encoding() != enc) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return nullptr;
    bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, enc, tv::Layout::ROW_MAJOR, total_bytes, /*owned=*/true);
    pool.set(key, bk);
  }
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);
  return static_cast<void *>(bk->buffer());
}

bool publish_resident_act(const std::string &name, const void *svm_ptr,
                          unsigned int n_elems, bool fp16) {
  if (!svm_ptr || n_elems == 0)
    return false;
  cl_mem buf =
    static_cast<cl_mem>(get_or_create_resident_backing(name, n_elems, fp16));
  if (!buf)
    return false;
  // GPU copy the SVM activation into the backing cl_mem (no host round-trip).
  v8c_copy_svm_to_clmem(svm_ptr, buf, n_elems, fp16);
  return true;
}

// =============================================================================
// Fused RMSNorm + v8c activation quant (paper §3.6 fused-kernel idea).
// =============================================================================
namespace {
struct FusedRmsqScratch {
  cl_mem gamma_cl = nullptr; // cached fp32 gamma (per gamma name)
  std::string gamma_name;
  unsigned int gamma_W = 0;
};
static FusedRmsqScratch &fused_rmsq_state() {
  static FusedRmsqScratch s;
  return s;
}

// CPU reference: RMSNorm + v8c-compatible asymmetric quant for one row.
// Used by NNTR_FUSED_RMSQ_CHECK=1 to validate the GPU kernel produces
// byte-identical outputs.
static void fused_rmsq_cpu_ref_row(const float *in_row, const float *gamma,
                                   float epsilon, unsigned int K,
                                   std::vector<int8_t> &out_i8,
                                   float &out_scale, int &out_zp,
                                   int &out_rs) {
  out_i8.assign(K, 0);
  // RMSNorm.
  double sumsq = 0.0;
  for (unsigned int k = 0; k < K; k++) sumsq += (double)in_row[k] * in_row[k];
  const float mean_sq = (float)(sumsq / K);
  const float inv_rms = 1.0f / std::sqrt(mean_sq + epsilon);
  // Min/max of normalized.
  std::vector<float> norm(K);
  float fmin = 0.0f, fmax = 0.0f;
  for (unsigned int k = 0; k < K; k++) {
    const float v = in_row[k] * inv_rms * gamma[k];
    norm[k] = v;
    if (v < fmin) fmin = v;
    if (v > fmax) fmax = v;
  }
  const float rmin = fmin < 0.0f ? fmin : 0.0f;
  const float rmax = fmax > 0.0f ? fmax : 0.0f;
  const float qmin = -128.0f, qmax = 127.0f;
  const float range = rmax - rmin;
  const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
  const float recip = range > 0.0f ? range / 255.0f : 1.0f;
  const float dmin = rmin * scale_q, dmax = rmax * scale_q;
  const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
  float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
  if (zp_f < qmin) zp_f = qmin;
  if (zp_f > qmax) zp_f = qmax;
  const int zp = (int)std::lrint(zp_f);
  out_scale = recip;
  out_zp = zp;
  int rs = 0;
  for (unsigned int k = 0; k < K; k++) {
    int q = (int)std::lrint(norm[k] * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    out_i8[k] = (int8_t)q;
    rs += q;
  }
  out_rs = rs;
}
} // anonymous namespace

bool fused_rmsnorm_quant_resident_fp32(const Tensor &input,
                                       const Tensor &gamma, float epsilon,
                                       unsigned int M, unsigned int K,
                                       const std::string &output_name,
                                       const void *output_host_ptr) {
  static const bool env_on = std::getenv("NNTR_FUSED_RMSQ") != nullptr;
  if (!env_on)
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP32)
    return false;
  if (M == 0 || K == 0 || K > 2048u)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t in_bytes = (size_t)M * K * sizeof(float);
  const size_t i8_bytes = (size_t)M * K * sizeof(int8_t);
  const size_t per_row4 = (size_t)M * sizeof(int32_t); // also fp32 alias

  auto &pool = tv::TensorBackingPool::Global();
  auto ensure_bk = [&](const std::string &name, size_t bytes,
                       tv::Encoding enc) -> std::shared_ptr<tv::TensorBacking> {
    auto bk = pool.get(name);
    if (!bk || bk->bytes() < bytes || bk->encoding() != enc) {
      cl_int err = CL_SUCCESS;
      cl_mem buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
      if (err != CL_SUCCESS || !buf) return nullptr;
      bk = std::make_shared<tv::TensorBacking>(
        ctx, buf, enc, tv::Layout::ROW_MAJOR, bytes, /*owned=*/true);
      pool.set(name, bk);
    }
    return bk;
  };
  auto bk_i8 = ensure_bk(output_name + ":fused_i8", i8_bytes, tv::Encoding::INT8);
  auto bk_sc = ensure_bk(output_name + ":fused_scale", per_row4, tv::Encoding::FP32);
  auto bk_zp = ensure_bk(output_name + ":fused_zp", per_row4, tv::Encoding::FP32);
  auto bk_rs = ensure_bk(output_name + ":fused_rs", per_row4, tv::Encoding::FP32);
  if (!bk_i8 || !bk_sc || !bk_zp || !bk_rs)
    return false;
  // Also register under ptr-keyed names so a downstream consumer (whose
  // Tensor instance shares the same data pointer via TensorPool reuse)
  // can find these entries without knowing the producer's tensor name.
  // See dotCl_v8c's existing ptr-based backing lookup.
  if (output_host_ptr != nullptr) {
    char k[80];
    std::snprintf(k, sizeof(k), "ptr:%p:fused_i8", output_host_ptr);
    pool.set(k, bk_i8);
    std::snprintf(k, sizeof(k), "ptr:%p:fused_scale", output_host_ptr);
    pool.set(k, bk_sc);
    std::snprintf(k, sizeof(k), "ptr:%p:fused_zp", output_host_ptr);
    pool.set(k, bk_zp);
    std::snprintf(k, sizeof(k), "ptr:%p:fused_rs", output_host_ptr);
    pool.set(k, bk_rs);
  }

  // Resolve input: prefer a TensorBacking from the input tensor (or the
  // pool keyed by host data ptr). Falls back to a fresh host upload.
  cl_mem in_cl = nullptr;
  cl_mem in_upload_owned = nullptr;
  std::shared_ptr<tv::TensorBacking> in_bk_pool_strong;
  if (const tv::TensorBacking *in_bk = input.getBacking();
      in_bk != nullptr && in_bk->encoding() == tv::Encoding::FP32 &&
      in_bk->bytes() >= in_bytes) {
    in_cl = in_bk->buffer();
  } else {
    const void *in_data_ptr = input.getData<uint8_t>();
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", in_data_ptr);
    in_bk_pool_strong = pool.get(std::string(key_buf));
    if (in_bk_pool_strong &&
        in_bk_pool_strong->encoding() == tv::Encoding::FP32 &&
        in_bk_pool_strong->bytes() >= in_bytes) {
      in_cl = in_bk_pool_strong->buffer();
    }
  }
  if (in_cl == nullptr) {
    cl_int err = CL_SUCCESS;
    in_upload_owned =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !in_upload_owned) return false;
    if (clEnqueueWriteBuffer(q, in_upload_owned, CL_TRUE, 0, in_bytes,
                             input.getData<uint8_t>(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      clReleaseMemObject(in_upload_owned);
      return false;
    }
    in_cl = in_upload_owned;
  }
  // First-trip diagnostic so we know whether the resident path fired.
  if (std::getenv("NNTR_FUSED_RMSQ_TRIP") != nullptr) {
    static int trip = 0;
    if (!trip) {
      trip = 1;
      std::fprintf(stderr,
                   "[FUSED-RMSQ] first call: name=%s M=%u K=%u  "
                   "input from %s\n",
                   output_name.c_str(), M, K,
                   in_upload_owned == nullptr ? "BACKING (resident)"
                                              : "HOST upload");
      std::fflush(stderr);
    }
  }

  // Gamma cache (1 buffer per gamma name).
  cl_mem gamma_cl = nullptr;
  {
    auto &st = fused_rmsq_state();
    const std::string &gn = gamma.getName();
    if (st.gamma_cl == nullptr || st.gamma_name != gn || st.gamma_W != K) {
      if (st.gamma_cl) clReleaseMemObject(st.gamma_cl);
      cl_int gerr = CL_SUCCESS;
      st.gamma_cl = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   (size_t)K * sizeof(float), nullptr, &gerr);
      if (gerr != CL_SUCCESS || !st.gamma_cl) {
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        st.gamma_cl = nullptr;
        return false;
      }
      if (clEnqueueWriteBuffer(q, st.gamma_cl, CL_TRUE, 0,
                               (size_t)K * sizeof(float),
                               gamma.getData<uint8_t>(), 0, nullptr,
                               nullptr) != CL_SUCCESS) {
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        clReleaseMemObject(st.gamma_cl);
        st.gamma_cl = nullptr;
        return false;
      }
      st.gamma_name = gn;
      st.gamma_W = K;
    }
    gamma_cl = st.gamma_cl;
  }

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(fused_rmsnorm_quant_kernel,
                              "fused_rmsnorm_quant_f32_par");
  if (!kp) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  cl_mem i8_buf = bk_i8->buffer();
  cl_mem sc_buf = bk_sc->buffer();
  cl_mem zp_buf = bk_zp->buffer();
  cl_mem rs_buf = bk_rs->buffer();
  int arg = 0;
  const int Mi = (int)M;
  const int Ki = (int)K;
  if (!kp->SetKernelArguments(arg++, &in_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &gamma_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &i8_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &sc_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &zp_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &rs_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &epsilon, sizeof(float)) ||
      !kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ki, sizeof(int))) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  constexpr int LWS = 64;
  const int wg_count[3] = {(int)M * LWS, 1, 1};
  const int wg_size[3] = {LWS, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wg_count, wg_size)) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }
  clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  // Self-check: full row sweep on first call, single-row summary on
  // subsequent calls (one per layer). Reads back ALL M*K bytes + the
  // per-row metadata and compares each row to a CPU reference.
  if (std::getenv("NNTR_FUSED_RMSQ_CHECK") != nullptr) {
    static int call = -1;
    call++;
    clFinish(q);
    std::vector<int8_t> gpu_i8((size_t)M * K);
    std::vector<float> gpu_scale(M);
    std::vector<int32_t> gpu_zp(M);
    std::vector<int32_t> gpu_rs(M);
    clEnqueueReadBuffer(q, i8_buf, CL_TRUE, 0, (size_t)M * K, gpu_i8.data(),
                        0, nullptr, nullptr);
    clEnqueueReadBuffer(q, sc_buf, CL_TRUE, 0, (size_t)M * 4,
                        gpu_scale.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(q, zp_buf, CL_TRUE, 0, (size_t)M * 4,
                        gpu_zp.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(q, rs_buf, CL_TRUE, 0, (size_t)M * 4,
                        gpu_rs.data(), 0, nullptr, nullptr);
    const float *in_data =
      reinterpret_cast<const float *>(input.getData<uint8_t>());
    const float *gm_data =
      reinterpret_cast<const float *>(gamma.getData<uint8_t>());
    unsigned int total_mismatch = 0, rows_with_mismatch = 0;
    int worst_diff = 0, worst_row = -1;
    std::vector<int8_t> cpu_i8;
    for (unsigned int m = 0; m < M; m++) {
      float cs = 0.0f;
      int cz = 0, crs = 0;
      fused_rmsq_cpu_ref_row(in_data + (size_t)m * K, gm_data, epsilon, K,
                             cpu_i8, cs, cz, crs);
      unsigned int row_mismatch = 0;
      int row_worst = 0;
      for (unsigned int k = 0; k < K; k++) {
        const int diff = std::abs((int)gpu_i8[(size_t)m * K + k] - (int)cpu_i8[k]);
        if (diff > 0) {
          row_mismatch++;
          if (diff > row_worst) row_worst = diff;
        }
      }
      // Allow 1 ulp difference in fp32 scale (results from non-bit-
      // equivalent reduction order in the rmsnorm sum-of-squares pass).
      // Exact equality for int zp/rs.
      const float scale_diff = std::fabs(gpu_scale[m] - cs);
      const float scale_eps = std::max(std::fabs(cs), std::fabs(gpu_scale[m]))
                              * 1e-6f;
      const bool meta_match =
        (scale_diff <= scale_eps) && (gpu_zp[m] == cz) && (gpu_rs[m] == crs);
      if (row_mismatch > 0 || !meta_match) {
        rows_with_mismatch++;
        total_mismatch += row_mismatch;
        if (row_worst > worst_diff) {
          worst_diff = row_worst;
          worst_row = (int)m;
        }
        if (call == 0 && rows_with_mismatch <= 3) {
          std::fprintf(stderr,
                       "  row %u mismatch: i8_diff_count=%u worst=%d  "
                       "scale cpu=%.9g gpu=%.9g (ulp=%.2g)  "
                       "zp cpu=%d gpu=%d  rs cpu=%d gpu=%d\n",
                       m, row_mismatch, row_worst, cs, gpu_scale[m],
                       (double)scale_diff,
                       cz, gpu_zp[m], crs, gpu_rs[m]);
        }
      }
    }
    std::fprintf(stderr,
                 "[FUSED-RMSQ-CHECK call=%d] %s M=%u K=%u : "
                 "rows_bad=%u/%u  total_i8_mismatch=%u  worst_diff=%d (row=%d)\n",
                 call, output_name.c_str(), M, K, rows_with_mismatch, M,
                 total_mismatch, worst_diff, worst_row);
    std::fflush(stderr);
  }

  if (in_upload_owned) {
    clFinish(q); // safe to release input upload
    clReleaseMemObject(in_upload_owned);
  }
  return true;
}

bool readback_backing_to_host(Tensor &t) {
  const tv::TensorBacking *bk = t.getBacking();
  if (bk == nullptr || bk->buffer() == nullptr)
    return false;
  const auto &dim = t.getDim();
  const size_t elems = (size_t)dim.batch() * dim.channel() * dim.height() *
                       dim.width();
  if (elems == 0)
    return false;
  const size_t elem_bytes =
    (t.getDataType() == ml::train::TensorDim::DataType::FP16) ? 2u :
    (t.getDataType() == ml::train::TensorDim::DataType::FP32) ? 4u : 0u;
  if (elem_bytes == 0)
    return false;
  const size_t bytes = elems * elem_bytes;
  if (bk->bytes() < bytes)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  clFinish(q);
  if (clEnqueueReadBuffer(q, bk->buffer(), CL_TRUE, 0, bytes,
                          t.getData<uint8_t>(), 0, nullptr,
                          nullptr) != CL_SUCCESS)
    return false;
  return true;
}

} // namespace nntrainer
