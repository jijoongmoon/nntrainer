// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_compute_ops.cpp
 * @date   22 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  CUDA ComputeOps subclass (mirror of ClComputeOps). P1 provides only
 *         the host-side copy ops so Tensor::copy() works on engine=cuda tensors
 *         (their memory is Unified/managed, hence host-addressable). The
 *         accelerator quantized GEMM/GEMV predicates are left at the base
 *         default (false), so float_tensor.cpp falls back to the CPU path until
 *         the CUDA kernels land in P3 (cuda_operations/).
 */

#include <compute_ops.h>
#include <cpu_ops_table.h>

#include <cstdlib>

#include <tensor.h>

#include <cuda_stream_manager.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_blas_manager.h>
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_fc_qint4.h>
#include <cuda_runtime.h>
#include <fp16.h>
#include <int4_utils.h>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#endif

namespace nntrainer {

// CudaComputeOps derives from CpuComputeOps (not the abstract ComputeOps base):
// engine=cuda tensors are Unified Memory (host-coherent), so every standard op
// runs correctly via the CPU implementations; this class only overrides the
// host-side copy ops for now. Inheriting CpuComputeOps means get_cuda_ops() can
// be installed without throwing on the un-accelerated ops (prereq for the CUDA
// op kernels in a later phase). [T6]
class CudaComputeOps : public CpuComputeOps {
public:
  // Plain elementwise copy (Y = X). Tensor::copy() calls this unconditionally
  // (no supports_*() guard); correct for host and (host-coherent) managed
  // pointers. A device-kernel copy is a later residency refinement.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }

#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<_FP16>(X[i * incX]);
  }
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<float>(X[i * incX]);
  }
#endif

  // ── Whole-op (Tensor-level) ───────────────────────────────────────────────
  // GeGLU: out = gelu_tanh(gate) * up. Device-resident fp16 kernel (opt-in via
  // NNTR_CUDA_GEGLU until the whole decode chain is on-GPU); otherwise the host
  // gelu loop on the host-coherent UVM tensors (CpuComputeOps::geglu). Matches
  // the former CudaGeGLULayer::gegluProcess byte-for-byte. [T7]
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override {
    const unsigned int dim2 = in1.width();
    const size_t elem_off = (size_t)row_offset * dim2;
    const size_t n = (size_t)active_rows * dim2;
    const auto dt = in1.getDataType();

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // GPU geglu (device-resident fp16): one kernel instead of the host loop, so
    // the FFN/PLE activation stays on the device. NNTR_CUDA_ASYNC governs the
    // drain.
    if (dt == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = std::getenv("NNTR_CUDA_GEGLU") != nullptr;
      if (gpu && n > 0) {
        auto *a =
          reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                   elem_off);
        auto *b =
          reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                   elem_off);
        auto *o = reinterpret_cast<unsigned short *>(out.getData<_FP16>() +
                                                     elem_off);
        const bool dev = nntrainer::cuda::dev_accessible(a);
        if (dev && cuda::cuda_geglu_fp16(a, b, o, (unsigned int)n))
          return;
      }
    }
#endif

    // Host gelu fallback: sync first so the host read of GPU-produced gate/up is
    // coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::geglu(in1, in2, out, active_rows, row_offset);
  }

  // FC GEMM: output = input * weight. The former CudaFcLayer::cudaFcGemm body
  // verbatim — QINT4 fused dequant-GEMM on device (KAI Section-A, w4a8 dp4a /
  // cuBLAS int8 IMMA) with host-weight/host-input staging, an FP32 cuBLAS path,
  // and a host Tensor::dot fallback (correct on the host-coherent UVM). [T7]
  void fc(Tensor &input, Tensor &weight, Tensor &output) override {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    using DT = ml::train::TensorDim::DataType;
    Tensor &input_ = input;
    Tensor &hidden_ = output;
    const DT wt = weight.getDataType();
    const DT at = input_.getDataType();

    const auto &id = input_.getDim();
    const auto &od = hidden_.getDim();
    const int K = (int)id.width();
    const int N = (int)od.width();
    const int M = (int)(id.batch() * id.channel() * id.height());

    static const bool fc_dbg = std::getenv("NNTR_FC_DEBUG") != nullptr;
    if (fc_dbg) {
      auto ptype = [](const void *p) {
        cudaPointerAttributes a{};
        bool ok = cudaPointerGetAttributes(&a, p) == cudaSuccess;
        cudaGetLastError();
        if (!ok)
          return 'u';
        switch (a.type) {
        case cudaMemoryTypeManaged: return 'm';
        case cudaMemoryTypeDevice: return 'd';
        case cudaMemoryTypeHost: return 'h';
        default: return '0';
        }
      };
      fprintf(stderr,
              "[FCDBG] wt=%d at=%d ot=%d M=%d N=%d K=%d in=%c w=%c out=%c\n",
              (int)wt, (int)at, (int)hidden_.getDataType(), M, N, K,
              ptype(input_.getData<float>()), ptype(weight.getData<uint8_t>()),
              ptype(hidden_.getData<float>()));
    }

    // QINT4 weight: fused dequant-GEMM on device (KAI Section-A layout). Default
    // ON; the host Tensor::dot path is NYI for QINT4 on x86 (KAI ARM-only).
    if ((wt == DT::QINT4 || wt == DT::QS4CX) &&
        (at == DT::FP32 || at == DT::FP16) && M > 0 && N > 0 && K > 0) {
      static const bool seca_enabled = []() {
        const char *e = std::getenv("NNTR_FC_CUDA_QINT4");
        return !(e != nullptr && e[0] == '0');
      }();
      if (seca_enabled && (int)weight.getDim().height() == K) {
        const uint8_t *W = weight.getData<uint8_t>();
        const uint16_t *S = weight.getScale<uint16_t>();
        // [weight 한벌] QS4CX plain weight -> reuse the QINT4 Section-A CUDA
        // path: resolve (once, cached) to a UVM Section-A + fp16 buffer so it is
        // device-resident exactly like a native QINT4 weight (prewarmed, no
        // per-call staging, same dp4a/cuBLAS fast path + repack cache).
        if (wt == DT::QS4CX) {
          const unsigned char *uW = nullptr;
          const unsigned short *uS = nullptr;
          if (cuda::cuda_fc_qs4cx_to_uvm_seca(W, weight.getScale<float>(),
                                              (unsigned)N, (unsigned)K, &uW,
                                              &uS)) {
            W = uW;
            S = uS;
          }
        }
        if (!nntrainer::cuda::dev_accessible(W)) {
          const uint8_t *dW = nullptr;
          const uint16_t *dS = nullptr;
          if (cuda::cuda_fc_qint4_stage_host_weight(W, S, (unsigned)N,
                                                    (unsigned)K, &dW, &dS)) {
            W = dW;
            S = dS;
          }
        }
        const bool fp16 = (at == DT::FP16);
        const void *Xp = fp16 ? (const void *)input_.getData<uint16_t>()
                              : (const void *)input_.getData<float>();
        void *Yp = fp16 ? (void *)hidden_.getData<uint16_t>()
                        : (void *)hidden_.getData<float>();
        static const bool use_dp4a = []() {
          const char *e = std::getenv("NNTR_FC_CUDA_DP4A");
          return !(e != nullptr && e[0] == '0');
        }();
        static const bool use_cublas_i8 = []() {
          const char *e = std::getenv("NNTR_FC_CUDA_CUBLAS");
          return e != nullptr && e[0] == '1';
        }();
        const bool x_dev = nntrainer::cuda::dev_accessible(Xp);
        const bool wy_dev = nntrainer::cuda::dev_accessible(W) &&
                            nntrainer::cuda::dev_accessible(Yp);
        bool all_dev = x_dev && wy_dev;
        if (!x_dev && wy_dev && fp16) {
          if (const uint16_t *Xd = cuda::cuda_fc_qint4_stage_host_x_fp16(
                (const uint16_t *)Xp, (unsigned)M, (unsigned)K)) {
            Xp = (const void *)Xd;
            all_dev = true;
          }
        }
        if (std::getenv("NNTR_FC_HOSTDBG") && !x_dev) {
          std::fprintf(stderr,
                       "[FC-HOSTDBG] host-input M=%u K=%u N=%u wy_dev=%d "
                       "fp16=%d staged=%d capturing=%d\n",
                       (unsigned)M, (unsigned)K, (unsigned)N, (int)wy_dev,
                       (int)fp16, (int)all_dev,
                       (int)cuda::StreamManager::Global().isCapturing());
        }
        bool ok = false;
        if (all_dev && fp16) {
          static const unsigned cublas_kmax = []() {
            const char *e = std::getenv("NNTR_FC_CUBLAS_KMAX");
            return e ? (unsigned)atoi(e) : (1u << 20);
          }();
          if (use_cublas_i8 && use_dp4a && M >= 32 && K <= (int)cublas_kmax)
            ok = cuda::cuda_fc_qint4_sectionA_cublas_i8_gemm_fp16(
              (const uint16_t *)Xp, W, S, (uint16_t *)Yp, (unsigned)M,
              (unsigned)N, (unsigned)K);
          if (!ok)
            ok = use_dp4a ? cuda::cuda_fc_qint4_sectionA_dp4a_gemm_fp16(
                              (const uint16_t *)Xp, W, S, (uint16_t *)Yp,
                              (unsigned)M, (unsigned)N, (unsigned)K)
                          : cuda::cuda_fc_qint4_sectionA_gemm_fp16_naive(
                              (const uint16_t *)Xp, W, S, (uint16_t *)Yp,
                              (unsigned)M, (unsigned)N, (unsigned)K);
        } else if (all_dev) {
          ok = use_dp4a ? cuda::cuda_fc_qint4_sectionA_dp4a_gemm_fp32(
                            (const float *)Xp, W, S, (float *)Yp, (unsigned)M,
                            (unsigned)N, (unsigned)K)
                        : cuda::cuda_fc_qint4_sectionA_gemm_fp32(
                            (const float *)Xp, W, S, (float *)Yp, (unsigned)M,
                            (unsigned)N, (unsigned)K);
        } else if (!fp16) {
          ok = cuda::cuda_fc_qint4_sectionA_gemm_fp32_resident(
            (const float *)Xp, W, S, (float *)Yp, (unsigned)M, (unsigned)N,
            (unsigned)K);
        }
        if (std::getenv("NNTR_FC_HOSTDBG") && !ok) {
          cudaPointerAttributes aw{}, ay{};
          cudaError_t ew = cudaPointerGetAttributes(&aw, W);
          cudaError_t ey = cudaPointerGetAttributes(&ay, Yp);
          cudaGetLastError();
          std::fprintf(
            stderr,
            "[FC-GPUFAIL] ok=0 -> HOST i8mm: M=%u K=%u N=%u x_dev=%d wy_dev=%d "
            "| W: err=%d type=%d  Y: err=%d type=%d  cap=%d\n",
            (unsigned)M, (unsigned)K, (unsigned)N, (int)x_dev, (int)wy_dev,
            (int)ew, (int)aw.type, (int)ey, (int)ay.type,
            (int)cuda::StreamManager::Global().isCapturing());
        }
        if (ok)
          return;
      }
    }

    // FP32 weight: cuBLAS SGEMM on the UVM pointers.
    if (wt == DT::FP32 && at == DT::FP32 && M > 0 && N > 0 && K > 0 &&
        nntrainer::cuda::dev_accessible(input_.getData<float>()) &&
        nntrainer::cuda::dev_accessible(weight.getData<float>()) &&
        nntrainer::cuda::dev_accessible(hidden_.getData<float>()) &&
        cuda::BlasManager::Global().sgemmRowMajor(
          M, N, K, input_.getData<float>(), weight.getData<float>(),
          hidden_.getData<float>())) {
      cuda::StreamManager::Global().maybeFinish();
      return;
    }
#endif

    // Host fallback: correct for FP16 / Q4_x / Q6_K / cross-engine host input
    // (and any GPU-path failure) on the host-coherent UVM tensors.
    CpuComputeOps::fc(input, weight, output);
  }
};

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer
