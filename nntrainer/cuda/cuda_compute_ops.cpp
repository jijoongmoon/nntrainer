// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_compute_ops.cpp
 * @date    29 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA ComputeOps subclass for the cuda context. Inherits
 *          CpuComputeOps: cuda tensors default to Unified Memory
 *          (host-coherent), so every op the CUDA backend does not accelerate
 *          runs correctly via the CPU implementation over the managed buffers.
 *          Overrides the element-wise decode dispatches (behind the same
 *          runtime gates the neutral layers used to open-code), the rms_norm
 *          whole-op, the FC GEMM dispatch, and the copy ops (the latter so a
 *          Tensor::copy is correct on the device-only activation pool).
 */

#include "cuda_compute_ops.h"

#include <cmath>
#include <stdexcept>

#include <acti_func.h> // ActivationType (apply_activation's ACT_NONE no-op)
#include <compute_ops.h>
#include <env_compat.h>
#include <tensor.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_fc_qint4.h>
#include <cuda_rmsnorm.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>

namespace nntrainer {

namespace {

using nntrainer::cuda::host_unreachable;

/**
 * @brief The ONE stream-drain spelling in this file.
 *
 * Every entry point here is a CudaComputeOps virtual, and this table is only
 * ever reached through a tensor whose ContextData carries it -- installed by
 * CudaContext::initialize() after cudaInit() succeeded. So the CUDA context
 * provably exists at every call: StreamManager::Global() cannot create one
 * that was not going to exist anyway, and the engine_selected() short-circuit
 * that cuda::drain_if_async() carries (it protects the SHARED layers, which
 * DO run on non-cuda runs of the unified binary -- see cuda_context_manager.h,
 * measured -55% XMX prefill) is dead weight here. Worse, it would make the
 * drain env-conditional in exactly the way host_unreachable() exists to stop.
 *
 * @param full true for the terminal drain (finish), false for the
 *             async-mode-only drain (finishIfAsync, a no-op in sync mode)
 */
void table_drain(bool full) {
  auto &sm = nntrainer::cuda::StreamManager::Global();
  if (full)
    sm.finish();
  else
    sm.finishIfAsync();
}

/**
 * @brief Will Tensor::copy(from) route through the scopy_* ops?
 *
 * An exact transcription of the branch condition in Tensor::copy
 * (tensor/tensor.cpp). When it holds, the copy lands in itensor_->copy ->
 * scopy_fp16 / scopy_fp32, which this table overrides and which stages a
 * device-only endpoint through cudaMemcpyAsync. When it does NOT hold,
 * Tensor::copy instead builds a fresh Tensor from from.getData<char>() -- a
 * plain HOST read -- and swaps that host allocation in as the destination's
 * backing store: no residency awareness anywhere on that branch.
 *
 * Deliberately an exact copy of the predicate rather than a stricter proxy
 * (e.g. getDim() equality): a stricter test would refuse copies Tensor::copy
 * really does handle on device, which turns a working run into a failing one.
 * It is coupled to Tensor::copy by construction -- if that branch condition
 * changes, this must change with it.
 */
bool copy_takes_scopy(const Tensor &to, const Tensor &from) {
  return from.size() != 0 && to.size() == from.size() &&
         to.scale_size() == from.scale_size() &&
         to.getDataType() == from.getDataType();
}

/**
 * @brief The one place this table admits a host body, and the one place it
 *        refuses.
 *
 * INVARIANT: no host math in this op table may dereference an operand that
 * lives in the device-only activation pool (cudaMalloc, armed by default on a
 * discrete GPU via NNTR_CUDA_DEV_ACT). Every override's fall-through to the
 * inherited CpuComputeOps body calls this first, so a missing/declined device
 * path surfaces as a named error naming the op AND the operand, instead of a
 * SIGSEGV several frames deep inside an AVX2 intrinsic.
 *
 * It also carries the coherence half: a host body that IS allowed to run must
 * see the last device write, hence the drain (a no-op in the default sync
 * mode).
 *
 * The residency probe is cuda::host_unreachable(), NOT cuda::dev_only(): the
 * latter short-circuits to false unless NNTR_ENGINE=="cuda", which is the
 * right gate for the shared layers (they must not boot cudart on a non-cuda
 * run) but the wrong one here -- this table only exists because a CudaContext
 * was constructed, and the pool that context armed is device-only regardless
 * of what the env says.
 *
 * This never turns a working run into a failing one: an operand that trips it
 * is memory the CPU cannot address, so the host body it guards would have
 * faulted on the very next instruction.
 *
 * @param op       op name for the message (no backend prefix; added here)
 * @param operands operands the host body will dereference; nullptr entries and
 *                 unallocated tensors are skipped
 */
void host_math_gate(const char *op,
                    std::initializer_list<const Tensor *> operands) {
  table_drain(false);
  const Tensor *blocked = nullptr;
  unsigned int blocked_idx = 0, idx = 0;
  for (const Tensor *t : operands) {
    const unsigned int here = idx++;
    if (t == nullptr)
      continue;
    if (host_unreachable(t->getData<char>())) {
      blocked = t;
      blocked_idx = here;
      break;
    }
  }
  if (blocked == nullptr)
    return;

  // A window view (getSharedDataTensor) carries no name of its own, so fall
  // back to the operand position -- an unnamed operand must still be located.
  std::string who = blocked->getName();
  if (who.empty()) {
    std::ostringstream w;
    w << "#" << blocked_idx;
    who = w.str();
  }
  std::ostringstream ss;
  ss << "CudaComputeOps::" << op
     << ": the host fallback cannot run on the device-only activation pool -- "
        "operand '"
     << who
     << "' is device memory the CPU cannot address. The device path for this "
        "op declined the call (unsupported dtype or shape, or its kernel gate "
        "is off). Re-run with NNTR_CUDA_DEV_ACT=0 for a host-coherent pool, "
        "or give this op a device implementation.";
  throw std::runtime_error(ss.str());
}

} // namespace

void CudaComputeOps::swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                            unsigned int active_rows, unsigned int row_offset) {
#ifdef ENABLE_FP16
  // engine=cuda device-resident fp16: one kernel instead of the host loop
  // (the host body below would fault on the device-only activation pool
  // under NNTR_CUDA_DEV_ACT). Gated on FP16 + batch/channel==1 +
  // row_offset==0 -- the batch/channel==1 gate mirrors the layer-side gate
  // this override replaces (with it, active_rows * width() equals the
  // (to - from) * width() element count the layer's former open-coded block
  // launched); falls through to the host body for non-device tensors.
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in1.batch() == 1 && in1.channel() == 1 && row_offset == 0) {
    const size_t n = (size_t)active_rows * in1.width();
    auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>());
    auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>());
    auto *o = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    const bool dev = a && nntrainer::cuda::dev_accessible(a);
    if (dev && n > 0 &&
        nntrainer::cuda::cuda_swiglu_fp16(a, b, o, (unsigned int)n))
      return;
  }
#endif
  host_math_gate("swiglu", {&in1, &in2, &out});
  CpuComputeOps::swiglu(in1, in2, out, active_rows, row_offset);
}

// GeGLU: out = gelu_tanh(gate) * up. Device-resident fp16 kernel (opt-in via
// NNTR_CUDA_GEGLU until the whole decode chain is on-GPU); otherwise the host
// gelu loop on the host-coherent UVM tensors (CpuComputeOps::geglu).
void CudaComputeOps::geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

#ifdef ENABLE_FP16
  // GPU geglu (device-resident fp16): one kernel instead of the host loop, so
  // the FFN/PLE activation stays on the device. NNTR_CUDA_ASYNC governs the
  // drain.
  if (dt == ml::train::TensorDim::DataType::FP16) {
    static const bool gpu = nntr_env_on("NNTR_CUDA_GEGLU");
    if (gpu && n > 0) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                         elem_off);
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                         elem_off);
      auto *o =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      const bool dev = nntrainer::cuda::dev_accessible(a);
      if (dev && nntrainer::cuda::cuda_geglu_fp16(a, b, o, (unsigned int)n))
        return;
    }
  }
#endif

  // Host gelu fallback: the gate syncs first so the host read of GPU-produced
  // gate/up is coherent under NNTR_CUDA_ASYNC (no-op in sync mode), and
  // refuses by name if an operand is device-only.
  host_math_gate("geglu", {&in1, &in2, &out});
  CpuComputeOps::geglu(in1, in2, out, active_rows, row_offset);
}

// Fused sigmoid gates on cuda (mirror of geglu above). A device-resident
// activation pool makes the DEVICE kernel the primary path (the base
// CpuComputeOps host loop faults on a device-only activation in runDecode).
// Host loop only for genuinely host tensors.
// Kill-switch: NNTR_CUDA_SIGMOID_GATE=0.
void CudaComputeOps::sigmoid_glu(const Tensor &in1, const Tensor &in2,
                                 Tensor &out, unsigned int active_rows,
                                 unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
#ifdef ENABLE_FP16
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 && n > 0) {
    static const bool gpu = []() {
      const char *e = std::getenv("NNTR_CUDA_SIGMOID_GATE");
      return !(e && e[0] == '0');
    }();
    if (gpu) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                         elem_off);
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                         elem_off);
      auto *o =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(a) &&
          nntrainer::cuda::cuda_sigmoid_glu_fp16(a, b, o, (unsigned int)n))
        return;
    }
  }
#endif
  host_math_gate("sigmoid_glu", {&in1, &in2, &out});
  CpuComputeOps::sigmoid_glu(in1, in2, out, active_rows, row_offset);
}

void CudaComputeOps::sigmoid_add(const Tensor &in1, const Tensor &in2,
                                 Tensor &out, unsigned int active_rows,
                                 unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
#ifdef ENABLE_FP16
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 && n > 0) {
    static const bool gpu = []() {
      const char *e = std::getenv("NNTR_CUDA_SIGMOID_GATE");
      return !(e && e[0] == '0');
    }();
    if (gpu) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                         elem_off);
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                         elem_off);
      auto *o =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(a) &&
          nntrainer::cuda::cuda_sigmoid_add_fp16(a, b, o, (unsigned int)n))
        return;
    }
  }
#endif
  host_math_gate("sigmoid_add", {&in1, &in2, &out});
  CpuComputeOps::sigmoid_add(in1, in2, out, active_rows, row_offset);
}

void CudaComputeOps::scalar_mul(const Tensor &in, Tensor &out, float scale) {
#ifdef ENABLE_FP16
  if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    static const bool gpu = nntr_env_on("NNTR_CUDA_ELTWISE");
    if (gpu) {
      auto *ip = reinterpret_cast<const unsigned short *>(in.getData<_FP16>());
      auto *op = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
      const bool dev = nntrainer::cuda::dev_accessible(ip);
      if (dev && nntrainer::cuda::cuda_scalar_mul_fp16(
                   ip, op, (unsigned int)in.size(), scale))
        return;
    }
  }
#endif
  // Host multiply reads the GPU-produced UVM input on the CPU; the gate syncs
  // first in async mode (no-op in default sync mode) and refuses by name on a
  // device-only operand.
  host_math_gate("scalar_mul", {&in, &out});
  CpuComputeOps::scalar_mul(in, out, scale);
}

void CudaComputeOps::softcap(const Tensor &in, Tensor &out, float cap,
                             int act_type) {
  // Terminal drain for the selective-sync (NNTR_CUDA_ASYNC) path: the softcap
  // input is the first host-read point of the lm_head logits, so the
  // one-per-token GPU pipeline drains here. Per call (the layer chunks are
  // per batch/channel); the drain is idempotent and a no-op in default mode
  // (every GPU op already drained). One spelling for the whole file
  // (table_drain): the former engine_selected() guard here was the file's last
  // env-conditional drain, and it cannot protect anything -- reaching this
  // virtual already proves the CUDA context exists.
  table_drain(true);
#ifdef ENABLE_FP16
  // Device-only activation pool: the logits are real device memory; the host
  // Tensor ops in the fallback would fault. out = cap * tanh(in / cap) in one
  // GPU kernel -- the kernel realizes tanh, the activation every reachable
  // configuration sets; the routing (device kernel regardless of act_type) is
  // the same the layer's former open-coded block applied.
  if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    auto *ip = reinterpret_cast<const unsigned short *>(in.getData<_FP16>());
    auto *op = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    cudaPointerAttributes pa{};
    // Accept Managed (UVM) too, not just Device: on integrated GPUs the
    // activation pool is cudaMallocManaged, so a Device-only gate sends the
    // softcap to the host fallback -- which, inside a CUDA-graph capture,
    // reads the not-yet-run lm_head logits (stale) and is itself not
    // captured -> garbage output. Managed pointers run the GPU kernel fine.
    // No engine_selected() term: the attribute probe below IS the gate (a
    // plain host pointer reports cudaMemoryTypeUnregistered and falls
    // through), and an env test here would send an SDK-path CudaContext --
    // built without NNTR_ENGINE=cuda but with the device-only pool armed --
    // straight into the host fallback on device logits.
    if (cudaPointerGetAttributes(&pa, ip) == cudaSuccess &&
        (pa.type == cudaMemoryTypeDevice || pa.type == cudaMemoryTypeManaged) &&
        nntrainer::cuda::cuda_softcap_fp16(ip, op, (unsigned int)in.size(),
                                           cap)) {
      cudaGetLastError();
      return;
    }
    cudaGetLastError();
  }
#endif
  host_math_gate("softcap", {&in, &out});
  CpuComputeOps::softcap(in, out, cap, act_type);
}

namespace {
// x * rsqrt(mean(x^2)+eps) * gamma, sum-of-squares accumulated in FP32 (no
// fp16 overflow). rows = leading dims folded, width = feature size.
template <typename T, typename G>
void rmsnorm_rows(const T *x, const G *g, T *y, unsigned int rows,
                  unsigned int width, float eps) {
  for (unsigned int r = 0; r < rows; ++r) {
    const T *xr = x + (size_t)r * width;
    T *yr = y + (size_t)r * width;
    float ss = 0.f;
    for (unsigned int k = 0; k < width; ++k) {
      float v = (float)xr[k];
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / (float)width + eps);
    for (unsigned int k = 0; k < width; ++k)
      yr[k] = (T)(((float)xr[k] * inv) * (float)g[k]);
  }
}

#ifdef ENABLE_FP16
bool dev_ok(const void *p) { return nntrainer::cuda::dev_accessible(p); }
#endif

void rmsnorm_dispatch(const Tensor &in, const Tensor &gamma, Tensor &out,
                      unsigned int rows, unsigned int width, float eps) {
  using DT = ml::train::TensorDim::DataType;
  const DT dt = in.getDataType();
  const DT gt = gamma.getDataType();
#ifdef ENABLE_FP16
  // GPU path: fp16 in/out/gamma all device-resident (UVM). Block-per-row, FP32
  // sum-of-squares. Used only for small row counts (decode, rows~1): the kernel
  // syncs per call, so for the wide prefill norm (rows=seq_len) the
  // multi-thread host norm wins -- gating by rows gives the decode speedup
  // without a prefill regression. NNTR_RMSNORM_CUDA_OFF disables; =all forces
  // all rows.
  static const int gpu_max_rows = []() {
    const char *e = std::getenv("NNTR_RMSNORM_CUDA_OFF");
    if (e && e[0] == 'a')
      return 1 << 30; // "all"
    if (e)
      return 0; // off
    return 32;  // decode-only default
  }();
  if (dt == DT::FP16 && out.getDataType() == DT::FP16 &&
      (gt == DT::FP16 || gt == DT::FP32) && (int)rows <= gpu_max_rows) {
    const unsigned short *xi =
      reinterpret_cast<const unsigned short *>(in.getData<_FP16>());
    unsigned short *yi =
      reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    // gamma is unquantized FP32 on disk and the RMSNorm layers request it as
    // FP32 for that reason, so an FP16 activation with an FP32 gamma is the
    // normal case -- the host tail below handles it explicitly. Requiring
    // gt == FP16 here therefore disabled the device norm outright instead of
    // narrowing it; bind a converted, cached fp16 gamma instead.
    const unsigned short *gi = nullptr;
    bool gamma_ok;
    if (gt == DT::FP16) {
      gi = reinterpret_cast<const unsigned short *>(gamma.getData<_FP16>());
      gamma_ok = dev_ok(gi);
    } else {
      gamma_ok =
        cuda::cuda_rmsnorm_gamma_to_fp16(gamma.getData<float>(), width, &gi);
    }
    if (gamma_ok && dev_ok(xi) && dev_ok(yi) &&
        cuda::cuda_rmsnorm_fp16(xi, gi, yi, eps, rows, width))
      return;
  }
#endif
  // Host rmsnorm fallback: the gate syncs first so the host read of
  // GPU-produced input is coherent under NNTR_CUDA_ASYNC (no-op in sync mode),
  // and refuses by name if in/out live in the device-only pool.
  host_math_gate("rms_norm", {&in, &out, &gamma});
  if (dt == DT::FP32 && gt == DT::FP32) {
    rmsnorm_rows(in.getData<float>(), gamma.getData<float>(),
                 out.getData<float>(), rows, width, eps);
#ifdef ENABLE_FP16
  } else if (dt == DT::FP16 && gt == DT::FP16) {
    rmsnorm_rows(in.getData<_FP16>(), gamma.getData<_FP16>(),
                 out.getData<_FP16>(), rows, width, eps);
  } else if (dt == DT::FP16 && gt == DT::FP32) {
    rmsnorm_rows(in.getData<_FP16>(), gamma.getData<float>(),
                 out.getData<_FP16>(), rows, width, eps);
  } else if (dt == DT::FP32 && gt == DT::FP16) {
    rmsnorm_rows(in.getData<float>(), gamma.getData<_FP16>(),
                 out.getData<float>(), rows, width, eps);
#endif
  } else {
    throw std::invalid_argument(
      "CudaComputeOps::rms_norm: unsupported data type");
  }
}
} // namespace

void CudaComputeOps::rms_norm(const Tensor &in, Tensor &out,
                              const Tensor &gamma, float epsilon,
                              unsigned int active_rows,
                              unsigned int row_offset) {
  // rmsnorm_dispatch consumes base pointers + a row count, so the
  // (active_rows, row_offset) window becomes a shared-data view at the row
  // offset. Every in-tree caller passes row_offset 0, where the views alias
  // the arguments' own buffers -- the same pointers the former per-backend
  // layer handed the dispatch.
  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  Tensor in_win = in.getSharedDataTensor(
    TensorDim(1, 1, active_rows, width, in.getDim().getTensorType()), elem_off,
    true);
  Tensor out_win = out.getSharedDataTensor(
    TensorDim(1, 1, active_rows, width, out.getDim().getTensorType()), elem_off,
    true);
  rmsnorm_dispatch(in_win, gamma, out_win, active_rows, width, epsilon);
}

// Reverse-RMSNorm (per-layer-embedding post_norm): y = (x*w / rms(x*w)) *
// out_scale, the per-feature weight folded INSIDE the denominator and the
// sum of squares accumulated in FP32. Mirrors ClComputeOps::rms_reverse_norm:
// same signature, same window arithmetic (data + row_offset*width), and like
// the CL kernel it does NOT fold the weight into `in` in place -- the doc'd
// contract says no graph consumer reads the reverse-norm input after this op,
// and the device kernel recomputes x*w in registers instead.
//
// The device kernel is FP16-ONLY (cuda_rms_reverse_norm_fp16 is the only one
// that exists): an FP32 ACTIVATION has no device path here and takes the
// inherited host body behind the named guard. What is NOT a reason to decline
// is an FP32 weight/out_scale -- see the dtype note in the body.
void CudaComputeOps::rms_reverse_norm(Tensor &in, Tensor &out,
                                      const Tensor &weight,
                                      const Tensor &out_scale, float epsilon,
                                      unsigned int active_rows,
                                      unsigned int row_offset) {
#ifdef ENABLE_FP16
  using DT = ml::train::TensorDim::DataType;
  // Kill-switch, so a suspected numeric regression here is one env var away
  // from a host-side A/B (with the device-only pool that A/B needs
  // NNTR_CUDA_DEV_ACT=0 too -- the gate below says so by name).
  static const bool gpu = []() {
    const char *e = std::getenv("NNTR_CUDA_RMS_REVERSE_NORM");
    return !(e && e[0] == '0');
  }();
  const unsigned int width = in.width();
  const DT wdt = weight.getDataType();
  const DT sdt = out_scale.getDataType();
  // The FP32-gamma lesson from rmsnorm_dispatch above, carried across: a norm
  // weight is unquantized on disk, so on a QUANTIZED package the layer resolves
  // it to FP32 while the activation stays FP16 (rms_norm_layer.cpp picks
  // getWeightDataType() only when that is itself a float type, else FP32).
  // Requiring wdt == FP16 therefore does not narrow the device path, it
  // DISABLES it for a whole class of packages -- exactly the defect the
  // rms_norm gate had to be repaired for. Accept FP32 too and bind a
  // converted, process-cached fp16 copy (same builder rms_norm uses; keyed on
  // the fp32 pointer, refused inside a graph capture). The ACTIVATION dtype is
  // the one real constraint: the kernel reads/writes fp16 only.
  const bool wdt_ok = (wdt == DT::FP16 || wdt == DT::FP32);
  const bool sdt_ok = (sdt == DT::FP16 || sdt == DT::FP32);
  if (gpu && active_rows > 0 && width > 0 && in.getDataType() == DT::FP16 &&
      out.getDataType() == DT::FP16 && wdt_ok && sdt_ok &&
      weight.width() == width && weight.size() == width &&
      out_scale.size() == 1) {
    const size_t elem_off = (size_t)row_offset * width;
    auto *x =
      reinterpret_cast<const unsigned short *>(in.getData<_FP16>() + elem_off);
    auto *y =
      reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
    const unsigned short *w = nullptr;
    const unsigned short *s = nullptr;
    bool w_ok, s_ok;
    if (wdt == DT::FP16) {
      w = reinterpret_cast<const unsigned short *>(weight.getData<_FP16>());
      w_ok = nntrainer::cuda::dev_accessible(w);
    } else {
      w_ok = cuda::cuda_rmsnorm_gamma_to_fp16(weight.getData<float>(), width,
                                              &w);
    }
    if (sdt == DT::FP16) {
      s = reinterpret_cast<const unsigned short *>(out_scale.getData<_FP16>());
      s_ok = nntrainer::cuda::dev_accessible(s);
    } else {
      // out_scale is [1,1,1,1]; the same width-N converter with N = 1.
      s_ok =
        cuda::cuda_rmsnorm_gamma_to_fp16(out_scale.getData<float>(), 1u, &s);
    }
    // The activations must additionally be device-readable: they come from the
    // device-only (or managed) activation pool. dev_accessible accepts Managed
    // and pinned-mapped too, so this engages on integrated / WDDM pools as
    // well. (The converter above already returns device-readable memory.)
    if (w_ok && s_ok && nntrainer::cuda::dev_accessible(x) &&
        nntrainer::cuda::dev_accessible(y) &&
        nntrainer::cuda::cuda_rms_reverse_norm_fp16(x, w, s, y, epsilon,
                                                    active_rows, width))
      return;
  }
#endif
  // FP32 activations (no device kernel), or a shape the kernel cannot bind:
  // the inherited host FP32-temp math is the DESIGNED path there (same routing
  // the OpenCL table takes), but it dereferences all four operands on the host.
  host_math_gate("rms_reverse_norm", {&in, &out, &weight, &out_scale});
  CpuComputeOps::rms_reverse_norm(in, out, weight, out_scale, epsilon,
                                  active_rows, row_offset);
}

// One residual-add operand. hidden = input is a copy, and Tensor::copy routes
// through scopy_fp16/scopy_fp32 -- overridden in this table, device-aware --
// so the inherited body is already correct for it. hidden += input is NOT:
// Tensor::add_i lands on ele_add_fp16, host math with no override here, which
// faults on a device-only activation. Give the accumulate form its own device
// kernel (the same cuda_add_fp16 the 2-input AdditionLayer fast path uses, so
// the numerics are the ones this backend already ships) and gate the rest.
void CudaComputeOps::residual_op(Tensor &hidden, const Tensor &input,
                                 bool accumulate) {
#ifdef ENABLE_FP16
  using DT = ml::train::TensorDim::DataType;
  if (accumulate && hidden.getDataType() == DT::FP16 &&
      input.getDataType() == DT::FP16 && hidden.size() == input.size() &&
      hidden.size() > 0) {
    static const bool gpu = nntr_env_on("NNTR_CUDA_ELTWISE");
    if (gpu) {
      auto *h = reinterpret_cast<unsigned short *>(hidden.getData<_FP16>());
      auto *i = reinterpret_cast<const unsigned short *>(input.getData<_FP16>());
      // In-place accumulate (dst aliases operand a): the kernel is pure
      // element-wise, one read and one write per index, so aliasing is safe.
      if (nntrainer::cuda::dev_accessible(h) &&
          nntrainer::cuda::dev_accessible(i) &&
          nntrainer::cuda::cuda_add_fp16(h, i, h, (unsigned int)hidden.size()))
        return;
    }
  }
#endif
  if (accumulate) {
    // add_i -> host ele_add_*: must not run on a device-only operand.
    host_math_gate("residual_op", {&hidden, &input});
  } else if (copy_takes_scopy(hidden, input)) {
    // Tensor::copy's MATCHING branch -> itensor_->copy -> scopy_*, overridden
    // in this table and device-aware. Only drain, so a host-coherent copy sees
    // the last device write.
    table_drain(false);
  } else {
    // Tensor::copy's MISMATCH branch is NOT device-aware: it builds
    // `Tensor t(from.getDim(), from.getData<char>())` -- a HOST read of the
    // source bytes -- and swaps that host allocation in as this tensor's
    // backing store. On a device-only operand that is a fault, and even when it
    // does not fault it silently replaces a pool tensor's storage with plain
    // host memory, so every later device op on `hidden` binds a pointer no
    // kernel can reach. Refuse by name.
    host_math_gate("residual_op", {&hidden, &input});
  }
  CpuComputeOps::residual_op(hidden, input, accumulate);
}

// Fused activation epilogue. No device kernel exists (and the LLM graphs never
// set a fused activation on an FC), so this override exists purely to hold the
// invariant: run the inherited host ActiFunc when the output is host
// reachable, refuse by name when it is not.
void CudaComputeOps::apply_activation(Tensor &out, int act_type) {
  // ACT_NONE is a no-op in every impl -- it must stay one here too, or the
  // guard would refuse a call that touches nothing. (Every in-tree caller
  // already filters it out; this keeps the op's contract self-contained.)
  if (static_cast<ActivationType>(act_type) == ActivationType::ACT_NONE)
    return;
  host_math_gate("apply_activation", {&out});
  CpuComputeOps::apply_activation(out, act_type);
}

// FC GEMM: output = input * weight. QS4CX weight -> fused dequant-GEMM on
// device, consuming the PLAIN nibble payload in place (single weight copy, no
// UVM copy). QINT4 never reaches here: layer_context coerces it to QS4CX at
// init.
void CudaComputeOps::fc(Tensor &input, Tensor &weight, Tensor &output) {
  using DT = ml::train::TensorDim::DataType;
  const DT wt = weight.getDataType();
  const DT at = input.getDataType();

  const auto &id = input.getDim();
  const auto &od = output.getDim();
  const int K = (int)id.width();
  const int N = (int)od.width();
  const int M = (int)(id.batch() * id.channel() * id.height());

  if (wt == DT::QS4CX && M > 0 && N > 0 && K > 0 &&
      (int)weight.getDim().height() == K) {
    const uint8_t *W = weight.getData<uint8_t>();
    // On a derived-cache HIT the dp4a and cuBLAS-i8 paths do not DEREFERENCE
    // the plain payload: they use its pointer VALUE as the key of the device
    // caches (packed int4 + rowsum; int8 [K,N]) that the load-time prewarm
    // already built. So "the derived cache exists" is as good an entry ticket
    // as device residency -- and it is the only one available under
    // NNTR_QS4CX_HEAP_BYPASS, where the payload is ordinary heap and
    // dev_accessible(W) is false by construction. Requiring residency there
    // sent every QS4CX FC to the host dot() tail below, which with
    // NNTR_CUDA_DROP_PLAIN reads pages that were discarded after the caches
    // were built: zeros, hence silently wrong logits rather than a crash.
    //
    // A cache MISS is the opposite: both builders bind the payload into a
    // device repack kernel (repack_plain_i4, repack_plain_i8_kn). The ticket
    // below only proves the DP4A cache exists -- the i8 [K,N] cache is a
    // separate map -- so the builders enforce device-readability themselves
    // (plain_bindable() in cuda_fc_qint4.cpp) and report failure, which this
    // chain's fall-through turns into a dp4a call that is a pure hit.
    const bool w_cached = cuda::cuda_fc_qs4cx_has_cache(W);
    // The NAIVE plain GEMM is the exception -- it binds W straight into the
    // kernel, so it still needs real device residency.
    const bool w_dev = nntrainer::cuda::dev_accessible(W);
    // The per-weight fp16 scale buffer the dequant kernel reads every call.
    const uint16_t *S = nullptr;
    if ((w_dev || w_cached) && cuda::cuda_fc_qs4cx_scales_to_uvm_fp16(
                                 weight.getScale<float>(), (unsigned)N, &S)) {
#ifdef ENABLE_FP16
      if (at == DT::FP16 && output.getDataType() == DT::FP16) {
        auto *Xh =
          reinterpret_cast<const unsigned short *>(input.getData<_FP16>());
        auto *Yh = reinterpret_cast<unsigned short *>(output.getData<_FP16>());
        // Prefill (M >= CUDA_FC_I8_PREFILL_MIN_M): w4a8 on the INT8 Tensor
        // Cores via cuBLAS (~10x the dp4a int-ALU GEMM, bit-identical). Then
        // the dp4a fast path, then the naive plain GEMM -- each falls to the
        // next on failure. The threshold is the header constant, not a local
        // literal, because the load-time prewarm decides whether to build the
        // i8 [K,N] cache from the same number.
        // This gate is the SHAPE only: the NNTR_FC_CUDA_CUBLAS=0 opt-out is
        // enforced inside cuda_fc_qs4cx_cublas_i8_gemm_fp16(), which then
        // reports failure and lets the dp4a path below take the call.
        const bool prefill = M >= (int)cuda::CUDA_FC_I8_PREFILL_MIN_M;
        if (nntrainer::cuda::dev_accessible(Xh) &&
            ((prefill &&
              cuda::cuda_fc_qs4cx_cublas_i8_gemm_fp16(
                Xh, W, S, Yh, (unsigned)M, (unsigned)N, (unsigned)K)) ||
             cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(Xh, W, S, Yh, (unsigned)M,
                                                (unsigned)N, (unsigned)K) ||
             (w_dev && cuda::cuda_fc_qs4cx_gemm_fp16_naive(
                         Xh, W, S, Yh, (unsigned)M, (unsigned)N, (unsigned)K))))
          return;
      }
#endif
      if (at == DT::FP32 && output.getDataType() == DT::FP32) {
        const float *X = input.getData<float>();
        float *Y = output.getData<float>();
        // w4a8 dp4a fast path; falls to the naive plain GEMM on failure.
        if (nntrainer::cuda::dev_accessible(X) &&
            (cuda::cuda_fc_qs4cx_dp4a_gemm_fp32(X, W, S, Y, (unsigned)M,
                                                (unsigned)N, (unsigned)K) ||
             (w_dev && cuda::cuda_fc_qs4cx_gemm_fp32(
                         X, W, S, Y, (unsigned)M, (unsigned)N, (unsigned)K))))
          return;
      }
    }
  }

  // Host fallback: the input is host-coherent UVM, so the CPU dot is correct.
  // Drain first in async mode so the host read sees the produced input.
  // NNTR_CUDA_FC_DBG=1 prints WHY a call fell off the device fast paths --
  // the fall-through above is silent by design (checklist B.15: a CUDA op
  // falling to the host loop is invisible without a runtime trace).
  static const bool fc_dbg = []() {
    const char *e = std::getenv("NNTR_CUDA_FC_DBG");
    return e && e[0] == '1';
  }();
  if (fc_dbg) {
    static int n_prints = 0;
    if (n_prints < 64) {
      ++n_prints;
      std::fprintf(
        stderr,
        "[CUDA-FC-DBG] host-dot fallback: wdt=%d adt=%d odt=%d M=%d N=%d "
        "K=%d w_h=%d dev(W)=%d dev(X)=%d\n",
        (int)wt, (int)at, (int)output.getDataType(), M, N, K,
        (int)weight.getDim().height(),
        (int)nntrainer::cuda::dev_accessible(weight.getData<uint8_t>()),
        (int)nntrainer::cuda::dev_accessible(input.getData<char>()));
    }
  }
  // The host dot() READS the weight bytes. If this payload's pages were
  // discarded (NNTR_CUDA_DROP_PLAIN, after the derived device caches were
  // built) they now read back as zeros, so the dot would produce a zero-weight
  // result -- correct-looking output, silently wrong numbers. Fail loudly
  // instead: reaching here with a dropped payload means a device path that was
  // supposed to be the only consumer of this weight declined the call.
  if (wt == DT::QS4CX &&
      cuda::cuda_fc_qs4cx_plain_dropped(weight.getData<uint8_t>()))
    throw std::runtime_error(
      "CudaComputeOps::fc: the QS4CX plain payload for this weight was "
      "dropped (NNTR_CUDA_DROP_PLAIN) but the call fell through to the host "
      "dot(), which would read zero-filled pages. Re-run with "
      "NNTR_CUDA_DROP_PLAIN=0, and use NNTR_CUDA_FC_DBG=1 to see why the "
      "device path declined.");
  // Same class of defect as the dropped payload above, one step earlier: the
  // host dot() also READS input and WRITES output through host pointers, which
  // is a fault (not silently wrong numbers) when the activation pool is
  // device-only. Refuse by name; the gate also carries the async drain.
  //
  // `weight` is in the operand list too, and is NOT exempt: dot() dereferences
  // the weight bytes on the host exactly as it does the activations. Weights
  // normally live in the managed pool (host-addressable), so this term is
  // expected never to fire -- but "expected" is not "enforced": the weight
  // residency policy is a separate, moving lever (NNTR_CUDA_WPREFETCH migrates
  // pages, NNTR_QS4CX_HEAP_BYPASS moves the payload out of the pool entirely),
  // and a future device-only weight pool would otherwise reopen this exact gap
  // silently. Probing it costs one cudaPointerGetAttributes on a path that is
  // already the slow fallback.
  host_math_gate("fc", {&input, &output, &weight});
  input.dot(weight, output, false, false);
}

// ── Copy ops (device-only aware) ─────────────────────────────────────────
// Under the device-only activation pool (NNTR_CUDA_DEV_ACT) an activation is
// real device memory; Tensor::copy() -> the CpuComputeOps host loop would
// fault on it. Route contiguous device-only copies through a stream-ordered
// cudaMemcpyAsync; host / host-coherent UVM keep the CPU path.
//
// Residency is probed with host_unreachable(), NOT cuda::dev_only(). Same
// reason as host_math_gate: dev_only() answers false unless
// NNTR_ENGINE=="cuda", while the device-only pool these ops exist for is armed
// by the CudaContext CONSTRUCTOR (NNTR_CUDA_DEV_ACT, cuda_context.cpp) --
// which also runs with NNTR_ENGINE unset on the library/SDK bring-up path. On
// such a run every probe below answered false, so the host element loops at
// the bottom of each op ran straight over cudaMalloc pointers: the very defect
// this table was written to make impossible, still live in its own copy ops.
static bool device_copy(const void *X, void *Y, size_t bytes, bool contiguous) {
  if (!(host_unreachable(X) || host_unreachable(Y)))
    return false;
  if (!contiguous)
    throw std::runtime_error(
      "CudaComputeOps: strided copy on device-only memory is unsupported");
  auto &sm = cuda::StreamManager::Global();
  if (cudaMemcpyAsync(Y, X, bytes, cudaMemcpyDefault, sm.GetStream()) !=
      cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(
      "CudaComputeOps: device copy (cudaMemcpyAsync) failed");
  }
  if (!host_unreachable(Y)) {
    if (sm.isCapturing())
      std::fprintf(
        stderr,
        "[CAP-AUDIT] scopy D2H (host-consumed) during capture: %zu bytes\n",
        bytes);
    sm.finish(); // D2H: the host consumes the destination immediately
  }
  return true;
}

void CudaComputeOps::scopy_fp32(const unsigned int N, const float *X,
                                const unsigned int incX, float *Y,
                                const unsigned int incY) {
  if (device_copy(X, Y, (size_t)N * sizeof(float), incX == 1 && incY == 1))
    return;
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}

#ifdef ENABLE_FP16
void CudaComputeOps::scopy_fp16(const unsigned int N, const _FP16 *X,
                                const unsigned int incX, _FP16 *Y,
                                const unsigned int incY) {
  if (device_copy(X, Y, (size_t)N * sizeof(_FP16), incX == 1 && incY == 1))
    return;
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}
// Converting copies with a device-only endpoint: stage through host temps
// (synchronous; these do not occur inside graph capture today). host_unreachable
// (not dev_only) for the same reason as device_copy above.
void CudaComputeOps::scopy_fp32_to_fp16(const unsigned int N, const float *X,
                                        const unsigned int incX, _FP16 *Y,
                                        const unsigned int incY) {
  if (host_unreachable(X) || host_unreachable(Y)) {
    if (incX != 1 || incY != 1)
      throw std::runtime_error(
        "CudaComputeOps: strided converting copy on device-only memory");
    if (cuda::StreamManager::Global().isCapturing())
      std::fprintf(stderr,
                   "[CAP-AUDIT] converting scopy fp32->fp16 during capture: "
                   "N=%u (host convert frozen into graph)\n",
                   N);
    table_drain(true);
    std::vector<float> xs;
    const float *xp = X;
    if (host_unreachable(X)) {
      xs.resize(N);
      cuda::copy_any(xs.data(), X, (size_t)N * sizeof(float));
      xp = xs.data();
    }
    std::vector<_FP16> ys(N);
    for (unsigned int i = 0; i < N; ++i)
      ys[i] = static_cast<_FP16>(xp[i]);
    if (host_unreachable(Y))
      cuda::copy_any(Y, ys.data(), (size_t)N * sizeof(_FP16));
    else
      std::memcpy(Y, ys.data(), (size_t)N * sizeof(_FP16));
    return;
  }
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<_FP16>(X[i * incX]);
}
void CudaComputeOps::scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                                        const unsigned int incX, float *Y,
                                        const unsigned int incY) {
  if (host_unreachable(X) || host_unreachable(Y)) {
    if (incX != 1 || incY != 1)
      throw std::runtime_error(
        "CudaComputeOps: strided converting copy on device-only memory");
    if (cuda::StreamManager::Global().isCapturing())
      std::fprintf(stderr,
                   "[CAP-AUDIT] converting scopy fp16->fp32 during capture: "
                   "N=%u (host convert frozen into graph)\n",
                   N);
    table_drain(true);
    std::vector<_FP16> xs;
    const _FP16 *xp = X;
    if (host_unreachable(X)) {
      xs.resize(N);
      cuda::copy_any(xs.data(), X, (size_t)N * sizeof(_FP16));
      xp = xs.data();
    }
    std::vector<float> ys(N);
    for (unsigned int i = 0; i < N; ++i)
      ys[i] = static_cast<float>(xp[i]);
    if (host_unreachable(Y))
      cuda::copy_any(Y, ys.data(), (size_t)N * sizeof(float));
    else
      std::memcpy(Y, ys.data(), (size_t)N * sizeof(float));
    return;
  }
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<float>(X[i * incX]);
}
#endif

// Load-time device-residency action, executed through the op-table prebuild
// seam: FullyConnectedLayerCl::read() calls w.getOps()->fc_prebuild_weight(w)
// per weight inside the parallel load worker, right after the weight bytes
// are read (skipped under FSU/opt_var). Only engine=cuda tensors resolve to
// this table, so no engine scan is needed to keep the call off gpu/cpu runs.
// Prebuild contract: a prebuild may create derived device state but must NOT
// invalidate the host payload -- cudaMemPrefetchAsync is a migration of the
// managed pages to the device, never an invalidation; the pointer stays
// host-accessible.
void CudaComputeOps::fc_prebuild_weight(Tensor &w) {
  if (w.getDataType() != ml::train::TensorDim::DataType::QS4CX)
    return;
  // NNTR_CUDA_WPREFETCH >= 2 opts in; unset -> 0 (default off).
  static const int wpf = []() {
    const char *e = std::getenv("NNTR_CUDA_WPREFETCH");
    return e ? atoi(e) : 0;
  }();
  if (wpf < 2)
    return;
  // The primitive is self-guarding (cuda_fc_qint4.cpp): integrated GPU ->
  // false, non-managed pointer -> false, and it computes its own byte extent
  // (the N*(K+1)/2 nibble payload + the N*4 fp32 scale tail).
  (void)cuda::cuda_fc_qs4cx_prefetch_weight(w.getData<uint8_t>(), w.width(),
                                            w.height());
}

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer
