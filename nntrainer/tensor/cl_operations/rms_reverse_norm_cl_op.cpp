// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rms_reverse_norm_cl_op.cpp
 * @date   29 July 2026
 * @brief  OpenCL reverse-RMSNorm whole-op kernel dispatch
 *         (out = out_scale * normalize(in * weight)).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Two relocations, no new math:
 *  - rms_reverse_norm_cl_fp16() moved VERBATIM from blas_kernels.cpp
 * (652e8f2e4) into an anonymous namespace here — same kernel
 *    (rms_reverse_norm_cl_fp16_coop in rmsnorm_fp16.cl, same program string),
 *    same argument binding, same RMSN_LWS=64 launch geometry.
 *  - The SVM residency contract moved out of
 *    RMSReverseNormLayer::incremental_forwarding's gpu_svm predicate, checked
 *    here in the SAME short-circuit order, each leg with a NAMED reject reason
 *    instead of the layer's silent fall-through. (N4).
 */

#include "rms_reverse_norm_cl_op.h"

#include <cl_kernels/cl_kernels.h>
#include <engine.h> // Engine::Global().getRegisteredContext("gpu")
#include <memory_data.h>
#include <opencl_buffer_manager.h>
#include <tensor.h>

namespace nntrainer {

// [divert tripwire] Why the last rms_reverse_norm_cl_op call on this thread
// handed the op back to its caller. Every `return false` goes through
// RRN_REJECT so the host bounce it causes can be NAMED (logged by
// ClComputeOps::rms_reverse_norm) instead of being silent — the silent
// four-predicate fall-through in the Layer body is the exact shape that hid
// the all `-0` PLE post_norm failure (F-B). Same mechanism as V8C_REJECT.
static const char *&rrn_reject_slot() {
  static thread_local const char *r = "none";
  return r;
}
#define RRN_REJECT(why)                                                        \
  do {                                                                         \
    rrn_reject_slot() = (why);                                                 \
    return false;                                                              \
  } while (0)

const char *rms_reverse_norm_cl_last_reject_reason() {
  return rrn_reject_slot();
}

#ifdef ENABLE_FP16
namespace {

// PLE reverse-RMSNorm GPU path: out = out_scale * normalize(in *
// weight). Mirrors rmsnorm_cl_fp16's coop dispatch (fp32 accumulation,
// SVM-direct or planner cl_mem sub-buffer bind), with the extra per-feature
// `weight` (arg 2, applied inside the RMS denom) and the post-norm SCALAR
// `out_scale` (arg 3). One workgroup (RMSN_LWS WIs) per row; W need not be %8
// (scalar per-element). [moved verbatim from blas_kernels.cpp; private now —
// the whole-op entry below is the only caller]
void rms_reverse_norm_cl_fp16(const _FP16 *input, const _FP16 *weight,
                              _FP16 out_scale, _FP16 *result,
                              const float epsilon, unsigned int height,
                              unsigned int width, bool use_svm, void *out_clmem,
                              void *in_clmem) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  const _FP16 eps_h = static_cast<_FP16>(epsilon);
  const size_t in_bytes = (size_t)height * width * sizeof(_FP16);
  cl_mem out_cl = static_cast<cl_mem>(out_clmem);
  cl_mem in_cl = static_cast<cl_mem>(in_clmem);
  const bool to_clmem = (out_cl != nullptr) && use_svm;
  const bool from_clmem = (in_cl != nullptr);
  const int n_rows = (int)height;
  const int w = (int)width;
  constexpr int RMSN_LWS = 64;

  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    rmsnorm_fp16_kernel, "rms_reverse_norm_cl_fp16_coop");
  if (!kp)
    return;

  if (to_clmem || from_clmem) {
    bool ok = true;
    if (from_clmem)
      ok = ok && kp->SetKernelArguments(0, &in_cl, sizeof(cl_mem));
    else
      ok = ok && kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input));
    if (to_clmem)
      ok = ok && kp->SetKernelArguments(1, &out_cl, sizeof(cl_mem));
    else
      ok = ok && kp->SetKernelSVMArguments(1, result);
    ok = ok && kp->SetKernelSVMArguments(2, const_cast<_FP16 *>(weight));
    if (!ok)
      return;
  } else if (use_svm) {
    if (!kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input)) ||
        !kp->SetKernelSVMArguments(1, result) ||
        !kp->SetKernelSVMArguments(2, const_cast<_FP16 *>(weight)))
      return;
  } else {
    auto &clbuf = ClBufferManager::Global();
    if (!clbuf.getInBufferA()->WriteDataRegion(blas_cc->command_queue_inst_,
                                               in_bytes, input) ||
        !clbuf.getInBufferB()->WriteDataRegion(blas_cc->command_queue_inst_,
                                               width * sizeof(_FP16), weight))
      return;
    if (!kp->SetKernelArguments(0, &clbuf.getInBufferA()->GetBuffer(),
                                sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &clbuf.getOutBufferA()->GetBuffer(),
                                sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &clbuf.getInBufferB()->GetBuffer(),
                                sizeof(cl_mem)))
      return;
  }
  // out_scale is a native _FP16; pass its 2 bytes straight through as a `half`
  // kernel arg (same as eps_h). static_cast<cl_half> would convert the VALUE to
  // a uint16 (0.0292 -> 0) and zero the scale -> all-zero output.
  if (!kp->SetKernelArguments(3, &out_scale, sizeof(cl_half)) ||
      !kp->SetKernelArguments(4, &eps_h, sizeof(cl_half)) ||
      !kp->SetKernelArguments(5, &n_rows, sizeof(int)) ||
      !kp->SetKernelArguments(6, &w, sizeof(int)))
    return;
  const int work_groups_count[3] = {RMSN_LWS * n_rows, 1, 1};
  const int work_group_size[3] = {RMSN_LWS, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, work_groups_count,
                                                    work_group_size))
    return;
  if (!use_svm && !to_clmem) {
    auto &clbuf = ClBufferManager::Global();
    clbuf.getOutBufferA()->ReadDataRegion(blas_cc->command_queue_inst_,
                                          in_bytes, result);
  }
}

} // namespace
#endif // ENABLE_FP16

bool rms_reverse_norm_cl_op(Tensor &in, Tensor &out, const Tensor &weight,
                            const Tensor &out_scale, float epsilon,
                            unsigned int active_rows, unsigned int row_offset) {
  // The FP32 activation path is the layer's DESIGNED host math (there is no
  // fp32 reverse-norm kernel; the cpu table runs it in place) — same routing
  // as before this refactor, now named.
  if (in.getDataType() != ml::train::TensorDim::DataType::FP16)
    RRN_REJECT("activation dtype not FP16 (designed host math path)");
#ifdef ENABLE_FP16
  // GPU-resident contract, moved verbatim from the Layer's gpu_svm predicate:
  // taken only when in/out/weight/out_scale are ALL SVM and the two scale
  // tensors carry FP16. Checked in the ORIGINAL short-circuit order so "which
  // predicate failed" means the same thing it did in the layer body.
  const auto in_md = in.getMemoryData();
  if (!(in_md && in_md->isSVM()))
    RRN_REJECT("input not SVM-resident");
  const auto out_md = out.getMemoryData();
  if (!(out_md && out_md->isSVM()))
    RRN_REJECT("output not SVM-resident");
  const auto w_md = weight.getMemoryData();
  if (!(w_md && w_md->isSVM()))
    RRN_REJECT("weight not SVM-resident");
  const auto os_md = out_scale.getMemoryData();
  if (!(os_md && os_md->isSVM()))
    RRN_REJECT("out_scale not SVM-resident");
  if (weight.getDataType() != ml::train::TensorDim::DataType::FP16)
    RRN_REJECT("weight dtype not FP16");
  if (out_scale.getDataType() != ml::train::TensorDim::DataType::FP16)
    RRN_REJECT("out_scale dtype not FP16");

  // The layer used to pass per-batch step VIEWS; the whole-op window is the
  // same memory: data + row_offset*W is exactly the old view base pointer, and
  // the old `view.getOffset()==0 && view.isClMem()` cl_mem candidacy is
  // `row_offset==0 && tensor.getOffset()==0 && tensor.isClMem()` on the whole
  // tensor (a view's offset accumulates the parent's).
  const size_t elem_off = (size_t)row_offset * in.width();
  void *in_cl = (row_offset == 0 && in.getOffset() == 0 && in.isClMem())
                  ? in.getClMem()
                  : nullptr;
  void *out_cl = (row_offset == 0 && out.getOffset() == 0 && out.isClMem())
                   ? out.getClMem()
                   : nullptr;
  rms_reverse_norm_cl_fp16(
    in.getData<_FP16>() + elem_off, weight.getData<_FP16>(),
    out_scale.getData<_FP16>()[0], out.getData<_FP16>() + elem_off, epsilon,
    active_rows, in.width(), /*use_svm=*/true, out_cl, in_cl);
  return true;
#else
  (void)out;
  (void)weight;
  (void)out_scale;
  (void)epsilon;
  (void)active_rows;
  (void)row_offset;
  RRN_REJECT("built without enable-fp16");
#endif
}

} // namespace nntrainer
