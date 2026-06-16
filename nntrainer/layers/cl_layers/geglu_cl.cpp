// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_cl.cpp
 * @date   08 Jun 2026
 * @brief  GPU GeGLU: gelu_tanh(gate) * up. Cloned from swiglu_cl.cpp with the
 *         SiLU gate replaced by the tanh-approx GELU (Gemma2).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "geglu_cl.h"
#include "nntrainer_log.h"
#include <blas_kernel_interface.h> // publish_resident_act (cl_mem residency)
#include <cl_kernels/geglu.h>
#include <cstdlib>
#ifdef ENABLE_FP16
#include <cl_kernels/geglu_fp16.h>
#endif
#include <iostream>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

bool GeGLULayerCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  // check if the kernels are already registered.
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for geglu_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_geglu_ptr = nullptr;

    kernel_geglu_ptr = cl_context.registerClKernel(geglu_kernel, "geglu_cl");

    if (!kernel_geglu_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_geglu_ptr);

#ifdef ENABLE_FP16
    kernel_geglu_ptr =
      cl_context.registerClKernel(geglu_fp16_kernel, "geglu_cl_fp16");

    if (!kernel_geglu_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_geglu_ptr);
#endif

    return true;
  } while (false);

  // clear all registered kernels if any error occurs during registration
  layer_kernel_ptrs.clear();

  return false;
}

void GeGLULayerCl::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void GeGLULayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  gegluProcess(in1, in2, out,
               in1.batch() * in1.channel() * in1.height(), 0);
}

void GeGLULayerCl::incremental_forwarding(RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
  }

  // [decode active-row fix] The activation buffers are NOT re-based at decode --
  // the current token lives at its ABSOLUTE row (to-1) of the full [INIT_SEQ,I]
  // tensor (confirmed empirically: processing rows [0,to) is token-identical to
  // processing the full height, while processing row 0 diverges). So process
  // only the live rows [from, to): at decode that is the single row `from`.
  // Without this, geglu re-ran the whole [1024,I] tensor every decode step -- a
  // ~1024x waste (the 806us/op decode cost on Gemma2 I=9216).
  //
  // SVM tensors: offset the data pointers to row `from` (getData()+from*width),
  //   process exactly (to-from) rows -> O(1) at decode regardless of position.
  // cl_mem tensors: getClMem() binds the whole buffer at index 0 (no pointer
  //   offset). The kernel now takes a row_off arg, so an all-cl_mem fp16 decode
  //   can still process ONLY the live row (to-1) by dispatching GWS=width with
  //   row_off=(to-1)*width -> O(1) (was [0,to) = O(position), the 804us/op cost).
  //   Mixed cl_mem/SVM or fp32 keep the [0,to) fallback (row_off can't serve a
  //   pointer-offset SVM arg and a base-bound cl_mem arg at once).
  const bool any_clmem = in1.isClMem() || in2.isClMem() || out.isClMem();
  const bool all_clmem = in1.isClMem() && in2.isClMem() && out.isClMem();
  const bool is_fp16 =
    in1.getDataType() == ml::train::TensorDim::DataType::FP16;
  if (from && all_clmem && is_fp16)
    gegluProcess(in1, in2, out, 1, 0);
  else if (any_clmem)
    gegluProcess(in1, in2, out, to, 0);
  else
    gegluProcess(in1, in2, out, to - from, from);
}

void GeGLULayerCl::gegluProcess(Tensor const &in1, Tensor const &in2,
                                Tensor &result, unsigned int active_rows,
                                unsigned int row_offset) {

  unsigned int dim1, dim2;
  dim1 = active_rows;
  dim2 = in1.width();

  // Element offset into the SVM pointers so the kernel processes rows
  // [row_offset, row_offset+active_rows). row_offset must be 0 for the cl_mem
  // path (getClMem() returns the whole buffer with no offset applied -- the
  // caller passes row_offset=0 + active_rows=to in that case).
  const size_t elem_off = (size_t)row_offset * dim2;

  // SVM-direct only when the tensors are GPU-resident (SVM pool). On the
  // default host (cpu) pool getData() returns host pointers, so use the
  // host-bounce path; passing host pointers as SVM kernel arguments produces
  // garbage. When the graph opts into the SVM pool (NNTR_GPU_SVM_POOL), this
  // becomes a zero-copy SVM-direct dispatch (residency).
  const auto md = in1.getMemoryData();
  const bool use_svm = md && md->isSVM();

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data1 = in1.getData() + elem_off;
    float *data2 = in2.getData() + elem_off;
    float *rdata = result.getData() + elem_off;
    geglu_cl(data1, data2, rdata, dim1, dim2, use_svm);
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data1 = in1.getData<_FP16>() + elem_off;
    _FP16 *data2 = in2.getData<_FP16>() + elem_off;
    _FP16 *rdata = result.getData<_FP16>() + elem_off;
    // Planner-decided STATIC residency: each of in1/in2/out binds the plane
    // its ResidencyClass picked at allocation -- the gate/up FC outputs and the
    // geglu output are GPU_CLMEM on the live Gemma2 path, so the kernel reads
    // and writes the planner cl_mem sub-buffers directly (no SVM map at all on
    // this op). Mixed cl_mem/SVM args are valid for partial classification.
    void *in1_cl = (use_svm && in1.isClMem()) ? in1.getClMem() : nullptr;
    void *in2_cl = (use_svm && in2.isClMem()) ? in2.getClMem() : nullptr;
    void *out_cl = (use_svm && result.isClMem()) ? result.getClMem() : nullptr;
    // [resident-act Step 1.5, legacy overlay] only when the static class did
    // not already place the output in cl_mem.
    void *res_backing = out_cl;
    if (res_backing == nullptr) {
      static const bool resident_act =
        std::getenv("NNTR_RESIDENT_ACT") != nullptr;
      const auto rmd = result.getMemoryData();
      if (resident_act && use_svm && rmd && rmd->isSVM())
        res_backing = static_cast<void *>(nntrainer::get_or_create_resident_backing(
          result.getName(), (unsigned int)result.size(), /*fp16=*/true));
    }
    // NNTR_DEVRES (legacy runtime overlay, default off): unchanged semantics
    // when no static class applies.
    static const bool devres_geglu = std::getenv("NNTR_DEVRES") != nullptr;
    const bool resident_edge = devres_geglu && use_svm && res_backing == nullptr;
    // When every buffer is bound as a whole cl_mem (index 0, no pointer offset),
    // the kernel applies elem_off itself so a single live row at row_offset is
    // processed in place. SVM/host args carry the offset on their pointer, so
    // row_off stays 0 there (mixing the two in one dispatch is gated out above).
    const unsigned int kern_row_off =
      (in1_cl && in2_cl && res_backing) ? (unsigned int)elem_off : 0u;
    geglu_cl_fp16(data1, data2, rdata, dim1, dim2, use_svm, res_backing,
                  /*skip_out_map=*/resident_edge, in1_cl, in2_cl, kern_row_off);
    if (devres_geglu)
      if (auto rmd = result.getMemoryData())
        rmd->setDeviceValid(resident_edge,
                            resident_edge ? result.getData<uint8_t>() : nullptr);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void GeGLULayerCl::geglu_cl(float *matAdata, float *vecXdata, float *vecYdata,
                            unsigned int dim1, unsigned int dim2, bool svm,
                            void *resident_out) {
  (void)resident_out; // FP32 path: residency overlay is FP16-only (no-op here)
  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_geglu_ptr = getLayerKernelPtrs()[Kernels::GEGLU_CL];
    int dim = int(dim1 * dim2);

    if (!svm) {
      bool write_result = true;

      write_result &= clbuffInstance.getInBufferA()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(float), matAdata);
      write_result &= clbuffInstance.getInBufferB()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(float), vecXdata);
      if (!write_result) {
        break;
      }

      auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
      auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
      auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

      bool set_result = true;
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
      if (!set_result) {
        break;
      }
    } else {
      bool map_result = true;
      map_result &=
        global_cl_context->command_queue_inst_.enqueueSVMUnmap(matAdata);
      map_result &=
        global_cl_context->command_queue_inst_.enqueueSVMUnmap(vecXdata);
      if (!map_result) {
        ml_loge("Failed to map svm");
        break;
      }

      bool set_svm_result = true;
      set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(0, matAdata);
      set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(1, vecXdata);
      // [resident-act] output direct to a cl_mem backing (device-resident, no
      // SVM intermediate) when the caller passed one; else SVM output.
      if (resident_out != nullptr) {
        cl_mem out_buf = static_cast<cl_mem>(resident_out);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(2, &out_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(2, vecYdata);
      }
      if (!set_svm_result) {
        ml_loge("Failed to set svm");
        break;
      }
    }

    // NOTE(mwlasiuk) : local size can not be larger than global
    const int32_t desired_local = 64;
    const bool can_use_desired = dim >= desired_local;
    const int32_t chosen_local = can_use_desired ? desired_local : dim;

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {chosen_local, 1, 1}; // test-value

    if (!global_cl_context->command_queue_inst_.DispatchCommand(
          kernel_geglu_ptr, work_groups_count, work_group_size)) {
      ml_loge("Failed to run");
      break;
    }

    if (!svm) {
      if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
            global_cl_context->command_queue_inst_, dim * sizeof(float),
            vecYdata)) {
        break;
      }
    } else {
      if (!global_cl_context->command_queue_inst_.enqueueSVMMap(
            vecYdata, dim * sizeof(float), true)) {
        ml_loge("Failed to unmap svm");
        break;
      }
    }

  } while (false);
}

#ifdef ENABLE_FP16
void GeGLULayerCl::geglu_cl_fp16(_FP16 *matAdata, _FP16 *vecXdata,
                                 _FP16 *vecYdata, unsigned int dim1,
                                 unsigned int dim2, bool svm,
                                 void *resident_out, bool skip_out_map,
                                 void *in1_clmem, void *in2_clmem,
                                 unsigned int row_off) {

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_geglu_ptr = getLayerKernelPtrs()[Kernels::GEGLU_CL_FP16];

    int dim = int(dim1 * dim2);

    if (!svm) {
      bool write_result = true;
      write_result &= clbuffInstance.getInBufferA()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(_FP16), matAdata);
      write_result &= clbuffInstance.getInBufferB()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(_FP16), vecXdata);
      if (!write_result) {
        break;
      }

      auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
      auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
      auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

      bool set_result = true;
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
      if (!set_result) {
        break;
      }
    } else {
      // Static cl_mem residency: a cl_mem-bound input (the gate/up FC wrote
      // the planner sub-buffer) needs NO SVM unmap -- its SVM shadow was never
      // written and unmapping it would flip SVM state one-sidedly.
      bool map_result = true;
      if (in1_clmem == nullptr)
        map_result &=
          global_cl_context->command_queue_inst_.enqueueSVMUnmap(matAdata);
      if (in2_clmem == nullptr)
        map_result &=
          global_cl_context->command_queue_inst_.enqueueSVMUnmap(vecXdata);
      if (!map_result) {
        ml_loge("Failed to map svm");
        break;
      }

      bool set_svm_result = true;
      if (in1_clmem != nullptr) {
        cl_mem in1_buf = static_cast<cl_mem>(in1_clmem);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(0, &in1_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(0, matAdata);
      }
      if (in2_clmem != nullptr) {
        cl_mem in2_buf = static_cast<cl_mem>(in2_clmem);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(1, &in2_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(1, vecXdata);
      }
      // [resident-act / static GPU_CLMEM] output direct to a cl_mem (the
      // tensor's planner sub-buffer): no SVM intermediate; else SVM output.
      if (resident_out != nullptr) {
        cl_mem out_buf = static_cast<cl_mem>(resident_out);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(2, &out_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(2, vecYdata);
      }
      if (!set_svm_result) {
        ml_loge("Failed to set svm");
        break;
      }
    }

    // Flat-index shift (arg 3): nonzero only for the all-cl_mem live-row decode
    // path, where every buffer is bound whole at index 0 and the kernel reads/
    // writes row (row_off/dim2). SVM/host paths carry the offset on their
    // pointer and pass 0. GWS below is dim1*dim2, so exactly dim1 rows run.
    if (!kernel_geglu_ptr->SetKernelArguments(3, &row_off, sizeof(int))) {
      ml_loge("Failed to set geglu row_off");
      break;
    }

    // NOTE(mwlasiuk) : local size can not be larger than global
    const int32_t desired_local = 64;
    const bool can_use_desired = dim >= desired_local;
    const int32_t chosen_local = can_use_desired ? desired_local : dim;

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {chosen_local, 1, 1}; // test-value

    if (!global_cl_context->command_queue_inst_.DispatchCommand(
          kernel_geglu_ptr, work_groups_count, work_group_size)) {
      ml_loge("Failed to run");
      break;
    }

    if (!svm) {
      if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
            global_cl_context->command_queue_inst_, dim * sizeof(_FP16),
            vecYdata)) {
        break;
      }
    } else if (resident_out == nullptr && !skip_out_map) {
      // async: geglu output is always consumed by the next GPU op (ffn_down
      // FC); in-order queue preserves ordering, no host read between.
      if (!global_cl_context->command_queue_inst_.enqueueSVMMap(
            vecYdata, dim * sizeof(_FP16), true, /*async=*/true)) {
        ml_loge("Failed to unmap svm");
        break;
      }
    }
    // NNTR_DEVRES Step 4: skip_out_map leaves vecYdata GPU-owned (unmapped) for
    // the device-resident ffn_down FC, which skips its matching unmap (the FC's
    // v8c_copy_svm_to_clmem) — the map/unmap PAIR is removed together (a one-
    // sided skip would crash on asymmetric SVM state). Coordinated via the
    // device_valid bit the producer set below.
    // resident_out: output is a cl_mem backing — no SVM map (consumer reads
    // it as cl_mem via the residency overlay).

  } while (false);
}
#endif

void GeGLULayerCl::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void GeGLULayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, geglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[GeGLULayerCl] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

std::vector<ClContext::SharedPtrClKernel> &GeGLULayerCl::getLayerKernelPtrs() {
  /**< kernel list relevant with this layer */
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} // namespace nntrainer
