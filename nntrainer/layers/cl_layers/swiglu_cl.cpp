// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   swiglu_cl.cpp
 * @date   6th June 2024
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "swiglu_cl.h"
#include "nntrainer_log.h"
#include <cl_kernels/swiglu.h>
#ifdef ENABLE_FP16
#include <cl_kernels/swiglu_fp16.h>
#endif
#include <iostream>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

bool SwiGLULayerCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  // check if the kernels are already registered.
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for swiglu_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_swiglu_ptr = nullptr;

    kernel_swiglu_ptr = cl_context.registerClKernel(swiglu_kernel, "swiglu_cl");

    if (!kernel_swiglu_ptr) {
      ml_loge("OpenCL Error: Fail to register swiglu_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_swiglu_ptr);

#ifdef ENABLE_FP16
    kernel_swiglu_ptr =
      cl_context.registerClKernel(swiglu_fp16_kernel, "swiglu_cl_fp16");

    if (!kernel_swiglu_ptr) {
      ml_loge("OpenCL Error: Fail to register swiglu_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_swiglu_ptr);
#endif

    return true;
  } while (false);

  // clear all registered kernels if any error occurs during registration
  layer_kernel_ptrs.clear();

  return false;
}

void SwiGLULayerCl::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SwiGLULayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  swigluProcess(in1, in2, out, in1.batch() * in1.channel() * in1.height(), 0);
}

void SwiGLULayerCl::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
  }

  // [decode active-row fix] mirrors GeGLULayerCl: the current token at decode
  // lives at its ABSOLUTE row (to-1) of the full [INIT_SEQ,I] tensor, so process
  // only the live rows [from, to). Without this, swiglu re-ran the whole
  // [INIT_SEQ,I] tensor every decode step -- a ~1024x waste (264us/op on Qwen3
  // I=3072). SVM tensors offset the data pointers (O(1) at decode); cl_mem
  // tensors bind the whole buffer at index 0, so process [0, to) to cover the
  // live row (O(position) fallback).
  const bool any_clmem = in1.isClMem() || in2.isClMem() || out.isClMem();
  if (any_clmem)
    swigluProcess(in1, in2, out, to, 0);
  else
    swigluProcess(in1, in2, out, to - from, from);
}

void SwiGLULayerCl::swigluProcess(Tensor const &in1, Tensor const &in2,
                                  Tensor &result, unsigned int active_rows,
                                  unsigned int row_offset) {

  unsigned int dim1, dim2;
  dim1 = active_rows;
  dim2 = in1.width();

  // Element offset into the SVM pointers so the kernel processes rows
  // [row_offset, row_offset+active_rows). row_offset is 0 for the cl_mem path.
  const size_t elem_off = (size_t)row_offset * dim2;

  // SVM-direct only when the tensors are GPU-resident (SVM pool). On the
  // default host (cpu) pool getData() returns host pointers, so use the
  // host-bounce path; passing host pointers as SVM kernel arguments produces
  // garbage. When the graph opts into the SVM pool (NNTR_GPU_SVM_POOL), this
  // becomes a zero-copy SVM-direct dispatch (residency).
  // See tensor/cl_operations/GPU_GENERALIZATION_PLAN.md Step 3.
  const auto md = in1.getMemoryData();
  const bool use_svm = md && md->isSVM();

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data1 = in1.getData() + elem_off;
    float *data2 = in2.getData() + elem_off;
    float *rdata = result.getData() + elem_off;
    swiglu_cl(data1, data2, rdata, dim1, dim2, use_svm);
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data1 = in1.getData<_FP16>() + elem_off;
    _FP16 *data2 = in2.getData<_FP16>() + elem_off;
    _FP16 *rdata = result.getData<_FP16>() + elem_off;
    // Planner-decided STATIC cl_mem residency (mirrors GeGLULayerCl): under
    // NNTR_GPU_CLMEM_POOL the gate/up FC outputs and the swiglu output are
    // GPU_CLMEM, so getData() is NOT host-addressable -- bind the planner
    // cl_mem sub-buffers directly. Without this the host-bounce path reads the
    // cl_mem handle as a host pointer => the swiglu output is never written =>
    // the FFN contributes nothing to the residual (measured: qwen3 MLP drops
    // out, layer-0 output collapses to embed+attention).
    void *in1_cl = (use_svm && in1.isClMem()) ? in1.getClMem() : nullptr;
    void *in2_cl = (use_svm && in2.isClMem()) ? in2.getClMem() : nullptr;
    void *out_cl = (use_svm && result.isClMem()) ? result.getClMem() : nullptr;
    swiglu_cl_fp16(data1, data2, rdata, dim1, dim2, use_svm, out_cl,
                   /*skip_out_map=*/false, in1_cl, in2_cl);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void SwiGLULayerCl::swiglu_cl(float *matAdata, float *vecXdata, float *vecYdata,
                              unsigned int dim1, unsigned int dim2, bool svm) {
  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_swiglu_ptr = getLayerKernelPtrs()[Kernels::SWIGLU_CL];
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
        kernel_swiglu_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
      set_result &=
        kernel_swiglu_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
      set_result &=
        kernel_swiglu_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
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
      set_svm_result &= kernel_swiglu_ptr->SetKernelSVMArguments(0, matAdata);
      set_svm_result &= kernel_swiglu_ptr->SetKernelSVMArguments(1, vecXdata);
      set_svm_result &= kernel_swiglu_ptr->SetKernelSVMArguments(2, vecYdata);
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
          kernel_swiglu_ptr, work_groups_count, work_group_size)) {
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
void SwiGLULayerCl::swiglu_cl_fp16(_FP16 *matAdata, _FP16 *vecXdata,
                                   _FP16 *vecYdata, unsigned int dim1,
                                   unsigned int dim2, bool svm,
                                   void *resident_out, bool skip_out_map,
                                   void *in1_clmem, void *in2_clmem) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_swiglu_ptr =
      getLayerKernelPtrs()[Kernels::SWIGLU_CL_FP16];

    int dim = int(dim1 * dim2);

    if (!svm) {
      // Host (cpu) pool: bounce host data through the shared scratch buffers.
      result = clbuffInstance.getInBufferA()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(_FP16), matAdata);
      if (!result) {
        break;
      }
      result = clbuffInstance.getInBufferB()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(_FP16), vecXdata);
      if (!result) {
        break;
      }
      auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
      auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
      auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

      bool set_result = true;
      set_result &=
        kernel_swiglu_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
      set_result &=
        kernel_swiglu_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
      set_result &=
        kernel_swiglu_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
      if (!set_result) {
        break;
      }
    } else {
      // GPU-resident pool: bind cl_mem-resident operands directly and SVM
      // operands via SVM args (mirrors GeGLULayerCl). A cl_mem-bound input
      // (the gate/up FC wrote its planner sub-buffer) must NOT be SVM-unmapped
      // -- its SVM shadow was never written.
      bool map_result = true;
      if (in1_clmem == nullptr)
        map_result &=
          global_cl_context->command_queue_inst_.enqueueSVMUnmap(matAdata);
      if (in2_clmem == nullptr)
        map_result &=
          global_cl_context->command_queue_inst_.enqueueSVMUnmap(vecXdata);
      if (!map_result) {
        ml_loge("swiglu: failed to unmap svm");
        break;
      }

      bool set_svm_result = true;
      if (in1_clmem != nullptr) {
        cl_mem in1_buf = static_cast<cl_mem>(in1_clmem);
        set_svm_result &=
          kernel_swiglu_ptr->SetKernelArguments(0, &in1_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_swiglu_ptr->SetKernelSVMArguments(0, matAdata);
      }
      if (in2_clmem != nullptr) {
        cl_mem in2_buf = static_cast<cl_mem>(in2_clmem);
        set_svm_result &=
          kernel_swiglu_ptr->SetKernelArguments(1, &in2_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_swiglu_ptr->SetKernelSVMArguments(1, vecXdata);
      }
      if (resident_out != nullptr) {
        cl_mem out_buf = static_cast<cl_mem>(resident_out);
        set_svm_result &=
          kernel_swiglu_ptr->SetKernelArguments(2, &out_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_swiglu_ptr->SetKernelSVMArguments(2, vecYdata);
      }
      if (!set_svm_result) {
        ml_loge("swiglu: failed to set svm/clmem args");
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

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_swiglu_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    if (!svm) {
      result = clbuffInstance.getOutBufferA()->ReadDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(_FP16), vecYdata);
      if (!result) {
        break;
      }
    } else if (resident_out == nullptr && !skip_out_map) {
      // async: swiglu output is consumed by the next GPU op (ffn_down FC); the
      // in-order queue preserves ordering, no host read between.
      if (!global_cl_context->command_queue_inst_.enqueueSVMMap(
            vecYdata, dim * sizeof(_FP16), true, /*async=*/true)) {
        ml_loge("swiglu: failed to map svm output");
        break;
      }
    }

  } while (false);
}
#endif

void SwiGLULayerCl::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void SwiGLULayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[SwigluLayerCl] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

std::vector<ClContext::SharedPtrClKernel> &SwiGLULayerCl::getLayerKernelPtrs() {
  /**< kernel list relevant with this layer */
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} // namespace nntrainer
