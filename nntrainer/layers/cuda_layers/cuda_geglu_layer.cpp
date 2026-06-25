// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_geglu_layer.cpp
 * @date   22 Jun 2026
 * @brief  Host-on-UVM GeGLU for the CUDA backend.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cuda_geglu_layer.h>

#include <cmath>
#include <cstdlib>

#include <cuda_stream_manager.h>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#endif

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; // gate
static constexpr size_t INPUT_IDX_2 = 1; // up

// gelu (tanh approximation, gelu_pytorch_tanh) -- same constants as geglu_cl.
static inline float gelu_tanh(float x) {
  const float k = 0.7978845608028654f; // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

void CudaGeGLULayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(geglu_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(geglu_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void CudaGeGLULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, geglu_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[CudaGeGLULayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

void CudaGeGLULayer::gegluProcess(const Tensor &in1, const Tensor &in2,
                                  Tensor &out, unsigned int rows) {
  const unsigned int dim2 = in1.width();
  const size_t n = (size_t)rows * dim2;
  const auto dt = in1.getDataType();

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  // GPU geglu (device-resident fp16): one kernel instead of the host loop, so
  // the FFN/PLE activation stays on the device. Opt-in (NNTR_CUDA_GEGLU) until
  // the whole decode chain is on-GPU. NNTR_CUDA_ASYNC governs the drain.
  if (dt == ml::train::TensorDim::DataType::FP16) {
    static const bool gpu = std::getenv("NNTR_CUDA_GEGLU") != nullptr;
    if (gpu && n > 0) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>());
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>());
      auto *o = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
      const bool dev = nntrainer::cuda::dev_accessible(a);
      if (dev && cuda::cuda_geglu_fp16(a, b, o, (unsigned int)n))
        return;
    }
  }
#endif

  // Host gelu fallback: sync first so the host read of GPU-produced gate/up is
  // coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
  cuda::StreamManager::Global().finishIfAsync();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>();
    const float *b = in2.getData<float>();
    float *o = out.getData<float>();
    for (size_t i = 0; i < n; ++i)
      o[i] = gelu_tanh(a[i]) * b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>();
    const _FP16 *b = in2.getData<_FP16>();
    _FP16 *o = out.getData<_FP16>();
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(gelu_tanh((float)a[i]) * (float)b[i]);
#endif
  } else {
    throw std::invalid_argument("CudaGeGLULayer: unsupported data type");
  }
}

void CudaGeGLULayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  gegluProcess(in1, in2, out, in1.batch() * in1.channel() * in1.height());
}

void CudaGeGLULayer::incremental_forwarding(RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  if (skip_prefill && from == 0)
    return;
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
  }
  // mirror GeGLULayerCl's host/non-cl_mem path: the live token lives at row 0
  // (the producers write row 0 at decode), so process (to-from) rows from base.
  gegluProcess(in1, in2, out, to - from);
}

void CudaGeGLULayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
