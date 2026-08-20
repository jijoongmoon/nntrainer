// SPDX-License-Identifier: Apache-2.0
/**
 * @file   broadcast_mul_layer.cpp
 * @date   2 July 2026
 * @brief  Row-broadcast multiply: out[b,c,h,:] = in0[b,c,h,:] * in1[b,c,h,0]
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude Code
 * @bug    No known bugs except for NYI items
 */

#include "broadcast_mul_layer.h"

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_A = 0;
static constexpr size_t INPUT_IDX_G = 1;

void BroadcastMulLayer::finalize(nntrainer::InitLayerContext &context) {
  auto dims = context.getInputDimensions();
  NNTR_THROW_IF(dims.size() != 2, std::invalid_argument)
    << "[broadcast_mul] needs exactly 2 inputs (a, gate)";
  NNTR_THROW_IF(dims[INPUT_IDX_G].width() != 1, std::invalid_argument)
    << "[broadcast_mul] gate input width must be 1, got " +
         std::to_string(dims[INPUT_IDX_G].width());
  context.setOutputDimensions({dims[INPUT_IDX_A]});
}

template <typename T>
static void bcast_mul_rows(nntrainer::Tensor &a, nntrainer::Tensor &g,
                           nntrainer::Tensor &out, unsigned int rows) {
  const unsigned int W = a.width();
  for (unsigned int b = 0; b < a.batch(); ++b) {
    for (unsigned int c = 0; c < a.channel(); ++c) {
      for (unsigned int h = 0; h < rows; ++h) {
        const T *ap = a.getData<T>() + a.getIndex(b, c, h, 0);
        const float gv =
          static_cast<float>(g.getData<T>()[g.getIndex(b, c, h, 0)]);
        T *op = out.getData<T>() + out.getIndex(b, c, h, 0);
        for (unsigned int w = 0; w < W; ++w)
          op[w] = static_cast<T>(static_cast<float>(ap[w]) * gv);
      }
    }
  }
}

static void bcast_mul_dispatch(nntrainer::Tensor &a, nntrainer::Tensor &g,
                               nntrainer::Tensor &out, unsigned int rows) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#ifdef ENABLE_FP16
  // Device path: one stream-ordered kernel. The host loop below needs a full
  // stream drain in front of it (40 nodes/chunk = 200 drains per 20K prefill)
  // and then multiplies the whole chunk on one core; both go away here.
  // Identical arithmetic: fp32 multiply, one fp16 rounding.
  static const bool g_ew_dev = []() {
    const char *e = std::getenv("NNTR_EW_DEV");
    return e == nullptr || e[0] != '0';
  }();
  // Capture override: see multiply_layer.cpp — a host fallback inside a
  // CUDA-graph capture reads un-executed producers and drops this node from
  // the graph (the 35B M2B decode corruption, 2026-08-21).
  const bool in_capture =
    nntrainer::cuda::StreamManager::Global().isCapturing();
  if (g_ew_dev && ((size_t)rows * a.width() >= 32768 || in_capture) &&
      a.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      a.batch() == 1 && a.channel() == 1) {
    const auto *ap =
      reinterpret_cast<const unsigned short *>(a.getData<_FP16>());
    const auto *gp =
      reinterpret_cast<const unsigned short *>(g.getData<_FP16>());
    auto *op = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    if (nntrainer::cuda::dev_accessible(ap) &&
        nntrainer::cuda::dev_accessible(gp) &&
        nntrainer::cuda::dev_accessible(op) &&
        nntrainer::cuda::cuda_bcast_mul_fp16(ap, gp, op, rows * a.width(),
                                             a.width()))
      return;
  }
#endif
  // Host fallback: may read a GPU-produced UVM input; sync first in async
  // mode (no-op in default sync mode).
  nntrainer::cuda::StreamManager::Global().finishIfAsync();
#endif
  if (a.getDataType() == ml::train::TensorDim::DataType::FP32) {
    bcast_mul_rows<float>(a, g, out, rows);
  } else if (a.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    bcast_mul_rows<_FP16>(a, g, out, rows);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  } else {
    NNTR_THROW_IF(true, std::invalid_argument)
      << "[broadcast_mul] unsupported data type";
  }
}

void BroadcastMulLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  nntrainer::Tensor &a = context.getInput(INPUT_IDX_A);
  nntrainer::Tensor &g = context.getInput(INPUT_IDX_G);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);
  bcast_mul_dispatch(a, g, out, a.height());
}

void BroadcastMulLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &a = context.getInput(INPUT_IDX_A);
  nntrainer::Tensor &g = context.getInput(INPUT_IDX_G);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);
  bcast_mul_dispatch(a, g, out, to - from);
}

void BroadcastMulLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim a_dim = context.getInput(INPUT_IDX_A).getDim();
  ml::train::TensorDim g_dim = context.getInput(INPUT_IDX_G).getDim();
  ml::train::TensorDim out_dim = context.getOutput(OUT_IDX).getDim();

  a_dim.height(input_dimensions[0].height());
  g_dim.height(input_dimensions[0].height());
  out_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_A, a_dim);
  context.updateInput(INPUT_IDX_G, g_dim);
  context.updateOutput(OUT_IDX, out_dim);
}

void BroadcastMulLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace causallm
