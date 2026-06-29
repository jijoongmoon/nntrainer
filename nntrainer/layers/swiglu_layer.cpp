// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   swiglu_layer.cpp
 * @date   29 June 2026
 * @brief  Backend-neutral SwiGLU activation: out = silu(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "swiglu_layer.h"

#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; // gate
static constexpr size_t INPUT_IDX_2 = 1; // up

void SwiGLULayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SwiGLULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[SwiGLULayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void SwiGLULayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  in1.getOps()->swiglu(in1, in2, out,
                       in1.batch() * in1.channel() * in1.height(),
                       /*row_offset=*/0);
}

void SwiGLULayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
  }

  // active-row decision (unifies the former SwiGLULayerCl branches; mirrors
  // GeGLULayer except the SVM/host path offsets the pointer to row `from`):
  //  - all-cl_mem fp16 decode: the producers write the live token to row 0, so
  //    process exactly 1 row at the buffer base (O(1); also avoids the
  //    one-row-out-of-bounds cl_mem write the old [0,to) branch could trigger).
  //  - any other cl_mem (mixed / fp32): process the whole [0,to) window.
  //  - SVM/host: process rows [from, to) by offsetting the pointer to `from`.
  const bool any_clmem = in1.isClMem() || in2.isClMem() || out.isClMem();
  const bool all_clmem = in1.isClMem() && in2.isClMem() && out.isClMem();
  const bool is_fp16 =
    in1.getDataType() == ml::train::TensorDim::DataType::FP16;

  unsigned int active_rows, row_offset;
  if (from && all_clmem && is_fp16) {
    active_rows = 1;
    row_offset = 0;
  } else if (any_clmem) {
    active_rows = to;
    row_offset = 0;
  } else {
    active_rows = to - from;
    row_offset = from;
  }

  in1.getOps()->swiglu(in1, in2, out, active_rows, row_offset);
}

void SwiGLULayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
