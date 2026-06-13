// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding_injection.cpp
 * @date   12 June 2026
 * @brief  Replace placeholder-token embeddings with externally computed ones.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstring>

#include "embedding_injection.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t EMBD_IDX = 0;
static constexpr size_t IDS_IDX = 1;
static constexpr size_t SIDE_IDX = 2;

void EmbeddingInjectionLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "embedding_injection expects 3 inputs (embeddings, ids, side)";
  NNTR_THROW_IF(token_ids.empty(), std::invalid_argument)
    << "embedding_injection requires the token_id property";

  const auto &in_dims = context.getInputDimensions();
  NNTR_THROW_IF(in_dims[EMBD_IDX].width() != in_dims[SIDE_IDX].width(),
                std::invalid_argument)
    << "embedding/side width mismatch: " << in_dims[EMBD_IDX].width() << " vs "
    << in_dims[SIDE_IDX].width();

  context.setOutputDimensions({in_dims[EMBD_IDX]});
}

void EmbeddingInjectionLayer::setProperty(
  const std::vector<std::string> &values) {
  for (const auto &value : values) {
    const auto pos = value.find('=');
    NNTR_THROW_IF(pos == std::string::npos, std::invalid_argument)
      << "embedding_injection: invalid property format: " << value;
    const std::string key = value.substr(0, pos);
    if (key == "token_id") {
      token_ids.clear();
      std::string list = value.substr(pos + 1);
      size_t start = 0;
      while (start < list.size()) {
        size_t comma = list.find(',', start);
        if (comma == std::string::npos)
          comma = list.size();
        token_ids.push_back(std::stoi(list.substr(start, comma - start)));
        start = comma + 1;
      }
    } else {
      NNTR_THROW_IF(true, std::invalid_argument)
        << "embedding_injection: unknown property: " << key;
    }
  }
}

void EmbeddingInjectionLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &embd = context.getInput(EMBD_IDX);
  nntrainer::Tensor &ids = context.getInput(IDS_IDX);
  nntrainer::Tensor &side = context.getInput(SIDE_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(embd.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "embedding_injection supports FP32 activations only";

  const unsigned int iter = to - from;
  const unsigned int dim = embd.width();
  const unsigned int side_rows = side.height();

  for (unsigned int b = 0; b < embd.batch(); ++b) {
    const float *ids_data = ids.getData<float>() + ids.getIndex(b, 0, 0, 0);
    for (unsigned int i = 0; i < iter; ++i) {
      const unsigned int global = from + i;
      const int id = static_cast<int>(std::lround(ids_data[i]));
      const bool is_placeholder =
        std::find(token_ids.begin(), token_ids.end(), id) != token_ids.end();

      float *dst = out.getData<float>() + out.getIndex(b, 0, i, 0);
      const float *src;
      if (is_placeholder && global < side_rows) {
        src = side.getData<float>() + side.getIndex(b, 0, global, 0);
      } else {
        src = embd.getData<float>() + embd.getIndex(b, 0, i, 0);
      }
      std::memcpy(dst, src, dim * sizeof(float));
    }
  }
}

void EmbeddingInjectionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim embd_dim = context.getInput(EMBD_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(OUT_IDX).getDim();

  embd_dim.height(input_dimensions[0].height());
  out_dim.height(input_dimensions[0].height());

  context.updateInput(EMBD_IDX, embd_dim);
  context.updateOutput(OUT_IDX, out_dim);
}

} // namespace causallm
