// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vision_attention.cpp
 * @date   13 June 2026
 * @brief  Non-causal windowed/full attention for the Qwen2.5 vision tower.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vision_attention.h"

namespace causallm {

static constexpr size_t Q_IDX = 0;
static constexpr size_t K_IDX = 1;
static constexpr size_t V_IDX = 2;
static constexpr size_t OUT_IDX = 0;

void VisionAttentionLayer::buildWindowIds() {
  const unsigned int merge = spatial_merge_size;
  const unsigned int frame = grid_h * grid_w;     // patches per frame
  const unsigned int seq = grid_t * frame;
  window_id.assign(seq, 0);

  // Attention never crosses temporal frames: HF builds cu_seqlens per frame
  // even for the full-attention layers. So full layers attend within a frame
  // (window = frame index); windowed layers further split each frame into
  // spatial windows.
  const unsigned int llm_h = grid_h / merge, llm_w = grid_w / merge;
  const unsigned int merged_per_frame = llm_h * llm_w;
  const unsigned int vmws = window_size / merge / patch_size; // 4
  const unsigned int num_win_h = (llm_h + vmws - 1) / vmws;
  const unsigned int num_win_w = (llm_w + vmws - 1) / vmws;
  const unsigned int win_per_frame = num_win_h * num_win_w;

  for (unsigned int p = 0; p < seq; ++p) {
    const unsigned int g = p / (merge * merge); // global merged unit
    const unsigned int t = g / merged_per_frame;
    if (is_full) {
      window_id[p] = static_cast<int>(t); // per-frame full attention
      continue;
    }
    const unsigned int sg = g % merged_per_frame;
    const unsigned int mh = sg / llm_w;
    const unsigned int mw = sg % llm_w;
    window_id[p] = static_cast<int>(
      t * win_per_frame + (mh / vmws) * num_win_w + (mw / vmws));
  }
}

void VisionAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "vision_attention expects 3 inputs (q, k, v)";
  NNTR_THROW_IF(num_heads == 0 || head_dim == 0 || grid_h == 0 || grid_w == 0,
                std::invalid_argument)
    << "vision_attention requires num_heads, head_dim, grid_h, grid_w";
  buildWindowIds();
  // output has q's shape
  context.setOutputDimensions({context.getInputDimensions()[Q_IDX]});
}

void VisionAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  for (const auto &value : values) {
    const auto pos = value.find('=');
    NNTR_THROW_IF(pos == std::string::npos, std::invalid_argument)
      << "vision_attention: invalid property: " << value;
    const std::string key = value.substr(0, pos);
    const std::string val = value.substr(pos + 1);
    if (key == "num_heads")
      num_heads = std::stoul(val);
    else if (key == "head_dim")
      head_dim = std::stoul(val);
    else if (key == "grid_h")
      grid_h = std::stoul(val);
    else if (key == "grid_w")
      grid_w = std::stoul(val);
    else if (key == "grid_t")
      grid_t = std::stoul(val);
    else if (key == "window_size")
      window_size = std::stoul(val);
    else if (key == "patch_size")
      patch_size = std::stoul(val);
    else if (key == "spatial_merge_size")
      spatial_merge_size = std::stoul(val);
    else if (key == "is_full")
      is_full = (val == "true" || val == "1");
    else
      NNTR_THROW_IF(true, std::invalid_argument)
        << "vision_attention: unknown property: " << key;
  }
}

void VisionAttentionLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &q = context.getInput(Q_IDX);
  nntrainer::Tensor &k = context.getInput(K_IDX);
  nntrainer::Tensor &v = context.getInput(V_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(q.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "vision_attention supports FP32 only";

  const unsigned int hd = head_dim;
  const unsigned int seq = to - from; // encoder: full sequence (from=0)
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  std::vector<float> scores(seq);
  for (unsigned int b = 0; b < q.batch(); ++b) {
    for (unsigned int h = 0; h < num_heads; ++h) {
      const size_t hoff = static_cast<size_t>(h) * hd;
      for (unsigned int i = 0; i < seq; ++i) {
        const float *qi = q.getData<float>() + q.getIndex(b, 0, i, 0) + hoff;
        // window_id encodes the attention grouping for both layer kinds:
        // full -> per temporal frame, windowed -> per spatial window of frame.
        // For images (grid_t=1) full layers put every token in window 0.
        const int wi = window_id[from + i];

        float maxs = -std::numeric_limits<float>::infinity();
        for (unsigned int j = 0; j < seq; ++j) {
          if (window_id[from + j] != wi) {
            scores[j] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const float *kj = k.getData<float>() + k.getIndex(b, 0, j, 0) + hoff;
          float dot = 0.0f;
          for (unsigned int d = 0; d < hd; ++d)
            dot += qi[d] * kj[d];
          dot *= scale;
          scores[j] = dot;
          if (dot > maxs)
            maxs = dot;
        }

        float denom = 0.0f;
        for (unsigned int j = 0; j < seq; ++j) {
          if (scores[j] == -std::numeric_limits<float>::infinity()) {
            scores[j] = 0.0f;
            continue;
          }
          scores[j] = std::exp(scores[j] - maxs);
          denom += scores[j];
        }
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;

        float *oi = out.getData<float>() + out.getIndex(b, 0, i, 0) + hoff;
        for (unsigned int d = 0; d < hd; ++d)
          oi[d] = 0.0f;
        for (unsigned int j = 0; j < seq; ++j) {
          const float w = scores[j] * inv;
          if (w == 0.0f)
            continue;
          const float *vj = v.getData<float>() + v.getIndex(b, 0, j, 0) + hoff;
          for (unsigned int d = 0; d < hd; ++d)
            oi[d] += w * vj[d];
        }
      }
    }
  }
}

void VisionAttentionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim q_dim = context.getInput(Q_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(OUT_IDX).getDim();
  q_dim.height(input_dimensions[0].height());
  out_dim.height(input_dimensions[0].height());
  context.updateInput(Q_IDX, q_dim);
  context.updateOutput(OUT_IDX, out_dim);
}

} // namespace causallm
