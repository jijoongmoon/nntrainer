// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vision_rope.cpp
 * @date   13 June 2026
 * @brief  Qwen2.5 vision 2D rotary position embedding (applied to q or k).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <stdexcept>

#include "vision_rope.h"

namespace causallm {

static constexpr size_t IN_IDX = 0;
static constexpr size_t OUT_IDX = 0;

void VisionRopeLayer::buildTables() {
  const unsigned int m = spatial_merge_size;
  const unsigned int frame = grid_h * grid_w;       // patches per frame
  const unsigned int seq = grid_t * frame;          // all temporal frames
  const unsigned int rot = head_dim / 2;            // 40 for head_dim 80
  const unsigned int half = rot / 2;                // 20 freqs per axis

  // inv_freq over the rotary half: 1/theta^(2i/rot)
  std::vector<float> inv_freq(half);
  for (unsigned int i = 0; i < half; ++i)
    inv_freq[i] =
      1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / rot);

  cos_tbl.assign(static_cast<size_t>(seq) * head_dim, 0.0f);
  sin_tbl.assign(static_cast<size_t>(seq) * head_dim, 0.0f);

  // The encoder rope is purely spatial (h, w); positions repeat per temporal
  // frame (matches rot_pos_emb's pos_ids.repeat(t, 1)). Build the per-frame
  // spatial pattern in merge-blocked order [h/m, w/m, m, m], then replicate.
  unsigned int sp = 0;
  for (unsigned int bh = 0; bh < grid_h / m; ++bh)
    for (unsigned int bw = 0; bw < grid_w / m; ++bw)
      for (unsigned int ih = 0; ih < m; ++ih)
        for (unsigned int iw = 0; iw < m; ++iw) {
          const unsigned int hpos = bh * m + ih;
          const unsigned int wpos = bw * m + iw;
          for (unsigned int i = 0; i < half; ++i) {
            const float fh = hpos * inv_freq[i];
            const float fw = wpos * inv_freq[i];
            const float ch = std::cos(fh), sh = std::sin(fh);
            const float cw = std::cos(fw), sw = std::sin(fw);
            for (unsigned int t = 0; t < grid_t; ++t) {
              const size_t row =
                (static_cast<size_t>(t) * frame + sp) * head_dim;
              cos_tbl[row + i] = ch;
              cos_tbl[row + half + i] = cw;
              cos_tbl[row + rot + i] = ch; // duplicate (second head_dim half)
              cos_tbl[row + rot + half + i] = cw;
              sin_tbl[row + i] = sh;
              sin_tbl[row + half + i] = sw;
              sin_tbl[row + rot + i] = sh;
              sin_tbl[row + rot + half + i] = sw;
            }
          }
          ++sp;
        }
}

void VisionRopeLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(grid_h == 0 || grid_w == 0 || head_dim == 0 || num_heads == 0,
                std::invalid_argument)
    << "vision_rope requires grid_h, grid_w, num_heads, head_dim";
  NNTR_THROW_IF(head_dim % 4 != 0, std::invalid_argument)
    << "vision_rope head_dim must be divisible by 4";
  buildTables();
  context.setOutputDimensions({context.getInputDimensions()[IN_IDX]});
}

void VisionRopeLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &value : values) {
    const auto pos = value.find('=');
    NNTR_THROW_IF(pos == std::string::npos, std::invalid_argument)
      << "vision_rope: invalid property: " << value;
    const std::string key = value.substr(0, pos);
    const std::string val = value.substr(pos + 1);
    if (key == "grid_h")
      grid_h = std::stoul(val);
    else if (key == "grid_w")
      grid_w = std::stoul(val);
    else if (key == "grid_t")
      grid_t = std::stoul(val);
    else if (key == "num_heads")
      num_heads = std::stoul(val);
    else if (key == "head_dim")
      head_dim = std::stoul(val);
    else if (key == "spatial_merge_size")
      spatial_merge_size = std::stoul(val);
    else if (key == "rope_theta")
      rope_theta = std::stof(val);
    else
      NNTR_THROW_IF(true, std::invalid_argument)
        << "vision_rope: unknown property: " << key;
  }
}

void VisionRopeLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &in = context.getInput(IN_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "vision_rope supports FP32 only";

  const unsigned int hd = head_dim;
  const unsigned int half = hd / 2;
  const unsigned int iter = to - from;

  for (unsigned int b = 0; b < in.batch(); ++b) {
    for (unsigned int s = 0; s < iter; ++s) {
      const unsigned int gpos = from + s;
      const float *crow = &cos_tbl[static_cast<size_t>(gpos) * hd];
      const float *srow = &sin_tbl[static_cast<size_t>(gpos) * hd];
      float *xrow = in.getData<float>() + in.getIndex(b, 0, s, 0);
      float *orow = out.getData<float>() + out.getIndex(b, 0, s, 0);
      for (unsigned int h = 0; h < num_heads; ++h) {
        float *x = xrow + static_cast<size_t>(h) * hd;
        float *o = orow + static_cast<size_t>(h) * hd;
        for (unsigned int i = 0; i < hd; ++i) {
          // rotate_half: first half <- -x[i+half], second half <- x[i-half]
          const float rh = (i < half) ? -x[i + half] : x[i - half];
          o[i] = x[i] * crow[i] + rh * srow[i];
        }
      }
    }
  }
}

void VisionRopeLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim in_dim = context.getInput(IN_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(OUT_IDX).getDim();
  in_dim.height(input_dimensions[0].height());
  out_dim.height(input_dimensions[0].height());
  context.updateInput(IN_IDX, in_dim);
  context.updateOutput(OUT_IDX, out_dim);
}

} // namespace causallm
