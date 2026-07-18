// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_attention.cpp
 * @date   18 July 2026
 * @brief  Block-diagonal non-causal attention for the Token2Wav DiT.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#endif

#include "dit_attention.h"

namespace causallm {

static constexpr size_t Q_IDX = 0;
static constexpr size_t K_IDX = 1;
static constexpr size_t V_IDX = 2;
static constexpr size_t OUT_IDX = 0;

void DiTAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "dit_attention expects 3 inputs (q, k, v)";
  NNTR_THROW_IF(num_heads == 0 || head_dim == 0 || block_size == 0,
                std::invalid_argument)
    << "dit_attention requires num_heads, head_dim, block_size";
  const auto &in = context.getInputDimensions();
  NNTR_THROW_IF(in[Q_IDX].width() != num_heads * head_dim,
                std::invalid_argument)
    << "dit_attention: q width " << in[Q_IDX].width()
    << " != num_heads*head_dim " << num_heads * head_dim;
  // output has q's shape
  context.setOutputDimensions({in[Q_IDX]});
}

void DiTAttentionLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &value : values) {
    const auto pos = value.find('=');
    NNTR_THROW_IF(pos == std::string::npos, std::invalid_argument)
      << "dit_attention: invalid property: " << value;
    const std::string key = value.substr(0, pos);
    const std::string val = value.substr(pos + 1);
    if (key == "num_heads")
      num_heads = std::stoul(val);
    else if (key == "head_dim")
      head_dim = std::stoul(val);
    else if (key == "block_size")
      block_size = std::stoul(val);
    else if (key == "look_ahead")
      look_ahead = std::stoul(val);
    else if (key == "look_backward")
      look_backward = std::stoul(val);
    else
      NNTR_THROW_IF(true, std::invalid_argument)
        << "dit_attention: unknown property: " << key;
  }
}

void DiTAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // producer FCs may be in-flight cuBLAS kernels under NNTR_CUDA_ASYNC
  nntrainer::cuda::drain_if_async();
#endif

  nntrainer::Tensor &q = context.getInput(Q_IDX);
  nntrainer::Tensor &k = context.getInput(K_IDX);
  nntrainer::Tensor &v = context.getInput(V_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(q.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "dit_attention supports FP32 only";

  const unsigned int hd = head_dim;
  const unsigned int seq = q.height();
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const int la = static_cast<int>(look_ahead);
  const int lb = static_cast<int>(look_backward);

  for (unsigned int b = 0; b < q.batch(); ++b) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int h = 0; h < num_heads; ++h) {
      std::vector<float> scores(seq);
      const size_t hoff = static_cast<size_t>(h) * hd;
      for (unsigned int i = 0; i < seq; ++i) {
        const float *qi = q.getData<float>() + q.getIndex(b, 0, i, 0) + hoff;
        const int bi = static_cast<int>(i / block_size);

        float maxs = -std::numeric_limits<float>::infinity();
        for (unsigned int j = 0; j < seq; ++j) {
          // keep(i,j) iff block_id[j]-block_id[i] in [-look_backward,
          // look_ahead] (modeling:3039-3040); same mask for every head.
          const int diff = static_cast<int>(j / block_size) - bi;
          if (diff < -lb || diff > la) {
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

} // namespace causallm
