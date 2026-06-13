// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding_injection.h
 * @date   12 June 2026
 * @brief  Replace placeholder-token embeddings with externally computed ones
 *         (e.g. audio/vision encoder outputs) — the masked_scatter step of
 *         multimodal HF models.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Inputs:
 *   0: token embeddings   [B, 1, S, D]   (output of the embedding layer)
 *   1: token ids          [B, 1, 1, S]   (the graph's input0, ids as float)
 *   2: side embeddings    [B, 1, S_side, D] (host-filled "input" layer)
 *
 * For each row i, if ids[i] == token_id the output row is copied from the
 * side input at the same GLOBAL position (from + i); otherwise the token
 * embedding passes through. The host therefore writes each encoder output
 * row at the exact position of its placeholder token.
 */

#ifndef __EMBEDDING_INJECTION_LAYER_H__
#define __EMBEDDING_INJECTION_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>
#include <vector>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief Embedding-injection layer for multimodal placeholder tokens
 */
WIN_EXPORT class EmbeddingInjectionLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT EmbeddingInjectionLayer() : Layer() {}

  WIN_EXPORT ~EmbeddingInjectionLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override {}

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void
  calcDerivative(nntrainer::RunLayerContext &context) override {}

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return EmbeddingInjectionLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   * @note supports "token_id=<id>[,<id>...]" (e.g. image and video tokens)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "embedding_injection";

private:
  std::vector<int> token_ids; /**< placeholder token ids to replace */
};

} // namespace causallm

#endif /* __EMBEDDING_INJECTION_LAYER_H__ */
