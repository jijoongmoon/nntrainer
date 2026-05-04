// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.h
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __EMBEDDING_LAYER_H__
#define __EMBEDDING_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace causallm {

namespace props {

/**
 * @brief Path to a JSON manifest describing a tensorwise 4-bit quantized
 *        embedding LUT. When set, the embedding layer skips the standard
 *        FP weight allocation and instead loads the packed nibble table
 *        described by the manifest, dequantizing per-token at forward time.
 *
 * Manifest schema:
 *   {
 *     "version":   1,
 *     "type":      "lut",
 *     "lut-path":  "<binary path, relative to manifest dir>",
 *     "size":      <out_dim>,
 *     "datatype":  "ufixed8",   // 4-bit values, 2 packed per uint8 byte
 *     "quant-param": { "scale": <float>, "offset": <int> }
 *   }
 */
class QuantizedLutPath final : public nntrainer::Property<std::string> {
public:
  QuantizedLutPath() = default;
  QuantizedLutPath(const std::string &v) { set(v); }
  static constexpr const char *key = "quantized_lut_path";
  using prop_tag = nntrainer::str_prop_tag;
};

} // namespace props

/**
 * @brief Tensorwise 4-bit quantized embedding LUT shared across networks.
 *        `packed` holds nibble-packed bytes (low nibble first) of length
 *        ceil(in_dim * out_dim / 2). Lives in a path-keyed weak cache so
 *        two graphs that load the same manifest see the same in-memory
 *        table.
 */
struct QuantLut {
  std::vector<uint8_t> packed;
  float scale = 1.0f;
  int offset = 0;
  size_t in_dim = 0;  ///< vocab size (rows)
  size_t out_dim = 0; ///< per-token feature dim (cols)
};

std::shared_ptr<QuantLut>
get_or_load_quant_lut(const std::string &manifest_path);


/**
 * @class   EmbeddingLayer
 * @brief   EmbeddingLayer
 * @todo    Support setBatch for EmbeddingLayer
 */
WIN_EXPORT class EmbeddingLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT EmbeddingLayer();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~EmbeddingLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] EmbeddingLayer &&
   */
  WIN_EXPORT EmbeddingLayer(EmbeddingLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs EmbeddingLayer to be moved.
   */
  WIN_EXPORT EmbeddingLayer &operator=(EmbeddingLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return EmbeddingLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "embedding_layer";

private:
  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             nntrainer::props::Scale, props::QuantizedLutPath>
    embedding_props;
  unsigned int weight_idx;

  // Tensorwise 4-bit LUT mode. Populated in finalize() when
  // QuantizedLutPath property is set; kept alive for the layer's lifetime
  // so the data stays valid across forward calls. The shared_ptr also
  // ensures cross-graph sharing via the path-keyed cache.
  std::shared_ptr<QuantLut> quant_lut_;
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __EMBEDDING_H__ */
