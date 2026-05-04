// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.cpp
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This embedding layer supports FP32/FP16/Q6_K data type only.
 */

#include <embedding_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "json.hpp"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

namespace {

// Path-keyed cache so two graphs (or two layers) that reference the same
// manifest share a single in-memory copy of the 4-bit LUT.
std::mutex                                                  g_lut_cache_mtx;
std::unordered_map<std::string, std::weak_ptr<QuantLut>>    g_lut_cache;

std::string dirname(const std::string &p) {
  auto pos = p.find_last_of('/');
  return (pos == std::string::npos) ? std::string() : p.substr(0, pos);
}

std::string resolve_relative(const std::string &path,
                             const std::string &base_dir) {
  if (path.empty() || path[0] == '/' || base_dir.empty())
    return path;
  return base_dir + "/" + path;
}

inline uint16_t clamp_u16(float v) {
  return static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, v)));
}

} // namespace

std::shared_ptr<QuantLut>
get_or_load_quant_lut(const std::string &manifest_path) {
  std::lock_guard<std::mutex> lk(g_lut_cache_mtx);

  auto it = g_lut_cache.find(manifest_path);
  if (it != g_lut_cache.end()) {
    if (auto sp = it->second.lock())
      return sp;
    g_lut_cache.erase(it);
  }

  std::ifstream f(manifest_path);
  NNTR_THROW_IF(!f.is_open(), std::runtime_error)
      << "Failed to open LUT manifest: " << manifest_path;

  nlohmann::json j;
  f >> j;

  const std::string lut_rel  = j.at("lut-path").get<std::string>();
  const int per_row          = j.at("size").get<int>();
  const std::string datatype = j.value("datatype", std::string("ufixed8"));
  const auto &qp             = j.at("quant-param");

  // Per the manifest convention used in this project, "ufixed8" means
  // two 4-bit values packed into one byte (NOT a single 8-bit code).
  NNTR_THROW_IF(datatype != "ufixed8", std::runtime_error)
      << "Only 'ufixed8' (4-bit packed in 8-bit) is supported, got: "
      << datatype;

  auto lut    = std::make_shared<QuantLut>();
  lut->scale  = qp.at("scale").get<float>();
  lut->offset = qp.at("offset").get<int>();
  lut->out_dim = static_cast<size_t>(per_row);

  const std::string lut_abs = resolve_relative(lut_rel, dirname(manifest_path));

  std::ifstream bin(lut_abs, std::ios::binary | std::ios::ate);
  NNTR_THROW_IF(!bin.is_open(), std::runtime_error)
      << "Failed to open LUT binary: " << lut_abs;
  const std::streamsize sz = bin.tellg();
  bin.seekg(0, std::ios::beg);
  lut->packed.resize(static_cast<size_t>(sz));
  bin.read(reinterpret_cast<char *>(lut->packed.data()), sz);

  // 4-bit packed: 2 elements per byte. Total nibbles = 2 * file_bytes.
  // in_dim = total_nibbles / out_dim.
  NNTR_THROW_IF(lut->out_dim == 0 || (2 * lut->packed.size()) % lut->out_dim,
                std::runtime_error)
      << "LUT binary size " << lut->packed.size()
      << " is not consistent with out_dim=" << lut->out_dim;
  lut->in_dim = (2 * lut->packed.size()) / lut->out_dim;

  ml_logi("Loaded shared 4-bit LUT '%s' (in_dim=%zu, out_dim=%zu, scale=%f, "
          "offset=%d, bytes=%zu)",
          manifest_path.c_str(), lut->in_dim, lut->out_dim, lut->scale,
          lut->offset, lut->packed.size());

  g_lut_cache[manifest_path] = lut;
  return lut;
}

EmbeddingLayer::EmbeddingLayer() :
  LayerImpl(),
  embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                  nntrainer::props::Scale(), props::QuantizedLutPath()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void EmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  // Accept FP32 by default (legacy path) and additionally UINT16/UINT32
  // for QNN-style pipelines where the activation dtype (and thus the
  // input layer feeding token IDs) is integer. Token IDs are read in
  // forwarding using the actual input dtype.
  const auto in_dtype = input_dim.getDataType();
  NNTR_THROW_IF(in_dtype != nntrainer::TensorDim::DataType::FP32 &&
                  in_dtype != nntrainer::TensorDim::DataType::UINT16 &&
                  in_dtype != nntrainer::TensorDim::DataType::UINT32,
                std::invalid_argument)
    << "Embedding layer input dtype must be FP32, UINT16, or UINT32";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  size_t in_dim =
    static_cast<size_t>(std::get<nntrainer::props::InDim>(embedding_props));
  size_t out_dim =
    static_cast<size_t>(std::get<nntrainer::props::OutDim>(embedding_props));

  // Tensorwise 4-bit LUT mode: load (or look up cached) shared LUT and
  // skip the standard managed weight allocation. The LUT is owned by
  // this layer via shared_ptr, and shared with any other layer that
  // references the same manifest path.
  auto &quant_path_prop =
    std::get<props::QuantizedLutPath>(embedding_props);
  if (!quant_path_prop.empty()) {
    quant_lut_ = get_or_load_quant_lut(quant_path_prop.get());

    NNTR_THROW_IF(quant_lut_->in_dim != in_dim, std::invalid_argument)
      << "in_dim mismatch: layer=" << in_dim
      << " manifest=" << quant_lut_->in_dim;
    NNTR_THROW_IF(quant_lut_->out_dim != out_dim, std::invalid_argument)
      << "out_dim mismatch: layer=" << out_dim
      << " manifest=" << quant_lut_->out_dim;
  }

  nntrainer::TensorDim output_dim = input_dim;

  // output_dim expected as hidden x num input (batch size)
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  if (quant_lut_) {
    // No managed weight in LUT mode — embedding rows live in quant_lut_.
    return;
  }

  nntrainer::TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void EmbeddingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, embedding_props);
  LayerImpl::setProperty(remain_props);
}

void EmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {}

void EmbeddingLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {

  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim = std::get<nntrainer::props::InDim>(embedding_props);
  unsigned int out_dim = std::get<nntrainer::props::OutDim>(embedding_props);
  float scale = std::get<nntrainer::props::Scale>(embedding_props).empty()
                  ? 1.0f
                  : std::get<nntrainer::props::Scale>(embedding_props).get();

  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  unsigned int b_size = input_.batch();

  // -------------------------------------------------------------------
  // Tensorwise 4-bit LUT path: dequantize the embedding row directly into
  // the output (UINT16 or float). Bypasses the managed-weight tensor
  // entirely; the packed table is shared with peer graphs via the
  // QuantLut shared_ptr.
  // -------------------------------------------------------------------
  if (quant_lut_) {
    NNTR_THROW_IF(out_dim != quant_lut_->out_dim, std::runtime_error)
      << "LUT out_dim drift";
    NNTR_THROW_IF(out_dim % 2 != 0, std::runtime_error)
      << "4-bit packed embedding requires out_dim to be even, got "
      << out_dim;

    const auto out_dtype = hidden_.getDataType();
    const auto in_dtype = input_.getDataType();
    const uint8_t *packed = quant_lut_->packed.data();
    const float lut_scale = quant_lut_->scale * scale;
    const int lut_offset = quant_lut_->offset;
    const size_t bytes_per_row = out_dim / 2;

    // Read a token id at position `i` of the current batch row,
    // honoring the actual input dtype (FP32 / UINT16 / UINT32).
    auto read_token = [&](const void *base, int i) -> size_t {
      switch (in_dtype) {
      case nntrainer::TensorDim::DataType::FP32:
        return static_cast<size_t>(static_cast<const float *>(base)[i]);
      case nntrainer::TensorDim::DataType::UINT16:
        return static_cast<size_t>(static_cast<const uint16_t *>(base)[i]);
      case nntrainer::TensorDim::DataType::UINT32:
        return static_cast<size_t>(static_cast<const uint32_t *>(base)[i]);
      default:
        throw std::runtime_error("Embedding: unsupported input dtype");
      }
    };

    for (unsigned int b = 0; b < b_size; ++b) {
      const size_t batch_off = b * input_.getDim().getFeatureLen();
      const void *in_base = nullptr;
      switch (in_dtype) {
      case nntrainer::TensorDim::DataType::FP32:
        in_base = input_.getAddress<float>(batch_off);
        break;
      case nntrainer::TensorDim::DataType::UINT16:
        in_base = input_.getAddress<uint16_t>(batch_off);
        break;
      case nntrainer::TensorDim::DataType::UINT32:
        in_base = input_.getAddress<uint32_t>(batch_off);
        break;
      default:
        throw std::runtime_error("Embedding: unsupported input dtype");
      }
      nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

      const int iter = static_cast<int>(to - from);

#pragma omp parallel for
      for (int i = 0; i < iter; ++i) {
        const size_t embed_idx = read_token(in_base, i);
        if (embed_idx >= in_dim) {
          throw std::invalid_argument(
            "input word index is greater than in_dim");
        }

        const uint8_t *row = packed + bytes_per_row * embed_idx;
        const size_t out_off = static_cast<size_t>(out_dim) * i;

        if (out_dtype == nntrainer::TensorDim::DataType::UINT16) {
          uint16_t *dst =
            batchsliced_hidden.getData<uint16_t>() + out_off;
          for (size_t k = 0; k < bytes_per_row; ++k) {
            const uint8_t byte = row[k];
            dst[2 * k] = clamp_u16(
              (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale);
            dst[2 * k + 1] = clamp_u16(
              (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
              lut_scale);
          }
        } else if (out_dtype == nntrainer::TensorDim::DataType::FP32) {
          float *dst = batchsliced_hidden.getData<float>() + out_off;
          for (size_t k = 0; k < bytes_per_row; ++k) {
            const uint8_t byte = row[k];
            dst[2 * k] =
              (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale;
            dst[2 * k + 1] =
              (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
              lut_scale;
          }
#ifdef ENABLE_FP16
        } else if (out_dtype == nntrainer::TensorDim::DataType::FP16) {
          _FP16 *dst = batchsliced_hidden.getData<_FP16>() + out_off;
          for (size_t k = 0; k < bytes_per_row; ++k) {
            const uint8_t byte = row[k];
            dst[2 * k] = static_cast<_FP16>(
              (static_cast<float>(byte & 0x0F) + lut_offset) * lut_scale);
            dst[2 * k + 1] = static_cast<_FP16>(
              (static_cast<float>((byte >> 4) & 0x0F) + lut_offset) *
              lut_scale);
          }
#endif
        } else {
          throw std::runtime_error(
            "EmbeddingLayer LUT mode: unsupported output dtype");
        }
      }
    }
    return;
  }

  // -------------------------------------------------------------------
  // Original non-LUT path (FP32/FP16 + Q4_0 / Q6_K block dequant).
  // -------------------------------------------------------------------
  nntrainer::Tensor &weight = context.getWeight(weight_idx);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  for (unsigned int b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

    int iter = to - from;

#pragma omp parallel for
    for (int i = 0; i < iter; ++i) {
      size_t embed_idx = static_cast<size_t>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      nntrainer::Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * (i));

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        nntrainer::dequantize_row_q6_K(
          (void *)((char *)weight.getData<uint8_t>() +
                   (210 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
        nntrainer::dequantize_row_q4_0(
          (void *)((char *)weight.getData<uint8_t>() +
                   (18 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else {
        out_tensor.copyData(cur_weight);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }
    }

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n weight: " << weight
              << "\n hidden: " << hidden_ << std::endl;
#endif
  }
}

void EmbeddingLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void EmbeddingLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void EmbeddingLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(embedding_props, method, this);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_embedding_layer() {
  auto layer = new EmbeddingLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_embedding_layer(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_embedding_layer,
                                                   destroy_embedding_layer};
}

#endif

} // namespace causallm
