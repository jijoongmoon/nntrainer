// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lm_head.cpp
 * @date   16 Jan 2026
 * @brief  This is lmhead layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <lm_head.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include "cpu_backend.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LmHeadParams {
  weight,
  bias,
};

LmHeadLayer::LmHeadLayer() :
  LayerImpl(), lmhead_props(nntrainer::props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void LmHeadLayer::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  auto unit = std::get<nntrainer::props::Unit>(lmhead_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<ml::train::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions.
  /// EffDimFlag shouldn't be fixed like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  /** set output dimensions */
  ///@note lm_head's output dimension (height is always 1 !)
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  if (is_nchw)
    output_dims[0].width(unit);
  else
    output_dims[0].channel(unit);
  output_dims[0].height(1);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ///@note LMHead layer's tensor dim is transposed dim of user-defined
  /// dim
  /// so it can reuse embedding layer.
  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[LmHeadParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[LmHeadParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }
}

void LmHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, lmhead_props);
  LayerImpl::setProperty(remain_props);
}

void LmHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  throw nntrainer::exception::not_supported(
    "Forwarding for LMHead layer is not supported");
}

void LmHeadLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {

  nntrainer::Tensor weight =
    context.getWeight(weight_idx[LmHeadParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();

  uint32_t K = weight.height();
  uint32_t N = weight.width();
  std::cout << "Height : " << K << std::endl;
  std::cout << "Width : " << N << std::endl;

  nntrainer::TensorDim dim_qint4(1, 1, K, N, nntrainer::TensorDim::Format::NCHW, nntrainer::Tdatatype::QINT4);
  nntrainer::Tensor W_qint4(dim_qint4);
  
  std::vector<float> weight_fp32 (N * K);

  std::vector<uint8_t> unpacked_weight (weight.getMemoryBytes());
  
  //std::vector<uint8_t> kai_quant_data(rhs_native_size_qs4cx);
  //std::vector<uint8_t> kai_quant_scale(rhs_scales_size_f32);

  
  nntrainer::unpack_q4_0((void *)weight.getData(), (void *)unpacked_weight.data(), weight.getMemoryBytes(), N, K);

  for (size_t row = 0; row < N; row++) {
    nntrainer::dequantize_row_q4_0(
        (void *)(unpacked_weight.data() + row * (K/32) * 18),
        weight_fp32.data() + row * K,
        K);
  }

  std::cout << weight_fp32.data()[0] << weight_fp32.data()[1] << weight_fp32.data()[2] << std::endl;
  
  size_t rhs_native_size_qs4cx = static_cast<size_t>(N) * (((K + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t); //nxk
  size_t rhs_scales_size_f32 = N * sizeof(float);
  

  std::vector<float> kai_quant_scale(N);
  std::vector <uint8_t> kai_quant_data (N * K / 2);
  size_t packed_size = nntrainer::nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
      N, K, 3, true);

  std::vector<uint8_t> packed_weights(packed_size);
  //std::cout << weight_fp32.data()[0] << " " << weight_fp32.data()[1] << " " << weight_fp32.data()[2] << std::endl;  
  nntrainer::nntr_quant_qs4cx_f32(N, K, (void *)weight_fp32.data(), (void *)kai_quant_data.data(), (void *)kai_quant_scale.data());
           
  //std::cout << kai_quant_scale.data()[0] << kai_quant_scale.data()[1] << kai_quant_scale.data()[2] << std::endl;          
  nntrainer::nntr_qsi4cxp_qs4cxs1s0_rhs_pack(N, K,
                                  packed_weights.data(),
                                  kai_quant_data.data(),
                                  kai_quant_scale.data(),
                                  3, true);

  //std::cout << kai_quant_scale.data()[0] << kai_quant_scale.data()[1] << kai_quant_scale.data()[2] << std::endl;
  
  memcpy(W_qint4.getData<uint8_t>(), packed_weights.data(), packed_size);



  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + (to - from - 1) * input_.width(), true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

   
    //input_step.dot(W_qint4, hidden_step, false, true);
    auto M = hidden_step.height();
    std::cout << M << std::endl;
    nntrainer::nntr_gemm_qai8dxp_qsi4cxp_packed(
      M, N, K,
      (void *)input_step.getData(),        // LHS (activations) - will be packed internally
      (void *)packed_weights.data(),       // RHS (weights) - assumed already packed in block-32 format
      hidden_step.getData(),               // Output
      3,
      true,                // transB
      -std::numeric_limits<float>::infinity(),  // lower_bound
      std::numeric_limits<float>::infinity()    // upper_bound
    );
    
    
    //input_step.dot(weight, hidden_step, false, false);

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[LmHeadParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void LmHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for LMHead layer is not supported");
}

void LmHeadLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for LMHead layer is not supported");
}

void LmHeadLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(lmhead_props, method, this);
}

void LmHeadLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();

  unsigned int height = input_dimensions[0].height();

  // output dim's height is always 1 !
  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_tie_word_embedding() {
  auto layer = new LmHeadLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_tie_word_embedding(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_tie_word_embedding,
                                                   destroy_tie_word_embedding};
}

#endif

} // namespace causallm
