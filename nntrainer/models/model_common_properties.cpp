// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   model_common_properties.cpp
 * @date   27 Aug 2021
 * @brief  This file contains common properties for model
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <model_common_properties.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

#ifdef ENABLE_FP16
#include <half_tensor.h>
#endif

namespace nntrainer::props {

namespace {

/**
 * @brief Can a matmul on this build take an FP16 activation and a QS4CX
 * weight?
 *
 * HalfTensor::dot() is the only host implementation of that product, and it
 * carries the QS4CX case only where NNTR_HAS_HOST_QS4CX_FP16_GEMM (declared
 * next to the case, in half_tensor.h) is 1. Without ENABLE_FP16 there is no
 * HalfTensor at all, so the answer is trivially no.
 */
constexpr bool hasHostQs4cxFp16Gemm() {
#if defined(ENABLE_FP16) && NNTR_HAS_HOST_QS4CX_FP16_GEMM
  return true;
#else
  return false;
#endif
}

} // namespace

Epochs::Epochs(unsigned int value) { set(value); }

bool LossType::isValid(const std::string &value) const {
  ml_logw("Model loss property is deprecated, use loss layer directly instead");
  return istrequal(value, "cross") || istrequal(value, "mse") ||
         istrequal(value, "kld");
}

TrainingBatchSize::TrainingBatchSize(unsigned int value) { set(value); }

ContinueTrain::ContinueTrain(bool value) { set(value); }

MemoryOptimization::MemoryOptimization(bool value) { set(value); }

Fsu::Fsu(bool value) { set(value); }

FsuPath::FsuPath(const std::string &value) { set(value); }

FsuLookahead::FsuLookahead(const unsigned int &value) { set(value); }
ModelTensorDataType::ModelTensorDataType(ModelTensorDataTypeInfo::Enum value) {
  set(value);
}

void ModelTensorDataType::set(const ModelTensorDataTypeInfo::Enum &value) {
  NNTR_THROW_IF(value == ModelTensorDataTypeInfo::Enum::WQS4CXA16 &&
                  !hasHostQs4cxFp16Gemm(),
                std::invalid_argument)
    << "model_tensor_type=QS4CX-FP16 needs a host GEMM that multiplies an "
       "FP16 activation by a QS4CX weight, and this build has none, so every "
       "QS4CX fully connected layer would throw at its first matmul. Build "
       "with enable-fp16 on x86, or use model_tensor_type=QS4CX-FP32.";

  EnumProperty<ModelTensorDataTypeInfo>::set(value);
}
LossScale::LossScale(float value) { set(value); }

bool LossScale::isValid(const float &value) const {
  bool is_valid = (std::fpclassify(value) != FP_ZERO);
  if (!is_valid)
    ml_loge("Loss scale cannot be 0");
  return is_valid;
}

} // namespace nntrainer::props
