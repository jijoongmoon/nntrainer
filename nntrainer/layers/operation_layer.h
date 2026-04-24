// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   operation_layer.h
 * @date   4 Oct 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is common class for operation layers
 *
 */
#ifndef __LAYER_OPERATION_H__
#define __LAYER_OPERATION_H__
#ifdef __cplusplus

#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @brief Base class for Unary Tensor Operation Layer
 *
 */
class UnaryOperationLayer : public Layer {
public:
  /**
   * @brief forwarding operation for unary input
   *
   */
  virtual void forwarding_operation(const Tensor &input, Tensor &hidden) = 0;

  /**
   * @brief copydoc Layer::forwarding(RunLayerContext &context, bool training)
   *
   */
  void forwarding(RunLayerContext &context, bool training) override {
    Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

    const Tensor input = context.getInput(0);
    forwarding_operation(input, hidden_);
  }

  static constexpr size_t SINGLE_INOUT_IDX = 0;
};

/**
 * @brief Base class for Binary Tensor Operation Layer
 *
 */
class BinaryOperationLayer : public Layer {
public:
  /**
   * @brief forwarding operation for binary inputs
   *
   */
  virtual void forwarding_operation(const Tensor &input0, const Tensor &input1,
                                    Tensor &hidden) = 0;

  /**
   * @brief copydoc Layer::forwarding(RunLayerContext &context, bool training)
   *
   */
  void forwarding(RunLayerContext &context, bool training) override {
    Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

    const Tensor &input0 = context.getInput(0);
    const Tensor &input1 = context.getInput(1);
    forwarding_operation(input0, input1, hidden_);
  }

  static constexpr size_t SINGLE_INOUT_IDX = 0;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_OPERATION_H__ */
