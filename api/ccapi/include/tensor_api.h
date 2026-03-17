// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api.h
 * @date   11 December 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Tensor interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_TENSOR_H__
#define __ML_TRAIN_TENSOR_H__

#if __cplusplus < MIN_CPP_VERSION
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus

#include <layer.h>
#include <memory>
#include <string>
#include <tensor_dim.h>

namespace ml {
namespace train {

/**
 * @class   Tensor
 * @brief   Public Tensor API using Pimpl pattern.
 *
 * Symbolic tensors (constructed with dim) represent graph placeholders.
 * Eager tensors (fromData/zeros/ones) hold actual data immediately.
 * After model compile, symbolic tensors get bound to internal storage.
 */
class Tensor {
public:
  /**
   * @brief Default constructor — creates an invalid/empty tensor
   */
  Tensor();

  /**
   * @brief Construct a symbolic tensor with given dimensions
   *
   * @param dim Tensor dimensions
   * @param name Optional name for graph identification
   */
  explicit Tensor(const TensorDim &dim, const std::string &name = "");

  /**
   * @brief Destructor
   */
  ~Tensor();

  /**
   * @brief Move constructor
   */
  Tensor(Tensor &&rhs) noexcept;

  /**
   * @brief Move assignment
   */
  Tensor &operator=(Tensor &&rhs) noexcept;

  /**
   * @brief Copy constructor (shallow — shares the same graph node)
   */
  Tensor(const Tensor &rhs);

  /**
   * @brief Copy assignment (shallow)
   */
  Tensor &operator=(const Tensor &rhs);

  /**
   * @brief Clone with deep copy of data (for eager tensors)
   *
   * @return Deep-copied Tensor
   */
  Tensor clone() const;

  /**
   * @brief Check if this tensor is valid (has been properly constructed)
   *
   * @return true if tensor has been constructed with dim or data
   */
  bool isValid() const;

  /**
   * @brief Get tensor dimensions
   *
   * @return TensorDim
   */
  const TensorDim &shape() const;

  /**
   * @brief Get tensor name
   *
   * @return name string
   */
  const std::string &name() const;

  /**
   * @brief Get data type
   *
   * @return TensorDim::DataType
   */
  TensorDim::DataType dtype() const;

  /**
   * @brief Set the source layer that produced this tensor
   *
   * @param l Source layer
   */
  void setSrcLayer(std::shared_ptr<Layer> l);

  /**
   * @brief Get the source layer that produced this tensor
   *
   * @return Source layer (may be nullptr)
   */
  std::shared_ptr<Layer> getSrcLayer() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace train
} // namespace ml
#endif // __ML_TRAIN_TENSOR_H__
