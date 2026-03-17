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
   * @brief Check if this tensor wraps external (user-managed) memory
   *
   * @return true if created via fromData()
   */
  bool isExternal() const;

  /**
   * @brief Check if this tensor has actual data (eager or bound after compile)
   *
   * @return true if data is accessible via data<T>() / getValue()
   */
  bool isMaterialized() const;

  /**
   * @brief Get read-only pointer to the underlying data
   *
   * @tparam T Data type (default: float)
   * @return Pointer to data buffer
   * @throws std::runtime_error if tensor is not materialized
   */
  template <typename T = float> const T *data() const;

  /**
   * @brief Get mutable pointer to the underlying data
   *
   * @tparam T Data type (default: float)
   * @return Pointer to mutable data buffer
   * @throws std::runtime_error if tensor is not materialized
   */
  template <typename T = float> T *mutable_data();

  /**
   * @brief Get value at specific location
   *
   * @param b batch index
   * @param c channel index
   * @param h height index
   * @param w width index
   * @return float value
   * @throws std::runtime_error if tensor is not materialized
   */
  float getValue(unsigned int b, unsigned int c, unsigned int h,
                 unsigned int w) const;

  /**
   * @brief Set value at specific location
   *
   * @param b batch index
   * @param c channel index
   * @param h height index
   * @param w width index
   * @param value value to set
   * @throws std::runtime_error if tensor is not materialized
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h,
                unsigned int w, float value);

  /**
   * @brief Copy data from external buffer into this tensor
   *
   * @param src Source buffer (must have at least shape().getDataLen() bytes)
   * @throws std::runtime_error if tensor is not materialized
   */
  void copyFrom(const void *src);

  /**
   * @brief Replace the external data pointer (fromData tensors only)
   *
   * @param new_ptr New data pointer (must outlive the tensor)
   * @throws std::runtime_error if tensor is not external
   */
  void setData(void *new_ptr);

  /**
   * @brief Create a tensor wrapping external user-managed memory (zero-copy)
   *
   * @param dim Tensor dimensions
   * @param data Pointer to user-managed buffer (must outlive the tensor)
   * @param name Optional name
   * @return Tensor wrapping the external buffer
   */
  static Tensor fromData(const TensorDim &dim, void *data,
                          const std::string &name = "");

  /**
   * @brief Create a zero-initialized eager tensor
   *
   * @param dim Tensor dimensions
   * @param name Optional name
   * @return Tensor with all elements set to 0
   */
  static Tensor zeros(const TensorDim &dim, const std::string &name = "");

  /**
   * @brief Create an eager tensor initialized to ones
   *
   * @param dim Tensor dimensions
   * @param name Optional name
   * @return Tensor with all elements set to 1
   */
  static Tensor ones(const TensorDim &dim, const std::string &name = "");

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
