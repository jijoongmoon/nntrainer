// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api.cpp
 * @date   11 December 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Tensor interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#include <tensor_api.h>

#include <tensor.h>

#include <cstring>
#include <stdexcept>

namespace ml {
namespace train {

/**
 * @brief Internal implementation of Tensor
 */
struct Tensor::Impl {
  TensorDim dim;
  std::string name;
  bool valid = false;
  bool external = false;

  std::shared_ptr<nntrainer::Tensor> eager_data;
  void *external_ptr = nullptr;

  std::shared_ptr<Layer> src_layer;

  Impl() = default;

  Impl(const TensorDim &d, const std::string &n) : dim(d), name(n), valid(true) {}
};

// --- Constructors / Destructor ---

Tensor::Tensor() : impl_(std::make_unique<Impl>()) {}

Tensor::Tensor(const TensorDim &dim, const std::string &name) :
  impl_(std::make_unique<Impl>(dim, name)) {}

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor &&rhs) noexcept = default;

Tensor &Tensor::operator=(Tensor &&rhs) noexcept = default;

Tensor::Tensor(const Tensor &rhs) :
  impl_(rhs.impl_ ? std::make_unique<Impl>(*rhs.impl_) : std::make_unique<Impl>()) {}

Tensor &Tensor::operator=(const Tensor &rhs) {
  if (this != &rhs) {
    impl_ = rhs.impl_ ? std::make_unique<Impl>(*rhs.impl_) : std::make_unique<Impl>();
  }
  return *this;
}

Tensor Tensor::clone() const {
  Tensor t;
  if (impl_) {
    t.impl_ = std::make_unique<Impl>(*impl_);
    if (impl_->eager_data && !impl_->external) {
      t.impl_->eager_data =
        std::make_shared<nntrainer::Tensor>(impl_->eager_data->clone());
    }
  }
  return t;
}

// --- Accessors ---

bool Tensor::isValid() const {
  return impl_ && impl_->valid;
}

const TensorDim &Tensor::shape() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get shape of invalid tensor");
  }
  return impl_->dim;
}

const std::string &Tensor::name() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get name of invalid tensor");
  }
  return impl_->name;
}

TensorDim::DataType Tensor::dtype() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get dtype of invalid tensor");
  }
  return impl_->dim.getDataType();
}

// --- State queries ---

bool Tensor::isExternal() const {
  return impl_ && impl_->external;
}

bool Tensor::isMaterialized() const {
  return impl_ && (impl_->eager_data != nullptr);
}

// --- Factory methods ---

Tensor Tensor::fromData(const TensorDim &dim, void *data,
                        const std::string &name) {
  if (!data) {
    throw std::invalid_argument("fromData: data pointer must not be null");
  }
  Tensor t;
  t.impl_->dim = dim;
  t.impl_->name = name;
  t.impl_->valid = true;
  t.impl_->external = true;
  t.impl_->external_ptr = data;
  t.impl_->eager_data =
    std::make_shared<nntrainer::Tensor>(dim, data);
  return t;
}

Tensor Tensor::zeros(const TensorDim &dim, const std::string &name) {
  Tensor t;
  t.impl_->dim = dim;
  t.impl_->name = name;
  t.impl_->valid = true;
  t.impl_->external = false;
  t.impl_->eager_data =
    std::make_shared<nntrainer::Tensor>(dim, true, nntrainer::Initializer::ZEROS, name);
  t.impl_->eager_data->initialize();
  return t;
}

Tensor Tensor::ones(const TensorDim &dim, const std::string &name) {
  Tensor t;
  t.impl_->dim = dim;
  t.impl_->name = name;
  t.impl_->valid = true;
  t.impl_->external = false;
  t.impl_->eager_data =
    std::make_shared<nntrainer::Tensor>(dim, true, nntrainer::Initializer::ONES, name);
  t.impl_->eager_data->initialize();
  return t;
}

// --- Source layer (backward compatible) ---

void Tensor::setSrcLayer(std::shared_ptr<Layer> l) {
  if (impl_) {
    impl_->src_layer = l;
  }
}

std::shared_ptr<Layer> Tensor::getSrcLayer() const {
  return impl_ ? impl_->src_layer : nullptr;
}

} // namespace train
} // namespace ml
