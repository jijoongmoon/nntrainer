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

#include <memory_data.h>
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

  // Graph edge info (for symbolic graph construction)
  std::shared_ptr<Layer> producing_layer;
  std::vector<Tensor> input_tensors;

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

// --- Data access ---

template <typename T> const T *Tensor::data() const {
  if (!impl_ || !impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getData<T>();
}

template <typename T> T *Tensor::mutable_data() {
  if (!impl_ || !impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getData<T>();
}

// Explicit instantiations
template const float *Tensor::data<float>() const;
template float *Tensor::mutable_data<float>();

float Tensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                       unsigned int w) const {
  if (!impl_ || !impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getValue<float>(b, c, h, w);
}

void Tensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  if (!impl_ || !impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  impl_->eager_data->setValue(b, c, h, w, value);
}

void Tensor::copyFrom(const void *src) {
  if (!src) {
    throw std::invalid_argument("copyFrom: source pointer must not be null");
  }
  if (!impl_ || !impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  std::memcpy(impl_->eager_data->getData(), src, impl_->eager_data->bytes());
}

void Tensor::setData(void *new_ptr) {
  if (!impl_ || !impl_->external) {
    throw std::runtime_error("setData: only supported for fromData tensors");
  }
  if (!new_ptr) {
    throw std::invalid_argument("setData: pointer must not be null");
  }
  impl_->external_ptr = new_ptr;
  impl_->eager_data->setData(
    std::make_shared<nntrainer::MemoryData>(new_ptr), 0, false);
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
  // Create internal tensor structure, then point to external memory (zero-copy)
  t.impl_->eager_data =
    std::make_shared<nntrainer::Tensor>(dim, true);
  t.impl_->eager_data->setData(
    std::make_shared<nntrainer::MemoryData>(data), 0, false);
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

// --- Graph info accessors ---

std::shared_ptr<Layer> Tensor::getProducingLayer() const {
  return impl_ ? impl_->producing_layer : nullptr;
}

std::vector<Tensor> Tensor::getInputTensors() const {
  if (impl_) {
    return impl_->input_tensors;
  }
  return {};
}

// --- Symbolic tensor operations (implicit layers) ---

static unsigned int implicit_layer_counter = 0;

static std::string nextImplicitName(const std::string &prefix) {
  return prefix + "_auto_" + std::to_string(implicit_layer_counter++);
}

Tensor Tensor::add(const Tensor &other) const {
  LayerHandle layer =
    createLayer("Addition", {"name=" + nextImplicitName("add")});
  return layer({*this, other});
}

Tensor Tensor::multiply(const Tensor &other) const {
  LayerHandle layer =
    createLayer("Multiply", {"name=" + nextImplicitName("mul")});
  return layer({*this, other});
}

Tensor Tensor::reshape(const TensorDim &new_shape) const {
  std::string target = std::to_string(new_shape.channel()) + ":" +
                       std::to_string(new_shape.height()) + ":" +
                       std::to_string(new_shape.width());
  LayerHandle layer = createLayer(
    "reshape", {"name=" + nextImplicitName("reshape"),
                "target_shape=" + target});
  Tensor output = layer({*this});
  // Override shape to match requested dimensions
  output.impl_->dim = TensorDim({shape().batch(), new_shape.channel(),
                                  new_shape.height(), new_shape.width()});
  return output;
}

// --- LayerHandle graph construction ---

/**
 * @brief Try to infer output dimensions from layer type and properties.
 *
 * This is a best-effort inference for common layer types.
 * Full shape inference happens during model.compile().
 */
static TensorDim inferOutputDim(const std::shared_ptr<Layer> &layer,
                                const std::vector<Tensor> &inputs) {
  if (inputs.empty() || !inputs[0].isValid()) {
    return TensorDim();
  }

  const TensorDim &in_dim = inputs[0].shape();
  std::string layer_type = layer->getType();

  // Fully connected: output = {batch, 1, 1, unit}
  if (layer_type == "fully_connected") {
    try {
      std::string unit_str = layer->getProperty("unit");
      if (!unit_str.empty()) {
        unsigned int unit = static_cast<unsigned int>(std::stoul(unit_str));
        return TensorDim({in_dim.batch(), 1, 1, unit});
      }
    } catch (...) {
      // Fall through to default
    }
  }

  // Most layers preserve shape (activation, normalization, dropout, etc.)
  return in_dim;
}

Tensor LayerHandle::operator()(const Tensor &input) {
  return (*this)(std::vector<Tensor>{input});
}

Tensor LayerHandle::operator()(const std::vector<Tensor> &inputs) {
  if (!ptr_) {
    throw std::runtime_error("LayerHandle: layer is null");
  }
  if (inputs.empty()) {
    throw std::invalid_argument("LayerHandle: at least one input required");
  }

  // Infer output dimensions
  TensorDim out_dim = inferOutputDim(ptr_, inputs);

  // Build output name from layer name
  std::string out_name;
  try {
    out_name = ptr_->getName();
    if (!out_name.empty()) {
      out_name += ":output";
    }
  } catch (...) {
    // getName() might throw if layer isn't fully initialized
  }

  // Create symbolic output tensor
  Tensor output;
  if (out_dim.batch() > 0 && out_dim.width() > 0) {
    output = Tensor(out_dim, out_name);
  } else {
    // Couldn't infer shape — create a valid but shapeless tensor
    output = Tensor(TensorDim(), out_name);
  }

  // Record graph edge
  output.impl_->producing_layer = ptr_;
  output.impl_->input_tensors = inputs;

  return output;
}

} // namespace train
} // namespace ml
