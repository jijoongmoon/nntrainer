// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api.cpp
 * @date   11 December 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Tensor interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#include <tensor_api.h>
#include <model.h>

#include <layer_context.h>
#include <memory_data.h>
#include <tensor.h>

#include <cstring>
#include <functional>
#include <stdexcept>
#include <set>
#include <unordered_set>

namespace ml {
namespace train {

/**
 * @brief Lightweight graph node for API-level symbolic graph construction.
 *
 * Unlike the internal GraphNode (used for execution with LayerNode),
 * this only holds connection info needed to build the graph before
 * model.compile(). Stored as shared_ptr so Tensor copies share graph
 * structure without recursive deep copies (O(N!) memory explosion).
 */
struct SymbolicGraphNode {
  std::shared_ptr<Layer> producing_layer;
  std::vector<std::shared_ptr<SymbolicGraphNode>> inputs;
  TensorDim dim;
  std::string name;
  int output_index = -1; ///< >=0 means indexed output, e.g. split(0)
  bool is_external = false;       ///< true if from Tensor::fromData
  void *external_ptr = nullptr;   ///< external data pointer if any
};

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

  // Graph edge (shared_ptr to avoid O(N!) deep-copy on Tensor copy)
  std::shared_ptr<SymbolicGraphNode> graph_edge;

  // Bound internal tensor (set after model compile+initialize)
  nntrainer::Tensor *bound_tensor = nullptr;

  // Lazy operation chain
  std::vector<std::function<void(nntrainer::Tensor &)>> call_chain;

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
  return impl_ && (impl_->eager_data != nullptr || impl_->bound_tensor != nullptr);
}

// --- Data access ---

template <typename T> const T *Tensor::data() const {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    return impl_->bound_tensor->getData<T>();
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getData<T>();
}

template <typename T> T *Tensor::mutable_data() {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    return impl_->bound_tensor->getData<T>();
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getData<T>();
}

// Explicit instantiations
template const float *Tensor::data<float>() const;
template float *Tensor::mutable_data<float>();

float Tensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                       unsigned int w) const {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    return impl_->bound_tensor->getValue<float>(b, c, h, w);
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getValue<float>(b, c, h, w);
}

void Tensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    impl_->bound_tensor->setValue(b, c, h, w, value);
    return;
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  impl_->eager_data->setValue(b, c, h, w, value);
}

void Tensor::copyFrom(const void *src) {
  if (!src) {
    throw std::invalid_argument("copyFrom: source pointer must not be null");
  }
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    std::memcpy(impl_->bound_tensor->getData(), src,
                impl_->bound_tensor->bytes());
    return;
  }
  if (!impl_->eager_data) {
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

// --- Lazy chaining ---

Tensor &Tensor::chain() {
  if (!impl_) {
    throw std::runtime_error("Cannot chain on invalid tensor");
  }
  impl_->call_chain.clear();
  return *this;
}

Tensor &Tensor::add_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot add_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.add_i(value); });
  return *this;
}

Tensor &Tensor::add_i(const Tensor &other, float alpha) {
  if (!impl_) {
    throw std::runtime_error("Cannot add_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back(
    [other_impl, alpha](nntrainer::Tensor &t) {
      nntrainer::Tensor *src = other_impl->bound_tensor
                                 ? other_impl->bound_tensor
                                 : other_impl->eager_data.get();
      if (!src)
        throw std::runtime_error("add_i: other tensor not materialized");
      t.add_i(*src, alpha);
    });
  return *this;
}

Tensor &Tensor::subtract_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot subtract_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.subtract_i(value); });
  return *this;
}

Tensor &Tensor::subtract_i(const Tensor &other) {
  if (!impl_) {
    throw std::runtime_error("Cannot subtract_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back(
    [other_impl](nntrainer::Tensor &t) {
      nntrainer::Tensor *src = other_impl->bound_tensor
                                 ? other_impl->bound_tensor
                                 : other_impl->eager_data.get();
      if (!src)
        throw std::runtime_error("subtract_i: other tensor not materialized");
      t.subtract_i(*src);
    });
  return *this;
}

Tensor &Tensor::multiply_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot multiply_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.multiply_i(value); });
  return *this;
}

Tensor &Tensor::multiply_i(const Tensor &other) {
  if (!impl_) {
    throw std::runtime_error("Cannot multiply_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back(
    [other_impl](nntrainer::Tensor &t) {
      nntrainer::Tensor *src = other_impl->bound_tensor
                                 ? other_impl->bound_tensor
                                 : other_impl->eager_data.get();
      if (!src)
        throw std::runtime_error("multiply_i: other tensor not materialized");
      t.multiply_i(*src);
    });
  return *this;
}

Tensor &Tensor::divide_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot divide_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.divide_i(value); });
  return *this;
}

Tensor &Tensor::divide_i(const Tensor &other) {
  if (!impl_) {
    throw std::runtime_error("Cannot divide_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back(
    [other_impl](nntrainer::Tensor &t) {
      nntrainer::Tensor *src = other_impl->bound_tensor
                                 ? other_impl->bound_tensor
                                 : other_impl->eager_data.get();
      if (!src)
        throw std::runtime_error("divide_i: other tensor not materialized");
      t.divide_i(*src);
    });
  return *this;
}

Tensor &Tensor::pow_i(float exponent) {
  if (!impl_) {
    throw std::runtime_error("Cannot pow_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [exponent](nntrainer::Tensor &t) { t.pow_i(exponent); });
  return *this;
}

Tensor &Tensor::inv_sqrt_i() {
  if (!impl_) {
    throw std::runtime_error("Cannot inv_sqrt_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [](nntrainer::Tensor &t) { t.inv_sqrt_i(); });
  return *this;
}

Tensor &Tensor::eval() {
  if (!impl_ || !isMaterialized()) {
    throw std::runtime_error(
      "Cannot eval: tensor is not materialized");
  }
  nntrainer::Tensor *target = impl_->bound_tensor
                                ? impl_->bound_tensor
                                : impl_->eager_data.get();
  for (auto &op : impl_->call_chain) {
    op(*target);
  }
  impl_->call_chain.clear();
  return *this;
}

// --- Private helpers ---

void *Tensor::getInternalPtr() const {
  if (!impl_)
    throw std::runtime_error("Tensor is not materialized");
  if (impl_->bound_tensor)
    return impl_->bound_tensor;
  if (impl_->eager_data)
    return impl_->eager_data.get();
  throw std::runtime_error("Tensor is not materialized");
}

Tensor Tensor::wrapResult(const void *internal_tensor) {
  const auto &internal =
    *static_cast<const nntrainer::Tensor *>(internal_tensor);
  Tensor result;
  result.impl_->dim = internal.getDim();
  result.impl_->valid = true;
  result.impl_->external = false;
  result.impl_->eager_data =
    std::make_shared<nntrainer::Tensor>(internal);
  return result;
}

// Convenience: cast getInternalPtr to nntrainer::Tensor*
static inline nntrainer::Tensor *asInternal(void *ptr) {
  return static_cast<nntrainer::Tensor *>(ptr);
}

// --- Eager operations returning new tensors ---

Tensor Tensor::add(float value) const {
  auto r = asInternal(getInternalPtr())->add(value);
  return wrapResult(&r);
}

Tensor Tensor::subtract(float value) const {
  auto r = asInternal(getInternalPtr())->subtract(value);
  return wrapResult(&r);
}

Tensor Tensor::subtract(const Tensor &other) const {
  auto r = asInternal(getInternalPtr())->subtract(
    *asInternal(other.getInternalPtr()));
  return wrapResult(&r);
}

Tensor Tensor::multiply(float value) const {
  auto r = asInternal(getInternalPtr())->multiply(value);
  return wrapResult(&r);
}

Tensor Tensor::divide(float value) const {
  auto r = asInternal(getInternalPtr())->divide(value);
  return wrapResult(&r);
}

Tensor Tensor::divide(const Tensor &other) const {
  auto r = asInternal(getInternalPtr())->divide(
    *asInternal(other.getInternalPtr()));
  return wrapResult(&r);
}

Tensor Tensor::dot(const Tensor &other, bool trans, bool trans_in) const {
  auto r = asInternal(getInternalPtr())->dot(
    *asInternal(other.getInternalPtr()), trans, trans_in);
  return wrapResult(&r);
}

Tensor Tensor::transpose(const std::string &direction) const {
  auto r = asInternal(getInternalPtr())->transpose(direction);
  return wrapResult(&r);
}

Tensor Tensor::pow(float exponent) const {
  auto r = asInternal(getInternalPtr())->pow(exponent);
  return wrapResult(&r);
}

Tensor Tensor::sum(unsigned int axis, float alpha) const {
  auto r = asInternal(getInternalPtr())->sum(axis, alpha);
  return wrapResult(&r);
}

Tensor Tensor::average(unsigned int axis) const {
  auto r = asInternal(getInternalPtr())->average(axis);
  return wrapResult(&r);
}

Tensor Tensor::average() const {
  auto r = asInternal(getInternalPtr())->average();
  return wrapResult(&r);
}

float Tensor::l2norm() const {
  return asInternal(getInternalPtr())->l2norm();
}

std::vector<unsigned int> Tensor::argmax() const {
  return asInternal(getInternalPtr())->argmax();
}

// --- Tensor manipulation ---

Tensor Tensor::getBatchSlice(unsigned int offset, unsigned int size) const {
  auto r = asInternal(getInternalPtr())->getBatchSlice(offset, size);
  return wrapResult(&r);
}

Tensor Tensor::getSharedDataTensor(const TensorDim &dim,
                                    size_t offset) const {
  auto r =
    asInternal(getInternalPtr())->getSharedDataTensor(dim, offset, false);
  return wrapResult(&r);
}

Tensor Tensor::apply(std::function<float(float)> f) const {
  auto *internal = asInternal(getInternalPtr());
  nntrainer::Tensor r = internal->clone();
  float *d = r.getData<float>();
  size_t len = r.size();
  for (size_t i = 0; i < len; ++i) {
    d[i] = f(d[i]);
  }
  return wrapResult(&r);
}

void Tensor::apply_i(std::function<float(float)> f) {
  auto *internal = asInternal(getInternalPtr());
  float *d = internal->getData<float>();
  size_t len = internal->size();
  for (size_t i = 0; i < len; ++i) {
    d[i] = f(d[i]);
  }
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat: tensors must not be empty");
  }

  std::vector<nntrainer::Tensor> internals;
  internals.reserve(tensors.size());
  for (auto &t : tensors) {
    internals.push_back(*asInternal(t.getInternalPtr()));
  }

  nntrainer::Tensor output;
  internals[0].concat(
    std::vector<nntrainer::Tensor>(internals.begin() + 1, internals.end()),
    axis, output);
  return wrapResult(&output);
}

// --- Immediate in-place operations ---

void Tensor::setZero() {
  asInternal(getInternalPtr())->setZero();
}

void Tensor::fill(const Tensor &from) {
  asInternal(getInternalPtr())->fill(
    *asInternal(from.getInternalPtr()));
}

void Tensor::copyData(const Tensor &from) {
  asInternal(getInternalPtr())->copyData(
    *asInternal(from.getInternalPtr()));
}

// --- Convenience dimension accessors ---

size_t Tensor::size() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get size of invalid tensor");
  }
  return impl_->dim.getDataLen();
}

bool Tensor::empty() const {
  return !impl_ || !impl_->valid || impl_->dim.getDataLen() == 0;
}

size_t Tensor::batch() const {
  return shape().batch();
}

size_t Tensor::channel() const {
  return shape().channel();
}

size_t Tensor::height() const {
  return shape().height();
}

size_t Tensor::width() const {
  return shape().width();
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
  return (impl_ && impl_->graph_edge) ? impl_->graph_edge->producing_layer
                                      : nullptr;
}

Tensor Tensor::output(unsigned int index) const {
  Tensor result;
  result.impl_ = std::make_unique<Impl>();
  result.impl_->valid = true;

  // Create a new graph edge that shares the same producing layer but with index
  auto indexed_edge = std::make_shared<SymbolicGraphNode>();
  if (impl_ && impl_->graph_edge) {
    indexed_edge->producing_layer = impl_->graph_edge->producing_layer;
    indexed_edge->inputs = impl_->graph_edge->inputs;
    indexed_edge->dim = impl_->graph_edge->dim;
    indexed_edge->name = impl_->graph_edge->name;
  }
  indexed_edge->output_index = static_cast<int>(index);
  result.impl_->graph_edge = indexed_edge;

  if (impl_) {
    result.impl_->src_layer = impl_->src_layer;
    result.impl_->dim = impl_->dim;
  }

  return result;
}

std::vector<Tensor> Tensor::getInputTensors() const {
  if (!impl_ || !impl_->graph_edge) {
    return {};
  }
  std::vector<Tensor> result;
  for (auto &edge : impl_->graph_edge->inputs) {
    Tensor t;
    t.impl_->graph_edge = edge;
    if (edge) {
      t.impl_->dim = edge->dim;
      t.impl_->name = edge->name;
      t.impl_->valid = true;
    }
    result.push_back(std::move(t));
  }
  return result;
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
  std::string layer_type = layer->getType();

  // Input layer: parse shape from input_shape property
  if (layer_type == "input") {
    try {
      std::string shape_str = layer->getProperty("input_shape");
      if (!shape_str.empty()) {
        // Parse "B:C:H:W" or "C:H:W" format
        std::vector<unsigned int> dims;
        std::stringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ':')) {
          dims.push_back(static_cast<unsigned int>(std::stoul(token)));
        }
        if (dims.size() == 4) {
          return TensorDim({dims[0], dims[1], dims[2], dims[3]});
        } else if (dims.size() == 3) {
          return TensorDim({1, dims[0], dims[1], dims[2]});
        }
      }
    } catch (...) {
      // Fall through
    }
  }

  if (inputs.empty() || !inputs[0].isValid()) {
    return TensorDim();
  }

  const TensorDim &in_dim = inputs[0].shape();

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

  // Record graph edge (shared, no deep copies)
  auto edge = std::make_shared<SymbolicGraphNode>();
  edge->producing_layer = ptr_;
  edge->dim = out_dim;
  edge->name = out_name;
  for (auto &inp : inputs) {
    if (inp.impl_ && inp.impl_->graph_edge) {
      edge->inputs.push_back(inp.impl_->graph_edge);
    } else {
      // Leaf tensor — create a leaf edge with no producer
      auto leaf = std::make_shared<SymbolicGraphNode>();
      if (inp.isValid()) {
        leaf->dim = inp.shape();
        leaf->name = inp.name();
        leaf->is_external = inp.impl_->external;
        leaf->external_ptr = inp.impl_->external_ptr;
      }
      edge->inputs.push_back(leaf);
    }
  }
  output.impl_->graph_edge = edge;

  return output;
}

// --- Model::compile(Tensor, Tensor) — graph extraction ---

int Model::compile(Tensor &input, Tensor &output, ExecutionMode mode) {
  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  int status = compile(inputs, outputs, mode);
  input = inputs[0];
  output = outputs[0];
  return status;
}

int Model::compile(Tensor &input, std::vector<Tensor> &outputs,
                   ExecutionMode mode) {
  std::vector<Tensor> inputs = {input};
  int status = compile(inputs, outputs, mode);
  input = inputs[0];
  return status;
}

int Model::compile(std::vector<Tensor> &inputs, std::vector<Tensor> &outputs,
                   ExecutionMode mode) {
  struct LayerInfo {
    std::shared_ptr<Layer> layer;
    std::vector<std::string> input_layer_names;
  };

  std::vector<LayerInfo> layers_in_order;
  std::unordered_set<Layer *> visited;

  // Collect input leaf names
  std::set<std::string> input_leaf_names;
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string name = inputs[i].name();
    if (name.empty()) {
      name = (inputs.size() == 1) ? "graph_input"
                                  : "graph_input_" + std::to_string(i);
    }
    input_leaf_names.insert(name);
  }

  // Track additional leaf tensors (e.g., fromData external caches)
  struct LeafInfo {
    TensorDim dim;
    bool is_external = false;       /**< true if from Tensor::fromData */
    void *external_ptr = nullptr;   /**< external data pointer if any */
  };
  std::map<std::string, LeafInfo> additional_leaves;
  int unnamed_leaf_counter = 0;

  // DFS on SymbolicGraphNode (post-order = topological order)
  std::function<void(const std::shared_ptr<SymbolicGraphNode> &)> dfs =
    [&](const std::shared_ptr<SymbolicGraphNode> &edge) {
      if (!edge || !edge->producing_layer) {
        return; // leaf
      }
      if (visited.count(edge->producing_layer.get())) {
        return;
      }
      visited.insert(edge->producing_layer.get());

      for (auto &inp : edge->inputs) {
        dfs(inp);
      }

      // Build input_layers names
      std::vector<std::string> input_names;
      for (auto &inp : edge->inputs) {
        if (inp && inp->producing_layer) {
          std::string layer_ref = inp->producing_layer->getName();
          if (inp->output_index >= 0) {
            layer_ref += "(" + std::to_string(inp->output_index) + ")";
          }
          input_names.push_back(layer_ref);
        } else {
          // Leaf tensor
          std::string leaf_name = inp ? inp->name : "";
          if (leaf_name.empty()) {
            // Empty leaf from Tensor() — this is the sentinel input to an
            // input layer. Check if the current edge's producing layer is an
            // input layer; if so, skip this leaf entirely since the input
            // layer itself serves as the graph entry point (no input_layers).
            if (edge->producing_layer) {
              try {
                if (edge->producing_layer->getType() == "input") {
                  // Input layer's sentinel leaf — skip, no input_layers needed
                  continue;
                }
              } catch (...) {}
            }
            leaf_name = "ext_input_" + std::to_string(unnamed_leaf_counter++);
          }
          if (input_leaf_names.count(leaf_name)) {
            input_names.push_back(leaf_name);
          } else {
            if (additional_leaves.find(leaf_name) == additional_leaves.end()) {
              bool ext = inp && inp->is_external;
              void *ext_ptr = ext ? inp->external_ptr : nullptr;
              additional_leaves[leaf_name] = {
                inp ? inp->dim : TensorDim(), ext, ext_ptr};
            }
            input_names.push_back(leaf_name);
          }
        }
      }

      layers_in_order.push_back({edge->producing_layer, input_names});
    };

  // Walk backwards from each output
  for (auto &output : outputs) {
    if (output.impl_ && output.impl_->graph_edge) {
      dfs(output.impl_->graph_edge);
    }
  }

  // 1. Create and add input layers (skip if already in DFS-discovered graph)
  //    When constructModel() already creates input layers via
  //    LayerHandle(createLayer("input",...)), the DFS walk will have found
  //    them. In that case, skip creating duplicate input layers.
  std::set<std::string> discovered_layer_names;
  for (auto &li : layers_in_order) {
    try {
      discovered_layer_names.insert(li.layer->getName());
    } catch (...) {}
  }

  int status;
  for (auto &inp : inputs) {
    std::string inp_name = inp.name();
    if (inp_name.empty()) {
      inp_name =
        (inputs.size() == 1)
          ? "graph_input"
          : "graph_input_" + std::to_string(&inp - &inputs[0]);
    }

    // If this input's producing layer was already discovered by DFS, skip
    if (inp.impl_ && inp.impl_->graph_edge &&
        inp.impl_->graph_edge->producing_layer) {
      std::string prod_name;
      try {
        prod_name = inp.impl_->graph_edge->producing_layer->getName();
      } catch (...) {}
      if (!prod_name.empty() && discovered_layer_names.count(prod_name)) {
        continue;  // Input layer already in the graph
      }
    }

    // Also skip if a layer with the same name was already discovered
    // (handles the case where inp_name matches an output name like "input0:output")
    bool already_has_input = false;
    for (auto &li : layers_in_order) {
      try {
        if (li.layer->getType() == "input" &&
            li.layer->getName() == inp_name) {
          already_has_input = true;
          break;
        }
      } catch (...) {}
    }
    if (already_has_input) continue;

    if (!inp.isValid()) continue;  // skip invalid tensors

    const TensorDim &dim = inp.shape();
    std::string shape_str = std::to_string(dim.channel()) + ":" +
                             std::to_string(dim.height()) + ":" +
                             std::to_string(dim.width());
    auto input_layer = createLayer(
      "input", {"name=" + inp_name, "input_shape=" + shape_str});
    status = addLayer(std::move(input_layer));
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  // 1b. Create input layers for additional leaf tensors.
  //     Skip external (fromData) tensors — they provide their own memory.
  for (auto &[leaf_name, leaf_info] : additional_leaves) {
    if (leaf_info.is_external) {
      // External tensor: no input layer needed. The tensor data is managed
      // externally (e.g., by KVCacheManager). The layer that consumes this
      // leaf will receive the external buffer directly.
      continue;
    }

    std::string leaf_shape = std::to_string(leaf_info.dim.channel()) + ":" +
                              std::to_string(leaf_info.dim.height()) + ":" +
                              std::to_string(leaf_info.dim.width());
    std::vector<std::string> leaf_props = {
      "name=" + leaf_name, "input_shape=" + leaf_shape};

    auto leaf_layer = createLayer("input", leaf_props);
    status = addLayer(std::move(leaf_layer));
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  // 2. Add each layer in topological order with input_layers set
  for (auto &info : layers_in_order) {
    if (!info.input_layer_names.empty()) {
      std::string input_layers_str;
      for (size_t i = 0; i < info.input_layer_names.size(); ++i) {
        if (i > 0)
          input_layers_str += ",";
        input_layers_str += info.input_layer_names[i];
      }
      info.layer->setProperty({"input_layers=" + input_layers_str});
    }
    status = addLayer(info.layer);
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  // 3. Compile the model
  status = compile(mode);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  // 4. Initialize the model
  status = initialize(mode);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  // 5. Allocate tensor memory
  status = allocate(mode);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  // 6. Materialize API tensors with allocated buffers
  for (auto &inp : inputs) {
    std::string inp_name = inp.name();
    if (inp_name.empty()) {
      inp_name =
        (inputs.size() == 1)
          ? "graph_input"
          : "graph_input_" + std::to_string(&inp - &inputs[0]);
    }
    const TensorDim &in_dim = inp.shape();
    inp.impl_->eager_data = std::make_shared<nntrainer::Tensor>(
      in_dim, true, nntrainer::Initializer::ZEROS, inp_name);
    inp.impl_->eager_data->initialize();
  }
  for (auto &output : outputs) {
    auto output_producer = output.getProducingLayer();
    if (!output_producer)
      continue;
    std::string output_layer_name = output_producer->getName();
    TensorDim out_dim;
    forEachLayer(
      [&](Layer &layer, nntrainer::RunLayerContext &rc, void *) {
        if (layer.getName() == output_layer_name && rc.getNumOutputs() > 0) {
          out_dim = rc.getOutput(0).getDim();
        }
      },
      nullptr);
    if (out_dim.getDataLen() > 0) {
      output.impl_->eager_data = std::make_shared<nntrainer::Tensor>(
        out_dim, true, nntrainer::Initializer::ZEROS, output_layer_name);
      output.impl_->eager_data->initialize();
      output.impl_->dim = out_dim;
    }
  }

  return ML_ERROR_NONE;
}

} // namespace train
} // namespace ml
