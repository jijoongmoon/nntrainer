// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_ccapi_tensor.cpp
 * @date        11 December 2023
 * @brief       cc API Tensor Unit tests.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <iostream>

#include <layer.h>
#include <model.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <tensor_api.h>

/**
 * @brief Original test — backward compatibility with Pimpl
 */
TEST(nntrainer_ccapi, tensor_01_p) {
  std::shared_ptr<ml::train::Layer> layer;
  ml::train::Tensor a;

  EXPECT_NO_THROW(layer =
                    ml::train::layer::Input({"name=input0", "input_shape=1:1:2",
                                             "normalization=true"}));

  EXPECT_NO_THROW(a.setSrcLayer(layer));

  std::shared_ptr<ml::train::Layer> layer_b = a.getSrcLayer();
  EXPECT_EQ(layer_b->getName(), "input0");
}

/**
 * @brief Default constructed tensor is invalid
 */
TEST(nntrainer_ccapi_tensor, default_construct_p) {
  ml::train::Tensor t;
  EXPECT_FALSE(t.isValid());
}

/**
 * @brief Symbolic tensor with dimension is valid
 */
TEST(nntrainer_ccapi_tensor, symbolic_construct_p) {
  ml::train::TensorDim dim({1, 1, 28, 28});
  ml::train::Tensor t(dim, "input");

  EXPECT_TRUE(t.isValid());
  EXPECT_EQ(t.name(), "input");
  EXPECT_EQ(t.shape().batch(), 1);
  EXPECT_EQ(t.shape().channel(), 1);
  EXPECT_EQ(t.shape().height(), 28);
  EXPECT_EQ(t.shape().width(), 28);
}

/**
 * @brief Symbolic tensor without name
 */
TEST(nntrainer_ccapi_tensor, symbolic_construct_no_name_p) {
  ml::train::TensorDim dim({1, 3, 32, 32});
  ml::train::Tensor t(dim);

  EXPECT_TRUE(t.isValid());
  EXPECT_EQ(t.name(), "");
  EXPECT_EQ(t.shape().channel(), 3);
}

/**
 * @brief dtype defaults to FP32
 */
TEST(nntrainer_ccapi_tensor, dtype_default_p) {
  ml::train::TensorDim dim({1, 1, 2, 2});
  ml::train::Tensor t(dim);

  EXPECT_EQ(t.dtype(), ml::train::TensorDim::DataType::FP32);
}

/**
 * @brief Move constructor transfers ownership
 */
TEST(nntrainer_ccapi_tensor, move_construct_p) {
  ml::train::TensorDim dim({1, 1, 28, 28});
  ml::train::Tensor a(dim, "original");
  ml::train::Tensor b(std::move(a));

  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "original");
  EXPECT_EQ(b.shape().height(), 28);
  // moved-from tensor should be invalid
  EXPECT_FALSE(a.isValid());
}

/**
 * @brief Move assignment transfers ownership
 */
TEST(nntrainer_ccapi_tensor, move_assign_p) {
  ml::train::TensorDim dim({1, 1, 10, 10});
  ml::train::Tensor a(dim, "src");
  ml::train::Tensor b;

  b = std::move(a);
  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "src");
  EXPECT_FALSE(a.isValid());
}

/**
 * @brief Copy constructor creates independent copy
 */
TEST(nntrainer_ccapi_tensor, copy_construct_p) {
  ml::train::TensorDim dim({1, 1, 28, 28});
  ml::train::Tensor a(dim, "shared");
  ml::train::Tensor b(a);

  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "shared");
  EXPECT_EQ(b.shape().width(), 28);
  // both should be valid
  EXPECT_TRUE(a.isValid());
}

/**
 * @brief Copy assignment creates independent copy
 */
TEST(nntrainer_ccapi_tensor, copy_assign_p) {
  ml::train::TensorDim dim({1, 1, 5, 5});
  ml::train::Tensor a(dim, "orig");
  ml::train::Tensor b;

  b = a;
  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "orig");
  EXPECT_TRUE(a.isValid());
}

/**
 * @brief Clone creates independent copy
 */
TEST(nntrainer_ccapi_tensor, clone_p) {
  ml::train::TensorDim dim({1, 1, 3, 3});
  ml::train::Tensor a(dim, "cloneable");
  ml::train::Tensor b = a.clone();

  EXPECT_TRUE(b.isValid());
  EXPECT_EQ(b.name(), "cloneable");
  EXPECT_EQ(b.shape().height(), 3);
}

/**
 * @brief Accessing shape on invalid tensor throws
 */
TEST(nntrainer_ccapi_tensor, invalid_shape_n) {
  ml::train::Tensor t;
  EXPECT_THROW(t.shape(), std::runtime_error);
}

/**
 * @brief Accessing name on invalid tensor throws
 */
TEST(nntrainer_ccapi_tensor, invalid_name_n) {
  ml::train::Tensor t;
  EXPECT_THROW(t.name(), std::runtime_error);
}

/**
 * @brief Accessing dtype on invalid tensor throws
 */
TEST(nntrainer_ccapi_tensor, invalid_dtype_n) {
  ml::train::Tensor t;
  EXPECT_THROW(t.dtype(), std::runtime_error);
}

/**
 * @brief setSrcLayer / getSrcLayer on default tensor
 */
TEST(nntrainer_ccapi_tensor, src_layer_default_p) {
  ml::train::Tensor t;
  EXPECT_EQ(t.getSrcLayer(), nullptr);
}

// ===== Step 1-2: fromData, zeros, ones =====

/**
 * @brief fromData wraps external buffer (zero-copy)
 */
TEST(nntrainer_ccapi_tensor, from_data_p) {
  float buf[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ml::train::TensorDim dim({1, 1, 3, 4});
  auto t = ml::train::Tensor::fromData(dim, buf);

  EXPECT_TRUE(t.isValid());
  EXPECT_TRUE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
  EXPECT_EQ(t.shape().height(), 3u);
  EXPECT_EQ(t.shape().width(), 4u);
}

/**
 * @brief fromData with name
 */
TEST(nntrainer_ccapi_tensor, from_data_named_p) {
  float buf[4] = {1, 2, 3, 4};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 2}, buf, "ext_cache");
  EXPECT_EQ(t.name(), "ext_cache");
  EXPECT_TRUE(t.isExternal());
}

/**
 * @brief fromData with null pointer throws
 */
TEST(nntrainer_ccapi_tensor, from_data_null_n) {
  EXPECT_THROW(ml::train::Tensor::fromData({1, 1, 2, 2}, nullptr),
               std::invalid_argument);
}

/**
 * @brief zeros creates materialized non-external tensor
 */
TEST(nntrainer_ccapi_tensor, zeros_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 3});
  EXPECT_TRUE(t.isValid());
  EXPECT_FALSE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
  EXPECT_EQ(t.shape().height(), 2u);
  EXPECT_EQ(t.shape().width(), 3u);
}

/**
 * @brief ones creates materialized non-external tensor
 */
TEST(nntrainer_ccapi_tensor, ones_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 3});
  EXPECT_TRUE(t.isValid());
  EXPECT_FALSE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
}

/**
 * @brief zeros with name
 */
TEST(nntrainer_ccapi_tensor, zeros_named_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 3, 3}, "zero_t");
  EXPECT_EQ(t.name(), "zero_t");
}

/**
 * @brief symbolic tensor is not materialized and not external
 */
TEST(nntrainer_ccapi_tensor, symbolic_not_materialized_p) {
  ml::train::Tensor t({1, 1, 28, 28});
  EXPECT_FALSE(t.isMaterialized());
  EXPECT_FALSE(t.isExternal());
}

/**
 * @brief default tensor is not materialized and not external
 */
TEST(nntrainer_ccapi_tensor, default_not_materialized_p) {
  ml::train::Tensor t;
  EXPECT_FALSE(t.isMaterialized());
  EXPECT_FALSE(t.isExternal());
}

// ===== Step 1-3: data access =====

/**
 * @brief data<float>() on fromData returns the original pointer (zero-copy)
 */
TEST(nntrainer_ccapi_tensor, data_from_data_zero_copy_p) {
  float buf[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 3}, buf);
  EXPECT_EQ(t.data<float>(), buf);
}

/**
 * @brief getValue on fromData tensor
 */
TEST(nntrainer_ccapi_tensor, get_value_from_data_p) {
  float buf[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 3}, buf);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 2), 3.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 2), 6.0f);
}

/**
 * @brief getValue on zeros tensor
 */
TEST(nntrainer_ccapi_tensor, get_value_zeros_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 0.0f);
}

/**
 * @brief getValue on ones tensor
 */
TEST(nntrainer_ccapi_tensor, get_value_ones_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 1.0f);
}

/**
 * @brief setValue modifies value in-place
 */
TEST(nntrainer_ccapi_tensor, set_value_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  t.setValue(0, 0, 1, 1, 42.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 42.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 0.0f);
}

/**
 * @brief mutable_data allows direct writes
 */
TEST(nntrainer_ccapi_tensor, mutable_data_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 1, 4});
  float *ptr = t.mutable_data<float>();
  ptr[0] = 10.0f;
  ptr[3] = 99.0f;
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 10.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 3), 99.0f);
}

/**
 * @brief setData replaces external pointer
 */
TEST(nntrainer_ccapi_tensor, set_data_replace_ptr_p) {
  float buf1[4] = {1, 2, 3, 4};
  float buf2[4] = {5, 6, 7, 8};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 2}, buf1);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);

  t.setData(buf2);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 8.0f);
  EXPECT_EQ(t.data<float>(), buf2);
}

/**
 * @brief setData on non-external (zeros) tensor throws
 */
TEST(nntrainer_ccapi_tensor, set_data_on_non_external_n) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  float buf[4] = {};
  EXPECT_THROW(t.setData(buf), std::runtime_error);
}

/**
 * @brief setData on symbolic tensor throws
 */
TEST(nntrainer_ccapi_tensor, set_data_on_symbolic_n) {
  ml::train::Tensor t({1, 1, 2, 2});
  float buf[4] = {};
  EXPECT_THROW(t.setData(buf), std::runtime_error);
}

/**
 * @brief setData with null throws
 */
TEST(nntrainer_ccapi_tensor, set_data_null_n) {
  float buf[4] = {1, 2, 3, 4};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 2}, buf);
  EXPECT_THROW(t.setData(nullptr), std::invalid_argument);
}

/**
 * @brief copyFrom copies data into materialized tensor
 */
TEST(nntrainer_ccapi_tensor, copy_from_p) {
  float src[4] = {10, 20, 30, 40};
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  t.copyFrom(src);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 10.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 40.0f);
}

/**
 * @brief copyFrom with null throws
 */
TEST(nntrainer_ccapi_tensor, copy_from_null_n) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  EXPECT_THROW(t.copyFrom(nullptr), std::invalid_argument);
}

/**
 * @brief data access on unmaterialized (symbolic) tensor throws
 */
TEST(nntrainer_ccapi_tensor, data_access_unmaterialized_n) {
  ml::train::Tensor t({1, 1, 28, 28});
  EXPECT_THROW(t.data<float>(), std::runtime_error);
  EXPECT_THROW(t.mutable_data<float>(), std::runtime_error);
  EXPECT_THROW(t.getValue(0, 0, 0, 0), std::runtime_error);
  EXPECT_THROW(t.setValue(0, 0, 0, 0, 1.0f), std::runtime_error);
}

/**
 * @brief clone of eager tensor creates independent copy
 */
TEST(nntrainer_ccapi_tensor, clone_eager_independent_p) {
  auto orig = ml::train::Tensor::zeros({1, 1, 2, 2});
  orig.setValue(0, 0, 0, 0, 99.0f);
  auto cloned = orig.clone();

  cloned.setValue(0, 0, 0, 0, 1.0f);
  EXPECT_FLOAT_EQ(orig.getValue(0, 0, 0, 0), 99.0f);
  EXPECT_FLOAT_EQ(cloned.getValue(0, 0, 0, 0), 1.0f);
}

// ===== Step 2-1: LayerHandle graph edge recording =====

/**
 * @brief LayerHandle wraps createLayer and enables operator()
 */
TEST(nntrainer_ccapi_tensor, layer_handle_construct_p) {
  ml::train::LayerHandle fc =
    ml::train::createLayer("fully_connected", {"unit=256", "name=fc1"});
  EXPECT_TRUE(static_cast<bool>(fc));
  EXPECT_EQ(fc->getName(), "fc1");
  EXPECT_EQ(fc->getType(), "fully_connected");
}

/**
 * @brief LayerHandle converts to shared_ptr<Layer> for backward compat
 */
TEST(nntrainer_ccapi_tensor, layer_handle_to_shared_ptr_p) {
  ml::train::LayerHandle fc =
    ml::train::createLayer("fully_connected", {"unit=128", "name=fc_conv"});
  std::shared_ptr<ml::train::Layer> layer_ptr = fc;
  EXPECT_EQ(layer_ptr->getName(), "fc_conv");
}

/**
 * @brief Layer call on symbolic tensor produces symbolic output
 */
TEST(nntrainer_ccapi_tensor, layer_call_symbolic_p) {
  using namespace ml::train;
  auto input = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto output = fc(input);

  EXPECT_TRUE(output.isValid());
  EXPECT_FALSE(output.isMaterialized());
  // Shape is propagated from input; full shape inference (e.g. FC unit)
  // will be resolved during model.compile()
  EXPECT_EQ(output.getProducingLayer()->getName(), "fc1");
}

/**
 * @brief Layer chain: fc1 -> fc2
 */
TEST(nntrainer_ccapi_tensor, layer_chain_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "x");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
  LayerHandle fc2 = createLayer("fully_connected", {"unit=10", "name=fc2"});
  auto h = fc1(x);
  auto y = fc2(h);

  EXPECT_TRUE(y.isValid());
  EXPECT_FALSE(y.isMaterialized());
  EXPECT_EQ(y.getProducingLayer()->getName(), "fc2");
  EXPECT_EQ(y.getInputTensors()[0].getProducingLayer()->getName(), "fc1");
}

/**
 * @brief Multi-input layer (Addition)
 */
TEST(nntrainer_ccapi_tensor, multi_input_layer_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 256}, "a");
  auto b = Tensor({1, 1, 1, 256}, "b");
  LayerHandle add = createLayer("Addition", {"name=add1"});
  auto added = add({a, b});

  EXPECT_TRUE(added.isValid());
  EXPECT_FALSE(added.isMaterialized());
}

/**
 * @brief Graph edge: output records producing layer
 */
TEST(nntrainer_ccapi_tensor, graph_edge_producing_layer_p) {
  using namespace ml::train;
  auto input = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto output = fc(input);

  auto producer = output.getProducingLayer();
  EXPECT_NE(producer, nullptr);
  EXPECT_EQ(producer->getName(), "fc1");
}

/**
 * @brief Graph edge: output records input tensors
 */
TEST(nntrainer_ccapi_tensor, graph_edge_input_tensors_p) {
  using namespace ml::train;
  auto input = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto output = fc(input);

  auto inputs = output.getInputTensors();
  EXPECT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].name(), "input");
}

/**
 * @brief Leaf/input tensor has no producing layer
 */
TEST(nntrainer_ccapi_tensor, graph_edge_leaf_tensor_p) {
  ml::train::Tensor input({1, 1, 1, 784}, "input");
  EXPECT_EQ(input.getProducingLayer(), nullptr);
  EXPECT_TRUE(input.getInputTensors().empty());
}

/**
 * @brief Chain graph traversal: output -> fc2 -> fc1 -> input
 */
TEST(nntrainer_ccapi_tensor, graph_chain_traversal_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "x");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
  LayerHandle fc2 = createLayer("fully_connected", {"unit=10", "name=fc2"});
  auto h = fc1(x);
  auto y = fc2(h);

  // y was produced by fc2
  EXPECT_EQ(y.getProducingLayer()->getName(), "fc2");
  // y's input is h
  auto y_inputs = y.getInputTensors();
  EXPECT_EQ(y_inputs.size(), 1u);
  // h was produced by fc1
  EXPECT_EQ(y_inputs[0].getProducingLayer()->getName(), "fc1");
  // h's input is x (leaf)
  auto h_inputs = y_inputs[0].getInputTensors();
  EXPECT_EQ(h_inputs.size(), 1u);
  EXPECT_EQ(h_inputs[0].name(), "x");
  EXPECT_EQ(h_inputs[0].getProducingLayer(), nullptr);
}

/**
 * @brief Null LayerHandle throws on call
 */
TEST(nntrainer_ccapi_tensor, layer_handle_null_call_n) {
  ml::train::LayerHandle null_handle;
  ml::train::Tensor input({1, 1, 1, 784}, "input");
  EXPECT_THROW(null_handle(input), std::runtime_error);
}

/**
 * @brief Empty inputs to operator() throws
 */
TEST(nntrainer_ccapi_tensor, layer_handle_empty_inputs_n) {
  ml::train::LayerHandle fc =
    ml::train::createLayer("fully_connected", {"unit=256", "name=fc1"});
  EXPECT_THROW(fc(std::vector<ml::train::Tensor>{}), std::invalid_argument);
}

// ===== Step 2-2: Tensor ops → implicit layers =====

/**
 * @brief add() creates implicit Addition layer
 */
TEST(nntrainer_ccapi_tensor, add_symbolic_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 256}, "a");
  auto b = Tensor({1, 1, 1, 256}, "b");
  auto c = a.add(b);

  EXPECT_TRUE(c.isValid());
  EXPECT_FALSE(c.isMaterialized());
  EXPECT_EQ(c.shape().width(), 256u);
  // Verify graph: c produced by Addition layer with 2 inputs
  EXPECT_NE(c.getProducingLayer(), nullptr);
  EXPECT_EQ(c.getProducingLayer()->getType(), "addition");
  EXPECT_EQ(c.getInputTensors().size(), 2u);
  EXPECT_EQ(c.getInputTensors()[0].name(), "a");
  EXPECT_EQ(c.getInputTensors()[1].name(), "b");
}

/**
 * @brief multiply() creates implicit Multiply layer
 */
TEST(nntrainer_ccapi_tensor, multiply_symbolic_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 128}, "ma");
  auto b = Tensor({1, 1, 1, 128}, "mb");
  auto c = a.multiply(b);

  EXPECT_TRUE(c.isValid());
  EXPECT_FALSE(c.isMaterialized());
  EXPECT_EQ(c.shape().width(), 128u);
  EXPECT_NE(c.getProducingLayer(), nullptr);
  EXPECT_EQ(c.getProducingLayer()->getType(), "multiply");
  EXPECT_EQ(c.getInputTensors().size(), 2u);
}

/**
 * @brief reshape() creates implicit Reshape layer
 */
TEST(nntrainer_ccapi_tensor, reshape_symbolic_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "flat");
  auto y = x.reshape(TensorDim({1, 1, 28, 28}));

  EXPECT_TRUE(y.isValid());
  EXPECT_FALSE(y.isMaterialized());
  EXPECT_EQ(y.shape().channel(), 1u);
  EXPECT_EQ(y.shape().height(), 28u);
  EXPECT_EQ(y.shape().width(), 28u);
  EXPECT_NE(y.getProducingLayer(), nullptr);
  EXPECT_EQ(y.getInputTensors().size(), 1u);
  EXPECT_EQ(y.getInputTensors()[0].name(), "flat");
}

/**
 * @brief Residual (skip) connection: x + fc(x)
 */
TEST(nntrainer_ccapi_tensor, residual_connection_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 256}, "x");
  LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc_res"});
  auto h = fc(x);
  auto out = x.add(h);

  EXPECT_TRUE(out.isValid());
  // out -> Addition -> [x, h]; h -> fc -> [x]
  auto inputs = out.getInputTensors();
  EXPECT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].name(), "x");
  EXPECT_EQ(inputs[1].getProducingLayer()->getName(), "fc_res");
}

/**
 * @brief Chained ops: a.add(b).multiply(c)
 */
TEST(nntrainer_ccapi_tensor, chained_ops_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 64}, "a");
  auto b = Tensor({1, 1, 1, 64}, "b");
  auto c = Tensor({1, 1, 1, 64}, "c");
  auto result = a.add(b).multiply(c);

  EXPECT_TRUE(result.isValid());
  EXPECT_EQ(result.getProducingLayer()->getType(), "multiply");
  auto mul_inputs = result.getInputTensors();
  EXPECT_EQ(mul_inputs.size(), 2u);
  // First input to multiply is the add result
  EXPECT_EQ(mul_inputs[0].getProducingLayer()->getType(), "addition");
}

// ===== Step 2-3: Model::compile(Tensor, Tensor) — graph extraction =====

/**
 * @brief Simple single FC layer compile from tensor graph
 */
TEST(nntrainer_ccapi_graph, simple_fc_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc = createLayer("fully_connected", {"unit=10", "name=fc"});
  auto y = fc(x);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
}

/**
 * @brief Multi-layer compile: fc1 -> relu -> fc2
 */
TEST(nntrainer_ccapi_graph, multi_layer_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 784}, "input");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
  LayerHandle relu = createLayer("activation", {"activation=relu", "name=relu1"});
  LayerHandle fc2 = createLayer("fully_connected", {"unit=10", "name=fc2"});

  auto h = fc1(x);
  h = relu(h);
  auto y = fc2(h);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
}

/**
 * @brief Residual connection compile: x -> fc1 -> add(x, fc1_out) -> fc_out
 */
TEST(nntrainer_ccapi_graph, residual_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 256}, "input");
  LayerHandle fc1 = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto h = fc1(x);
  auto out = x.add(h);
  LayerHandle fc_out = createLayer("fully_connected", {"unit=10", "name=fc_out"});
  auto y = fc_out(out);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
}

/**
 * @brief Existing addLayer-based workflow still works
 */
TEST(nntrainer_ccapi_graph, existing_add_layer_still_works_p) {
  using namespace ml::train;
  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  model->addLayer(
    createLayer("input", {"name=in", "input_shape=1:1:784"}));
  model->addLayer(
    createLayer("fully_connected", {"name=fc", "unit=10", "input_layers=in"}));
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
}
