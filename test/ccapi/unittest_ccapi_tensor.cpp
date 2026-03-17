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
