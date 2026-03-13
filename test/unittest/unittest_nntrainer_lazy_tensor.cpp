// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   unittest_nntrainer_lazy_tensor.cpp
 * @date   05 Jun 2020
 * @brief  A unittest for nntrainer_lazy_tensor
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <fstream>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

/**
 * @brief The nntrainer_LazyTensorOpsTest class provides a test fixture for
 *        nntrainer Lazy Tensor Operations.
 */
class nntrainer_LazyTensorOpsTest : public ::testing::Test {
protected:
  nntrainer_LazyTensorOpsTest() {}

  virtual void SetUp() {
    target = nntrainer::Tensor(batch, channel, height, width);
    GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + l);

    original.copy(target);
  }

  // virtual void TearDown()
  /**
   * @brief return a tensor filled with contant value
   */
  nntrainer::Tensor constant_(float value) {
    nntrainer::Tensor t(batch, channel, height, width);
    return t.apply<float>([value](float) { return value; });
  }

  nntrainer::Tensor target;
  nntrainer::Tensor original;
  nntrainer::Tensor expected;

private:
  int batch = 3;
  int height = 2;
  int width = 10;
  int channel = 1;
};

// LazyTensor init test
TEST(nntrainer_LazyTensor, LazyTensor_01_p) {
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + l);

  nntrainer::LazyTensor delayed(target);

  nntrainer::Tensor result;
  result = delayed.run();

  float *expected = target.getData();
  ASSERT_NE(expected, (float *)NULL);
  float *current = result.getData();
  ASSERT_NE(current, (float *)NULL);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(current[i], expected[i]);
  }
}

// Simple chain and run
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_01_p) {
  EXPECT_TRUE(target.chain().run() == original);
}

// Simple chain and add_i(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_02_p) {
  expected = original.add(2.1f);
  EXPECT_TRUE(target.chain().add_i(2.1f).run() == expected);
}

// chain and add_i(float) add_i(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_03_p) {
  expected = original.add(4.2f);
  EXPECT_TRUE(target.chain().add_i(2.1f).add_i(2.1f).run() == expected);
}

// chain and add_i(float) add_i(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_04_p) {
  expected = original.add(4.2f);
  EXPECT_TRUE(target.chain().add_i(2.1f).add_i(2.1f).run() == expected);
}

// chain and add_i(float) add_i(Tensor)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_05_p) {
  expected = original.add(6.1f);
  EXPECT_TRUE(target.chain().add_i(2.1f).add_i(constant_(2.0f), 2.0f).run() ==
              expected);
}

// chain and add_i(float) subtract(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_06_p) {
  EXPECT_TRUE(target.chain().add_i(2.1f).subtract_i(2.1f).run() == original);
}

// other basic operations (positive)...
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_07_p) {
  target = constant_(1.0);
  expected = constant_(2.0);
  EXPECT_TRUE(target.chain().multiply_i(2.0f).run() == expected);
  EXPECT_TRUE(target.chain().multiply_i(constant_(2.0f)).run() == expected);

  target = constant_(1.0f);
  expected = constant_(0.5f);
  EXPECT_TRUE(target.chain().divide_i(2.0f).run() == expected);
  EXPECT_TRUE(target.chain().divide_i(constant_(2.0f)).run() == expected);
}

// other basic operations (negative)...
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_07_n) {
  EXPECT_THROW(target.chain().add_i(constant(2.0f, 9, 9, 9, 9)).run(),
               std::runtime_error);

  EXPECT_THROW(target.chain().subtract_i(constant(2.0f, 9, 9, 9, 9)).run(),
               std::runtime_error);

  EXPECT_THROW(target.chain().multiply_i(constant(2.0f, 9, 9, 9, 9)).run(),
               std::runtime_error);

  EXPECT_THROW(target.chain().divide_i(constant(2.0f, 9, 9, 9, 9)).run(),
               std::runtime_error);
}

// pow_i
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_pow_01_p) {
  target = constant_(2.0f);
  expected = constant_(8.0f);
  EXPECT_TRUE(target.chain().pow_i(3.0f).run() == expected);
}

// sqrt_i
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_sqrt_01_p) {
  target = constant_(4.0f);
  expected = constant_(2.0f);
  EXPECT_TRUE(target.chain().sqrt_i().run() == expected);
}

// inv_sqrt_i
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_inv_sqrt_01_p) {
  target = constant_(4.0f);
  expected = constant_(0.5f);
  EXPECT_TRUE(target.chain().inv_sqrt_i().run() == expected);
}

// neg
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_neg_01_p) {
  target = constant_(3.0f);
  expected = constant_(-3.0f);
  EXPECT_TRUE(target.chain().neg().run() == expected);
}

// abs
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_abs_01_p) {
  target = constant_(-5.0f);
  expected = constant_(5.0f);
  EXPECT_TRUE(target.chain().abs().run() == expected);
}

// chaining new ops with arithmetic
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_chain_new_01_p) {
  // sqrt(4) * 3 + 1 = 7
  target = constant_(4.0f);
  expected = constant_(7.0f);
  EXPECT_TRUE(
    target.chain().sqrt_i().multiply_i(3.0f).add_i(1.0f).run() == expected);
}

// chaining pow with arithmetic
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_chain_new_02_p) {
  // (2^3 - 1) / 2 = 3.5
  target = constant_(2.0f);
  expected = constant_(3.5f);
  EXPECT_TRUE(
    target.chain().pow_i(3.0f).subtract_i(1.0f).divide_i(2.0f).run() ==
    expected);
}

// neg then abs should give positive
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_neg_abs_01_p) {
  target = constant_(7.0f);
  expected = constant_(7.0f);
  EXPECT_TRUE(target.chain().neg().abs().run() == expected);
}

// exp_i: exp(0) = 1, exp(1) ≈ 2.71828
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_exp_01_p) {
  target = constant_(0.0f);
  expected = constant_(1.0f);
  EXPECT_TRUE(target.chain().exp_i().run() == expected);
}

// log_i: log(1) = 0
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_log_01_p) {
  target = constant_(1.0f);
  expected = constant_(0.0f);
  EXPECT_TRUE(target.chain().log_i().run() == expected);
}

// exp then log should be identity
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_exp_log_01_p) {
  target = constant_(2.0f);
  expected = constant_(2.0f);
  EXPECT_TRUE(target.chain().exp_i().log_i().run() == expected);
}

// clamp_i
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_clamp_01_p) {
  target = constant_(10.0f);
  expected = constant_(5.0f);
  EXPECT_TRUE(target.chain().clamp_i(-5.0f, 5.0f).run() == expected);
}

// clamp_i lower bound
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_clamp_02_p) {
  target = constant_(-10.0f);
  expected = constant_(-3.0f);
  EXPECT_TRUE(target.chain().clamp_i(-3.0f, 3.0f).run() == expected);
}

// clamp in chain: exp(0) = 1, clamp(0.5, 2.0) -> 1.0
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_clamp_chain_01_p) {
  target = constant_(0.0f);
  expected = constant_(1.0f);
  EXPECT_TRUE(
    target.chain().exp_i().clamp_i(0.5f, 2.0f).run() == expected);
}

// sum()
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_08_p) {
  target = constant(1.0f, 4, 4, 4, 4);
  expected = constant(64.0f, 4, 1, 1, 1);
  EXPECT_TRUE(target.chain().sum_by_batch().run() == expected);

  expected = constant(4.0f, 1, 4, 4, 4);
  EXPECT_TRUE(target.chain().sum(0).run() == expected);

  expected = constant(4.0f, 4, 1, 4, 4);
  EXPECT_TRUE(target.chain().sum(1).run() == expected);

  expected = constant(4.0f, 4, 4, 1, 4);
  EXPECT_TRUE(target.chain().sum(2).run() == expected);

  expected = constant(4.0f, 4, 4, 4, 1);
  EXPECT_TRUE(target.chain().sum(3).run() == expected);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
