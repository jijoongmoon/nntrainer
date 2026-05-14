// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_weight_dtype_map.cpp
 * @date   14 May 2026
 * @brief  Unit tests for the per-role weight_dtype_map property.
 *
 *         The map lets a single layer have weights of mixed dtypes — e.g.
 *         FC `weight` as Q8_0 while FC `bias` stays FP32. The test exercises
 *         the InitLayerContext::getDataTypeForRole(role, fallback) lookup
 *         path directly so it can validate the parser without spinning up
 *         a whole model graph.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include <layer_context.h>
#include <tensor_dim.h>

using DT = nntrainer::TensorDim::DataType;

namespace {

// Build a minimal InitLayerContext seeded with a known
// (format, weight_dt, activation_dt) triple. Inputs/outputs are dummy.
nntrainer::InitLayerContext makeCtx(const std::string &weight_dt_default,
                                    const std::string &activation_dt_default) {
  std::vector<nntrainer::TensorDim> in_dims = {
    nntrainer::TensorDim(1, 1, 1, 32,
                         {nntrainer::Tformat::NCHW,
                          nntrainer::TensorDim::DataType::FP32})};
  std::vector<bool> req_out_connected = {true};
  return nntrainer::InitLayerContext(
    in_dims, req_out_connected, /*is_inplace=*/false, "dummy_layer", "scope",
    /*max_norm=*/0.0f,
    {"NCHW", weight_dt_default, activation_dt_default},
    /*loss_scale=*/1.0f, ml::train::ExecutionMode::INFERENCE,
    ml::train::LayerComputeEngine::CPU);
}

} // namespace

TEST(WeightDtypeMap, empty_map_returns_fallback) {
  // No map set -> getDataTypeForRole always returns the explicit fallback.
  auto ctx = makeCtx("Q8_0", "FP32");
  ctx.setWeightDtypeMap("");

  EXPECT_EQ(ctx.getDataTypeForRole("weight", ctx.getWeightDataType()), DT::Q8_0);
  EXPECT_EQ(ctx.getDataTypeForRole("bias",   ctx.getActivationDataType()),
            DT::FP32);
  EXPECT_EQ(ctx.getDataTypeForRole("gamma",  ctx.getWeightDataType()), DT::Q8_0);
}

TEST(WeightDtypeMap, single_role_override) {
  auto ctx = makeCtx("Q8_0", "FP32");
  ctx.setWeightDtypeMap("weight:Q4_0");

  // weight -> Q4_0 (from map), bias -> FP32 (fallback), gamma -> Q8_0 (fallback)
  EXPECT_EQ(ctx.getDataTypeForRole("weight", ctx.getWeightDataType()), DT::Q4_0);
  EXPECT_EQ(ctx.getDataTypeForRole("bias",   ctx.getActivationDataType()),
            DT::FP32);
  EXPECT_EQ(ctx.getDataTypeForRole("gamma",  ctx.getWeightDataType()), DT::Q8_0);
}

TEST(WeightDtypeMap, multi_role_override) {
  // Classic mixed-precision: FC weight=Q8_0, FC bias=FP32 explicitly stated.
  auto ctx = makeCtx("FP32", "FP32");
  ctx.setWeightDtypeMap("weight:Q8_0,bias:FP32");

  EXPECT_EQ(ctx.getDataTypeForRole("weight", ctx.getWeightDataType()), DT::Q8_0);
  EXPECT_EQ(ctx.getDataTypeForRole("bias",   ctx.getActivationDataType()),
            DT::FP32);
  // unknown role -> fallback path
  EXPECT_EQ(ctx.getDataTypeForRole("loraA",  ctx.getWeightDataType()), DT::FP32);
}

TEST(WeightDtypeMap, tolerates_whitespace_and_trailing_comma) {
  auto ctx = makeCtx("FP32", "FP32");
  ctx.setWeightDtypeMap("  weight : Q4_0 ,   bias : FP16  ,");

  EXPECT_EQ(ctx.getDataTypeForRole("weight", DT::FP32), DT::Q4_0);
  EXPECT_EQ(ctx.getDataTypeForRole("bias",   DT::FP32), DT::FP16);
}

TEST(WeightDtypeMap, layernorm_style_gamma_beta) {
  auto ctx = makeCtx("Q4_0", "FP32");
  // Pretend a LayerNorm wants gamma in FP32 instead of Q4_0 the model picked.
  ctx.setWeightDtypeMap("gamma:FP32,beta:FP32");

  EXPECT_EQ(ctx.getDataTypeForRole("gamma", ctx.getWeightDataType()), DT::FP32);
  EXPECT_EQ(ctx.getDataTypeForRole("beta",  ctx.getActivationDataType()),
            DT::FP32);
}

TEST(WeightDtypeMap, missing_colon_throws) {
  auto ctx = makeCtx("FP32", "FP32");
  ctx.setWeightDtypeMap("weight=Q4_0");  // wrong separator
  EXPECT_THROW(ctx.getDataTypeForRole("weight", DT::FP32),
               std::invalid_argument);
}

TEST(WeightDtypeMap, unknown_dtype_throws) {
  auto ctx = makeCtx("FP32", "FP32");
  ctx.setWeightDtypeMap("weight:BOGUS");
  EXPECT_THROW(ctx.getDataTypeForRole("weight", DT::FP32),
               std::invalid_argument);
}

TEST(WeightDtypeMap, q8_0_now_accepted) {
  // Regression for the matching Q8_0 entry we added to TensorDataTypeInfo
  // alongside this feature: previously "Q8_0" would throw "No matching
  // enum" because the dtype enum was registered but the property dictionary
  // wasn't updated.
  auto ctx = makeCtx("FP32", "FP32");
  ctx.setWeightDtypeMap("weight:Q8_0");
  EXPECT_EQ(ctx.getDataTypeForRole("weight", DT::FP32), DT::Q8_0);
}

TEST(WeightDtypeMap, lazy_parsing_only_once) {
  // The cache should survive multiple lookups (parser runs once).
  auto ctx = makeCtx("FP32", "FP32");
  ctx.setWeightDtypeMap("weight:Q4_0,bias:FP16");
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(ctx.getDataTypeForRole("weight", DT::FP32), DT::Q4_0);
    EXPECT_EQ(ctx.getDataTypeForRole("bias",   DT::FP32), DT::FP16);
  }
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    ::testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS" << std::endl;
  }
  return result;
}
