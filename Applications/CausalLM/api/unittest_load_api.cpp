// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_load_api.cpp
 * @brief   Focused tests for the CausalLM load API contracts: loading the
 *          model already in memory is a no-op, and a load that fails does not
 *          take the loaded model down with it.
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "causal_lm.h"
#include "causal_lm_api.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

namespace causal_lm_api_test {
void setModelForTest(std::unique_ptr<causallm::Transformer> model,
                     const std::string &architecture);
void setLoadedBuiltinRequestForTest(BackendType compute, ModelType model_type,
                                    ModelQuantizationType quant_type);
void setLoadedPackageRequestForTest(BackendType compute,
                                    const std::string &model_dir);
void resetForTest();
} // namespace causal_lm_api_test

namespace {

constexpr const char *kFakeOutput = "load-api-test-output";

/**
 * @brief Stand-in for a loaded model.
 *
 * It answers with a marker string, so a test can tell "the model I loaded is
 * still the one running" from "something reloaded underneath me" without
 * needing weights on disk.
 */
class FakeCausalLM final : public causallm::CausalLM {
public:
  FakeCausalLM() : CausalLM() {}

  void initialize() override { is_initialized = true; }

  void load_weight(const std::string &) override {}

  void run(const WSTR, bool = false, const WSTR = "", const WSTR = "",
           bool = true) override {
    if (output_list.empty())
      output_list.push_back(kFakeOutput);
    else
      output_list[0] = kFakeOutput;
    has_run_ = true;
  }
};

std::unique_ptr<causallm::Transformer> makeFakeModel() {
  return std::make_unique<FakeCausalLM>();
}

/**
 * @brief An empty directory of our own to run in.
 *
 * loadModel() resolves a built-in ::ModelType to "./models/<name>-<quant>", so
 * running from a directory that has no models/ subtree is what makes "a load
 * that really read weights would have failed" true, and therefore what makes
 * a successful load proof that no weights were read.
 */
std::filesystem::path makeSandboxDirectory() {
  const auto stamp =
    std::chrono::steady_clock::now().time_since_epoch().count();
  const auto base = std::filesystem::temp_directory_path();
  for (int attempt = 0; attempt < 100; ++attempt) {
    const auto candidate =
      base / ("causal_lm_load_api_" + std::to_string(stamp) + "_" +
              std::to_string(attempt));
    std::error_code ec;
    if (std::filesystem::create_directory(candidate, ec))
      return candidate;
  }
  return {};
}

/** @brief Puts a fake loaded model, and an empty working directory, in place */
class CausalLmLoadApiTest : public ::testing::Test {
protected:
  void SetUp() override {
    previous_directory_ = std::filesystem::current_path();
    sandbox_ = makeSandboxDirectory();
    ASSERT_FALSE(sandbox_.empty()) << "could not create a sandbox directory";
    std::filesystem::current_path(sandbox_);

    causal_lm_api_test::resetForTest();
    causal_lm_api_test::setModelForTest(makeFakeModel(), "LoadApiTestCausalLM");
  }

  void TearDown() override {
    causal_lm_api_test::resetForTest();
    std::filesystem::current_path(previous_directory_);
    std::error_code ec;
    std::filesystem::remove_all(sandbox_, ec);
  }

  /** @brief Asserts the fake model is the one still answering prompts. */
  void expectLoadedModelStillRuns() {
    const char *output = nullptr;
    ASSERT_EQ(runModel("load api test prompt", &output), CAUSAL_LM_ERROR_NONE);
    ASSERT_NE(output, nullptr);
    EXPECT_STREQ(output, kFakeOutput);
  }

  std::filesystem::path previous_directory_;
  std::filesystem::path sandbox_;
};

} // namespace

/**
 * The request matches the model in memory, and there is no model package to
 * read, so returning success can only mean the weights already in memory were
 * kept. Before loading became idempotent this returned
 * CAUSAL_LM_ERROR_MODEL_LOAD_FAILED.
 */
TEST_F(CausalLmLoadApiTest, RepeatedLoadOfTheSameModelReusesIt) {
  causal_lm_api_test::setLoadedBuiltinRequestForTest(
    CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
    CAUSAL_LM_QUANTIZATION_W8A16);

  EXPECT_EQ(loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
                      CAUSAL_LM_QUANTIZATION_W8A16),
            CAUSAL_LM_ERROR_NONE);

  expectLoadedModelStillRuns();
}

/** A repeat of a directory load is reused on the same terms. */
TEST_F(CausalLmLoadApiTest, RepeatedLoadOfTheSamePackageDirectoryReusesIt) {
  const std::string package = (sandbox_ / "package").string();
  causal_lm_api_test::setLoadedPackageRequestForTest(CAUSAL_LM_BACKEND_CPU,
                                                     package);

  EXPECT_EQ(loadModelFromPath(CAUSAL_LM_BACKEND_CPU, package.c_str()),
            CAUSAL_LM_ERROR_NONE);

  expectLoadedModelStillRuns();
}

/** A different quantization is a different model, so it must not be reused. */
TEST_F(CausalLmLoadApiTest, DifferentQuantizationIsNotReused) {
  causal_lm_api_test::setLoadedBuiltinRequestForTest(
    CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
    CAUSAL_LM_QUANTIZATION_W8A16);

  EXPECT_EQ(loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
                      CAUSAL_LM_QUANTIZATION_W16A16),
            CAUSAL_LM_ERROR_MODEL_LOAD_FAILED);
}

/** So is a different backend, even though every other argument matches. */
TEST_F(CausalLmLoadApiTest, DifferentBackendIsNotReused) {
  causal_lm_api_test::setLoadedBuiltinRequestForTest(
    CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
    CAUSAL_LM_QUANTIZATION_W8A16);

  EXPECT_EQ(loadModel(CAUSAL_LM_BACKEND_GPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
                      CAUSAL_LM_QUANTIZATION_W8A16),
            CAUSAL_LM_ERROR_MODEL_LOAD_FAILED);
}

/** And so is a different package directory. */
TEST_F(CausalLmLoadApiTest, DifferentPackageDirectoryIsNotReused) {
  causal_lm_api_test::setLoadedPackageRequestForTest(
    CAUSAL_LM_BACKEND_CPU, (sandbox_ / "package").string());

  const std::string other = (sandbox_ / "other-package").string();
  EXPECT_EQ(loadModelFromPath(CAUSAL_LM_BACKEND_CPU, other.c_str()),
            CAUSAL_LM_ERROR_INVALID_PARAMETER);
}

/**
 * The load above failed. The model that was loaded before it must still be
 * loaded and still answer prompts -- the caller asked for a second model and
 * did not get it, which is not a reason to lose the first.
 */
TEST_F(CausalLmLoadApiTest, FailedReloadLeavesTheLoadedModelRunnable) {
  causal_lm_api_test::setLoadedBuiltinRequestForTest(
    CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
    CAUSAL_LM_QUANTIZATION_W8A16);
  expectLoadedModelStillRuns();

  ASSERT_EQ(loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
                      CAUSAL_LM_QUANTIZATION_W16A16),
            CAUSAL_LM_ERROR_MODEL_LOAD_FAILED);

  expectLoadedModelStillRuns();
}

/** The same, for the directory entry point. */
TEST_F(CausalLmLoadApiTest, FailedDirectoryLoadLeavesTheLoadedModelRunnable) {
  causal_lm_api_test::setLoadedPackageRequestForTest(
    CAUSAL_LM_BACKEND_CPU, (sandbox_ / "package").string());
  expectLoadedModelStillRuns();

  const std::string other = (sandbox_ / "other-package").string();
  ASSERT_EQ(loadModelFromPath(CAUSAL_LM_BACKEND_CPU, other.c_str()),
            CAUSAL_LM_ERROR_INVALID_PARAMETER);

  expectLoadedModelStillRuns();
}

/**
 * A model injected without a request answers no request: the guard must not
 * hand a caller some other model just because one happens to be loaded.
 */
TEST_F(CausalLmLoadApiTest, LoadedModelOfUnknownIdentityIsNotReused) {
  EXPECT_EQ(loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
                      CAUSAL_LM_QUANTIZATION_W16A16),
            CAUSAL_LM_ERROR_MODEL_LOAD_FAILED);
}
