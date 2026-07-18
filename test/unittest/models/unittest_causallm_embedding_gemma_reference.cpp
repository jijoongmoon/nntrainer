// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_embedding_gemma_reference.cpp
 * @date   16 June 2026
 * @brief  Differential test for the tiny EmbeddingGemma model.
 *
 * Compares nntrainer's EmbeddingGemma output (Gemma3 transformer + mean pooling
 * + L2 normalize) against a golden fixture from HuggingFace transformers
 * (generate_embedding_gemma_reference.py). EmbeddingGemma sanitizes its configs
 * in the constructor, so a thin subclass initializes the (virtual) Transformer
 * base with the processed configs. The test skips when the fixture is absent.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <embedding_gemma.h>

#include <memory>

namespace {

/**
 * @brief Tiny EmbeddingGemma differential adapter
 *
 * EmbeddingGemma must sanitize its configs before initializing the virtual
 * Transformer base, so this thin subclass owns that mem-initializer (the one in
 * EmbeddingTestAdapter is ignored per virtual-base rules).
 */
class EmbeddingGemmaRefAdapter final
  : public causallm_test::EmbeddingTestAdapter<causallm::EmbeddingGemma> {
public:
  EmbeddingGemmaRefAdapter(causallm::json &cfg, causallm::json &generation_cfg,
                           causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::EMBEDDING),
    causallm_test::EmbeddingTestAdapter<causallm::EmbeddingGemma>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Differential model descriptor for the tiny EmbeddingGemma fixture
 */
causallm_test::DifferentialModel embeddingGemmaModel() {
  return {
    "embedding_gemma_tiny",
    [](causallm::json &cfg, causallm::json &gen_cfg, causallm::json &nntr_cfg) {
      return std::make_unique<EmbeddingGemmaRefAdapter>(cfg, gen_cfg, nntr_cfg);
    },
  };
}

/**
 * @brief FP32 output embedding matches the HF reference (atol + cosine)
 */
TEST(EmbeddingGemmaDifferentialTest, FP32MatchesHFReference) {
  causallm_test::runFp32EmbeddingDifferentialChecks(embeddingGemmaModel());
}

/**
 * @brief Q4_0 quantized embedding is close to the HF FP32 reference
 */
TEST(EmbeddingGemmaDifferentialTest, Q40CloseToFP32Reference) {
  causallm_test::runQ40EmbeddingDifferentialChecks(embeddingGemmaModel());
}

} // namespace
