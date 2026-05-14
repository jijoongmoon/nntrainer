// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_lfm2_vl_vision_transformer.cpp
 * @date   13 May 2026
 * @brief  Unit tests for Lfm2VlVisionTransformer (CLIP/SigLIP-style ViT encoder).
 *         These tests exercise structure-only behavior: config parsing,
 *         parameter defaults, and that initialize() builds + compiles the
 *         symbolic graph without throwing on a tiny toy configuration.
 *         No GGUF / pretrained weights are required.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <lfm2_vl_vision_transformer.h>

namespace {

/**
 * @brief Test-only subclass that exposes a handful of the base Transformer's
 *        protected fields. We only need read access — the test does not
 *        mutate any internal state.
 */
class TestableLfm2Vl : public causallm::Lfm2VlVisionTransformer {
public:
  using causallm::Lfm2VlVisionTransformer::Lfm2VlVisionTransformer;

  unsigned int imageSize() const { return IMAGE_SIZE; }
  unsigned int patchSize() const { return PATCH_SIZE; }
  unsigned int numChannels() const { return NUM_CHANNELS; }
  unsigned int numPatches() const { return NUM_PATCHES; }
  int hiddenDim() const { return DIM; }
  int numLayers() const { return NUM_LAYERS; }
  int numHeads() const { return NUM_HEADS; }
  int headDim() const { return HEAD_DIM; }
  int intermediateSize() const { return INTERMEDIATE_SIZE; }
  unsigned int initSeqLen() const { return INIT_SEQ_LEN; }
  bool isCausal() const { return IS_CAUSAL; }
  unsigned int ropeTheta() const { return ROPE_THETA; }
  bool initialized() const { return is_initialized; }
};

/**
 * @brief Helper: build the three JSON blobs Lfm2VlVisionTransformer expects.
 *        Sized small enough for the graph to compile in a few hundred ms.
 */
struct ToyConfig {
  causallm::json cfg;
  causallm::json generation_cfg;
  causallm::json nntr_cfg;
};

ToyConfig makeToyConfig() {
  ToyConfig c;
  c.cfg = {{"hidden_size", 16},
           {"num_attention_heads", 2},
           {"num_hidden_layers", 2},
           {"intermediate_size", 32},
           {"image_size", 32},
           {"patch_size", 16},
           {"num_channels", 3},
           {"layer_norm_eps", 1e-6}};
  c.generation_cfg = causallm::json::object();
  // Provide every field the base Transformer::setupParameters() reads from
  // nntr_cfg via the unchecked [] operator (it will throw otherwise).
  c.nntr_cfg = {{"batch_size", 1},
                {"model_type", "embedding"},
                {"model_tensor_type", "FP32-FP32"},
                {"embedding_dtype", "FP32"},
                {"fc_layer_dtype", "FP32"},
                {"init_seq_len", 4},
                {"max_seq_len", 4},
                {"num_to_generate", 0}};
  return c;
}

causallm::json makeFullNntrCfg() {
  return {{"batch_size", 1},
          {"model_type", "embedding"},
          {"model_tensor_type", "FP32-FP32"},
          {"embedding_dtype", "FP32"},
          {"fc_layer_dtype", "FP32"},
          {"init_seq_len", 256},
          {"max_seq_len", 256},
          {"num_to_generate", 0}};
}

} // namespace

TEST(Lfm2VlVisionTransformer, setup_parameters_picks_up_overrides) {
  auto toy = makeToyConfig();
  TestableLfm2Vl vit(toy.cfg, toy.generation_cfg, toy.nntr_cfg);

  EXPECT_EQ(vit.imageSize(), 32u);
  EXPECT_EQ(vit.patchSize(), 16u);
  EXPECT_EQ(vit.numChannels(), 3u);
  // (image_size / patch_size) ^ 2 = (32/16)^2 = 4.
  EXPECT_EQ(vit.numPatches(), 4u);
  EXPECT_EQ(vit.hiddenDim(), 16);
  EXPECT_EQ(vit.numHeads(), 2);
  // head_dim defaults to hidden_size / num_attention_heads = 16 / 2 = 8.
  EXPECT_EQ(vit.headDim(), 8);
  EXPECT_EQ(vit.numLayers(), 2);
  EXPECT_EQ(vit.intermediateSize(), 32);

  // ViT is non-causal, no RoPE.
  EXPECT_FALSE(vit.isCausal());
  EXPECT_EQ(vit.ropeTheta(), 0u);

  // INIT_SEQ_LEN mirrors NUM_PATCHES.
  EXPECT_EQ(vit.initSeqLen(), 4u);
}

TEST(Lfm2VlVisionTransformer, setup_parameters_uses_defaults_when_fields_missing) {
  // Empty cfg: every field must fall back to the LFM2.5-VL / SigLIP2 defaults
  // baked into setupParameters().
  causallm::json cfg = causallm::json::object();
  causallm::json generation_cfg = causallm::json::object();
  causallm::json nntr_cfg = makeFullNntrCfg();
  TestableLfm2Vl vit(cfg, generation_cfg, nntr_cfg);

  EXPECT_EQ(vit.imageSize(), 256u);
  EXPECT_EQ(vit.patchSize(), 16u);
  EXPECT_EQ(vit.numChannels(), 3u);
  EXPECT_EQ(vit.numPatches(), 256u); // (256/16)^2
  EXPECT_EQ(vit.hiddenDim(), 768);
  EXPECT_EQ(vit.numHeads(), 12);
  EXPECT_EQ(vit.headDim(), 64); // 768 / 12
  EXPECT_EQ(vit.numLayers(), 12);
  EXPECT_EQ(vit.intermediateSize(), 3072);

  EXPECT_FALSE(vit.isCausal());
  EXPECT_EQ(vit.ropeTheta(), 0u);
  EXPECT_EQ(vit.initSeqLen(), 256u);
}

TEST(Lfm2VlVisionTransformer, initialize_builds_graph_without_exception) {
  // This is the structural smoke test: any mismatch between layer-name
  // uniqueness, shape inference (Conv2D -> Reshape -> mha_core ->
  // residual addition) or property parsing will surface here as a thrown
  // exception inside compile().
  auto toy = makeToyConfig();
  TestableLfm2Vl vit(toy.cfg, toy.generation_cfg, toy.nntr_cfg);

  ASSERT_NO_THROW(vit.initialize());
  EXPECT_TRUE(vit.initialized());
}

TEST(Lfm2VlVisionTransformer, initialize_supports_larger_grid) {
  // A slightly larger grid (4x4 patches, 4 heads, 3 layers) to catch
  // regressions that only manifest at non-trivial NUM_HEADS / NUM_LAYERS.
  causallm::json cfg = {{"hidden_size", 32},
                        {"num_attention_heads", 4},
                        {"num_hidden_layers", 3},
                        {"intermediate_size", 64},
                        {"image_size", 64},
                        {"patch_size", 16},
                        {"num_channels", 3},
                        {"layer_norm_eps", 1e-6}};
  causallm::json generation_cfg = causallm::json::object();
  causallm::json nntr_cfg = makeFullNntrCfg();
  nntr_cfg["init_seq_len"] = 16;
  nntr_cfg["max_seq_len"] = 16;

  TestableLfm2Vl vit(cfg, generation_cfg, nntr_cfg);
  EXPECT_EQ(vit.numPatches(), 16u); // (64/16)^2

  ASSERT_NO_THROW(vit.initialize());
  EXPECT_TRUE(vit.initialized());
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
