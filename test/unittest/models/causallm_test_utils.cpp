// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   causallm_test_utils.cpp
 * @date   15 May 2026
 * @brief  Shared helpers for tiny CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

using json = causallm::json;

namespace causallm_test {

namespace {

/**
 * @brief Sanitize a string for use in a temporary file name
 */
std::string sanitizeName(std::string name) {
  std::replace_if(
    name.begin(), name.end(),
    [](unsigned char c) { return !std::isalnum(c) && c != '_' && c != '-'; },
    '_');
  return name;
}

/**
 * @brief Write a string into a file
 */
void writeFile(const std::filesystem::path &path, const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open " + path.string());
  }

  file << content;
  if (!file.good()) {
    throw std::runtime_error("failed to write " + path.string());
  }
}

/**
 * @brief Write a tiny tokenizer file for CausalLM tests
 */
std::filesystem::path writeTinyTokenizer(const std::filesystem::path &dir) {
  auto tokenizer_path = dir / "tokenizer.json";

  std::ostringstream vocab;
  vocab << "      \"<unk>\": 0,\n";
  vocab << "      \"hello\": 1,\n";
  vocab << "      \"world\": 2,\n";
  for (unsigned int i = 3; i < 31; ++i) {
    vocab << "      \"tok" << i << "\": " << i << ",\n";
  }
  vocab << "      \"<eos>\": 31\n";

  std::ostringstream tokenizer;
  tokenizer << R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 31,
      "content": "<eos>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
)" << vocab.str()
            << R"(    },
    "unk_token": "<unk>"
  }
})";

  writeFile(tokenizer_path, tokenizer.str());
  return tokenizer_path;
}

/**
 * @brief Create a loaded target-dtype model from deterministic FP32 weights
 */
std::unique_ptr<TinyCausalLMRunner>
makeLoadedDeterministicModel(const TinyCausalLMCase &test_case,
                             const TinyCausalLMFiles &files) {
  TinyCausalLMDataType fp32_data_type = makeTinyFp32DataType();
  // Use the case's nntrainer config factory (if any) for the source model too,
  // so that the model graph is compiled with the correct
  // init_seq_len/max_seq_len.
  auto source_nntr_cfg =
    test_case.make_nntrainer_config
      ? test_case.make_nntrainer_config(files.tokenizer_path, fp32_data_type)
      : makeTinyNntrainerConfig(files.tokenizer_path, fp32_data_type);
  TinyCausalLMConfig source_config = {
    test_case.make_model_config(),
    makeTinyGenerationConfig(),
    source_nntr_cfg,
  };

  auto source = test_case.create_model(
    source_config.model, source_config.generation, source_config.nntrainer);
  source->initializeModel();
  test_case.setup_weights(*source);
  source->saveWeightWithDtype(
    files.weight_path.string(),
    test_case.make_layer_dtype_map(test_case.data_type));

  auto loaded_config = makeTinyCausalLMConfig(test_case, files.tokenizer_path);
  auto loaded = test_case.create_model(
    loaded_config.model, loaded_config.generation, loaded_config.nntrainer);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

} // namespace

/**
 * @brief Make files for one tiny CausalLM test invocation
 */
TinyCausalLMFiles makeTinyCausalLMFiles(const std::string &suite_name,
                                        const std::string &test_name,
                                        const std::string &case_name) {
  std::string name = "nntrainer_causallm_tiny";
  name += "_";
  name += sanitizeName(suite_name);
  name += "_";
  name += sanitizeName(test_name);
  name += "_";
  name += sanitizeName(case_name);

  auto dir = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  TinyCausalLMFiles files;
  files.dir = dir;
  files.tokenizer_path = writeTinyTokenizer(dir);
  files.weight_path = dir / "causallm_tiny.bin";
  return files;
}

/**
 * @brief Make minimal generation config shared by tiny CausalLM tests
 */
causallm::json makeTinyGenerationConfig() {
  return {
    {"bos_token_id", 0}, {"eos_token_id", 31}, {"do_sample", false},
    {"top_k", 1},        {"top_p", 1.0},       {"temperature", 1.0},
  };
}

/**
 * @brief Make FP32 data type variant
 */
TinyCausalLMDataType makeTinyFp32DataType() {
  return {
    "FP32", "FP32", "FP32", "FP32", "FP32-FP32",
  };
}

/**
 * @brief Make Q4_0 weights with FP32 activations data type variant
 */
TinyCausalLMDataType makeTinyQ40Fp32DataType() {
  return {
    "Q40_FP32", "Q4_0", "Q4_0", "Q4_0", "Q4_0-FP32",
  };
}

/**
 * @brief Make Q4_0 weights with FP16 activations data type variant
 */
TinyCausalLMDataType makeTinyQ40Fp16DataType() {
  return {
    "Q40_FP16", "Q4_0", "Q4_0", "Q4_0", "Q4_0-FP16",
  };
}

/**
 * @brief Convert a test dtype string to an nntrainer tensor data type
 */
ml::train::TensorDim::DataType toTensorDataType(const std::string &dtype) {
  if (dtype == "FP32")
    return ml::train::TensorDim::DataType::FP32;
  if (dtype == "FP16")
    return ml::train::TensorDim::DataType::FP16;
  if (dtype == "Q4_0")
    return ml::train::TensorDim::DataType::Q4_0;
  if (dtype == "NONE")
    return ml::train::TensorDim::DataType::NONE;

  throw std::invalid_argument("unsupported tiny CausalLM dtype: " + dtype);
}

/**
 * @brief Make minimal nntrainer config shared by tiny CausalLM tests
 */
causallm::json
makeTinyNntrainerConfig(const std::filesystem::path &tokenizer_path,
                        const TinyCausalLMDataType &data_type) {
  return {
    {"bad_word_ids", std::vector<unsigned int>{}},
    {"batch_size", 1},
    {"embedding_dtype", data_type.embedding_dtype},
    {"fc_layer_dtype", data_type.fc_layer_dtype},
    {"init_seq_len", 4},
    {"lmhead_dtype", data_type.lmhead_dtype},
    {"max_seq_len", 8},
    {"model_tensor_type", data_type.model_tensor_type},
    {"model_type", "CausalLM"},
    {"num_to_generate", 1},
    {"tokenizer_file", tokenizer_path.string()},
  };
}

/**
 * @brief Make a construct-only nntrainer config for config-plumbing tests
 */
causallm::json makeTinyCtorOnlyNntrainerConfig() {
  auto nntr_cfg =
    makeTinyNntrainerConfig(std::filesystem::path(), makeTinyFp32DataType());
  nntr_cfg.erase("tokenizer_file");
  return nntr_cfg;
}

/**
 * @brief Make complete tiny configs for one model case
 */
TinyCausalLMConfig
makeTinyCausalLMConfig(const TinyCausalLMCase &test_case,
                       const std::filesystem::path &tokenizer_path) {
  auto nntr_cfg =
    test_case.make_nntrainer_config
      ? test_case.make_nntrainer_config(tokenizer_path, test_case.data_type)
      : makeTinyNntrainerConfig(tokenizer_path, test_case.data_type);
  return {
    test_case.make_model_config(),
    makeTinyGenerationConfig(),
    nntr_cfg,
  };
}

/**
 * @brief Verify greedy decoding chooses the maximum logit token
 */
void expectGreedyGenerationSelectsArgmax(TinyCausalLMRunner &model) {
  std::vector<float> logits(32, -2.0f);
  logits[2] = 1.0f;
  logits[3] = 5.0f;
  logits[4] = 4.0f;
  unsigned int input_ids[4] = {1, 0, 0, 0};

  auto ids = model.generateFromLogits(logits.data(), false, 1.0f, input_ids, 1);

  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 3u);
}

/**
 * @brief Verify save/load round-trip preserves tiny model logits
 */
void expectWeightRoundTripProducesSameLogits(const TinyCausalLMCase &test_case,
                                             const TinyCausalLMFiles &files) {
  auto first = makeLoadedDeterministicModel(test_case, files);
  auto second = makeLoadedDeterministicModel(test_case, files);

  std::vector<float> first_logits;
  std::vector<float> second_logits;
  ASSERT_NO_THROW(first_logits =
                    first->prefillLogits(test_case.expected_logits.prompt));
  ASSERT_NO_THROW(second_logits =
                    second->prefillLogits(test_case.expected_logits.prompt));

  ASSERT_EQ(first_logits.size(), second_logits.size());
  for (size_t i = 0; i < first_logits.size(); ++i)
    EXPECT_NEAR(first_logits[i], second_logits[i],
                test_case.expected_logits.logits_tolerance);
}

/**
 * @brief Verify a tiny model emits the expected logits for a prompt
 */
void expectPromptProducesExpectedLogits(const TinyCausalLMCase &test_case,
                                        const TinyCausalLMFiles &files) {
  auto model = makeLoadedDeterministicModel(test_case, files);

  std::vector<float> logits;
  ASSERT_NO_THROW(logits =
                    model->prefillLogits(test_case.expected_logits.prompt));

  ASSERT_EQ(logits.size(), test_case.expected_logits.logits.size());
  for (size_t i = 0; i < test_case.expected_logits.logits.size(); ++i)
    EXPECT_NEAR(logits[i], test_case.expected_logits.logits[i],
                test_case.expected_logits.logits_tolerance)
      << "logit mismatch at index " << i;
}

/**
 * @brief Locate the fixture directory for a fixture name
 */
std::filesystem::path findFixtureDir(const std::string &fixture_name) {
  if (const char *env = std::getenv("NNTRAINER_CAUSALLM_FIXTURE_DIR")) {
    return std::filesystem::path(env) / fixture_name;
  }
  // Resolve relative to this source file: .../test/unittest/models/
  std::filesystem::path src(__FILE__);
  return src.parent_path() / "causallm_reference" / fixture_name;
}

/**
 * @brief Load reference fixture from disk
 */
ReferenceFixture loadReferenceFixture(const std::filesystem::path &dir) {
  ReferenceFixture fix;
  fix.dir = dir;
  fix.logits_atol_fp32 = 1e-2f;
  fix.logits_atol_q40 = 5.0f;
  fix.prefix_match_min = 2;
  fix.prompt = "hello tok4";
  fix.embedding_atol = 1e-2f;
  fix.cosine_min = 0.999f;
  fix.embedding_atol_q40 = 0.1f;
  fix.cosine_min_q40 = 0.99f;

  auto meta_path = dir / "meta.json";
  if (std::filesystem::exists(meta_path)) {
    std::ifstream f(meta_path);
    json meta = json::parse(f);
    if (meta.contains("logits_atol_fp32"))
      fix.logits_atol_fp32 = meta["logits_atol_fp32"].get<float>();
    if (meta.contains("logits_atol_q40"))
      fix.logits_atol_q40 = meta["logits_atol_q40"].get<float>();
    if (meta.contains("prefix_match_min"))
      fix.prefix_match_min = meta["prefix_match_min"].get<size_t>();
    if (meta.contains("prompt"))
      fix.prompt = meta["prompt"].get<std::string>();
    if (meta.contains("embedding_atol"))
      fix.embedding_atol = meta["embedding_atol"].get<float>();
    if (meta.contains("cosine_min"))
      fix.cosine_min = meta["cosine_min"].get<float>();
    if (meta.contains("embedding_atol_q40"))
      fix.embedding_atol_q40 = meta["embedding_atol_q40"].get<float>();
    if (meta.contains("cosine_min_q40"))
      fix.cosine_min_q40 = meta["cosine_min_q40"].get<float>();
  }

  auto load_json_array = [&](const std::string &name) -> json {
    auto p = dir / name;
    if (!std::filesystem::exists(p))
      return json{};
    std::ifstream f(p);
    return json::parse(f);
  };

  auto ids_j = load_json_array("input_ids.json");
  for (auto &v : ids_j)
    fix.input_ids.push_back(v.get<unsigned int>());

  auto logits_j = load_json_array("reference_logits.json");
  for (auto &v : logits_j)
    fix.reference_logits.push_back(v.get<float>());

  auto tokens_j = load_json_array("reference_tokens.json");
  for (auto &v : tokens_j)
    fix.reference_tokens.push_back(v.get<unsigned int>());

  auto embedding_j = load_json_array("reference_embedding.json");
  for (auto &v : embedding_j)
    fix.reference_embedding.push_back(v.get<float>());

  return fix;
}

/**
 * @brief Assert logit vectors agree within absolute tolerance
 */
void expectLogitsNear(const std::vector<float> &got,
                      const std::vector<float> &ref, float atol) {
  ASSERT_EQ(got.size(), ref.size()) << "logit vector size mismatch";

  // argmax must match
  auto got_max = std::max_element(got.begin(), got.end());
  auto ref_max = std::max_element(ref.begin(), ref.end());
  EXPECT_EQ(std::distance(got.begin(), got_max),
            std::distance(ref.begin(), ref_max))
    << "argmax mismatch: nntrainer=" << std::distance(got.begin(), got_max)
    << " ref=" << std::distance(ref.begin(), ref_max);

  for (size_t i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(got[i], ref[i], atol) << "logit mismatch at index " << i;
}

/**
 * @brief Assert greedy token prefix agreement
 */
void expectTokenPrefixMatch(const std::vector<unsigned int> &got,
                            const std::vector<unsigned int> &ref,
                            size_t min_match) {
  ASSERT_FALSE(got.empty()) << "nntrainer produced no tokens";
  ASSERT_FALSE(ref.empty()) << "reference has no tokens";

  size_t match = 0;
  size_t check = std::min(got.size(), ref.size());
  for (size_t i = 0; i < check; ++i) {
    if (got[i] == ref[i])
      ++match;
    else
      break;
  }

  EXPECT_GE(match, min_match)
    << "greedy prefix match " << match << " < min " << min_match
    << " (got[0]=" << (got.empty() ? -1 : (int)got[0])
    << " ref[0]=" << (ref.empty() ? -1 : (int)ref[0]) << ")";
}

/**
 * @brief Assert embedding vectors agree element-wise and by cosine similarity
 */
void expectEmbeddingNear(const std::vector<float> &got,
                         const std::vector<float> &ref, float atol,
                         float cosine_min) {
  ASSERT_EQ(got.size(), ref.size()) << "embedding vector size mismatch";
  ASSERT_FALSE(ref.empty()) << "reference embedding is empty";

  double dot = 0.0, norm_got = 0.0, norm_ref = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    dot += static_cast<double>(got[i]) * ref[i];
    norm_got += static_cast<double>(got[i]) * got[i];
    norm_ref += static_cast<double>(ref[i]) * ref[i];
    EXPECT_NEAR(got[i], ref[i], atol) << "embedding mismatch at index " << i;
  }

  double cosine = (norm_got > 0.0 && norm_ref > 0.0)
                    ? dot / (std::sqrt(norm_got) * std::sqrt(norm_ref))
                    : 0.0;
  EXPECT_GE(cosine, cosine_min)
    << "cosine similarity " << cosine << " < min " << cosine_min;
}

namespace {

constexpr const char *QUANTIZE_BIN_ENV = "NNTR_QUANTIZE_BIN";

/**
 * @brief The path-valued keys of nntr_config.json
 *
 * INVARIANT (the contract every consumer of a package must reproduce):
 * a package is RELOCATABLE -- nntr_quantize writes these keys as bare
 * filenames relative to the package directory, never as absolute paths -- and
 * NOTHING in the model layer resolves them. Transformer's constructor receives
 * a json, not a directory, so it opens whatever string it is handed relative
 * to the PROCESS CWD. Resolution is therefore the caller's job, and every
 * caller that hands a loaded config to a model must do it:
 *
 *   Applications/CausalLM/main.cpp                (resolveNntrConfigPath)
 *   Applications/CausalLM/api/causal_lm_api.cpp   (three load sites)
 *   this file                                     (three load sites, below)
 *
 * Hand-absolutizing one key (as this file used to do for tokenizer_file only)
 * is what hides the omission: a generated single-file package names no other
 * path, so the harness passes -- until the package is a sidecar package, whose
 * embedding_file_name / ple_file_name then resolve against the runner's CWD
 * and the load dies on "Failed to open LUT manifest".
 */
constexpr const char *NNTR_CONFIG_PATH_KEYS[] = {
  "tokenizer_file", "embedding_file_name", "ple_file_name",
  "module_config_path"};

/**
 * @brief Resolve every relative path-valued key against the package directory
 * @param nntr_cfg Loaded nntr_config.json
 * @param dir Directory the config was loaded from
 * @note Mirrors resolveNntrConfigPath() in main.cpp / causal_lm_api.cpp:
 *       absolute and empty values are left alone, so it is idempotent.
 */
void resolvePackagePaths(json &nntr_cfg, const std::filesystem::path &dir) {
  for (const char *key : NNTR_CONFIG_PATH_KEYS) {
    if (!nntr_cfg.contains(key) || !nntr_cfg[key].is_string())
      continue;

    std::filesystem::path value = nntr_cfg[key].get<std::string>();
    if (value.empty() || value.is_absolute())
      continue;

    nntr_cfg[key] = (dir / value).string();
  }
}

/**
 * @brief nntrainer configs loaded from a fixture directory
 */
struct FixtureConfigs {
  json model_cfg;
  json gen_cfg;
  json nntr_cfg;
  std::filesystem::path weight_path;
};

/**
 * @brief Load model/generation/nntrainer configs from a fixture directory
 *
 * Resolves every relative path key against @a dir, then pins tokenizer_file to
 * the tokenizer shipped in that directory so the test is self-contained even
 * when a generator baked an absolute tokenizer path into the fixture.
 */
FixtureConfigs loadFixtureConfigs(const std::filesystem::path &dir) {
  FixtureConfigs fc;
  fc.model_cfg = causallm::LoadJsonFile((dir / "config.json").string());
  fc.gen_cfg =
    causallm::LoadJsonFile((dir / "generation_config.json").string());
  fc.nntr_cfg = causallm::LoadJsonFile((dir / "nntr_config.json").string());

  resolvePackagePaths(fc.nntr_cfg, dir);
  fc.nntr_cfg["tokenizer_file"] = (dir / "tokenizer.json").string();

  std::string bin_name = fc.nntr_cfg["model_file_name"].get<std::string>();
  fc.weight_path = dir / bin_name;
  return fc;
}

/**
 * @brief Run the nntr_quantize binary on a fixture dir into an output dir
 */
bool runQuantize(const std::string &quantize_bin,
                 const std::filesystem::path &fp32_dir,
                 const std::filesystem::path &out_dir,
                 const std::string &extra_args = "") {
  std::string cmd = quantize_bin + " " + fp32_dir.string() + " -o " +
                    out_dir.string() + " --fc_dtype Q4_0";
  if (!extra_args.empty())
    cmd += " " + extra_args;
  cmd += " 2>&1";
  int ret = std::system(cmd.c_str());
  return ret == 0;
}

/**
 * @brief Locate and load a fixture, returning false (and reason) if absent
 */
bool tryLoadFixture(const DifferentialModel &model,
                    std::filesystem::path &fixture_dir,
                    ReferenceFixture &fixture, std::string &skip_reason) {
  fixture_dir = findFixtureDir(model.fixture_name);

  if (!std::filesystem::exists(fixture_dir) ||
      !std::filesystem::exists(fixture_dir / "nntr_config.json") ||
      !std::filesystem::exists(fixture_dir / "reference_logits.json")) {
    skip_reason = "Fixtures absent — run the generate_*_reference.py script";
    return false;
  }

  auto fc = loadFixtureConfigs(fixture_dir);
  if (!std::filesystem::exists(fc.weight_path)) {
    skip_reason = "FP32 weight file absent: " + fc.weight_path.string();
    return false;
  }

  fixture = loadReferenceFixture(fixture_dir);
  if (fixture.input_ids.empty() || fixture.reference_logits.empty() ||
      fixture.reference_tokens.empty()) {
    skip_reason = "Fixture JSON arrays are empty";
    return false;
  }
  return true;
}

/**
 * @brief Locate and load an embedding fixture, returning false (and reason)
 *        if the fixture or its FP32 weights are absent
 */
bool tryLoadEmbeddingFixture(const DifferentialModel &model,
                             std::filesystem::path &fixture_dir,
                             ReferenceFixture &fixture,
                             std::string &skip_reason) {
  fixture_dir = findFixtureDir(model.fixture_name);

  if (!std::filesystem::exists(fixture_dir) ||
      !std::filesystem::exists(fixture_dir / "nntr_config.json") ||
      !std::filesystem::exists(fixture_dir / "reference_embedding.json")) {
    skip_reason = "Fixtures absent — run the generate_*_reference.py script";
    return false;
  }

  auto fc = loadFixtureConfigs(fixture_dir);
  if (!std::filesystem::exists(fc.weight_path)) {
    skip_reason = "FP32 weight file absent: " + fc.weight_path.string();
    return false;
  }

  fixture = loadReferenceFixture(fixture_dir);
  if (fixture.reference_embedding.empty()) {
    skip_reason = "Fixture reference_embedding.json is empty";
    return false;
  }
  return true;
}

// ===========================================================================
// mmap sidecar package helpers
// ===========================================================================

/**
 * @brief Read a whole file into a byte string
 */
std::string readFileBytes(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("failed to open " + path.string());

  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

/**
 * @brief config.json section that carries the model dimensions
 * @note Gemma-family fixtures nest them under "text_config".
 */
const json &modelDimSection(const json &cfg) {
  return cfg.contains("text_config") ? cfg["text_config"] : cfg;
}

/**
 * @brief Bytes one quantized row occupies, mirroring loadGgmlManifest()
 * @return 0 when @a datatype / @a size has no sidecar row-block encoding
 */
size_t sidecarRowBytes(const std::string &datatype, unsigned int size) {
  if (datatype == "q4_0")
    return (size % 32 == 0) ? static_cast<size_t>(size) / 32 * 18 : 0;
  if (datatype == "q6_k")
    return (size % 256 == 0) ? static_cast<size_t>(size) / 256 * 210 : 0;
  return 0;
}

/**
 * @brief Build an UNTIED FP32 source package from a committed fixture
 *
 * Every committed CausalLM fixture is tie_word_embeddings=true, and the mmap
 * sidecar split is defined only for an UNTIED lookup table -- a tied
 * embedding-0 is a "tie_word_embeddings" node, which structurally has no
 * sidecar path -- so a sidecar package cannot be generated from a fixture as
 * committed. Untying is config.json's tie_word_embeddings (the sole owner of
 * the rule, for base Transformer and Gemma4 alike), plus one bin edit, because
 * an untied graph reads a head record that a tied graph does not:
 *
 *   embedding0          "embedding_layer", weight [vocab, dim]
 *   output_of_causallm  "lm_head",         weight [dim, vocab]  (transposed;
 *                                          lm_head.cpp:92 "LMHead layer's
 *                                          tensor dim is transposed dim")
 *
 * In a tied checkpoint lm_head.weight IS embed_tokens.weight, so the head
 * record is the transpose of the embedding table -- exactly what the
 * production converters write for an untied checkpoint
 * (res/qwen3/qwen3-0.6b/weight_converter.py:108, transpose=True). Which edit
 * that takes depends on what the fixture generator already put in the bin:
 *
 *  APPEND_TRANSPOSED_HEAD   the tied bin has no head record at all
 *      (weight_converter.py:107 skips it when tied) -- append one.
 *
 *  REPLACE_TRAILING_HEAD    the tied bin ends with a head record that is the
 *      embedding table saved UNTRANSPOSED (generate_gemma4_reference.py:217
 *      writes embed_tokens.weight as-is). A tied graph never reads it, so the
 *      orientation went unnoticed; an untied graph does, so replace the
 *      trailing record with the transpose.
 *
 * Either way the result is numerically the fixture's own model, which the
 * caller proves by re-running the fixture's HF reference logits through it
 * before anything is quantized -- so a wrong edit here fails loudly instead of
 * silently making the rest of the test agree with itself.
 *
 * @return false with @a reason set when the fixture cannot supply what the
 *         chosen strategy needs
 */
bool makeUntiedSourcePackage(const std::filesystem::path &fixture_dir,
                             const std::filesystem::path &out_dir,
                             SidecarUntieStrategy strategy,
                             std::string &reason) {
  auto model_cfg =
    causallm::LoadJsonFile((fixture_dir / "config.json").string());
  auto nntr_cfg =
    causallm::LoadJsonFile((fixture_dir / "nntr_config.json").string());

  const std::string bin_name = nntr_cfg["model_file_name"].get<std::string>();
  const auto src_bin = fixture_dir / bin_name;
  if (!std::filesystem::exists(src_bin)) {
    reason = "FP32 weight file absent: " + src_bin.string();
    return false;
  }

  const json &dims = modelDimSection(model_cfg);
  if (!dims.contains("vocab_size") || !dims.contains("hidden_size")) {
    reason = "fixture config.json has no vocab_size / hidden_size";
    return false;
  }

  const auto vocab = dims["vocab_size"].get<unsigned int>();
  const auto dim = dims["hidden_size"].get<unsigned int>();
  const size_t elements = static_cast<size_t>(vocab) * dim;
  const size_t table_bytes = elements * sizeof(float);

  std::string bytes = readFileBytes(src_bin);
  if (bytes.size() < table_bytes) {
    reason = "fixture bin is smaller than its embedding table";
    return false;
  }

  std::vector<float> table(elements);
  std::memcpy(table.data(), bytes.data(), table_bytes);

  std::vector<float> transposed(elements);
  for (unsigned int row = 0; row < vocab; ++row)
    for (unsigned int col = 0; col < dim; ++col)
      transposed[static_cast<size_t>(col) * vocab + row] =
        table[static_cast<size_t>(row) * dim + col];

  if (strategy == SidecarUntieStrategy::REPLACE_TRAILING_HEAD)
    bytes.resize(bytes.size() - table_bytes);
  bytes.append(reinterpret_cast<const char *>(transposed.data()), table_bytes);
  model_cfg["tie_word_embeddings"] = false;

  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  for (const char *aux : {"generation_config.json", "tokenizer.json"})
    std::filesystem::copy_file(
      fixture_dir / aux, out_dir / aux,
      std::filesystem::copy_options::overwrite_existing);

  writeFile(out_dir / bin_name, bytes);
  writeFile(out_dir / "config.json", model_cfg.dump(2));
  writeFile(out_dir / "nntr_config.json", nntr_cfg.dump(2));
  return true;
}

} // namespace

/**
 * @brief Run the FP32 differential checks for a model against its fixture
 */
void runFp32DifferentialChecks(const DifferentialModel &model) {
  std::filesystem::path fixture_dir;
  ReferenceFixture fixture;
  std::string skip_reason;
  if (!tryLoadFixture(model, fixture_dir, fixture, skip_reason))
    GTEST_SKIP() << skip_reason;

  auto fc = loadFixtureConfigs(fixture_dir);

  // Use separate model instances for prefill-logits and greedy-generate so
  // that internal KV-cache state from the first pass does not affect the
  // second.
  auto m_logits = model.make_model(fc.model_cfg, fc.gen_cfg, fc.nntr_cfg);
  m_logits->initializeModel();
  m_logits->loadWeight(fc.weight_path.string());

  std::vector<float> got_logits;
  ASSERT_NO_THROW(got_logits =
                    m_logits->prefillLogitsFromIds(fixture.input_ids));
  expectLogitsNear(got_logits, fixture.reference_logits,
                   fixture.logits_atol_fp32);

  auto m_gen = model.make_model(fc.model_cfg, fc.gen_cfg, fc.nntr_cfg);
  m_gen->initializeModel();
  m_gen->loadWeight(fc.weight_path.string());

  std::vector<unsigned int> got_tokens;
  ASSERT_NO_THROW(got_tokens = m_gen->greedyGenerateFromIds(
                    fixture.input_ids, fixture.reference_tokens.size()));
  expectTokenPrefixMatch(got_tokens, fixture.reference_tokens,
                         fixture.reference_tokens.size());
}

/**
 * @brief Run the Q4_0 differential checks for a model against its fixture
 */
void runQ40DifferentialChecks(const DifferentialModel &model) {
  std::filesystem::path fixture_dir;
  ReferenceFixture fixture;
  std::string skip_reason;
  if (!tryLoadFixture(model, fixture_dir, fixture, skip_reason))
    GTEST_SKIP() << skip_reason;

  const char *quantize_bin_env = std::getenv(QUANTIZE_BIN_ENV);
  if (!quantize_bin_env || std::string(quantize_bin_env).empty())
    GTEST_SKIP() << "NNTR_QUANTIZE_BIN not set — Q4_0 test skipped";
  const std::string quantize_bin(quantize_bin_env);

  auto q4_dir = std::filesystem::temp_directory_path() /
                ("nntrainer_" + model.fixture_name + "_q40_ref");
  std::filesystem::remove_all(q4_dir);
  std::filesystem::create_directories(q4_dir);

  ASSERT_TRUE(runQuantize(quantize_bin, fixture_dir, q4_dir))
    << "nntr_quantize failed — check that the FP32 fixture is valid";

  auto q4_nntr_cfg_path = q4_dir / "nntr_config.json";
  ASSERT_TRUE(std::filesystem::exists(q4_nntr_cfg_path))
    << "nntr_quantize did not produce nntr_config.json";

  std::string q4_bin_name;
  {
    std::ifstream f(q4_nntr_cfg_path);
    auto q4_cfg_j = json::parse(f);
    q4_bin_name = q4_cfg_j["model_file_name"].get<std::string>();
  }
  auto q4_bin_path = q4_dir / q4_bin_name;
  ASSERT_TRUE(std::filesystem::exists(q4_bin_path))
    << "Q4_0 weight file not found: " << q4_bin_path;

  // --- Load Q4_0 model ---
  // The generated config names its files relative to the package directory
  // (see NNTR_CONFIG_PATH_KEYS): resolve them the way the runtime loaders do,
  // then pin the tokenizer to the fixture copy.
  auto fc = loadFixtureConfigs(fixture_dir);
  auto q4_nntr_cfg = causallm::LoadJsonFile(q4_nntr_cfg_path.string());
  resolvePackagePaths(q4_nntr_cfg, q4_dir);
  q4_nntr_cfg["tokenizer_file"] = (fixture_dir / "tokenizer.json").string();

  auto q4_model = model.make_model(fc.model_cfg, fc.gen_cfg, q4_nntr_cfg);
  q4_model->initializeModel();
  q4_model->loadWeight(q4_bin_path.string());

  std::vector<float> q4_logits;
  ASSERT_NO_THROW(q4_logits =
                    q4_model->prefillLogitsFromIds(fixture.input_ids));

  // Q4_0 logits vs HF FP32 reference
  expectLogitsNear(q4_logits, fixture.reference_logits,
                   fixture.logits_atol_q40);

  // --- Load nntrainer FP32 model for a direct comparison ---
  auto fp32_fc = loadFixtureConfigs(fixture_dir);
  auto fp32_model =
    model.make_model(fp32_fc.model_cfg, fp32_fc.gen_cfg, fp32_fc.nntr_cfg);
  fp32_model->initializeModel();
  fp32_model->loadWeight(fp32_fc.weight_path.string());

  std::vector<float> fp32_logits;
  ASSERT_NO_THROW(fp32_logits =
                    fp32_model->prefillLogitsFromIds(fixture.input_ids));

  // Q4_0 logits vs nntrainer FP32 logits
  expectLogitsNear(q4_logits, fp32_logits, fixture.logits_atol_q40);

  // --- Greedy prefix check ---
  std::vector<unsigned int> q4_tokens;
  ASSERT_NO_THROW(q4_tokens = q4_model->greedyGenerateFromIds(
                    fixture.input_ids, fixture.reference_tokens.size()));
  expectTokenPrefixMatch(q4_tokens, fixture.reference_tokens,
                         fixture.prefix_match_min);
}

/**
 * @brief Run the FP32 differential checks for an embedding model
 */
void runFp32EmbeddingDifferentialChecks(const DifferentialModel &model) {
  std::filesystem::path fixture_dir;
  ReferenceFixture fixture;
  std::string skip_reason;
  if (!tryLoadEmbeddingFixture(model, fixture_dir, fixture, skip_reason))
    GTEST_SKIP() << skip_reason;

  auto fc = loadFixtureConfigs(fixture_dir);

  // Drive the model in embedding mode. Point module_config_path at the
  // fixture's modules.json (when present) so the SentenceTransformer pooling /
  // normalize pipeline resolves relative to the fixture directory.
  fc.nntr_cfg["model_type"] = "Embedding";
  auto modules_path = fixture_dir / "modules.json";
  if (std::filesystem::exists(modules_path))
    fc.nntr_cfg["module_config_path"] = modules_path.string();

  auto m = model.make_model(fc.model_cfg, fc.gen_cfg, fc.nntr_cfg);
  m->initializeModel();
  m->loadWeight(fc.weight_path.string());

  std::vector<float> got;
  ASSERT_NO_THROW(
    got = m->embedPrompt(fixture.prompt, fixture.reference_embedding.size()));
  expectEmbeddingNear(got, fixture.reference_embedding, fixture.embedding_atol,
                      fixture.cosine_min);
}

/**
 * @brief Run Q4_0 differential checks for an embedding model
 *
 * Quantizes the FP32 fixture with nntr_quantize, then checks:
 *  1. Q4_0 embedding vs HF FP32 reference (cosine + atol)
 *  2. Q4_0 embedding vs nntrainer FP32 embedding (cosine + atol)
 *
 * Skips gracefully when NNTR_QUANTIZE_BIN is unset or when the model
 * architecture is not supported by nntr_quantize (e.g. BERT, DeBERTa).
 */
void runQ40EmbeddingDifferentialChecks(const DifferentialModel &model) {
  std::filesystem::path fixture_dir;
  ReferenceFixture fixture;
  std::string skip_reason;
  if (!tryLoadEmbeddingFixture(model, fixture_dir, fixture, skip_reason))
    GTEST_SKIP() << skip_reason;

  const char *quantize_bin_env = std::getenv(QUANTIZE_BIN_ENV);
  if (!quantize_bin_env || std::string(quantize_bin_env).empty())
    GTEST_SKIP() << "NNTR_QUANTIZE_BIN not set — Q4_0 test skipped";
  const std::string quantize_bin(quantize_bin_env);

  auto q4_dir = std::filesystem::temp_directory_path() /
                ("nntrainer_" + model.fixture_name + "_q40_emb");
  std::filesystem::remove_all(q4_dir);
  std::filesystem::create_directories(q4_dir);

  if (!runQuantize(quantize_bin, fixture_dir, q4_dir))
    GTEST_SKIP() << "nntr_quantize does not support this architecture — Q4_0 "
                    "test skipped";

  auto q4_nntr_cfg_path = q4_dir / "nntr_config.json";
  ASSERT_TRUE(std::filesystem::exists(q4_nntr_cfg_path))
    << "nntr_quantize did not produce nntr_config.json";

  std::string q4_bin_name;
  {
    std::ifstream f(q4_nntr_cfg_path);
    auto q4_cfg_j = json::parse(f);
    q4_bin_name = q4_cfg_j["model_file_name"].get<std::string>();
  }
  auto q4_bin_path = q4_dir / q4_bin_name;
  ASSERT_TRUE(std::filesystem::exists(q4_bin_path))
    << "Q4_0 weight file not found: " << q4_bin_path;

  // --- Load Q4_0 model in embedding mode ---
  // Same package-relative path contract as runQ40DifferentialChecks.
  auto fc = loadFixtureConfigs(fixture_dir);
  auto q4_nntr_cfg = causallm::LoadJsonFile(q4_nntr_cfg_path.string());
  resolvePackagePaths(q4_nntr_cfg, q4_dir);
  q4_nntr_cfg["tokenizer_file"] = (fixture_dir / "tokenizer.json").string();
  q4_nntr_cfg["model_type"] = "Embedding";
  auto modules_path = fixture_dir / "modules.json";
  if (std::filesystem::exists(modules_path))
    q4_nntr_cfg["module_config_path"] = modules_path.string();

  auto q4_model = model.make_model(fc.model_cfg, fc.gen_cfg, q4_nntr_cfg);
  q4_model->initializeModel();
  q4_model->loadWeight(q4_bin_path.string());

  std::vector<float> q4_got;
  ASSERT_NO_THROW(q4_got = q4_model->embedPrompt(
                    fixture.prompt, fixture.reference_embedding.size()));

  // Q4_0 embedding vs HF FP32 reference
  expectEmbeddingNear(q4_got, fixture.reference_embedding,
                      fixture.embedding_atol_q40, fixture.cosine_min_q40);

  // --- Load nntrainer FP32 model for a direct comparison ---
  auto fp32_fc = loadFixtureConfigs(fixture_dir);
  fp32_fc.nntr_cfg["model_type"] = "Embedding";
  if (std::filesystem::exists(modules_path))
    fp32_fc.nntr_cfg["module_config_path"] = modules_path.string();

  auto fp32_model =
    model.make_model(fp32_fc.model_cfg, fp32_fc.gen_cfg, fp32_fc.nntr_cfg);
  fp32_model->initializeModel();
  fp32_model->loadWeight(fp32_fc.weight_path.string());

  std::vector<float> fp32_got;
  ASSERT_NO_THROW(fp32_got = fp32_model->embedPrompt(
                    fixture.prompt, fixture.reference_embedding.size()));

  // Q4_0 embedding vs nntrainer FP32 embedding
  expectEmbeddingNear(q4_got, fp32_got, fixture.embedding_atol_q40,
                      fixture.cosine_min_q40);
}

/**
 * @brief Run the mmap-sidecar package checks for a model against its fixture
 */
void runSidecarPackageChecks(const DifferentialModel &model,
                             SidecarUntieStrategy strategy) {
  std::filesystem::path fixture_dir;
  ReferenceFixture fixture;
  std::string skip_reason;
  if (!tryLoadFixture(model, fixture_dir, fixture, skip_reason))
    GTEST_SKIP() << skip_reason;

  const char *quantize_bin_env = std::getenv(QUANTIZE_BIN_ENV);
  if (!quantize_bin_env || std::string(quantize_bin_env).empty())
    GTEST_SKIP() << "NNTR_QUANTIZE_BIN not set — sidecar package test skipped";
  const std::string quantize_bin(quantize_bin_env);

  const auto tmp_dir = std::filesystem::temp_directory_path();
  const auto src_dir =
    tmp_dir / ("nntrainer_" + model.fixture_name + "_sidecar_src");
  const auto side_dir =
    tmp_dir / ("nntrainer_" + model.fixture_name + "_sidecar_pkg");
  const auto single_dir =
    tmp_dir / ("nntrainer_" + model.fixture_name + "_sidecar_single");

  std::string reason;
  if (!makeUntiedSourcePackage(fixture_dir, src_dir, strategy, reason))
    GTEST_SKIP() << reason;

  // --- (1) The untied source is the fixture's model -------------------------
  // Without this the rest of the test could agree with itself on a
  // mis-assembled package and prove nothing.
  {
    auto src_fc = loadFixtureConfigs(src_dir);
    auto src_model =
      model.make_model(src_fc.model_cfg, src_fc.gen_cfg, src_fc.nntr_cfg);
    src_model->initializeModel();
    src_model->loadWeight(src_fc.weight_path.string());

    std::vector<float> src_logits;
    ASSERT_NO_THROW(src_logits =
                      src_model->prefillLogitsFromIds(fixture.input_ids));
    expectLogitsNear(src_logits, fixture.reference_logits,
                     fixture.logits_atol_fp32);
  }

  // --- (2) One source, two packages ----------------------------------------
  for (const auto &dir : {side_dir, single_dir}) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
  }

  ASSERT_TRUE(runQuantize(quantize_bin, src_dir, side_dir,
                          "--embd_dtype Q4_0 --sidecar on"))
    << "nntr_quantize --sidecar on failed on the untied source package";
  ASSERT_TRUE(runQuantize(quantize_bin, src_dir, single_dir,
                          "--embd_dtype Q4_0 --no-sidecar"))
    << "nntr_quantize --no-sidecar failed on the untied source package";

  auto side_cfg =
    causallm::LoadJsonFile((side_dir / "nntr_config.json").string());
  auto single_cfg =
    causallm::LoadJsonFile((single_dir / "nntr_config.json").string());

  // --- (3) The split happened, and the twin's did not -----------------------
  ASSERT_TRUE(side_cfg.contains("embedding_file_name"))
    << "--sidecar on produced a package with no embedding_file_name";
  EXPECT_FALSE(single_cfg.contains("embedding_file_name"));
  EXPECT_FALSE(single_cfg.contains("ple_file_name"));

  // --- (4) Manifest arithmetic, and the byte partition ----------------------
  // The union of the emitted files is byte-for-byte the file the single-file
  // package holds: every payload occurs in it exactly once, and what remains
  // once they are spliced out is the slim bin. This is the invariant the whole
  // rung rests on, checked on real generated bytes rather than asserted.
  const auto side_bin =
    side_dir / side_cfg["model_file_name"].get<std::string>();
  const auto single_bin =
    single_dir / single_cfg["model_file_name"].get<std::string>();

  const std::string slim = readFileBytes(side_bin);
  const std::string single = readFileBytes(single_bin);
  std::string residue = single;
  size_t payload_total = 0;

  for (const char *key : {"embedding_file_name", "ple_file_name"}) {
    if (!side_cfg.contains(key))
      continue;

    const auto manifest_path =
      side_dir / side_cfg[key].get<std::string>();
    ASSERT_TRUE(std::filesystem::exists(manifest_path))
      << "sidecar manifest named by " << key << " does not exist: "
      << manifest_path;

    auto manifest = causallm::LoadJsonFile(manifest_path.string());
    const auto rows = manifest["rows"].get<unsigned int>();
    const auto size = manifest["size"].get<unsigned int>();
    const auto datatype = manifest["datatype"].get<std::string>();
    const size_t row_bytes = sidecarRowBytes(datatype, size);
    ASSERT_NE(row_bytes, 0u)
      << "manifest datatype '" << datatype << "' with size " << size
      << " has no row-block encoding the loader accepts";

    const auto payload_path =
      side_dir / manifest["lut-path"].get<std::string>();
    ASSERT_TRUE(std::filesystem::exists(payload_path))
      << "sidecar payload does not exist: " << payload_path;

    const std::string payload = readFileBytes(payload_path);
    EXPECT_EQ(payload.size(), static_cast<size_t>(rows) * row_bytes)
      << "payload " << payload_path.filename() << " disagrees with rows("
      << rows << ") * row_bytes(" << row_bytes << ")";
    payload_total += payload.size();

    const auto at = residue.find(payload);
    ASSERT_NE(at, std::string::npos)
      << "sidecar payload " << payload_path.filename()
      << " does not occur in the single-file package";
    residue.erase(at, payload.size());
  }

  EXPECT_EQ(slim.size() + payload_total, single.size());
  EXPECT_TRUE(residue == slim)
    << "the sidecar package is not a byte partition of the single-file package";

  // --- (5) Both packages load, and answer identically ----------------------
  // Loading goes through the same package-relative path resolution every
  // runtime loader performs; the CWD here is the test runner's, never the
  // package directory, so an unresolved embedding_file_name fails right here.
  auto load_logits = [&](const std::filesystem::path &dir) {
    auto cfg = causallm::LoadJsonFile((dir / "config.json").string());
    auto gen_cfg =
      causallm::LoadJsonFile((dir / "generation_config.json").string());
    auto nntr_cfg = causallm::LoadJsonFile((dir / "nntr_config.json").string());
    resolvePackagePaths(nntr_cfg, dir);
    nntr_cfg["tokenizer_file"] = (dir / "tokenizer.json").string();

    const auto bin =
      dir / nntr_cfg["model_file_name"].get<std::string>();
    auto runner = model.make_model(cfg, gen_cfg, nntr_cfg);
    runner->initializeModel();
    runner->loadWeight(bin.string());
    return runner->prefillLogitsFromIds(fixture.input_ids);
  };

  std::vector<float> side_logits;
  ASSERT_NO_THROW(side_logits = load_logits(side_dir))
    << "the generated SIDECAR package failed to load from " << side_dir;

  std::vector<float> single_logits;
  ASSERT_NO_THROW(single_logits = load_logits(single_dir))
    << "the generated single-file package failed to load from " << single_dir;

  ASSERT_EQ(side_logits.size(), single_logits.size());
  size_t differing = 0;
  for (size_t i = 0; i < side_logits.size(); ++i)
    if (side_logits[i] != single_logits[i])
      ++differing;
  EXPECT_EQ(differing, 0u)
    << differing << " of " << side_logits.size()
    << " logits differ between the sidecar package and its single-file twin";

  expectLogitsNear(side_logits, fixture.reference_logits,
                   fixture.logits_atol_q40);
}

} // namespace causallm_test
