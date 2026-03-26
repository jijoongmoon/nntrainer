/**
 * @file   test_causallm.cpp
 * @brief  Test CausalLM model construction, compilation, and initialization
 *         for all converted model architectures (Qwen3, Qwen2, Gemma3, Llama).
 *         Verifies the Symbolic Tensor API graph construction works correctly.
 */
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "gemma3_causallm.h"
#include "qwen2_causallm.h"
#include "qwen3_causallm.h"

using json = nlohmann::json;

bool test_model(const std::string &name,
                std::unique_ptr<causallm::CausalLM> model,
                const std::string &weight_path) {
  std::cout << "\n====== Testing " << name << " ======\n";

  try {
    // Step 1: Initialize (constructModel + compile + init)
    std::cout << "  [1] Initializing..." << std::flush;
    model->initialize();
    std::cout << " OK\n";

    // Step 2: Save weights (initialized with "ones" initializer)
    std::cout << "  [2] Saving weights..." << std::flush;
    model->save_weight(weight_path);
    std::cout << " OK\n";

    // Step 3: Load weights back
    std::cout << "  [3] Loading weights..." << std::flush;
    model->load_weight(weight_path);
    std::cout << " OK\n";

    // Step 4: Run forward pass
    std::cout << "  [4] Running inference..." << std::flush;
    model->run("hello", false);
    std::cout << " OK\n";

    std::cout << "  ====== " << name << " PASSED ======\n";
    return true;

  } catch (const std::exception &e) {
    std::cerr << "\n  [FAIL] " << name << ": " << e.what() << "\n";
    return false;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <tokenizer.json> <tmp_dir> [arch...]\n"
              << "  arch: qwen3 qwen2 gemma3 llama (default: all)\n";
    return 1;
  }

  std::string tokenizer_path = argv[1];
  std::string tmp_dir = argv[2];
  std::string weight_path = tmp_dir + "/test_weights.bin";

  // Determine which models to test
  std::vector<std::string> test_archs;
  if (argc > 3) {
    for (int i = 3; i < argc; ++i)
      test_archs.push_back(argv[i]);
  } else {
    test_archs = {"qwen3", "qwen2", "gemma3", "llama"};
  }

  // Common configs
  int hidden_size = 64;
  int num_heads = 4;
  int num_kv_heads = 2;
  int head_dim = hidden_size / num_heads;
  int intermediate_size = 128;
  int vocab_size = 256;
  int num_layers = 2;

  auto make_cfg = [&](const std::string &arch) {
    json cfg;
    cfg["architectures"] = std::vector<std::string>{arch};
    cfg["hidden_size"] = hidden_size;
    cfg["num_hidden_layers"] = num_layers;
    cfg["num_attention_heads"] = num_heads;
    cfg["num_key_value_heads"] = num_kv_heads;
    cfg["intermediate_size"] = intermediate_size;
    cfg["vocab_size"] = vocab_size;
    cfg["max_position_embeddings"] = 128;
    cfg["rope_theta"] = 10000;
    cfg["rms_norm_eps"] = 1e-5;
    cfg["tie_word_embeddings"] = true;
    cfg["head_dim"] = head_dim;
    return cfg;
  };

  json gen;
  gen["eos_token_id"] = std::vector<unsigned int>{2};
  gen["bos_token_id"] = 1;
  gen["do_sample"] = false;

  json nntr;
  nntr["batch_size"] = 1;
  nntr["model_tensor_type"] = "FP32-FP32";
  nntr["init_seq_len"] = 4;
  nntr["max_seq_len"] = 16;
  nntr["num_to_generate"] = 2;
  nntr["embedding_dtype"] = "fp32";
  nntr["fc_layer_dtype"] = "fp32";
  nntr["lmhead_dtype"] = "fp32";
  nntr["tokenizer_file"] = tokenizer_path;
  nntr["model_file_name"] = "test_weights.bin";
  nntr["model_type"] = "CausalLM";
  nntr["bad_word_ids"] = std::vector<unsigned int>{};

  int pass = 0, fail = 0;

  for (const auto &arch : test_archs) {
    std::unique_ptr<causallm::CausalLM> model;
    std::string model_name;

    if (arch == "qwen3") {
      model_name = "Qwen3CausalLM";
      json cfg = make_cfg("Qwen3ForCausalLM");
      model = std::make_unique<causallm::Qwen3CausalLM>(cfg, gen, nntr);
    } else if (arch == "qwen2") {
      model_name = "Qwen2CausalLM";
      json cfg = make_cfg("Qwen2ForCausalLM");
      cfg["is_causal"] = true;
      model = std::make_unique<causallm::Qwen2CausalLM>(cfg, gen, nntr);
    } else if (arch == "gemma3") {
      model_name = "Gemma3CausalLM";
      json cfg = make_cfg("Gemma3ForCausalLM");
      cfg["attn_logit_softcapping"] = 50.0;
      cfg["layer_types"] =
        std::vector<std::string>{"sliding_attention", "global_attention"};
      cfg["sliding_window"] = 64;
      model = std::make_unique<causallm::Gemma3CausalLM>(cfg, gen, nntr);
    } else if (arch == "llama") {
      model_name = "CausalLM (Llama)";
      json cfg = make_cfg("LlamaForCausalLM");
      model = std::make_unique<causallm::CausalLM>(
        cfg, gen, nntr, causallm::ModelType::CAUSALLM);
    } else {
      std::cerr << "Unknown arch: " << arch << "\n";
      fail++;
      continue;
    }

    if (test_model(model_name, std::move(model), weight_path))
      pass++;
    else
      fail++;
  }

  std::cout << "\n======================================\n";
  std::cout << "  RESULTS: " << pass << " passed, " << fail << " failed\n";
  std::cout << "======================================\n";

  return fail > 0 ? 1 : 0;
}
