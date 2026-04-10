// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.cpp
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "quick_dot_ai_api.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "causal_lm.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "json.hpp"
#include "model_config_internal.h"
#include "qwen2_causallm.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include <factory.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

/**
 * @brief Per-handle state for a loaded CausalLM model instance.
 *
 * The handle-based API (loadModelHandle / runModelHandle / ...) allocates
 * one of these per loaded model, which allows multiple models to live
 * simultaneously and multiple threads to drive different handles in
 * parallel without blocking each other.
 *
 * Note: the legacy non-handle API (loadModel / runModel / ...) is
 * implemented on top of a single static "default" instance of this struct
 * so that existing callers (e.g. test_api) keep working unchanged.
 */
struct CausalLmModel {
  std::mutex mtx;
  std::unique_ptr<causallm::Transformer> model;
  std::string architecture;
  std::string last_output;
  double initialization_duration_ms = 0.0;
  bool initialized = false;
};

// Globals shared across all handles — options set via setOptions() apply
// process-wide regardless of which handle is active.
static std::mutex g_registry_mutex;
static bool g_use_chat_template = false;
static bool g_verbose = false;

// Default handle backing the legacy non-handle API.
static CausalLmModel &get_default_handle() {
  static CausalLmModel instance;
  return instance;
}

static std::map<std::string, std::string> g_model_path_map = {
  {"QWEN3-0.6B", "qwen3-0.6b"},
};

/**
 * @brief RegisteredModel
 */
struct RegisteredModel {
  std::string arch_name;
  ModelRuntimeConfig config;
};
static std::map<std::string, RegisteredModel> g_model_registry;
static std::map<std::string, ModelArchConfig> g_arch_config_map;

// Internal C++ registration functions — called from model_config.cpp
// These bypass extern "C" PLT and write directly to our static maps.
namespace quick_dot_ai {

void register_arch(const char *arch_name, ModelArchConfig config) {
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
}

void register_model(const char *model_name, const char *arch_name,
                    ModelRuntimeConfig config) {
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);
  g_model_registry[name] = {aname, config};
}

} // namespace quick_dot_ai

// Helper to register models (similar to main.cpp)
// ensuring factory is populated.
// @note: Factory registration is singleton and persistent, but we do it once
// here to be sure. Since main.cpp is not linked, we must duplicate registration
// or share it. Assuming this lib is used independently of main.cpp.
static void register_models() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    causallm::Factory::Instance().registerModel(
      "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                    nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                         nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                         nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3SlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3CachedSlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssForCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "GptOssCachedSlimCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    // Register built-in configurations
    register_builtin_model_configs();
  });
}

static const char *get_model_name_from_type(ModelType type) {
  switch (type) {
  case CAUSAL_LM_MODEL_QWEN3_0_6B:
    return "QWEN3-0.6B";
  default:
    return nullptr;
  }
}

static std::string apply_chat_template(const std::string &architecture,
                                       const std::string &input) {
  if (architecture == "LlamaForCausalLM") {
    // Llama 2/3 chat format: [INST] {prompt} [/INST]
    return "[INST] " + input + " [/INST]";
  } else if (architecture == "Qwen2ForCausalLM" ||
             architecture == "Qwen3ForCausalLM" ||
             architecture == "Qwen3MoeForCausalLM" ||
             architecture == "Qwen3SlimMoeForCausalLM" ||
             architecture == "Qwen3CachedSlimMoeForCausalLM") {
    // Qwen chat format
    // <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    // Note: assuming model handles tokenizer specific special tokens or we
    // might need to handle them raw if tokenizer enabled
    return "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
  } else if (architecture == "Gemma3ForCausalLM") {
    // Gemma chat format:
    // <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    return "<start_of_turn>user\n" + input +
           "<end_of_turn>\n<start_of_turn>model\n";
  }
  return input;
}

static std::string get_quantization_suffix(ModelQuantizationType type) {
  switch (type) {
  case CAUSAL_LM_QUANTIZATION_W4A32:
    return "-w4a32";
  case CAUSAL_LM_QUANTIZATION_W16A16:
    return "-w16a16";
  case CAUSAL_LM_QUANTIZATION_W8A16:
    return "-w8a16";
  case CAUSAL_LM_QUANTIZATION_W32A32:
    return "-w32a32";
  default: // W4A32 by default
    return "-w4a32";
  }
}

static std::string resolve_model_path(const std::string &model_key,
                                      ModelQuantizationType quant_type) {
  std::string path_upper = model_key;
  std::transform(path_upper.begin(), path_upper.end(), path_upper.begin(),
                 ::toupper);

  std::string base_dir_name = "";

  // 1. Try to find base directory name from map
  if (g_model_path_map.find(path_upper) != g_model_path_map.end()) {
    base_dir_name = g_model_path_map[path_upper];
  } else {
    // Fallback: use lowercased key as base dir name if not found in map
    // or just return empty? For restricted API, we should probably fail
    // earlier, but here we can return constructed path.
    base_dir_name = path_upper;
    std::transform(base_dir_name.begin(), base_dir_name.end(),
                   base_dir_name.begin(), ::tolower);
  }

  std::string model_path =
    "./models/" + base_dir_name + get_quantization_suffix(quant_type);

  return model_path;
}

static bool check_file_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static void validate_models() {
  std::cout << "[DEBUG] Validating model files..." << std::endl;
  // Iterate over all known model names in map
  for (auto const &[key, val] : g_model_path_map) {
    // We want to check for each Quantization Type if it exists
    // List of quant types to check: UNKNOWN (default), W4A32, W16A16, W32A32
    std::vector<ModelQuantizationType> quant_types = {
      CAUSAL_LM_QUANTIZATION_UNKNOWN, CAUSAL_LM_QUANTIZATION_W4A32,
      CAUSAL_LM_QUANTIZATION_W16A16, CAUSAL_LM_QUANTIZATION_W32A32};

    for (auto qt : quant_types) {
      std::string quant_suffix = get_quantization_suffix(qt);

      std::string lookup_key = key;
      if (qt != CAUSAL_LM_QUANTIZATION_UNKNOWN) {
        std::transform(quant_suffix.begin(), quant_suffix.end(),
                       quant_suffix.begin(), ::toupper); // "-W4A32"
        lookup_key += quant_suffix;
      }

      // Resolve path for this combination
      std::string resolved_path = resolve_model_path(key, qt);

      if (g_model_registry.find(lookup_key) != g_model_registry.end()) {
        // CASE 1: Configuration is registered in model_config.cpp
        // For these models, we only check if the binary weight file exists.
        // The configurations (config.json, etc.) are embedded in the library.
        RegisteredModel &rm = g_model_registry[lookup_key];
        std::string bin_file_name = rm.config.model_file_name;
        std::string full_path = resolved_path + "/" + bin_file_name;

        if (check_file_exists(full_path)) {
          std::cout << "  [OK] Reg Config: " << lookup_key << " -> "
                    << full_path << std::endl;
        } else {
          std::cout << "  [FAIL] Reg Config: " << lookup_key
                    << " -> Missing binary: " << full_path << std::endl;
        }

      } else {
        // CASE 2: No internal config, but model type exists (via map
        // iteration). For these models, we require external configuration files
        // (config.json, nntr_config.json) to be present in the directory.
        if (check_file_exists(resolved_path)) {
          bool has_config = check_file_exists(resolved_path + "/config.json");
          bool has_nntr =
            check_file_exists(resolved_path + "/nntr_config.json");

          if (has_config && has_nntr) {
            std::cout << "  [OK] External Config: " << lookup_key << " -> "
                      << resolved_path << std::endl;
            // Optional: Parse nntr_config to check bin
            try {
              json nntr =
                causallm::LoadJsonFile(resolved_path + "/nntr_config.json");
              if (nntr.contains("model_file_name")) {
                std::string bin = nntr["model_file_name"];
                if (check_file_exists(resolved_path + "/" + bin)) {
                  std::cout << "       (Binary confirmed: " << bin << ")"
                            << std::endl;
                } else {
                  std::cout << "       (MISSING BINARY: " << bin << ")"
                            << std::endl;
                }
              }
            } catch (...) {
            }
          } else {
            std::cout << "  [FAIL] External Config: " << lookup_key
                      << " -> Missing configs in " << resolved_path
                      << std::endl;
          }
        }
      }
    }
  }
}

ErrorCode setOptions(Config config) {
  // Currently no options are being handled
  g_use_chat_template = config.use_chat_template;
  g_verbose = config.verbose;
  if (config.debug_mode) {
    // Ensure models are registered so we can validate them
    register_models();
    validate_models();
  }
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModelArchitecture(const char *arch_name,
                                    ModelArchConfig config) {
  if (arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModel(const char *model_name, const char *arch_name,
                        ModelRuntimeConfig config) {
  if (model_name == nullptr || arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);

  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);

  g_model_registry[name] = {aname, config};
  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core loader shared by loadModel and loadModelHandle.
 *
 * Populates the given handle's model / architecture / init-duration fields
 * on success. Takes the handle's own mutex so two concurrent loads on the
 * same handle are serialized, while loads on different handles run in
 * parallel. A separate registry mutex protects g_model_registry /
 * g_arch_config_map during lookup.
 */
static ErrorCode load_into_handle(CausalLmModel &h, BackendType compute,
                                  ModelType modeltype,
                                  ModelQuantizationType quant_type) {

  auto start_init = std::chrono::high_resolution_clock::now();

  const char *target_model_name = get_model_name_from_type(modeltype);
  if (target_model_name == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // Ensure models/configs are registered (thread-safe via call_once)
  register_models();

  std::lock_guard<std::mutex> lock(h.mtx);
  try {

    // Check if it's a registered in-memory config
    std::string input_name = std::string(target_model_name);
    std::string input_name_upper = input_name;
    std::transform(input_name_upper.begin(), input_name_upper.end(),
                   input_name_upper.begin(), ::toupper);

    std::string quant_suffix = "";
    switch (quant_type) {
    case CAUSAL_LM_QUANTIZATION_W4A32:
      quant_suffix = "-W4A32";
      break;
    case CAUSAL_LM_QUANTIZATION_W16A16:
      quant_suffix = "-W16A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W8A16:
      quant_suffix = "-W8A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W32A32:
      quant_suffix = "-W32A32";
      break;
    default:
      break;
    }
    std::string lookup_name = input_name_upper + quant_suffix;

    json cfg;
    json generation_cfg;
    json nntr_cfg;
    std::string model_dir_path;

    // Snapshot registry entries under the registry mutex so concurrent
    // loads on different handles don't race with each other (or with
    // registerModel / registerModelArchitecture).
    std::lock_guard<std::mutex> reg_lock(g_registry_mutex);

    // Check in-memory map first
    if (g_model_registry.find(lookup_name) != g_model_registry.end()) {
      // ------------------------------------------------------------------------
      // CASE 1: Model Configuration is Internal (Registered in
      // model_config.cpp)
      // ------------------------------------------------------------------------
      // In this case, we do NOT load config.json or nntr_config.json from disk.
      // We only locate the binary weight file.
      RegisteredModel &rm = g_model_registry[lookup_name];

      // Find architecture config
      if (g_arch_config_map.find(rm.arch_name) == g_arch_config_map.end()) {
        std::cerr << "Architecture '" << rm.arch_name
                  << "' not found for model '" << lookup_name << "'"
                  << std::endl;
        return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
      }
      ModelArchConfig &ac = g_arch_config_map[rm.arch_name];
      ModelRuntimeConfig &rc = rm.config;

      // Strategy: Resolve path to find the weight file
      model_dir_path = resolve_model_path(target_model_name, quant_type);

      // Populate JSONs from Arch Struct
      cfg["vocab_size"] = ac.vocab_size;
      cfg["hidden_size"] = ac.hidden_size;
      cfg["intermediate_size"] = ac.intermediate_size;
      cfg["num_hidden_layers"] = ac.num_hidden_layers;
      cfg["num_attention_heads"] = ac.num_attention_heads;
      cfg["head_dim"] = ac.head_dim;
      cfg["num_key_value_heads"] = ac.num_key_value_heads > 0
                                     ? ac.num_key_value_heads
                                     : ac.num_attention_heads;
      cfg["max_position_embeddings"] = ac.max_position_embeddings;
      cfg["rope_theta"] = ac.rope_theta;
      cfg["rms_norm_eps"] = ac.rms_norm_eps;
      cfg["tie_word_embeddings"] = ac.tie_word_embeddings;
      if (ac.sliding_window != UINT_MAX) {
        cfg["sliding_window"] = ac.sliding_window;
      } else {
        cfg["sliding_window"] = nullptr;
      }
      cfg["sliding_window_pattern"] = ac.sliding_window_pattern;
      cfg["architectures"] = {std::string(ac.architecture)};

      if (ac.num_eos_token_ids > 0) {
        std::vector<unsigned int> eos_ids;
        for (unsigned int i = 0; i < ac.num_eos_token_ids; ++i)
          eos_ids.push_back(ac.eos_token_ids[i]);
        generation_cfg["eos_token_id"] = eos_ids;
      }
      generation_cfg["bos_token_id"] = ac.bos_token_id;

      // Populate JSONs from Runtime Struct
      generation_cfg["top_k"] = rc.top_k;
      generation_cfg["top_p"] = rc.top_p;
      generation_cfg["temperature"] = rc.temperature;
      generation_cfg["do_sample"] = false;

      nntr_cfg["batch_size"] = rc.batch_size;
      nntr_cfg["model_type"] = std::string(rc.model_type);
      nntr_cfg["model_tensor_type"] = std::string(rc.model_tensor_type);
      nntr_cfg["init_seq_len"] = rc.init_seq_len;
      nntr_cfg["max_seq_len"] = rc.max_seq_len;
      nntr_cfg["num_to_generate"] = rc.num_to_generate;
      nntr_cfg["fsu"] = rc.fsu;
      nntr_cfg["fsu_lookahead"] = rc.fsu_lookahead;
      nntr_cfg["embedding_dtype"] = std::string(rc.embedding_dtype);
      nntr_cfg["fc_layer_dtype"] = std::string(rc.fc_layer_dtype);
      nntr_cfg["model_file_name"] = std::string(rc.model_file_name);

      std::string t_file = rc.tokenizer_file;
      nntr_cfg["tokenizer_file"] = model_dir_path + "/" + t_file;

      if (strlen(rc.lmhead_dtype) > 0) {
        nntr_cfg["lmhead_dtype"] = std::string(rc.lmhead_dtype);
      }

      std::vector<unsigned int> bad_ids;
      for (unsigned int i = 0; i < rc.num_bad_word_ids; ++i)
        bad_ids.push_back(rc.bad_word_ids[i]);
      nntr_cfg["bad_word_ids"] = bad_ids;

    } else {
      // --------------------------------------------------
      // CASE 2: External Model Configuration (File-based)
      // --------------------------------------------------
      // The model type is registered (enum), but specific configuration for
      // this quantization is not in memory. We must load config.json and
      // nntr_config.json from the model directory
      model_dir_path = resolve_model_path(target_model_name, quant_type);

      // Load configuration files
      cfg = causallm::LoadJsonFile(model_dir_path + "/config.json");
      generation_cfg =
        causallm::LoadJsonFile(model_dir_path + "/generation_config.json");
      nntr_cfg = causallm::LoadJsonFile(model_dir_path + "/nntr_config.json");

      if (nntr_cfg.contains("tokenizer_file")) {
        std::string t_file = nntr_cfg["tokenizer_file"];
        nntr_cfg["tokenizer_file"] = model_dir_path + "/" + t_file;
      }
    }

    // Construct weight file path
    std::string weight_file_name;
    if (nntr_cfg.contains("model_file_name")) {
      weight_file_name = nntr_cfg["model_file_name"].get<std::string>();
    } else {
      weight_file_name =
        "pytorch_model.bin"; // Default fallback if not specified
    }

    const std::string weight_file = model_dir_path + "/" + weight_file_name;

    // Determine architecture from config or ModelType
    // Priority: Config file architecture > ModelType mapping (fallback)
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else {
      // No fallback mapping from specific ModelType instances to generic
      // architecture strings for now, as specific types should have config or
      // be loaded from valid file with config.json
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }

    h.model = causallm::Factory::Instance().create(architecture, cfg,
                                                   generation_cfg, nntr_cfg);
    if (!h.model) {
      return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
    }

    h.model->initialize();
    h.model->load_weight(weight_file);

    h.initialized = true;
    h.architecture = architecture;

    auto finish_init = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_init - start_init);
    h.initialization_duration_ms = init_duration.count();

  } catch (const std::exception &e) {
    std::cerr << "Exception in loadModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  } catch (...) {
    std::cerr << "Unknown exception in loadModel" << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core runner shared by runModel and runModelHandle.
 */
static ErrorCode run_on_handle(CausalLmModel &h, const char *inputTextPrompt,
                               const char **outputText) {
  if (inputTextPrompt == nullptr || outputText == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || !h.model) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    std::string input(inputTextPrompt);

    if (g_use_chat_template) {
      input = apply_chat_template(h.architecture, input);
    }

// We assume single batch request for this API
#if defined(_WIN32)
    h.model->run(std::wstring(input.begin(), input.end()), false, L"", L"",
                 g_verbose);
#else
    h.model->run(input, false, "", "", g_verbose);
#endif

    auto causal_lm_model = dynamic_cast<causallm::CausalLM *>(h.model.get());
    h.last_output = "";
    if (causal_lm_model) {
      h.last_output = causal_lm_model->getOutput(0);
    }

    *outputText = h.last_output.c_str();

  } catch (const std::exception &e) {
    std::cerr << "Exception in runModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core metrics fetcher shared by getPerformanceMetrics and its
 *        handle-based counterpart.
 */
static ErrorCode metrics_on_handle(CausalLmModel &h,
                                   PerformanceMetrics *metrics) {
  if (metrics == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || !h.model) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    auto causal_lm_model = dynamic_cast<causallm::CausalLM *>(h.model.get());
    if (!causal_lm_model) {
      return CAUSAL_LM_ERROR_UNKNOWN;
    }
    if (!causal_lm_model->hasRun()) {
      return CAUSAL_LM_ERROR_INFERENCE_NOT_RUN;
    }
    auto internal_metrics = causal_lm_model->getPerformanceMetrics();
    metrics->prefill_tokens = internal_metrics.prefill_tokens;
    metrics->prefill_duration_ms = internal_metrics.prefill_duration_ms;
    metrics->generation_tokens = internal_metrics.generation_tokens;
    metrics->generation_duration_ms = internal_metrics.generation_duration_ms;
    metrics->total_duration_ms = internal_metrics.total_duration_ms;
    metrics->peak_memory_kb = internal_metrics.peak_memory_kb;
    metrics->initialization_duration_ms = h.initialization_duration_ms;
  } catch (const std::exception &e) {
    std::cerr << "Exception in getPerformanceMetrics: " << e.what()
              << std::endl;
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/*============================================================================
 * Legacy non-handle API implementation
 *============================================================================*/

ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type) {
  return load_into_handle(get_default_handle(), compute, modeltype, quant_type);
}

ErrorCode runModel(const char *inputTextPrompt, const char **outputText) {
  return run_on_handle(get_default_handle(), inputTextPrompt, outputText);
}

ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics) {
  return metrics_on_handle(get_default_handle(), metrics);
}

/*============================================================================
 * Handle-based API implementation
 *============================================================================*/

ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                          ModelQuantizationType quant_type,
                          CausalLmHandle *out_handle) {
  if (out_handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  auto *h = new (std::nothrow) CausalLmModel();
  if (h == nullptr) {
    return CAUSAL_LM_ERROR_UNKNOWN;
  }
  ErrorCode ec = load_into_handle(*h, compute, modeltype, quant_type);
  if (ec != CAUSAL_LM_ERROR_NONE) {
    delete h;
    *out_handle = nullptr;
    return ec;
  }
  *out_handle = h;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModelHandle(CausalLmHandle handle, const char *inputTextPrompt,
                         const char **outputText) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return run_on_handle(*handle, inputTextPrompt, outputText);
}

ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                      PerformanceMetrics *metrics) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return metrics_on_handle(*handle, metrics);
}

ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                  const char *inputTextPrompt,
                                  CausalLmTokenCallback callback,
                                  void *user_data) {
  if (handle == nullptr || inputTextPrompt == nullptr || callback == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || !h.model) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  // Streaming is only wired up for causallm::CausalLM subclasses (the
  // setStreamer / registerOutputs hook lives on that class). If the
  // loaded model is something else (e.g. a future sentence-transformer
  // style model) the caller should fall back to runModelHandle().
  auto *causal = dynamic_cast<causallm::CausalLM *>(h.model.get());
  if (causal == nullptr) {
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  // Safe upcast: CallbackStreamer embeds BaseStreamer as its first
  // field (C-style inheritance), so &streamer.base yields a valid
  // BaseStreamer pointer without reinterpret_cast.
  causal->setStreamer(&streamer.base);

  // RAII detach: make sure the dangling stack pointer never survives
  // the return of this function, no matter which exception path we
  // exit through.
  struct Detach {
    causallm::CausalLM *c;
    ~Detach() { c->setStreamer(nullptr); }
  } detach_guard{causal};

  try {
    std::string input(inputTextPrompt);
    if (g_use_chat_template) {
      input = apply_chat_template(h.architecture, input);
    }

#if defined(_WIN32)
    h.model->run(std::wstring(input.begin(), input.end()), false, L"", L"",
                 g_verbose);
#else
    h.model->run(input, false, "", "", g_verbose);
#endif

    h.last_output = causal->getOutput(0);
  } catch (const std::exception &e) {
    std::cerr << "Exception in runModelHandleStreaming: " << e.what()
              << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    std::cerr << "Unknown exception in runModelHandleStreaming" << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode destroyModelHandle(CausalLmHandle handle) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_NONE;
  }
  // Take the mutex to make sure no in-flight call on this handle is still
  // running, then release and delete. Any caller that still holds a pointer
  // to the output buffer returned by runModelHandle is reading freed memory
  // after this point — documented as "valid until destroy".
  {
    std::lock_guard<std::mutex> lock(handle->mtx);
    handle->model.reset();
    handle->initialized = false;
  }
  delete handle;
  return CAUSAL_LM_ERROR_NONE;
}