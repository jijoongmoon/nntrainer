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
#include <cxxabi.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "causal_lm.h"
#include "chat_template.h"
#include "gauss2_5_causallm.h"
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
#include "multilingual_tinybert_16mb.h"
#include <factory.h>
#ifdef ENABLE_QNN
#include "gauss3_6_qnn.h"
#include "gauss3_8_qnn.h"
#include "gauss3_8_vision_encoder_qnn.h"

#endif
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

using json = nlohmann::json;
using causallm::multimodal_pointer;

/**
 * @brief Per-handle state for a loaded CausalLM model instance.
 *
 * Each handle may carry one or more sub-models so that compositions like
 * vision-encoder + LLM can live behind a single handle. The vectors are
 * kept parallel: models[i] ↔ architectures[i] ↔ model_dirs[i] ↔
 * initialization_duration_ms[i]. The single-model API paths
 * (runModelHandle / runModelHandleStreaming) operate on models[0] and
 * ignore the rest; the multimodal API drives the full set.
 *
 * Note: the legacy non-handle API (loadModel / runModel / ...) is
 * implemented on top of a single static "default" instance of this struct
 * so that existing callers (e.g. test_api) keep working unchanged.
 */
struct CausalLmModel
{
  std::mutex mtx;
  std::vector<std::unique_ptr<causallm::Transformer>> models;
  std::vector<std::string> architectures;
  std::vector<std::string> model_dirs;
  std::string last_output;
  std::string native_lib_dir;
  std::vector<double> initialization_duration_ms;
  bool initialized = false;
};

// Globals shared across all handles — options set via setOptions() apply
// process-wide regardless of which handle is active.
static std::mutex g_registry_mutex;
static bool g_use_chat_template = true;
static bool g_verbose = false;
static std::string g_last_output = "";
static double g_initialization_duration_ms = 0.0;
static causallm::ChatTemplate g_chat_template;
static std::string g_formatted_template;
static std::string g_chat_template_name = "default";

// Default handle backing the legacy non-handle API.
static CausalLmModel &get_default_handle()
{
  static CausalLmModel instance;
  return instance;
}

static std::map<std::string, std::string> g_model_path_map = {
    {"QWEN3-0.6B", "qwen3-0.6b"},
    {"GAUSS2.5-1B", "gauss2.5-1b"},
    {"QWEN3-1.7B-Q40", "qwen3-1.7b-q40-arm"},
    {"GAUSS3.6", "gauss-3.6"},
    {"TINY_BERT", "tiny_bert"},

#ifdef ENABLE_QNN
    {"GAUSS3.6-QNN", "gauss-3.6-qnn"},
    {"GAUSS3.8-QNN", "gauss-3.8-qnn"},
    {"GAUSS3.8-VE-QNN", "gauss-3.8-vencoder-qnn"},
    {"GAUSS3.8-VIT-QNN", "gauss-3.8-vit-qnn"},
#endif
};

/**
 * @brief RegisteredModel
 */
struct RegisteredModel
{
  std::string arch_name;
  ModelRuntimeConfig config;
};
static std::map<std::string, RegisteredModel> g_model_registry;
static std::map<std::string, ModelArchConfig> g_arch_config_map;

// Internal C++ registration functions — called from model_config.cpp
// These bypass extern "C" PLT and write directly to our static maps.
namespace quick_dot_ai
{

  void register_arch(const char *arch_name, ModelArchConfig config)
  {
    std::string name(arch_name);
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    g_arch_config_map[name] = config;
  }

  void register_model(const char *model_name, const char *arch_name,
                      ModelRuntimeConfig config)
  {
    std::string name(model_name);
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    std::string aname(arch_name);
    std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);
    g_model_registry[name] = {aname, config};
  }

} // namespace quick_dot_ai

// Helper to register models (similar to main.cpp) ensuring factory is
// populated. Factory registration is singleton and persistent, but we do it
// once here to be sure. Since mquiain.cpp is not linked, we must duplicate
// registration or share it. Assuming this lib is used independently of
// main.cpp.
static void register_models()
{
  static std::once_flag flag;
  std::call_once(flag, []()
                 {
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
        "Qwen3MoeForCausalLM",
        [](json cfg, json generation_cfg, json nntr_cfg) {
          return std::make_unique<causallm::Qwen3MoECausalLM>(
              cfg, generation_cfg, nntr_cfg);
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
    causallm::Factory::Instance().registerModel(
        "Gauss2_5ForCausalLM",
        [](json cfg, json generation_cfg, json nntr_cfg) {
          return std::make_unique<causallm::Gauss2_5_Causallm>(
              cfg, generation_cfg, nntr_cfg);
        });
  causallm::Factory::Instance().registerModel(
    "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::MultilingualTinyBert>(
        cfg, generation_cfg, nntr_cfg);
    });

#ifdef ENABLE_QNN
    causallm::Factory::Instance().registerModel(
        "Gauss_3_6_QNN", [](json cfg, json generation_cfg, json nntr_cfg) {
          return std::make_unique<causallm::Gauss3_6_QNN>(cfg, generation_cfg,
                                                          nntr_cfg);
        });
    causallm::Factory::Instance().registerModel(
        "Gauss_3_8_QNN", [](json cfg, json generation_cfg, json nntr_cfg) {
          return std::make_unique<causallm::Gauss3_8_QNN>(cfg, generation_cfg,
                                                          nntr_cfg);
        });
    // causallm::Factory::Instance ().registerModel ("Gauss_3_8_Visual_VIT_QNN",
    //     [] (json cfg, json generation_cfg, json nntr_cfg) {
    //       return std::make_unique<causallm::Gauss3_8_VIT_QNN> (
    //           cfg, generation_cfg, nntr_cfg);
    //     });
    causallm::Factory::Instance().registerModel(
        "Gauss_3_8_VEncoder_QNN",
        [](json cfg, json generation_cfg, json nntr_cfg) {
          return std::make_unique<causallm::Gauss3_8_Vision_Encoder_QNN>(
              cfg, generation_cfg, nntr_cfg);
        });
#endif
    // Register built-in configurations
    quick_dot_ai::register_builtin_configs(); });
}

static const char *get_model_name_from_type(ModelType type)
{
  switch (type)
  {
  case CAUSAL_LM_MODEL_QWEN3_0_6B:
    return "QWEN3-0.6B";
  case CAUSAL_LM_MODEL_GAUSS2_5:
    return "GAUSS2.5-1B";
  case CAUSAL_LM_MODEL_QWEN3_1_7B_Q40:
    return "QWEN3-1.7B-Q40";
  case CAUSAL_LM_MODEL_GAUSS3_6:
    return "GAUSS3.6";
  case CAUSAL_LM_MODEL_TINY_BERT:
    return "TINY_BERT";
#ifdef ENABLE_QNN
  case CAUSAL_LM_MODEL_GAUSS3_6_QNN:
    return "GAUSS3.6-QNN";
  case CAUSAL_LM_MODEL_GAUSS3_8_QNN:
    return "GAUSS3.8-QNN";
  case CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN:
    return "GAUSS3.8-VE-QNN";
  case CAUSAL_LM_MODEL_GAUSS3_8_VIT_QNN:
    return "GAUSS3.8-VIT-QNN";
#endif
  default:
    return nullptr;
  }
}

static std::string apply_chat_template(const std::string &architecture,
                                       const std::string &input)
{
  // Use dynamic chat template from tokenizer_config.json if available
  if (g_chat_template.isAvailable())
  {
    return g_chat_template.apply(input);
  }

  LOGE("----------------APPLY CHAT FALLBACKS!!!!!!-------------");

  // Fallback: hardcoded per-architecture templates
  if (architecture == "LlamaForCausalLM")
  {
    // Llama 2/3 chat format: [INST] {prompt} [/INST]
    return "[INST] " + input + " [/INST]";
  }
  else if (architecture == "Qwen2ForCausalLM" ||
           architecture == "Qwen3ForCausalLM" ||
           architecture == "Qwen3MoeForCausalLM" ||
           architecture == "Qwen3SlimMoeForCausalLM" ||
           architecture == "Qwen3CachedSlimMoeForCausalLM")
  {
    // Qwen chat format
    // <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    return "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
  }
  else if (architecture == "Gemma3ForCausalLM")
  {
    // Gemma chat format:
    // <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    return "<start_of_turn>user\n" + input +
           "<end_of_turn>\n<start_of_turn>model\n";
  }
  else if (architecture == "Gauss_3_6_QNN" ||
           architecture == "Gauss_3_8_QNN")
  {
    return "<|begin_of_text|><|turn_start|>System\n<|turn_end|>\n<|turn_start|>"
           "User\n" +
           input + "\n<|turn_end|>\n<|turn_start|>Assistant\n";
  }
  return input;
}

static std::string get_quantization_suffix(ModelQuantizationType type)
{
  return "";
  switch (type)
  {
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
                                      ModelQuantizationType quant_type)
{
  std::string path_upper = model_key;
  std::transform(path_upper.begin(), path_upper.end(), path_upper.begin(),
                 ::toupper);

  std::string base_dir_name = "";

  // 1. Try to find base directory name from map
  if (g_model_path_map.find(path_upper) != g_model_path_map.end())
  {
    base_dir_name = g_model_path_map[path_upper];
  }
  else
  {
    // Fallback: use lowercased key as base dir name if not found in map
    // or just return empty? For restricted API, we should probably fail
    // earlier, but here we can return constructed path.
    base_dir_name = path_upper;
    std::transform(base_dir_name.begin(), base_dir_name.end(),
                   base_dir_name.begin(), ::tolower);
  }

  std::string model_path =
      "/models/" + base_dir_name + get_quantization_suffix(quant_type);

  return model_path;
}

/**
 * @brief Rebase path-like keys of a sub-model nntr_config.json onto @p sub_dir.
 *
 * Called once per sub-model inside the multi-model branch of
 * load_into_handle so that downstream code (Factory::create, load_weight)
 * sees absolute paths — mirrors the inline fixups the single-model path
 * already performs for model_file_name / binary_config_path / ...
 *
 * Absolute values (leading '/') are left untouched so the caller can
 * override a specific file with a system-wide path if they want.
 */
static void fix_paths(json &nntr_cfg, const std::string &sub_dir)
{
  static const char *kKeys[] = {
      "tokenizer_file",
      "model_file_name",
      "binary_config_path",
      "image_newline_path",
      "embedding_file_name",
  };
  for (const char *k : kKeys)
  {
    if (!nntr_cfg.contains(k) || !nntr_cfg[k].is_string())
      continue;
    std::string v = nntr_cfg[k].get<std::string>();
    if (v.empty() || v[0] == '/')
      continue;
    nntr_cfg[k] = sub_dir + "/" + v;
  }
}

static bool check_file_exists(const std::string &path)
{
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static void validate_models()
{
  LOGD("[DEBUG] Validating model files...");
  // Iterate over all known model names in map
  for (auto const &[key, val] : g_model_path_map)
  {
    // We want to check for each Quantization Type if it exists
    // List of quant types to check: UNKNOWN (default), W4A32, W16A16, W32A32
    std::vector<ModelQuantizationType> quant_types = {
        CAUSAL_LM_QUANTIZATION_UNKNOWN, CAUSAL_LM_QUANTIZATION_W4A32,
        CAUSAL_LM_QUANTIZATION_W16A16, CAUSAL_LM_QUANTIZATION_W32A32};

    for (auto qt : quant_types)
    {
      std::string quant_suffix = get_quantization_suffix(qt);

      std::string lookup_key = key;
      if (qt != CAUSAL_LM_QUANTIZATION_UNKNOWN)
      {
        std::transform(quant_suffix.begin(), quant_suffix.end(),
                       quant_suffix.begin(), ::toupper); // "-W4A32"
        lookup_key += quant_suffix;
      }

      // Resolve path for this combination
      std::string resolved_path = "." + resolve_model_path(key, qt);

      if (g_model_registry.find(lookup_key) != g_model_registry.end())
      {
        // CASE 1: Configuration is registered in model_config.cpp
        // For these models, we only check if the binary weight file exists.
        // The configurations (config.json, etc.) are embedded in the library.
        RegisteredModel &rm = g_model_registry[lookup_key];
        std::string bin_file_name = rm.config.model_file_name;
        std::string full_path = resolved_path + "/" + bin_file_name;

        if (check_file_exists(full_path))
        {
          LOGD("  [OK] Reg Config: %s -> %s", lookup_key.c_str(), full_path.c_str());
        }
        else
        {
          LOGD("  [FAIL] Reg Config: %s -> Missing binary: %s", lookup_key.c_str(), full_path.c_str());
        }
      }
      else
      {
        // CASE 2: No internal config, but model type exists (via map
        // iteration). For these models, we require external configuration files
        // (config.json, nntr_config.json) to be present in the directory.
        if (check_file_exists(resolved_path))
        {
          bool has_config = check_file_exists(resolved_path + "/config.json");
          bool has_nntr =
              check_file_exists(resolved_path + "/nntr_config.json");

          if (has_config && has_nntr)
          {
            LOGD("  [OK] External Config: %s -> %s", lookup_key.c_str(), resolved_path.c_str());
            // Optional: Parse nntr_config to check bin
            try
            {
              json nntr =
                  causallm::LoadJsonFile(resolved_path + "/nntr_config.json");
              if (nntr.contains("model_file_name"))
              {
                std::string bin = nntr["model_file_name"];
                if (check_file_exists(resolved_path + "/" + bin))
                {
                  LOGD("       (Binary confirmed: %s)", bin.c_str());
                }
                else
                {
                  LOGD("       (MISSING BINARY: %s)", bin.c_str());
                }
              }
            }
            catch (...)
            {
            }
          }
          else
          {
            LOGD("  [FAIL] External Config: %s -> Missing configs in %s", lookup_key.c_str(), resolved_path.c_str());
          }
        }
      }
    }
  }
}

ErrorCode setOptions(Config config)
{
  // Currently no options are being handled
  g_use_chat_template = config.use_chat_template;
  g_verbose = config.verbose;
  g_chat_template_name = (config.chat_template_name != nullptr)
                             ? config.chat_template_name
                             : "default";
  if (config.debug_mode)
  {
    // Ensure models are registered so we can validate them
    register_models();
    validate_models();
  }
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModelArchitecture(const char *arch_name,
                                    ModelArchConfig config)
{
  if (arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModel(const char *model_name, const char *arch_name,
                        ModelRuntimeConfig config)
{
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
 * Populates the given handle's model / architecture / init-duration
 * vectors on success. Takes the handle's own mutex so two concurrent
 * loads on the same handle are serialized, while loads on different
 * handles run in parallel. A separate registry mutex protects
 * g_model_registry / g_arch_config_map during lookup.
 *
 * Dispatch in CASE 2 (file-based):
 *   - If the top-level nntr_config.json has both "architectures" (string
 *     array) and "model_dirs" (string array) of equal non-zero length,
 *     loads one sub-model per entry (e.g. vision encoder + LLM).
 *   - Otherwise loads a single model from the resolved directory using
 *     the pre-existing flow.
 */
static ErrorCode load_into_handle(CausalLmModel &h, BackendType compute,
                                  ModelType modeltype,
                                  ModelQuantizationType quant_type,
                                  const char *native_lib_dir)
{
  LOGD("[DEBUG] load_into_handle: START");
  LOGD("[DEBUG]   compute: %d", compute);
  LOGD("[DEBUG]   modeltype: %d", modeltype);
  LOGD("[DEBUG]   quant_type: %d", quant_type);

  auto start_init = std::chrono::high_resolution_clock::now();

  const char *target_model_name = get_model_name_from_type(modeltype);
  if (target_model_name == nullptr)
  {
    LOGE("[DEBUG] load_into_handle: Invalid modeltype");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  LOGD("[DEBUG] load_into_handle: target_model_name = %s %d", target_model_name,
       modeltype);

  // Ensure models/configs are registered (thread-safe via call_once)
  LOGD("[DEBUG] load_into_handle: Calling register_models...");
  register_models();
  LOGD("[DEBUG] load_into_handle: register_models done");

  std::lock_guard<std::mutex> lock(h.mtx);
  try
  {

    // Check if it's a registered in-memory config
    std::string input_name = std::string(target_model_name);
    std::string input_name_upper = input_name;
    std::transform(input_name_upper.begin(), input_name_upper.end(),
                   input_name_upper.begin(), ::toupper);
    LOGD("[DEBUG] load_into_handle: input_name = %s", input_name.c_str());

    std::string quant_suffix = "";
    switch (quant_type)
    {
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
    LOGD("[DEBUG] load_into_handle: lookup_name = %s", lookup_name.c_str());

    json cfg;
    json generation_cfg;
    json nntr_cfg;
    std::string model_dir_path;
    std::string abs_model_dir;
    std::string base_dir =
        "/sdcard/Android/data/com.example.sampletestapp/files";

    // Snapshot registry entries under the registry mutex so concurrent
    // loads on different handles don't race with each other (or with
    // registerModel / registerModelArchitecture).
    std::lock_guard<std::mutex> reg_lock(g_registry_mutex);

    // Check in-memory map first
    // if (g_model_registry.find(lookup_name) != g_model_registry.end()) {

    // always goto case2
    if (0)
    {
      LOGD("[DEBUG] load_into_handle: CASE 1 - Internal config found for %s", lookup_name.c_str());
      // ------------------------------------------------------------------------
      // CASE 1: Model Configuration is Internal (Registered in
      // model_config.cpp)
      // ------------------------------------------------------------------------
      // In this case, we do NOT load config.json or nntr_config.json from disk.
      // We only locate the binary weight file.
      RegisteredModel &rm = g_model_registry[lookup_name];

      // Find architecture config
      if (g_arch_config_map.find(rm.arch_name) == g_arch_config_map.end())
      {
        LOGE("[DEBUG] load_into_handle: Architecture '%s' not found for model '%s'",
             rm.arch_name.c_str(), lookup_name.c_str());
        return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
      }
      LOGD("[DEBUG] load_into_handle: arch_name = %s", rm.arch_name.c_str());
      ModelArchConfig &ac = g_arch_config_map[rm.arch_name];
      ModelRuntimeConfig &rc = rm.config;

      // Strategy: Resolve path to find the weight file
      model_dir_path = "." + resolve_model_path(target_model_name, quant_type);
      LOGD("[DEBUG] load_into_handle: model_dir_path = %s", model_dir_path.c_str());

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
      if (ac.sliding_window != UINT_MAX)
      {
        cfg["sliding_window"] = ac.sliding_window;
      }
      else
      {
        cfg["sliding_window"] = nullptr;
      }
      cfg["sliding_window_pattern"] = ac.sliding_window_pattern;
      cfg["architectures"] = {std::string(ac.architecture)};

      if (ac.num_eos_token_ids > 0)
      {
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

      // tokenizer_file path is set later from abs_model_dir in the shared
      // post-processing block below.
      (void)rc.tokenizer_file;

      if (strlen(rc.lmhead_dtype) > 0)
      {
        nntr_cfg["lmhead_dtype"] = std::string(rc.lmhead_dtype);
      }

      std::vector<unsigned int> bad_ids;
      for (unsigned int i = 0; i < rc.num_bad_word_ids; ++i)
        bad_ids.push_back(rc.bad_word_ids[i]);
      nntr_cfg["bad_word_ids"] = bad_ids;
    }
    else
    {
      LOGD("[DEBUG] load_into_handle: CASE 2 - External config (file-based)");
      // --------------------------------------------------
      // CASE 2: External Model Configuration (File-based)
      // --------------------------------------------------
      // The model type is registered (enum), but specific configuration for
      // this quantization is not in memory. We must load config.json and
      // nntr_config.json from the model directory
      model_dir_path = resolve_model_path(target_model_name, quant_type);
      LOGD("[DEBUG] load_into_handle: model_dir_path = %s",
           model_dir_path.c_str());

      abs_model_dir = base_dir + model_dir_path;
      LOGD("[DEBUG] load_into_handle: abs_model_dir = %s",
           abs_model_dir.c_str());

      // Top-level nntr_config.json is read once and used for both
      //   (a) multi-model dispatch (architectures[] + model_dirs[]), and
      //   (b) the single-model fallback below.
      json top_nntr =
          causallm::LoadJsonFile(abs_model_dir + "/nntr_config.json");

      LOGD("[DEBUG] load_into_handle: abs_model_dir = %s",
           abs_model_dir.c_str());

      LOGD("[DEBUG] load_into_handle: top_nntr = %s",
           (abs_model_dir + "/nntr_config.json").c_str());

      const bool is_multi =
          top_nntr.contains("architectures") &&
          top_nntr["architectures"].is_array() &&
          top_nntr.contains("model_dirs") &&
          top_nntr["model_dirs"].is_array() &&
          !top_nntr["architectures"].empty() &&
          top_nntr["architectures"].size() == top_nntr["model_dirs"].size();

      if(top_nntr.contains("use_chat_template"))
      {
        g_use_chat_template = top_nntr["use_chat_template"].get<bool>();
      }

      LOGD("[DEBUG] load_into_handle: abs_model_dir = %d %d %d %d %d %d", top_nntr.contains("architectures"), top_nntr["architectures"].is_array(), top_nntr.contains("model_dirs"), top_nntr["model_dirs"].is_array(), top_nntr["architectures"].size(), top_nntr["model_dirs"].size());

      if (is_multi)
      {
        // ----------------------------------------------------------------
        // Multi-model branch.
        //
        //   top_nntr_config.json:
        //     { "architectures": ["ArchA", "ArchB"],
        //       "model_dirs":   ["sub_a",  "sub_b"] }
        //
        // Each sub_dir = abs_model_dir + "/" + model_dirs[i] owns its own
        // config.json / generation_config.json / nntr_config.json +
        // weights. The top-level architectures[i] wins over any
        // "architectures" entry inside sub-config — one source of truth.
        // ----------------------------------------------------------------
        auto archs =
            top_nntr["architectures"].get<std::vector<std::string>>();
        auto dirs = top_nntr["model_dirs"].get<std::vector<std::string>>();
        LOGD("[DEBUG] load_into_handle: MULTI-MODEL spec (N=%zu)",
             archs.size());

        for (size_t i = 0; i < archs.size(); ++i)
        {
          const std::string &arch_i = archs[i];
          const std::string sub_dir = abs_model_dir + "/" + dirs[i];
          LOGD("[DEBUG]   [%zu] arch=%s dir=%s", i, arch_i.c_str(),
               sub_dir.c_str());

          json sub_cfg = causallm::LoadJsonFile(sub_dir + "/config.json");

          json sub_gen;
          if (check_file_exists(sub_dir + "/generation_config.json"))
          {
            sub_gen =
                causallm::LoadJsonFile(sub_dir + "/generation_config.json");
          }

          json sub_nntr =
              causallm::LoadJsonFile(sub_dir + "/nntr_config.json");

          fix_paths(sub_nntr, sub_dir);

          // Optional per-sub-model overrides from the top-level config.
          // Lets callers flip flags like uses_embedding / add keys like
          // embedding_file_name without duplicating the sub-model's own
          // nntr_config.json. fix_paths is run again so any newly
          // introduced path-like key (e.g. embedding_file_name) is
          // resolved relative to sub_dir just like the native keys.
          if (top_nntr.contains("model_options") &&
              top_nntr["model_options"].is_array() &&
              i < top_nntr["model_options"].size() &&
              top_nntr["model_options"][i].is_object())
          {
            for (auto it = top_nntr["model_options"][i].begin();
                 it != top_nntr["model_options"][i].end(); ++it)
            {
              sub_nntr[it.key()] = it.value();
              LOGD("[DEBUG]   override sub[%zu] %s", i, it.key().c_str());
            }
            fix_paths(sub_nntr, sub_dir);
          }
          if (sub_nntr.contains("lora_path"))
          {
            LOGD("lora_path : %s",
                 sub_nntr["lora_path"].get<std::string>().c_str());
            std::string lora_path =
                sub_dir + "/" + sub_nntr["lora_path"].get<std::string>();
            sub_nntr["lora_path"] = lora_path;
            LOGD("lora_path is now %s", lora_path.c_str());
          }

          auto m = causallm::Factory::Instance().create(arch_i, sub_cfg,
                                                        sub_gen, sub_nntr);
          if (!m)
          {
            LOGE("[DEBUG] load_into_handle: Factory::create returned nullptr "
                 "for sub-model %zu (arch=%s)",
                 i, arch_i.c_str());
            return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
          }

          auto sub_t0 = std::chrono::high_resolution_clock::now();
          if (native_lib_dir != nullptr && strlen(native_lib_dir) > 0)
          {
            setenv("ADSP_LIBRARY_PATH", native_lib_dir, 1);
            m->initialize(std::string(native_lib_dir));
          }
          else
          {
            m->initialize();
          }

          std::string weight_file =
              sub_nntr.contains("model_file_name")
                  ? sub_nntr["model_file_name"].get<std::string>()
                  : (sub_dir + "/pytorch_model.bin");
          m->load_weight(weight_file);
          auto sub_t1 = std::chrono::high_resolution_clock::now();
          double sub_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(sub_t1 -
                                                                    sub_t0)
                  .count();

          h.models.push_back(std::move(m));
          h.architectures.push_back(arch_i);
          h.model_dirs.push_back(sub_dir);
          h.initialization_duration_ms.push_back(sub_ms);
          LOGD("[DEBUG]   [%zu] loaded (%.1f ms)", i, sub_ms);

          // Load chat template from tokenizer_config.json if available.
          std::string tc_path = sub_dir + "/tokenizer_config.json";
          if (check_file_exists(tc_path))
          {
            g_chat_template =
                causallm::ChatTemplate::fromFile(tc_path, g_chat_template_name);
            if (g_chat_template.isAvailable())
            {
              std::cout
                  << "[Info] Chat template loaded from tokenizer_config.json"
                  << std::endl;
            }
            else
            {
              std::cerr << "[Warning] tokenizer_config.json found but chat "
                           "template could "
                           "not be loaded. Falling back to hardcoded templates."
                        << std::endl;
            }
          }
        }

        if (native_lib_dir != nullptr)
          h.native_lib_dir = native_lib_dir;
        h.initialized = true;

        auto finish_init = std::chrono::high_resolution_clock::now();
        auto e2e = std::chrono::duration_cast<std::chrono::milliseconds>(
                       finish_init - start_init)
                       .count();
        LOGD("[DEBUG] load_into_handle: MULTI-MODEL SUCCESS "
             "(%zu models, %ld ms e2e)",
             h.models.size(), e2e);
        return CAUSAL_LM_ERROR_NONE;
      }

      // -------------------- single-model fallback --------------------
      LOGD("single cfg : %s", (abs_model_dir + "/config.json").c_str());
      cfg = causallm::LoadJsonFile(abs_model_dir + "/config.json");

      if (check_file_exists(abs_model_dir + "/generation_config.json"))
      {
        generation_cfg =
            causallm::LoadJsonFile(abs_model_dir + "/generation_config.json");
      }

      nntr_cfg = std::move(top_nntr);

      if (nntr_cfg.contains("lora_path"))
      {
        nntr_cfg["lora_path"] = "";
      }

      LOGD("single tokenizer : %s",
           (abs_model_dir + "/tokenizer.json").c_str());

      if (nntr_cfg.contains("tokenizer_file"))
      {
        nntr_cfg["tokenizer_file"] = abs_model_dir + "/tokenizer.json";
      }
    }

    // Load chat template from tokenizer_config.json if available.
    std::string tc_path = abs_model_dir + "/tokenizer_config.json";
    if (check_file_exists(tc_path))
    {
      g_chat_template =
          causallm::ChatTemplate::fromFile(tc_path, g_chat_template_name);
      if (g_chat_template.isAvailable())
      {
        LOGD("[Info] Chat template loaded from tokenizer_config.json");
      }
      else
      {
        LOGE("[Warning] tokenizer_config.json found but chat template could not be loaded. Falling back to hardcoded templates.");
      }
    }
    else
    {
      g_chat_template = causallm::ChatTemplate();
      LOGE("[Warning] tokenizer_config.json not found in %s. Using hardcoded chat templates.", model_dir_path.c_str());
    }

    // Construct weight file path
    std::string weight_file_name;
    if (nntr_cfg.contains("model_file_name"))
    {
      weight_file_name = nntr_cfg["model_file_name"].get<std::string>();
    }
    else
    {
      weight_file_name = "pytorch_model.bin";
    }

    const std::string weight_file = abs_model_dir + "/" + weight_file_name;
    LOGD("[DEBUG] load_into_handle: weight_file = %s", weight_file.c_str());

    nntr_cfg["model_file_name"] = weight_file;
    if (nntr_cfg.contains("binary_config_path"))
    {
      std::string str = nntr_cfg["binary_config_path"].get<std::string>();
      nntr_cfg["binary_config_path"] = abs_model_dir + "/" + str;
      LOGD("[DEBUG] bianry config data: file = %s",
           nntr_cfg["binary_config_path"].get<std::string>().c_str());
    }
    if (nntr_cfg.contains("image_newline_path"))
    {
      std::string str = nntr_cfg["image_newline_path"].get<std::string>();
      nntr_cfg["image_newline_path"] = abs_model_dir + "/" + str;
      LOGD("[DEBUG] new line config data: file = %s",
           nntr_cfg["image_newline_path"].get<std::string>().c_str());
    }
    if (nntr_cfg.contains("embedding_file_name"))
    {
      std::string str = nntr_cfg["embedding_file_name"].get<std::string>();
      nntr_cfg["embedding_file_name"] = abs_model_dir + "/" + str;
    }

    LOGD("[DEBUG] -------------------------- asdfasdfasdfasdfasdfasdf ");

    // Determine architecture from config or ModelType
    // Priority: Config file architecture > ModelType mapping (fallback)
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty())
    {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    }
    else
    {
      // No fallback mapping from specific ModelType instances to generic
      // architecture strings for now, as specific types should have config or
      // be loaded from valid file with config.json
      LOGE("[DEBUG] load_into_handle: No architecture found in config");
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
    LOGD("[DEBUG] load_into_handle: architecture = %s", architecture.c_str());

    LOGD("[DEBUG] load_into_handle: Creating model via Factory...%s ",
         architecture.c_str());

    auto m = causallm::Factory::Instance().create(architecture, cfg,
                                                  generation_cfg, nntr_cfg);
    if (!m)
    {
      LOGE("[DEBUG] load_into_handle: Factory::create returned nullptr");
      return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
    }
    LOGD("[DEBUG] load_into_handle: Model created successfully");

    if (native_lib_dir != nullptr)
      h.native_lib_dir = native_lib_dir;

    LOGD("[DEBUG] load_into_handle: Calling model->initialize()...");
    if (native_lib_dir != nullptr && strlen(native_lib_dir) > 0)
    {
      setenv("ADSP_LIBRARY_PATH", native_lib_dir, 1);
      m->initialize(std::string(native_lib_dir));
    }
    else
    {
      m->initialize();
    }
    LOGD("[DEBUG] load_into_handle: model->initialize() done");

    LOGD("[DEBUG] load_into_handle: Calling model->load_weight()...");
    m->load_weight(weight_file);
    LOGD("[DEBUG] load_into_handle: model->load_weight() done");

    auto finish_init = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        finish_init - start_init);

    h.models.push_back(std::move(m));
    h.architectures.push_back(architecture);
    h.model_dirs.push_back(abs_model_dir);
    h.initialization_duration_ms.push_back(
        static_cast<double>(init_duration.count()));
    h.initialized = true;

    LOGD("[DEBUG] load_into_handle: SINGLE SUCCESS (init took %ld ms)",
         init_duration.count());
  }
  catch (...)
  {
    // RTTI may not match across shared libraries — query the current
    // exception's typeinfo directly via the Itanium ABI hook. This
    // works even when catching by concrete types fails due to typeinfo
    // duplication between libnntrainer.so and libquick_dot_ai_api.so.
    const std::type_info *ti = abi::__cxa_current_exception_type();
    const char *raw = ti ? ti->name() : "(null)";
    int status = 0;
    char *demangled =
        (ti != nullptr) ? abi::__cxa_demangle(raw, nullptr, nullptr, &status)
                        : nullptr;
    LOGE("[DEBUG] load_into_handle: unknown exception, type=%s",
         demangled ? demangled : raw);
    std::free(demangled);

    // Also try once more via rethrow — in case std::exception RTTI does
    // match from this catch-site (we already tried above but leaving
    // this as a second chance is cheap).
    try
    {
      throw;
    }
    catch (const std::exception &e)
    {
      LOGE("[DEBUG] load_into_handle: rethrown std::exception what()=%s",
           e.what());
    }
    catch (const std::invalid_argument &e)
    {
      LOGE("[DEBUG] load_into_handle: rethrown std::exception what()=%s",
           e.what());
    }
    catch (...)
    {
      LOGE("[DEBUG] load_into_handle: rethrown still non-std");
    }
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }
  LOGD("[DEBUG] load_into_handle: END (returning CAUSAL_LM_ERROR_NONE)");
  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core runner shared by runModel and runModelHandle.
 */
static ErrorCode run_on_handle(CausalLmModel &h, const char *inputTextPrompt,
                               const char **outputText)
{
  if (inputTextPrompt == nullptr || outputText == nullptr)
  {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty() || !h.models[0])
  {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try
  {
    auto &model = *h.models[0];
    const std::string &architecture = h.architectures[0];

    std::string input(inputTextPrompt);

    if (g_use_chat_template)
    {
      input = apply_chat_template(architecture, input);
    }

// We assume single batch request for this API
#if defined(_WIN32)
    model.run(std::wstring(input.begin(), input.end()), false, L"", L"",
              g_verbose);
#else
    model.run(input, false, "", "", g_verbose);
#endif

    h.last_output = model.getOutput(0);
    *outputText = h.last_output.c_str();
  }
  catch (const std::exception &e)
  {
    LOGE("Exception in runModel: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core metrics fetcher shared by getPerformanceMetrics and its
 *        handle-based counterpart.
 *
 * Reports models[0] runtime metrics. initialization_duration_ms is the
 * sum over all sub-models this handle owns.
 */
static ErrorCode metrics_on_handle(CausalLmModel &h,
                                   PerformanceMetrics *metrics)
{
  if (metrics == nullptr)
  {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty() || !h.models[0])
  {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try
  {
    auto *model = h.models[0].get();
    if (!model->hasRun())
    {
      return CAUSAL_LM_ERROR_INFERENCE_NOT_RUN;
    }
    auto im = model->getPerformanceMetrics();
    metrics->prefill_tokens = im.prefill_tokens;
    metrics->prefill_duration_ms = im.prefill_duration_ms;
    metrics->generation_tokens = im.generation_tokens;
    metrics->generation_duration_ms = im.generation_duration_ms;
    metrics->total_duration_ms = im.total_duration_ms;
    metrics->peak_memory_kb = im.peak_memory_kb;

    double total_init = 0.0;
    for (double d : h.initialization_duration_ms)
      total_init += d;
    metrics->initialization_duration_ms = total_init;
  }
  catch (const std::exception &e)
  {
    LOGE("Exception in getPerformanceMetrics: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/*****************************************************************************
 * Chat Template API - role + content message support
 *****************************************************************************/

static std::vector<causallm::ChatMessage>
convertMessages(const CausalLMChatMessage *messages, size_t num_messages)
{
  std::vector<causallm::ChatMessage> result;
  result.reserve(num_messages);
  for (size_t i = 0; i < num_messages; ++i)
  {
    causallm::ChatMessage msg;
    msg.role = messages[i].role ? messages[i].role : "";
    msg.content = messages[i].content ? messages[i].content : "";
    result.push_back(std::move(msg));
  }
  return result;
}

/**
 * @brief Apply chat template to messages with hardcoded fallback
 */
static std::string
apply_chat_template_messages(const std::string &architecture,
                             const std::vector<causallm::ChatMessage> &messages,
                             bool add_generation_prompt)
{
  if (g_chat_template.isAvailable())
  {
    return g_chat_template.apply(messages, add_generation_prompt);
  }

  std::string result;

  if (architecture == "LlamaForCausalLM")
  {
    for (const auto &msg : messages)
    {
      if (msg.role == "system")
      {
        result += "<<SYS>>\n" + msg.content + "\n<</SYS>>\n\n";
      }
      else if (msg.role == "user")
      {
        result += "[INST] " + msg.content + " [/INST]";
      }
      else if (msg.role == "assistant")
      {
        result += msg.content + "\n";
      }
    }
  }
  else if (architecture == "Qwen2ForCausalLM" ||
           architecture == "Qwen3ForCausalLM" ||
           architecture == "Qwen3MoeForCausalLM" ||
           architecture == "Qwen3SlimMoeForCausalLM" ||
           architecture == "Qwen3CachedSlimMoeForCausalLM")
  {
    for (const auto &msg : messages)
    {
      result += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    if (add_generation_prompt)
    {
      result += "<|im_start|>assistant\n";
    }
  }
  else if (architecture == "Gemma3ForCausalLM")
  {
    for (const auto &msg : messages)
    {
      if (msg.role == "user")
      {
        result += "<start_of_turn>user\n" + msg.content + "<end_of_turn>\n";
      }
      else if (msg.role == "assistant")
      {
        result += "<start_of_turn>model\n" + msg.content + "<end_of_turn>\n";
      }
    }
    if (add_generation_prompt)
    {
      result += "<start_of_turn>model\n";
    }
  }
  else
  {
    for (const auto &msg : messages)
    {
      result += msg.content + "\n";
    }
  }

  return result;
}

ErrorCode applyChatTemplate(const CausalLMChatMessage *messages,
                            size_t num_messages, bool add_generation_prompt,
                            const char **formattedText)
{
  if (messages == nullptr || num_messages == 0 || formattedText == nullptr)
  {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try
  {
    auto &h = get_default_handle();
    std::lock_guard<std::mutex> lock(h.mtx);

    auto chat_messages = convertMessages(messages, num_messages);
    std::string arch =
        h.architectures.empty() ? std::string() : h.architectures[0];
    g_formatted_template =
        apply_chat_template_messages(arch, chat_messages, add_generation_prompt);

    *formattedText = g_formatted_template.c_str();
  }
  catch (const std::exception &e)
  {
    LOGE("Exception in applyChatTemplate: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModelWithMessages(const CausalLMChatMessage *messages,
                               size_t num_messages, bool add_generation_prompt,
                               const char **outputText)
{
  if (outputText == nullptr)
  {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  const char *formattedInput = nullptr;
  ErrorCode err = applyChatTemplate(messages, num_messages,
                                    add_generation_prompt, &formattedInput);
  if (err != CAUSAL_LM_ERROR_NONE)
  {
    return err;
  }

  return runModel(formattedInput, outputText);
}
/*============================================================================
 * Legacy non-handle API implementation
 *============================================================================*/

ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type)
{
  return load_into_handle(get_default_handle(), compute, modeltype, quant_type, nullptr);
}

ErrorCode runModel(const char *inputTextPrompt, const char **outputText)
{
  return run_on_handle(get_default_handle(), inputTextPrompt, outputText);
}

ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics)
{
  return metrics_on_handle(get_default_handle(), metrics);
}

/*============================================================================
 * Handle-based API implementation
 *============================================================================*/

ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                          ModelQuantizationType quant_type,
                          const char *native_lib_dir,
                          CausalLmHandle *out_handle)
{
  LOGD("[DEBUG] loadModelHandle:%d START", __LINE__);
  LOGD("[DEBUG] loadModelHandle:%d   compute: %d", __LINE__, compute);
  LOGD("[DEBUG] loadModelHandle:%d   modeltype: %d", __LINE__, modeltype);
  LOGD("[DEBUG] loadModelHandle:%d   quant_type: %d", __LINE__, quant_type);
  LOGD("[DEBUG] loadModelHandle:%d   native_lib_dir: %s", __LINE__,
       native_lib_dir ? native_lib_dir : "(null)");
  LOGD("[DEBUG] loadModelHandle:%d   out_handle ptr: %p", __LINE__,
       (void *)out_handle);

  if (out_handle == nullptr)
  {
    LOGE("[DEBUG] loadModelHandle:%d out_handle is nullptr", __LINE__);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  auto *h = new (std::nothrow) CausalLmModel();
  if (h == nullptr)
  {
    LOGE("[DEBUG] loadModelHandle:%d Failed to allocate CausalLmModel",
         __LINE__);
    return CAUSAL_LM_ERROR_UNKNOWN;
  }
  LOGD("[DEBUG] loadModelHandle:%d CausalLmModel allocated at %p", __LINE__,
       (void *)h);

  LOGD("[DEBUG] loadModelHandle:%d Calling load_into_handle...", __LINE__);
  ErrorCode ec =
      load_into_handle(*h, compute, modeltype, quant_type, native_lib_dir);
  LOGD("[DEBUG] loadModelHandle:%d load_into_handle returned: %d", __LINE__,
       ec);

  if (ec != CAUSAL_LM_ERROR_NONE)
  {
    LOGE("[DEBUG] loadModelHandle:%d load_into_handle failed, deleting handle",
         __LINE__);
    delete h;
    *out_handle = nullptr;
    return ec;
  }
  *out_handle = h;
  LOGD("[DEBUG] loadModelHandle:%d SUCCESS, handle set to %p", __LINE__,
       (void *)h);
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModelHandle(CausalLmHandle handle, const char *inputTextPrompt,
                         const char **outputText)
{
  if (handle == nullptr)
  {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return run_on_handle(*handle, inputTextPrompt, outputText);
}

ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                      PerformanceMetrics *metrics)
{
  if (handle == nullptr)
  {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return metrics_on_handle(*handle, metrics);
}

ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                  const char *inputTextPrompt,
                                  CausalLmTokenCallback callback,
                                  void *user_data)
{
  LOGD("[DEBUG] runModelHandleStreaming: START");
  LOGD("[DEBUG]   handle: %p", (void *)handle);
  LOGD("[DEBUG]   inputTextPrompt: %s",
       inputTextPrompt ? inputTextPrompt : "(null)");
  LOGD("[DEBUG]   callback: %p", (void *)callback);
  LOGD("[DEBUG]   user_data: %p", user_data);

  if (handle == nullptr || inputTextPrompt == nullptr || callback == nullptr)
  {
    LOGE("[DEBUG] runModelHandleStreaming: INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  LOGD("[DEBUG] runModelHandleStreaming: Acquiring mutex lock...");
  std::lock_guard<std::mutex> lock(h.mtx);
  LOGD("[DEBUG] runModelHandleStreaming: Mutex lock acquired");

  if (!h.initialized || h.models.empty() || !h.models[0])
  {
    LOGE("[DEBUG] runModelHandleStreaming: NOT_INITIALIZED "
         "(initialized=%d, size=%zu)",
         h.initialized, h.models.size());
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  auto *m = h.models[0].get();
  const std::string &architecture = h.architectures[0];
  LOGD("[DEBUG] runModelHandleStreaming: Model is initialized, architecture=%s",
       architecture.c_str());

  // Set up streaming via Transformer interface (works for both CausalLM and
  // QNN models)
  LOGD("[DEBUG] runModelHandleStreaming: Initializing callback streamer...");
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  LOGD("[DEBUG] runModelHandleStreaming: Callback streamer initialized");

  // Safe upcast: CallbackStreamer embeds BaseStreamer as its first
  // field (C-style inheritance), so &streamer.base yields a valid
  // BaseStreamer pointer without reinterpret_cast.
  LOGD("[DEBUG] runModelHandleStreaming: Setting streamer on model...");
  m->setStreamer(&streamer.base);
  LOGD("[DEBUG] runModelHandleStreaming: Streamer set successfully");

  // RAII detach: make sure the dangling stack pointer never survives
  // the return of this function, no matter which exception path we
  // exit through.
  struct Detach
  {
    causallm::Transformer *t;
    ~Detach()
    {
      LOGD("[DEBUG] runModelHandleStreaming::Detach: Clearing streamer");
      t->setStreamer(nullptr);
    }
  } detach_guard{m};

  try
  {
    LOGD("[DEBUG] runModelHandleStreaming: Preparing input text...");
    std::string input(inputTextPrompt);
    LOGD("[DEBUG]   raw input length: %zu", input.length());
    LOGD("[DEBUG]   g_use_chat_template: %d", g_use_chat_template);

    if (g_use_chat_template)
    {
      LOGD("[DEBUG] runModelHandleStreaming: Applying chat template...");
      input = apply_chat_template("", input);
      LOGD("[DEBUG]   templated input length: %zu", input.length());
      LOGD("[DEBUG]   templated input: %s", input.c_str());
      // LOGD("[DEBUG]   templated input preview: %.100s%s", input.c_str(),
      //      input.length() > 100 ? "..." : "");
    }

    LOGD("[DEBUG] runModelHandleStreaming: Calling model->run()...");
#if defined(_WIN32)
    m->run(std::wstring(input.begin(), input.end()), false, L"", L"",
           g_verbose);
#else
    m->run(input, false, "", "", true);
#endif
    LOGD("[DEBUG] runModelHandleStreaming: model->run() completed");

    LOGD("[DEBUG] runModelHandleStreaming: Getting output...");
    h.last_output = m->getOutput(0);
    LOGD("[DEBUG]   output length: %zu", h.last_output.length());
    LOGD("[DEBUG]   output: %s", h.last_output.c_str());

    // Log performance metrics after successful run
    if (m->hasRun())
    {
      auto im = m->getPerformanceMetrics();
      double total_init = 0.0;
      for (double d : h.initialization_duration_ms)
        total_init += d;

      LOGD("[PERF] Performance Metrics:");
      LOGD("[PERF]   prefill_tokens: %u", im.prefill_tokens);
      LOGD("[PERF]   prefill_duration_ms: %.2f", im.prefill_duration_ms);
      LOGD("[PERF]   generation_tokens: %u", im.generation_tokens);
      LOGD("[PERF]   generation_duration_ms: %.2f", im.generation_duration_ms);
      LOGD("[PERF]   total_duration_ms: %.2f", im.total_duration_ms);
      LOGD("[PERF]   peak_memory_kb: %.2f", im.peak_memory_kb);
      LOGD("[PERF]   initialization_duration_ms: %.2f", total_init);

      // Calculate tokens per second for prefill
      if (im.prefill_duration_ms > 0)
      {
        double tokens_per_sec = (im.prefill_tokens * 1000.0) / im.prefill_duration_ms;
        LOGD("[PERF]   prefill_tokens_per_sec: %.2f", tokens_per_sec);
      }
      // Calculate tokens per second for generation
      if (im.generation_duration_ms > 0)
      {
        double tokens_per_sec = (im.generation_tokens * 1000.0) / im.generation_duration_ms;
        LOGD("[PERF]   generation_tokens_per_sec: %.2f", tokens_per_sec);
      }
    }
  }
  catch (const std::exception &e)
  {
    LOGE("[DEBUG] runModelHandleStreaming: Exception caught: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  catch (...)
  {
    LOGE("[DEBUG] runModelHandleStreaming: Unknown exception caught");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  LOGD("[DEBUG] runModelHandleStreaming: END (SUCCESS)");
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode unloadModelHandle(CausalLmHandle handle)
{
  if (handle == nullptr)
  {
    return CAUSAL_LM_ERROR_NONE;
  }
  std::lock_guard<std::mutex> lock(handle->mtx);
  handle->models.clear();
  handle->architectures.clear();
  handle->model_dirs.clear();
  handle->initialization_duration_ms.clear();
  handle->initialized = false;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode destroyModelHandle(CausalLmHandle handle)
{
  if (handle == nullptr)
  {
    return CAUSAL_LM_ERROR_NONE;
  }
  // Take the mutex to make sure no in-flight call on this handle is still
  // running, then release and delete. Any caller that still holds a pointer
  // to the output buffer returned by runModelHandle is reading freed memory
  // after this point — documented as "valid until destroy".
  {
    std::lock_guard<std::mutex> lock(handle->mtx);
    handle->models.clear();
    handle->architectures.clear();
    handle->model_dirs.clear();
    handle->initialization_duration_ms.clear();
    handle->initialized = false;
  }
  delete handle;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode cancelModelHandle(CausalLmHandle handle)
{
  LOGD("[DEBUG] cancelModelHandle: handle=%p", (void *)handle);

  if (handle == nullptr)
  {
    LOGE("[DEBUG] cancelModelHandle: handle is nullptr, returning INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // NOTE: We intentionally do NOT take the mutex here to avoid blocking
  // when run() is holding the lock. The requestStop() method is thread-safe
  // (uses atomic<bool>), and the models vector is not modified during run()
  // (only during load/unload which do take the mutex). This allows immediate
  // cancellation from any thread (e.g., UI cancel button handler).
  LOGD("[DEBUG] cancelModelHandle: checking state without mutex, initialized=%d, models.size=%zu",
       handle->initialized, handle->models.size());

  if (!handle->initialized || handle->models.empty())
  {
    LOGE("[DEBUG] cancelModelHandle: not initialized, returning NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  // Set stop flag on all models (primarily affects models[0] for LLM)
  for (size_t i = 0; i < handle->models.size(); ++i)
  {
    if (handle->models[i])
    {
      LOGD("[DEBUG] cancelModelHandle: calling requestStop() on model[%zu]", i);
      handle->models[i]->requestStop();
    }
  }

  LOGD("[DEBUG] cancelModelHandle: returning NONE (success)");
  return CAUSAL_LM_ERROR_NONE;
}

/*============================================================================
 * Multimodal API Implementation
 *
 * Preconditions: the handle must have been loaded from a multi-model
 * nntr_config.json carrying at least two sub-models. The first sub-model
 * is expected to be the vision encoder and the second the LLM, though
 * the concrete integration (vision encoding + embedding fusion + LLM
 * generation) is still TODO. Single-model handles return
 * CAUSAL_LM_ERROR_UNSUPPORTED.
 *============================================================================*/

#ifdef ENABLE_QNN
/**
 * @brief Shared multimodal pipeline: tokenize, compose
 *        [text_pre | image | text_post] embeddings, attach streamer,
 *        drive llm->run_with_embeddings(). Assumes h.mtx is held and
 *        `llm` and `image_embeds` are valid.
 *
 * Ownership: takes ownership of image_embeds.first (frees via
 * std::free before returning, on both success and failure paths).
 */
static ErrorCode
execute_multimodal_llm(CausalLmModel &h, causallm::Gauss3_8_QNN *llm,
                       causallm::multimodal_pointer image_embeds,
                       const std::string &prompt,
                       CausalLmTokenCallback callback, void *user_data)
{
  auto *tok = llm->getTokenizer();
  if (tok == nullptr)
  {
    LOGE("[DEBUG] execute_multimodal_llm: LLM has no tokenizer");
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  std::vector<int> text_ids = tok->Encode(prompt);
  int32_t image_token_id = tok->TokenToId("<|image|>");

  const size_t bpt = llm->embeddingBytesPerToken();
  if (bpt == 0)
  {
    LOGE("[DEBUG] execute_multimodal_llm: embedding table not loaded "
         "(set uses_embedding=false + embedding_file_name on LLM config)");
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  if (image_embeds.second % bpt != 0)
  {
    LOGE("[DEBUG] execute_multimodal_llm: image_embeds.size=%zu not a "
         "multiple of bpt=%zu",
         image_embeds.second, bpt);
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  const size_t n_image = image_embeds.second / bpt;

  // Locate <|image|> placeholder; if absent, prepend image embeddings.
  auto it_img =
      (image_token_id >= 0)
          ? std::find(text_ids.begin(), text_ids.end(), image_token_id)
          : text_ids.end();
  const bool has_placeholder = (it_img != text_ids.end());
  const size_t img_pos =
      has_placeholder
          ? static_cast<size_t>(std::distance(text_ids.begin(), it_img))
          : 0;
  const size_t n_text_kept = text_ids.size() - (has_placeholder ? 1 : 0);
  const size_t n_total = n_text_kept + n_image;
  LOGD("[DEBUG] execute_multimodal_llm: text=%zu image=%zu total=%zu "
       "placeholder=%d pos=%zu",
       text_ids.size(), n_image, n_total, has_placeholder, img_pos);

  // Compose combined buffer: [pre-image text | image | post-image text].
  std::vector<uint8_t> combined(n_total * bpt);
  uint8_t *dst = combined.data();
  auto copy_text_range = [&](size_t start, size_t end) -> bool
  {
    for (size_t i = start; i < end; ++i)
    {
      const void *e = llm->lookupEmbedding(text_ids[i]);
      LOGD("text_ids[i]: %d", text_ids[i]);
      if (e == nullptr)
      {
        LOGE("[DEBUG] execute_multimodal_llm: lookupEmbedding(%d) null",
             text_ids[i]);
        return false;
      }
      std::memcpy(dst, e, bpt);
      dst += bpt;
    }
    return true;
  };
  if (!copy_text_range(0, img_pos))
  {
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  std::memcpy(dst, image_embeds.first, n_image * bpt);
  dst += n_image * bpt;
  const size_t after_start = has_placeholder ? img_pos + 1 : img_pos;
  if (!copy_text_range(after_start, text_ids.size()))
  {
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  std::free(image_embeds.first);
  image_embeds.first = nullptr;

  // Attach streamer and drive generation.
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  llm->setStreamer(&streamer.base);
  struct Detach
  {
    causallm::Transformer *t;
    ~Detach() { t->setStreamer(nullptr); }
  } detach_guard{llm};

  try
  {
    llm->run_with_embeddings(combined.data(), n_total, text_ids,
                             /*do_sample=*/false,
                             /*log_output=*/g_verbose);
  }
  catch (const std::exception &e)
  {
    LOGE("[DEBUG] execute_multimodal_llm: llm threw: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}
#endif // ENABLE_QNN

ErrorCode runMultimodalHandleStreaming(CausalLmHandle handle,
                                       const char *prompt,
                                       const float *pixelValues, int numPatches,
                                       int originalHeight, int originalWidth,
                                       CausalLmTokenCallback callback,
                                       void *user_data)
{
  LOGD("[DEBUG] runMultimodalHandleStreaming: START");
  LOGD("[DEBUG]   handle=%p", handle);
  LOGD("[DEBUG]   prompt=%s", prompt ? prompt : "(null)");
  LOGD("[DEBUG]   pixelValues=%p", pixelValues);
  LOGD("[DEBUG]   numPatches=%d", numPatches);
  LOGD("[DEBUG]   originalHeight=%d", originalHeight);
  LOGD("[DEBUG]   originalWidth=%d", originalWidth);
  LOGD("[DEBUG]   callback=%p", (void *)callback);
  LOGD("[DEBUG]   user_data=%p", user_data);

  if (handle == nullptr || prompt == nullptr || pixelValues == nullptr ||
      callback == nullptr)
  {
    LOGE("[DEBUG] runMultimodalHandleStreaming: INVALID_PARAMETER"
         " handle=%p prompt=%s pixelValues=%p callback=%p",
         handle, prompt, pixelValues, (void *)callback);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty())
  {
    LOGE("[DEBUG] runMultimodalHandleStreaming: NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  // Multimodal expects the handle to be loaded from a multi-model
  // nntr_config.json (architectures[] + model_dirs[]) with at least
  // [vision_encoder, llm]. A single-model handle cannot drive this path.
  if (h.models.size() < 2)
  {
    LOGE("[DEBUG] runMultimodalHandleStreaming: need >=2 sub-models "
         "(got %zu). Load with multi-model nntr_config.json.",
         h.models.size());
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  LOGD("[DEBUG] runMultimodalHandleStreaming: %zu sub-models loaded",
       h.models.size());
  for (size_t i = 0; i < h.architectures.size(); ++i)
  {
    LOGD("[DEBUG]   models[%zu]: arch=%s dir=%s", i,
         h.architectures[i].c_str(), h.model_dirs[i].c_str());
  }

  // Log pixel values summary (first few values)
  // Note: patch size is fixed at 512x512
  const int PATCH_SIZE = 512;
  long long totalValues = 1LL * numPatches * 3 * PATCH_SIZE * PATCH_SIZE;
  LOGD("[DEBUG]   totalPixelValues=%lld", totalValues);
  if (totalValues > 0 && pixelValues != nullptr)
  {
    LOGD("[DEBUG]   pixelValues[0..4]=%f, %f, %f, %f, %f", pixelValues[0],
         pixelValues[1], pixelValues[2],
         (totalValues > 3 ? pixelValues[3] : 0.0f),
         (totalValues > 4 ? pixelValues[4] : 0.0f));
  }

#ifdef ENABLE_QNN
  auto *vision =
      dynamic_cast<causallm::Gauss3_8_Vision_Encoder_QNN *>(h.models[0].get());
  auto *llm = dynamic_cast<causallm::Gauss3_8_QNN *>(h.models[1].get());
  if (vision == nullptr || llm == nullptr)
  {
    LOGE("[DEBUG] runMultimodalHandleStreaming: unexpected sub-model types "
         "(arch[0]=%s arch[1]=%s)",
         h.architectures.size() > 0 ? h.architectures[0].c_str() : "?",
         h.architectures.size() > 1 ? h.architectures[1].c_str() : "?");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  auto embedding_info = llm->get_embedding_info();
  vision->set_quant_param(embedding_info.first, embedding_info.second);

  // --- Step 1: vision encoder -> image embeddings ---
  const size_t pixel_bytes = static_cast<size_t>(numPatches) * 3 * PATCH_SIZE *
                             PATCH_SIZE * sizeof(float);
  causallm::multimodal_pointer image_in{const_cast<float *>(pixelValues),
                                        pixel_bytes};
  causallm::multimodal_pointer image_embeds{nullptr, 0};
  try
  {
    LOGD("[DEBUG] runMultimodalHandleStreaming: vision->run_image() ...");
    image_embeds = vision->run_image(
        std::string(prompt), image_in, originalHeight, originalWidth,
        /*do_sample=*/false, /*system=*/"", /*tail=*/"",
        /*log_output=*/g_verbose);
    LOGD("[DEBUG] runMultimodalHandleStreaming: vision done ptr=%p size=%zu",
         image_embeds.first, image_embeds.second);
  }
  catch (const std::exception &e)
  {
    LOGE("[DEBUG] runMultimodalHandleStreaming: vision threw: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  LOGD("[DEBUG] runMultimodalHandleStreaming: Preparing input text...");
  std::string input(prompt);
  LOGD("[DEBUG]   raw input length: %zu", input.length());
  LOGD("[DEBUG]   g_use_chat_template: %d", g_use_chat_template);

  if (g_use_chat_template)
  {
    LOGD("[DEBUG] runMultimodalHandleStreaming: Applying chat template...");
    input = apply_chat_template(h.architectures[1], input);
    LOGD("[DEBUG]   templated input length: %zu", input.length());
    LOGD("[DEBUG]   templated input preview: %.100s%s", input.c_str(),
         input.length() > 100 ? "..." : "");
  }

  return execute_multimodal_llm(h, llm, image_embeds, input, callback,
                                user_data);
#else
  LOGE("[DEBUG] runMultimodalHandleStreaming: built without ENABLE_QNN");
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
}

ErrorCode runMultimodalHandle(CausalLmHandle handle,
                              const char *prompt,
                              const float *pixelValues,
                              int numPatches,
                              int originalHeight,
                              int originalWidth,
                              const char **outputText)
{
  LOGD("[DEBUG] runMultimodalHandle: START");
  LOGD("[DEBUG]   handle=%p", handle);
  LOGD("[DEBUG]   prompt=%s", prompt ? prompt : "(null)");
  LOGD("[DEBUG]   pixelValues=%p", pixelValues);
  LOGD("[DEBUG]   numPatches=%d", numPatches);
  LOGD("[DEBUG]   originalHeight=%d", originalHeight);
  LOGD("[DEBUG]   originalWidth=%d", originalWidth);
  LOGD("[DEBUG]   outputText=%p", outputText);

  if (handle == nullptr || prompt == nullptr || pixelValues == nullptr ||
      outputText == nullptr)
  {
    LOGE("[DEBUG] runMultimodalHandle: INVALID_PARAMETER"
         " handle=%p prompt=%s pixelValues=%p outputText=%p",
         handle, prompt, pixelValues, outputText);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty())
  {
    LOGE("[DEBUG] runMultimodalHandle: NOT_INITIALIZED");
    *outputText = nullptr;
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  if (h.models.size() < 2)
  {
    LOGE("[DEBUG] runMultimodalHandle: need >=2 sub-models (got %zu)",
         h.models.size());
    *outputText = nullptr;
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  LOGD("[DEBUG] runMultimodalHandle: %zu sub-models loaded", h.models.size());
  for (size_t i = 0; i < h.architectures.size(); ++i)
  {
    LOGD("[DEBUG]   models[%zu]: arch=%s dir=%s", i,
         h.architectures[i].c_str(), h.model_dirs[i].c_str());
  }

  // Log pixel values summary (first few values)
  // Note: patch size is fixed at 512x512
  const int PATCH_SIZE = 512;
  long long totalValues = 1LL * numPatches * 3 * PATCH_SIZE * PATCH_SIZE;
  LOGD("[DEBUG]   totalPixelValues=%lld", totalValues);
  if (totalValues > 0 && pixelValues != nullptr)
  {
    LOGD("[DEBUG]   pixelValues[0..4]=%f, %f, %f, %f, %f", pixelValues[0],
         pixelValues[1], pixelValues[2],
         (totalValues > 3 ? pixelValues[3] : 0.0f),
         (totalValues > 4 ? pixelValues[4] : 0.0f));
  }

#ifdef ENABLE_QNN
  auto *vision =
      dynamic_cast<causallm::Gauss3_8_Vision_Encoder_QNN *>(h.models[0].get());
  auto *llm = dynamic_cast<causallm::Gauss3_8_QNN *>(h.models[1].get());
  if (vision == nullptr || llm == nullptr)
  {
    LOGE("[DEBUG] runMultimodalHandle: unexpected sub-model types");
    *outputText = nullptr;
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  const size_t pixel_bytes = static_cast<size_t>(numPatches) * 3 * PATCH_SIZE *
                             PATCH_SIZE * sizeof(float);
  causallm::multimodal_pointer image_in{const_cast<float *>(pixelValues),
                                        pixel_bytes};
  causallm::multimodal_pointer image_embeds{nullptr, 0};
  try
  {
    image_embeds = vision->run_image(
        std::string(prompt), image_in, originalHeight, originalWidth,
        /*do_sample=*/false, "", "", /*log_output=*/g_verbose);
  }
  catch (const std::exception &e)
  {
    LOGE("[DEBUG] runMultimodalHandle: vision threw: %s", e.what());
    *outputText = nullptr;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  // Blocking path = streaming path + accumulator callback that appends
  // each delta into h.last_output. *outputText is then served from that
  // same string so it remains valid until the next run/destroy.
  h.last_output.clear();
  auto accumulate_cb = +[](const char *delta, void *user_data) -> int
  {
    auto *s = static_cast<std::string *>(user_data);
    if (delta != nullptr)
      s->append(delta);
    return 0;
  };

  ErrorCode ec = execute_multimodal_llm(h, llm, image_embeds,
                                        std::string(prompt), accumulate_cb,
                                        static_cast<void *>(&h.last_output));
  if (ec != CAUSAL_LM_ERROR_NONE)
  {
    *outputText = nullptr;
    return ec;
  }
  *outputText = h.last_output.c_str();
  return CAUSAL_LM_ERROR_NONE;
#else
  LOGE("[DEBUG] runMultimodalHandle: built without ENABLE_QNN");
  *outputText = nullptr;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
}