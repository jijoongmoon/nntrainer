/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file    causal_lm_api.cpp
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "causal_lm_api.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "causal_lm.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "json.hpp"
#include "qwen2_causallm.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include <factory.h>

using json = nlohmann::json;

static std::unique_ptr<causallm::Transformer> g_model;
static std::mutex g_mutex;
static bool g_initialized = false;
static std::string g_architecture = "";
static bool g_use_chat_template = false;

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
  });
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

ErrorCode setOptions(Config config) {
  // Currently no options are being handled
  g_use_chat_template = config.use_chat_template;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    const char *path) {
  std::lock_guard<std::mutex> lock(g_mutex);

  if (path == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    register_models();

    std::string model_path = path;

    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // Construct weight file path
    std::string weight_file_name;
    if (nntr_cfg.contains("model_file_name")) {
      weight_file_name = nntr_cfg["model_file_name"].get<std::string>();
    } else {
      weight_file_name =
        "pytorch_model.bin"; // Default fallback if not specified
    }

    const std::string weight_file = model_path + "/" + weight_file_name;

    // Determine architecture from config or ModelType
    // Priority: Config file architecture > ModelType mapping (fallback)
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else {
      // Fallback mapping
      switch (modeltype) {
      case CAUSAL_LM_MODEL_LLAMA:
        architecture = "LlamaForCausalLM";
        break;
      case CAUSAL_LM_MODEL_QWEN2:
        architecture = "Qwen2ForCausalLM";
        break;
      case CAUSAL_LM_MODEL_QWEN3:
        architecture = "Qwen3ForCausalLM";
        break;
      case CAUSAL_LM_MODEL_QWEN3_MOE:
        architecture = "Qwen3MoeForCausalLM";
        break;
      case CAUSAL_LM_MODEL_GPT_OSS:
        architecture = "GptOssForCausalLM";
        break;
      case CAUSAL_LM_MODEL_GEMMA3:
        architecture = "Gemma3ForCausalLM";
        break;
      default:
        return CAUSAL_LM_ERROR_INVALID_PARAMETER;
      }
    }

    g_model = causallm::Factory::Instance().create(architecture, cfg,
                                                   generation_cfg, nntr_cfg);
    if (!g_model) {
      return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
    }

    g_model->initialize();
    g_model->load_weight(weight_file);

    g_initialized = true;
    g_architecture = architecture;

  } catch (const std::exception &e) {
    std::cerr << "Exception in loadModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModel(const char *inputTextPrompt, char *outputText,
                   size_t output_size) {
  if (!g_initialized || !g_model) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  if (inputTextPrompt == nullptr || outputText == nullptr || output_size == 0) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    std::lock_guard<std::mutex> lock(g_mutex);

    std::string input(inputTextPrompt);

    if (g_use_chat_template) {
      input = apply_chat_template(g_architecture, input);
    }

// We assume single batch request for this API
#if defined(_WIN32)
    g_model->run(std::wstring(input.begin(), input.end()), false, L"", L"",
                 false);
#else
    g_model->run(input, false, "", "", false);
#endif

    auto causal_lm_model = dynamic_cast<causallm::CausalLM *>(g_model.get());
    std::string output = "";
    if (causal_lm_model) {
      output = causal_lm_model->getOutput(0);
    }

    if (output.length() >= output_size) {
      // Truncate if buffer is too small
      std::memcpy(outputText, output.c_str(), output_size - 1);
      outputText[output_size - 1] = '\0';
    } else {
      std::strcpy(outputText, output.c_str());
    }

  } catch (const std::exception &e) {
    std::cerr << "Exception in runModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics) {
  if (!g_initialized || !g_model) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  if (metrics == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto causal_lm_model = dynamic_cast<causallm::CausalLM *>(g_model.get());

    if (causal_lm_model) {
      if (!causal_lm_model->hasRun()) {
        return CAUSAL_LM_ERROR_INFERENCE_NOT_RUN;
      }
      *metrics = causal_lm_model->getPerformanceMetrics();
    } else {
      return CAUSAL_LM_ERROR_UNKNOWN;
    }

  } catch (const std::exception &e) {
    std::cerr << "Exception in getPerformanceMetrics: " << e.what()
              << std::endl;
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}
