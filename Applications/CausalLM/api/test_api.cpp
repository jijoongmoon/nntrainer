// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

 * @file   test_api.cpp
 * @date   21 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @brief  Simple application to test CausalLM API
 * @bug    No known bugs except for NYI items
 *
 */

#include "causal_lm_api.h"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [prompt]" << std::endl;
    return 1;
  }

  const char *model_path = argv[1];
  const char *prompt = (argc >= 3) ? argv[2] : "Hello, how are you?";
  bool use_chat_template = true;
  if (argc >= 4) {
    use_chat_template =
      (std::string(argv[3]) == "1" || std::string(argv[3]) == "true");
  }

  std::string quant_str = "UNKNOWN";
  ModelQuantizationType quant_type = CAUSAL_LM_QUANTIZATION_UNKNOWN;
  if (argc >= 5) {
    quant_str = std::string(argv[4]);
    if (quant_str == "W4A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
    else if (quant_str == "W16A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
    else if (quant_str == "W8A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
    else if (quant_str == "W32A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
  }

  std::cout << "Loading model from: " << model_path << std::endl;
  std::cout << "Use chat template: " << (use_chat_template ? "true" : "false")
            << std::endl;
  std::cout << "Quantization: " << quant_str << std::endl;

  // 1. Set Options (Optional)
  Config config;
  config.reserved = 0;
  config.use_chat_template = use_chat_template;
  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Failed to set options: " << err << std::endl;
    return 1;
  }

  // 2. Load model
  ModelType model_type = CAUSAL_LM_MODEL_UNKNOWN;
  const char *path_arg = model_path;

  // Simple mapping for test purposes
  if (std::string(model_path) == "QWEN3-0.6B") {
    model_type = CAUSAL_LM_MODEL_QWEN3_0_6B;
    path_arg = nullptr;
    std::cout << "Testing known model type: QWEN3-0.6B" << std::endl;
  }

  err = loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_UNKNOWN, quant_type,
                  model_path);
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Failed to load model: " << err << std::endl;
    return 1;
  }
  std::cout << "Model loaded successfully." << std::endl;

  // 3. Run Inference
  const char *outputText = nullptr;
  std::cout << "Running inference with prompt: " << prompt << std::endl;

  err = runModel(prompt, &outputText);
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Failed to run model: " << err << std::endl;
    return 1;
  }

  if (outputText) {
    std::cout << "Output: " << outputText << std::endl;
  } else {
    std::cout << "Output: (null)" << std::endl;
  }

  // 4. Get Metrics
  // 4. Get Metrics
  PerformanceMetrics metrics;
  err = getPerformanceMetrics(&metrics);
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Failed to get metrics: " << err << std::endl;
  } else {
    std::cout << "\nPerformance Metrics:" << std::endl;
    std::cout << "  Prefill: " << metrics.prefill_tokens << " tokens in "
              << metrics.prefill_duration_ms << " ms ("
              << (metrics.prefill_tokens / metrics.prefill_duration_ms * 1000.0)
              << " TPS)" << std::endl;
    std::cout << "  Generation: " << metrics.generation_tokens << " tokens in "
              << metrics.generation_duration_ms << " ms ("
              << (metrics.generation_tokens / metrics.generation_duration_ms *
                  1000.0)
              << " TPS)" << std::endl;
  }

  return 0;
}
