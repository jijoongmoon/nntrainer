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

  std::cout << "Loading model from: " << model_path << std::endl;
  std::cout << "Use chat template: " << (use_chat_template ? "true" : "false")
            << std::endl;

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
  // Auto-detect model type or assume one. For test, we pass UNKNOWN or a
  // specific one if known. The implementation falls back to config.json
  // architecture if provided.
  err = loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_UNKNOWN, model_path);
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
