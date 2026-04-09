// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   test_litert_gpu2.cpp
 * @date   09 Apr 2026
 * @brief  CLI test for LiteRT-LM GPU2 backend via CausalLM API
 * @see    https://github.com/nntrainer/nntrainer
 */

#include "causal_lm_api.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char *argv[]) {
  const char *prompt = "What is AI?";
  const char *model_base_path =
      "/data/local/tmp/nntrainer/causallm/models/";

  if (argc >= 2)
    prompt = argv[1];
  if (argc >= 3)
    model_base_path = argv[2];

  printf("=== LiteRT-LM GPU2 Backend Test ===\n");
  printf("Model base path: %s\n", model_base_path);
  printf("Prompt: %s\n\n", prompt);

  // Set model base path
  ErrorCode err = setModelBasePath(model_base_path);
  if (err != CAUSAL_LM_ERROR_NONE) {
    fprintf(stderr, "setModelBasePath failed: %d\n", err);
    return 1;
  }

  // Configure options
  Config config;
  config.use_chat_template = false;
  config.debug_mode = true;
  config.verbose = true;
  err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    fprintf(stderr, "setOptions failed: %d\n", err);
    return 1;
  }

  // Load model with GPU2 backend (LiteRT-LM)
  printf("Loading model (GPU2 / Gemma4-E2B)...\n");
  err = loadModel(CAUSAL_LM_BACKEND_GPU2, CAUSAL_LM_MODEL_GEMMA4_E2B,
                  CAUSAL_LM_QUANTIZATION_UNKNOWN);
  if (err != CAUSAL_LM_ERROR_NONE) {
    fprintf(stderr, "loadModel failed: %d\n", err);
    return 1;
  }
  printf("Model loaded successfully.\n\n");

  // Run inference
  printf("Running inference...\n");
  const char *output = NULL;
  err = runModel(prompt, &output);
  if (err != CAUSAL_LM_ERROR_NONE) {
    fprintf(stderr, "runModel failed: %d\n", err);
    unloadModel();
    return 1;
  }

  printf("\n--- Output ---\n%s\n--------------\n\n", output ? output : "(null)");

  // Performance metrics
  PerformanceMetrics metrics;
  err = getPerformanceMetrics(&metrics);
  if (err == CAUSAL_LM_ERROR_NONE) {
    printf("Performance:\n");
    printf("  Prefill:    %u tokens / %.1f ms\n", metrics.prefill_tokens,
           metrics.prefill_duration_ms);
    printf("  Generation: %u tokens / %.1f ms\n", metrics.generation_tokens,
           metrics.generation_duration_ms);
    printf("  Total:      %.1f ms\n", metrics.total_duration_ms);
    printf("  Peak mem:   %zu KB\n", metrics.peak_memory_kb);
  }

  // Cleanup
  unloadModel();
  printf("\nDone.\n");
  return 0;
}
