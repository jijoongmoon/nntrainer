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
 * @file    causal_lm_api.h
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __CAUSAL_LM_API_H__
#define __CAUSAL_LM_API_H__

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Error codes
 */
typedef enum {
  CAUSAL_LM_ERROR_NONE = 0,
  CAUSAL_LM_ERROR_INVALID_PARAMETER = 1,
  CAUSAL_LM_ERROR_MODEL_LOAD_FAILED = 2,
  CAUSAL_LM_ERROR_INFERENCE_FAILED = 3,
  CAUSAL_LM_ERROR_NOT_INITIALIZED = 4,
  CAUSAL_LM_ERROR_INFERENCE_NOT_RUN = 5,
  CAUSAL_LM_ERROR_UNKNOWN = 99
} ErrorCode;

/**
 * @brief Backend compute type
 */
typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1, /// < @todo: support gpu
  CAUSAL_LM_BACKEND_NPU = 2, /// < @todo: support npu
} BackendType;

/**
 * @brief Model type
 */
typedef enum {
  CAUSAL_LM_MODEL_UNKNOWN = 0,
  CAUSAL_LM_MODEL_LLAMA = 1,
  CAUSAL_LM_MODEL_QWEN2 = 2,
  CAUSAL_LM_MODEL_QWEN3 = 3,
  CAUSAL_LM_MODEL_QWEN3_MOE = 4,
  CAUSAL_LM_MODEL_GPT_OSS = 5,
  CAUSAL_LM_MODEL_GEMMA3 = 6
} ModelType;

/**
 * @brief Configuration structure
 */
typedef struct {
  // Add configuration options here as needed
  int reserved;
  bool use_chat_template; /// < @brief Whther to apply chat template to input
} Config;

/**
 * @brief Set global options
 * @param config Configuration object
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode setOptions(Config config);

/**
 * @brief Load a model
 * @param compute Backend compute type
 * @param modeltype Model type
 * @param path Path to the model directory
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModel(BackendType compute, ModelType modeltype,
                               const char *path);

/**
 * @brief Performance Metrics
 */
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
} PerformanceMetrics;

/**
 * @brief Get performance metrics of the last run
 * @param metrics Pointer to PerformanceMetrics struct to be filled
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);

/**
 * @brief Run inference
 * @param inputTextPrompt Input prompt
 * @param outputText Buffer to store output text
 * @param output_size Size of the output buffer
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModel(const char *inputTextPrompt, char *outputText,
                              size_t output_size);

#ifdef __cplusplus
}
#endif

#endif // __CAUSAL_LM_API_H__
