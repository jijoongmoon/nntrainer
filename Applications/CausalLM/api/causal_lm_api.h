// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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

#include "causal_lm_callback.h"

#include <stddef.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performance Metrics
 */
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
  /* [perf-split] Appended, never reordered: prefill_duration_ms and
   * generation_duration_ms take the prefill->decode boundary before the device
   * queue drains, so on an async backend a bigger prefill block inflates
   * "prefill TPS" and deflates "decode TPS" by the same amount. These are the
   * rows that do not move: ttft_ms is its own measurement (prompt submit ->
   * first token to the caller), first_token_ms is the first decode step alone,
   * and decode_steady_ms is the median of tokens 2..N. prefill_duration_ms is
   * additionally queue-corrected when NNTR_PERF_SPLIT_DRAIN=1 is set (the
   * measurement path only); prefill_drain_ms is what that drain cost, and 0.0
   * means the drain was not armed and that time is inside first_token_ms.
   * See Applications/CausalLM/models/performance_metrics.h. */
  double ttft_ms;
  double first_token_ms;
  double decode_steady_ms;
  unsigned int decode_steady_tokens;
  double prefill_drain_ms;
} PerformanceMetrics;

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
 * @note  Enable only when your library supports the model
 */
typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
} ModelType;

/**
 * @brief Configuration structure
 */
typedef struct {
  // Add configuration options here as needed
  bool use_chat_template; /// < @brief Whether to apply chat template to input
  bool debug_mode; /// < @brief Check model file validity during initialization
  bool verbose;    /// < @brief Whether to print output during generation
} Config;

/**
 * @brief Set global options
 * @param config Configuration object
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode setOptions(Config config);

/**
 * @brief Model Quantization type
 */
typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,  ///< 4-bit weights, 32-bit activations
  CAUSAL_LM_QUANTIZATION_W16A16 = 2, ///< 16-bit weights, 16-bit activations
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,  ///< 8-bit weights, 16-bit activations
  CAUSAL_LM_QUANTIZATION_W32A32 = 4, ///< 32-bit weights, 32-bit activations
} ModelQuantizationType;

/**
 * @brief Load a model
 * @param compute Backend compute type
 * @param modeltype Model type
 * @param quant_type Model quantization type
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModel(BackendType compute, ModelType modeltype,
                               ModelQuantizationType quant_type);

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
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModel(const char *inputTextPrompt,
                              const char **outputText);

/**
 * @brief Run synchronous inference and stream decoded output deltas.
 * @param inputTextPrompt Input prompt
 * @param outputText Buffer to store final output text
 * @param callback Token delta callback, returning non-zero to stop generation.
 * The callback is invoked synchronously during generation and must not call the
 * CausalLM C API reentrantly.
 * @param user_data Opaque pointer passed to callback
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelStreaming(const char *inputTextPrompt,
                                       const char **outputText,
                                       CausalLmTokenCallback callback,
                                       void *user_data);

/**
 * @brief Request cancellation of the active inference run, if any.
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode cancelModel(void);

#ifdef __cplusplus
}
#endif

#endif // __CAUSAL_LM_API_H__
