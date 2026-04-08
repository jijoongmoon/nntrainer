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

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

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

/*============================================================================
 * Handle-based API (for parallel multi-model execution)
 *
 * The non-handle API above operates on a single process-wide model instance
 * protected by one global mutex, which serializes every call and prevents
 * loading more than one model at a time. The handle-based API below lets a
 * caller load several models simultaneously and run them in parallel from
 * different threads, with per-handle state so that different handles never
 * block each other. Each handle owns its own model, its own last-output
 * buffer, and its own mutex.
 *
 * Typical usage:
 *   CausalLmHandle h = NULL;
 *   loadModelHandle(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
 *                   CAUSAL_LM_QUANTIZATION_W4A32, &h);
 *   const char *out = NULL;
 *   runModelHandle(h, "Hello", &out);
 *   // ... use out (owned by h, valid until the next run or destroy) ...
 *   destroyModelHandle(h);
 *============================================================================*/

/**
 * @brief Opaque handle to a loaded CausalLM model instance.
 */
typedef struct CausalLmModel *CausalLmHandle;

/**
 * @brief Load a model and return a newly-allocated handle.
 *
 * Calling this multiple times with different parameters returns independent
 * handles, each with its own model state. The caller must eventually call
 * destroyModelHandle on the returned handle to release resources.
 *
 * @param compute    Backend compute type
 * @param modeltype  Model type enum
 * @param quant_type Quantization type
 * @param out_handle Out-parameter that receives the new handle on success
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                                     ModelQuantizationType quant_type,
                                     CausalLmHandle *out_handle);

/**
 * @brief Run inference on a specific handle.
 *
 * The returned outputText pointer is owned by the handle and remains valid
 * until the next runModelHandle call on the same handle or until the handle
 * is destroyed. Different handles are safe to call concurrently from
 * different threads; the same handle is serialized by its own internal
 * mutex.
 *
 * @param handle          Handle returned by loadModelHandle
 * @param inputTextPrompt Input prompt
 * @param outputText      Out-parameter that receives a pointer to the output
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelHandle(CausalLmHandle handle,
                                    const char *inputTextPrompt,
                                    const char **outputText);

/**
 * @brief Retrieve performance metrics of the last run for a given handle.
 * @param handle  Handle returned by loadModelHandle
 * @param metrics Pointer to a PerformanceMetrics struct to be filled
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode
getPerformanceMetricsHandle(CausalLmHandle handle, PerformanceMetrics *metrics);

/**
 * @brief Release all resources owned by a handle.
 *
 * Passing a NULL handle is a no-op and returns CAUSAL_LM_ERROR_NONE.
 *
 * @param handle Handle returned by loadModelHandle
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode destroyModelHandle(CausalLmHandle handle);

#ifdef __cplusplus
}
#endif

#endif // __CAUSAL_LM_API_H__
