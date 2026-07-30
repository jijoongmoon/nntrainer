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
 * @note  Unchanged surface: @c use_chat_template keeps its exact meaning for
 *        every existing caller, including one that zero-initialises this
 *        struct (false = feed the prompt raw). Per-call control lives in
 *        ::GenerationOptions instead, so no field here changed its sense.
 */
typedef struct {
  // Add configuration options here as needed
  bool use_chat_template; /// < @brief Whether to apply chat template to input
  bool debug_mode; /// < @brief Check model file validity during initialization
  bool verbose;    /// < @brief Whether to print output during generation
} Config;

/**
 * @brief Per-call generation options.
 *
 * Passing NULL wherever a @c GenerationOptions is accepted selects the
 * documented defaults: the model package's chat template IS applied, and the
 * package's own render context is used. That default is deliberate -- a caller
 * hands this API a user turn, and a chat model needs its turn markers to answer
 * instead of drift. A caller that has ALREADY templated its text (or wants a
 * bare completion) sets @c apply_chat_template to false and gets its bytes
 * through verbatim; nothing wraps an already-wrapped prompt.
 */
typedef struct {
  /**
   * @brief Apply the model package's chat template to the prompt.
   *        Defaults to true when the options pointer is NULL.
   */
  bool apply_chat_template;
  /**
   * @brief Optional UTF-8 JSON object of extra template render keys, merged
   *        over the model package's own "chat_template_context" defaults.
   *        NULL (or empty) = use the package defaults unchanged.
   */
  const char *chat_context_json;
  /**
   * @brief Sampling policy: negative = follow the model package's
   *        generation_config.json "do_sample" (the default, and what the
   *        command line runner does), 0 = greedy, positive = sample.
   *
   * Following the package matters for front-end agreement: a package that
   * asks for sampling decodes differently from a greedy run, so an SDK that
   * forced greedy answered the same prompt differently than the runner did.
   */
  int do_sample;
} GenerationOptions;

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
 * @brief Load a model from a model-package directory.
 *
 * The directory owns the whole specification -- config.json,
 * nntr_config.json, the weight file, the tokenizer and the chat template --
 * exactly as the command line runner reads it, so an SDK consumer and the
 * runner can be pointed at one package and get one answer. This is the entry
 * point to use for any model that is not one of the built-in ::ModelType
 * values (that enum stays as it is for existing callers).
 *
 * @param compute   Backend compute type; selects the nntrainer engine and
 *                  must be chosen before the first load in the process.
 * @param model_dir Path to the model package directory (UTF-8).
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModelFromPath(BackendType compute,
                                       const char *model_dir);

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
 * @brief Run inference with per-call options.
 * @param inputTextPrompt Input prompt (UTF-8). A user turn when the chat
 *        template is applied, otherwise the verbatim model input.
 * @param options Per-call options, or NULL for the documented defaults
 *        (chat template applied, package render context, package sampling
 *        policy).
 * @param outputText Buffer to store output text
 * @return ErrorCode; CAUSAL_LM_ERROR_INFERENCE_FAILED when a chat template is
 *         present but cannot render the prompt -- the prompt is never quietly
 *         downgraded to raw text behind the caller's back.
 */
WIN_EXPORT ErrorCode runModelWithOptions(const char *inputTextPrompt,
                                         const GenerationOptions *options,
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
