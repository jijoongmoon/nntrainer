// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.h
 * @date    20 Mar 2026
 * @brief   C API for src (extension of CausalLM)
 *
 *          This header is self-contained: if causal_lm_api.h has already
 *          been included its types are reused; otherwise fallback
 *          definitions are provided so that this single header is
 *          sufficient for application code.
 *
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __QUICK_DOT_AI_API_H__
#define __QUICK_DOT_AI_API_H__

/* ── Extended model types (src additions) ────────────────────── */
#ifdef __CAUSAL_LM_API_H__
/* Model types already defined from causal_lm_api.h */
#else /* causal_lm_api.h not included — provide full definitions */

#define __CAUSAL_LM_API_H__

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include "callback_streamer.h"
#include "streamer.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef enum {
  CAUSAL_LM_ERROR_NONE = 0,
  CAUSAL_LM_ERROR_INVALID_PARAMETER = 1,
  CAUSAL_LM_ERROR_MODEL_LOAD_FAILED = 2,
  CAUSAL_LM_ERROR_INFERENCE_FAILED = 3,
  CAUSAL_LM_ERROR_NOT_INITIALIZED = 4,
  CAUSAL_LM_ERROR_INFERENCE_NOT_RUN = 5,
  CAUSAL_LM_ERROR_UNKNOWN = 99
} ErrorCode;

typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1,
  CAUSAL_LM_BACKEND_NPU = 2,
} BackendType;

typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
} ModelType;

typedef struct {
  bool use_chat_template;
  bool debug_mode;
  bool verbose;
} Config;

WIN_EXPORT ErrorCode setOptions(Config config);

typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,
  CAUSAL_LM_QUANTIZATION_W16A16 = 2,
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,
  CAUSAL_LM_QUANTIZATION_W32A32 = 4,
} ModelQuantizationType;

WIN_EXPORT ErrorCode loadModel(BackendType compute, ModelType modeltype,
                               ModelQuantizationType quant_type);

typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} PerformanceMetrics;

WIN_EXPORT ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);

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

/**
 * @brief Streaming counterpart of runModelHandle.
 *
 * Synchronously drives inference on @p handle and invokes @p callback
 * once per decoded-token boundary with a UTF-8 delta string. The call
 * blocks on the invoking thread until generation finishes, hits an EOS
 * token, reaches NUM_TO_GENERATE, the callback returns non-zero (which
 * requests cancellation at the next token boundary), or an error
 * occurs.
 *
 * The @p delta pointer passed into the callback is owned by the
 * streaming runtime and is only valid for the duration of the callback
 * invocation. Callers that need to retain the text must copy it.
 *
 * After a successful return the handle's "last output" buffer holds
 * the full concatenated generation (or the partial output on a
 * cancelled run), so a subsequent getPerformanceMetricsHandle() call
 * returns valid metrics and the same handle can be reused for another
 * run — identical semantics to runModelHandle.
 *
 * Streaming is currently only supported on models whose underlying
 * C++ implementation derives from causallm::CausalLM (all the Qwen
 * variants and Llama do; non-CausalLM models return
 * CAUSAL_LM_ERROR_UNKNOWN). See AsyncAndStreaming.md §3.4 at the repo
 * root for the full design.
 *
 * @param handle          Handle returned by loadModelHandle.
 * @param inputTextPrompt Input prompt (UTF-8, NUL-terminated).
 * @param callback        Token delta callback. Must be non-NULL.
 * @param user_data       Opaque pointer forwarded verbatim to the
 *                        callback on every invocation. May be NULL.
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode
runModelHandleStreaming(CausalLmHandle handle, const char *inputTextPrompt,
                        CausalLmTokenCallback callback, void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* __CAUSAL_LM_API_H__ */

#endif /* __QUICK_DOT_AI_API_H__ */