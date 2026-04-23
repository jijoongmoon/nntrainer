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
  CAUSAL_LM_ERROR_UNSUPPORTED = 6,
  CAUSAL_LM_ERROR_UNKNOWN = 99
} ErrorCode;

typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1,
  CAUSAL_LM_BACKEND_NPU = 2,
} BackendType;

typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
  CAUSAL_LM_MODEL_GAUSS2_5 = 1,
  CAUSAL_LM_MODEL_GAUSS3_6_QNN = 2,
  CAUSAL_LM_MODEL_GAUSS3_8_QNN = 3,
  CAUSAL_LM_MODEL_QWEN3_1_7B_Q40 = 4,
  CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN = 5,
  CAUSAL_LM_MODEL_GAUSS3_8_VIT_QNN = 6,
  CAUSAL_LM_MODEL_GAUSS3_6 = 7,
  CAUSAL_LM_MODEL_TINY_BERT = 8
} ModelType;

typedef struct {
  // Add configuration options here as needed
  bool use_chat_template; /// < @brief Whether to apply chat template to input
  bool debug_mode; /// < @brief Check model file validity during initialization
  bool verbose;    /// < @brief Whether to print output during generation
  const char
      *chat_template_name; /// < @brief Template name to select from array
                           ///  (e.g., "default", "tool_use"). NULL for
                           ///  "default".
} Config;

WIN_EXPORT ErrorCode setOptions(Config config);

typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,
  CAUSAL_LM_QUANTIZATION_W16A16 = 2,
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,
  CAUSAL_LM_QUANTIZATION_W32A32 = 4,
} ModelQuantizationType;

/**
 * @brief Chat message structure for chat template formatting
 * @note  Compatible with HuggingFace apply_chat_template() format
 */
typedef struct {
  const char *role;    /**< Message role: "system", "user", or "assistant" */
  const char *content; /**< Message content text */
} CausalLMChatMessage;

/**
 * @brief Load a model
 * @param compute Backend compute type
 * @param modeltype Model type
 * @param quant_type Model quantization type
 * @return ErrorCode
 */
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

/**
 * @brief Run inference with chat template formatted messages
 * @param messages Array of chat messages with role and content
 * @param num_messages Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param outputText Buffer to store output text (owned by the library)
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelWithMessages(const CausalLMChatMessage *messages,
                                          size_t num_messages,
                                          bool add_generation_prompt,
                                          const char **outputText);

/**
 * @brief Apply chat template to messages without running inference
 * @param messages Array of chat messages with role and content
 * @param num_messages Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param formattedText Buffer to store formatted text (owned by the library)
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode applyChatTemplate(const CausalLMChatMessage *messages,
                                       size_t num_messages,
                                       bool add_generation_prompt,
                                       const char **formattedText);
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
 * A single handle may internally carry multiple sub-models (e.g. vision
 * encoder + LLM) when loaded from a top-level nntr_config.json that
 * specifies "architectures" and "model_dirs" arrays. The single-model
 * run API (runModelHandle / runModelHandleStreaming) drives models[0]
 * only; the multimodal API (runMultimodalHandle*) drives the full set.
 *
 * Typical usage:
 *   CausalLmHandle h = NULL;
 *   loadModelHandle(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
 *                   CAUSAL_LM_QUANTIZATION_W4A32, NULL, &h);
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
 * @param compute         Backend compute type
 * @param modeltype       Model type enum
 * @param quant_type      Quantization type
 * @param native_lib_dir  Native library directory path (from Android
 *                        ApplicationInfo.nativeLibraryDir). May be NULL.
 * @param out_handle      Out-parameter that receives the new handle on success
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                                     ModelQuantizationType quant_type,
                                     const char *native_lib_dir,
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
 * Single-model API: drives models[0] only even when the handle was
 * populated with multiple sub-models. Use runMultimodalHandle for
 * compositions such as vision-encoder + LLM.
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
WIN_EXPORT ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                                 PerformanceMetrics *metrics);

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
 * @brief Request cancellation of an in-progress run on a handle.
 *
 * Sets the stop flag on the model, causing the token generation loop
 * to exit at the next token boundary. Thread-safe: can be called from
 * any thread (e.g., from a UI cancel button handler).
 *
 * If no run is in progress, this function is a no-op.
 *
 * @param handle Handle returned by loadModelHandle
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode cancelModelHandle(CausalLmHandle handle);

/**
 * @brief Unload the model from a handle without destroying the handle.
 *
 * Releases the model weights and internal state but keeps the handle
 * struct alive. After a successful unload, the handle's initialized flag
 * is cleared and subsequent run / metrics calls will return
 * CAUSAL_LM_ERROR_NOT_INITIALIZED. The handle can be destroyed later
 * with destroyModelHandle, or (in future) re-loaded.
 *
 * Passing a NULL handle is a no-op and returns CAUSAL_LM_ERROR_NONE.
 *
 * @param handle Handle returned by loadModelHandle
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode unloadModelHandle(CausalLmHandle handle);

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
WIN_EXPORT ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                             const char *inputTextPrompt,
                                             CausalLmTokenCallback callback,
                                             void *user_data);

/*============================================================================
 * Multimodal API
 *
 * These functions extend the handle-based API to support image+text inputs.
 * The pixel values are passed as preprocessed FloatArray (CHW format) from
 * the Kotlin image processor (LlavaNextImageProcessor).
 *
 * The handle must have been loaded from a multi-model nntr_config.json
 * (architectures[] + model_dirs[]) with at least [vision_encoder, llm];
 * a single-model handle returns CAUSAL_LM_ERROR_UNSUPPORTED.
 *
 * Vision Encoder integration is planned for future implementation.
 * Currently these functions return CAUSAL_LM_ERROR_UNSUPPORTED as stubs
 * once the multi-model precondition is satisfied.
 *============================================================================*/

/**
 * @brief Streaming multimodal inference on a specific handle.
 *
 * @param handle         Handle returned by loadModelHandle
 * @param prompt         Text prompt (UTF-8, NUL-terminated)
 * @param pixelValues    Preprocessed image patches in CHW format
 * @param numPatches     Number of image patches
 * @param originalHeight Original image height before preprocessing
 * @param originalWidth  Original image width before preprocessing
 * @param callback       Token delta callback. Must be non-NULL.
 * @param user_data      Opaque pointer forwarded to callback
 * @return ErrorCode (CAUSAL_LM_ERROR_UNSUPPORTED until Vision Encoder
 * implemented)
 */
WIN_EXPORT ErrorCode runMultimodalHandleStreaming(
    CausalLmHandle handle, const char *prompt, const float *pixelValues,
    int numPatches, int originalHeight, int originalWidth,
    CausalLmTokenCallback callback, void *user_data);

/**
 * @brief Blocking multimodal inference on a specific handle.
 *
 * @param handle         Handle returned by loadModelHandle
 * @param prompt         Text prompt (UTF-8, NUL-terminated)
 * @param pixelValues    Preprocessed image patches in CHW format
 * @param numPatches     Number of image patches
 * @param originalHeight Original image height before preprocessing
 * @param originalWidth  Original image width before preprocessing
 * @param outputText     Out-parameter that receives a pointer to the output
 * @return ErrorCode (CAUSAL_LM_ERROR_UNSUPPORTED until Vision Encoder
 * implemented)
 */
WIN_EXPORT ErrorCode runMultimodalHandle(CausalLmHandle handle,
                                         const char *prompt,
                                         const float *pixelValues,
                                         int numPatches, int originalHeight,
                                         int originalWidth,
                                         const char **outputText);

#ifdef __cplusplus
}
#endif

#endif /* __CAUSAL_LM_API_H__ */

#endif /* __QUICK_DOT_AI_API_H__ */