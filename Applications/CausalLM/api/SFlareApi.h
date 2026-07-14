// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   SFlareApi.h
 * @brief  Samsung FLARE (Fast & Light-weight AI Runtime Engine) API v2.
 *         SFlare-branded SDK facade rebound onto the CausalLM factory:
 *         model architecture, dtype, tokenizer, template and generation
 *         parameters all come from the model directory's config.json /
 *         nntr_config.json — the API only carries paths, backend choice
 *         and runtime knobs.
 * @date   14 July 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note   v2 changes vs the 2025-04 SFlareApi (Gauss3 binding):
 *         - Text I/O is UTF-8 `const char *` on every platform (the old
 *           header used wchar_t* on Windows and std::string& elsewhere).
 *         - Model hyper-parameters are no longer hard-coded per enum value;
 *           `SFlareConfig::model_path` must point to a directory holding
 *           config.json + nntr_config.json + weights (+ tokenizer/template).
 *         - GPU backends are first-class (OpenCL / CUDA), selected at
 *           loadSFlareLLMModel(); the engine choice is process-wide and
 *           fixed by the FIRST load (nntrainer engine contexts are
 *           process-lifetime singletons).
 *         - Streaming generation via a token callback.
 *         - LoRA and FSU are not wired in v2 (SFLARE_UNDEFIND / ignored).
 */

#ifndef __SFLARE_API_H__
#define __SFLARE_API_H__

#include <cstddef>
#include <string>

#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

namespace SFlareApi {

/**
 * @brief Compute backend requested for the model.
 * @note  CPU -> nntrainer engine "cpu", GPU* -> "gpu" (OpenCL),
 *        GPU_NVIDIA -> "cuda". The GPU_INTEL / GPU_ADRENO variants
 *        additionally apply the validated per-device OpenCL kernel-path
 *        environment bundle; plain GPU applies only the common OpenCL set.
 *        The engine is fixed process-wide by the first successful load.
 */
enum class BackendType { CPU, GPU, NPU, GPU_INTEL, GPU_NVIDIA, GPU_ADRENO };

/**
 * @brief Foundation model family. Informational in v2 — the actual
 *        architecture is read from config.json in model_path.
 */
enum class GaussFModelType { GAUSS1B, GAUSS3B, GAUSS3_3B, GAUSS4 };

/**
 * @brief Application type. Only INSTRUCT is wired in v2.
 */
enum class ApplicationType { INSTRUCT, SFT, LORA };

/**
 * @brief Model data type. Informational in v2 — the executable dtype
 *        contract lives in nntr_config.json (model_tensor_type /
 *        fc_layer_dtype / embedding_dtype) and cannot be switched at
 *        runtime (QS4CX act-dtype contract).
 */
enum class ModelDataType {
  DTYPE_W4KA32,
  DTYPE_W40A32,
  DTYPE_W8A32,
  DTYPE_W16A32,
  DTYPE_W32A32,
  DTYPE_WQ4A32,
  DTYPE_W4A16,      /**< QS4CX weights + FP16 activations (gauss4 release) */
  DTYPE_FROM_CONFIG /**< default: trust nntr_config.json verbatim */
};

/**
 * @brief Memory/speed policy applied as environment defaults at first load.
 *        User-provided environment variables always win (set with
 *        overwrite=0). MINIMAL = smallest steady residency with no
 *        measured TPS loss (JIT-i8 / plain-drop lever family);
 *        PERFORMANCE = cached derived weights; BALANCED = engine defaults.
 */
enum class MemoryProfile { MINIMAL, BALANCED, PERFORMANCE };

/**
 * @brief Error codes. The first five values keep the 2025-04 order.
 */
enum class ErrorCode {
  SFLARE_SUCCESS,
  SFLARE_FAIL,
  SFLARE_INVALID_CONFIG,
  SFLARE_INVALID_INPUT,
  SFLARE_UNDEFIND, /**< kept for source compatibility */
  SFLARE_UNDEFINED = SFLARE_UNDEFIND,
  SFLARE_NOT_LOADED,
  SFLARE_BUFFER_TOO_SMALL
};

extern "C" {
/**
 * @brief Token callback invoked for each UTF-8 decoded delta during
 *        streaming generation. The delta pointer is valid only for the
 *        duration of the call; copy it if needed later.
 * @return 0 to continue, non-zero to request a cooperative stop.
 */
typedef int (*SFlareTokenCallback)(const char *delta_utf8, void *user_data);
}

/**
 * @brief Per-call generation parameters. Pass nullptr for defaults.
 */
struct GenParams {
  unsigned int max_new_tokens = 0; /**< 0 = nntr_config.json num_to_generate */
  bool do_sample = false;          /**< false = greedy decoding */
  bool apply_chat_template = true; /**< wrap input with the model's chat
                                        template when one exists (raw mode
                                        when false) */
};

/**
 * @brief SFlare configuration. Only model_path is required in v2; the
 *        model directory's config.json / nntr_config.json carry the spec.
 */
struct SFlareConfig {
  GaussFModelType llm_model = GaussFModelType::GAUSS4; /**< informational */
  ModelDataType data_type = ModelDataType::DTYPE_FROM_CONFIG; /**< info */
  int fsu_lookahead = 0;              /**< ignored in v2 (FSU not wired) */
  const char *tokenizer_path = nullptr; /**< optional override of
                                             nntr_config tokenizer_file */
  const char *model_path = nullptr;     /**< REQUIRED model directory */
  MemoryProfile memory_profile = MemoryProfile::MINIMAL;
};

/**
 * @brief Structured performance report of the last completed run.
 */
struct SFlarePerformance {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  double prefill_tps;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double generation_tps;
  double initialization_duration_ms; /**< initialize+load_weight+repack */
  size_t peak_memory_kb;
};

/**
 * @brief SFlareContext — the interface an application drives.
 *        Lifecycle: initSFlare() -> setSFlareOptions() ->
 *        loadSFlareLLMModel() -> executeSFlareLLM()* ->
 *        DestroySFlareContext().
 * @note  One context binds one loaded model. KV state persists across
 *        executeSFlareLLM() calls on the same context (multi-turn);
 *        reloading the model is the reset path.
 */
WIN_EXPORT class SFlareContext {

public:
  SFlareContext() = default;

  WIN_EXPORT virtual ~SFlareContext();

  /**
   * @brief Set the SFlare options (validates model_path contents).
   * @return SFLARE_INVALID_CONFIG when model_path is missing or lacks
   *         config.json / nntr_config.json.
   */
  WIN_EXPORT virtual ErrorCode setSFlareOptions(SFlareConfig config) = 0;

  /**
   * @brief Create + initialize + load + repack the model on @p compute.
   * @param compute   Backend; fixed process-wide by the first load.
   * @param enable_fsu Ignored in v2 (kept for source compatibility).
   * @param app_type  Only INSTRUCT is supported in v2.
   * @param lora_path Not supported in v2 (must be nullptr).
   */
  WIN_EXPORT virtual ErrorCode
  loadSFlareLLMModel(BackendType compute, bool enable_fsu = false,
                     ApplicationType app_type = ApplicationType::INSTRUCT,
                     const char *lora_path = nullptr) = 0;

  /**
   * @brief Synchronous generation; final text copied into @p output_utf8.
   * @return SFLARE_BUFFER_TOO_SMALL when the output was truncated to fit
   *         @p output_size (the truncated text is still NUL-terminated).
   */
  WIN_EXPORT virtual ErrorCode
  executeSFlareLLM(const char *input_utf8, char *output_utf8,
                   size_t output_size,
                   const GenParams *params = nullptr) = 0;

  /**
   * @brief Synchronous generation with per-delta streaming callback.
   *        The callback runs on the calling thread and must not call the
   *        SFlare API reentrantly; returning non-zero stops generation.
   */
  WIN_EXPORT virtual ErrorCode
  executeSFlareLLM(const char *input_utf8, SFlareTokenCallback callback,
                   void *user_data, const GenParams *params = nullptr) = 0;

  /**
   * @brief Resume generation from a KV cache saved by saveKVcache():
   *        reload @p kvcache_path at token position @p prev_idx, prefill
   *        @p input_utf8 after it, then generate (legacy reRun).
   *        One-shot: the armed cache is consumed by this call.
   */
  WIN_EXPORT virtual ErrorCode
  executeSFlareLLM(const char *input_utf8, char *output_utf8,
                   size_t output_size, unsigned int prev_idx,
                   const char *kvcache_path,
                   const GenParams *params = nullptr) = 0;

  /**
   * @brief Cooperatively stop the in-flight executeSFlareLLM() (call from
   *        another thread). @p idx receives the current KV token position
   *        (final position is authoritative only after execute returns).
   */
  WIN_EXPORT virtual ErrorCode pauseSFlareLLM(unsigned int &idx) = 0;

  /**
   * @brief Save the KV cache up to the current position.
   */
  WIN_EXPORT virtual ErrorCode saveKVcache(const char *path) = 0;

  /**
   * @brief Arm a saved KV cache for the next executeSFlareLLM() call
   *        (equivalent to the reRun overload's cache arguments).
   * @param token_len Token position @p path was saved at.
   */
  WIN_EXPORT virtual ErrorCode loadKVcache(const char *path,
                                           unsigned int token_len) = 0;

  /**
   * @brief Legacy CSV performance line:
   *        "[Tokenizer ms], [Prefill Tokens], [Prefill ms], [Prefill TPS],
   *         [Gen Tokens], [Gen ms], [Gen TPS]"
   */
  WIN_EXPORT virtual ErrorCode getPerformance(char *perf_csv,
                                              size_t size) = 0;

  /**
   * @brief Structured performance report of the last run.
   */
  WIN_EXPORT virtual ErrorCode getPerformance(SFlarePerformance &out) = 0;
};

/**
 * @brief Create an SFlare context. Model registration is handled
 *        internally (idempotent); @p registered is always set to true and
 *        kept only for source compatibility with the 2025-04 API.
 * @return Context pointer on success, nullptr on failure.
 */
extern "C" WIN_EXPORT SFlareContext *initSFlare(bool &registered);

/**
 * @brief Destroy an SFlare context. Unlike the 2025-04 API this does NOT
 *        release the process-global nntrainer Engine (other contexts and
 *        future loads stay valid).
 */
extern "C" WIN_EXPORT ErrorCode DestroySFlareContext(SFlareContext *context);

} // namespace SFlareApi

#endif /* __SFLARE_API_H__ */
