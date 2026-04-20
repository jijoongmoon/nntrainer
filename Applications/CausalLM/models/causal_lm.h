// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.h
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Qwen3
 *          - Qwen3-MoE
 * @note   This CausalLM assumes the Decoder-based model, which structure is
 *
 *        [Transformer]
 *              |
 *           [LMHead]
 */

#ifndef __CAUSAL_LM_H__
#define __CAUSAL_LM_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <atomic>
#include <transformer.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

// Forward-declare the C streamer type from api/streamer.h. We keep this
// a bare forward declaration so causal_lm.h does not pull the API
// header into the rest of the CausalLM models tree — the full
// definition is only needed in causal_lm.cpp where the vtable is
// actually invoked.
extern "C" {
struct BaseStreamer;
}

namespace causallm {

/**
 * @brief CausalLM Class
 */
WIN_EXPORT class CausalLM : virtual public Transformer {

public:
  /**
   * @brief Construct a new CausalLM object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation
   * (generation_config.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   */
  CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Destroy the CausalLM object
   */
  virtual ~CausalLM() {
    if (ids_history)
      free(ids_history);
  }

  /**
   * @brief run the CausalLM model
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  /**
   * @brief Get the generated output text
   * @param batch_idx Index of the batch item
   * @return Generated text string
   */
  std::string getOutput(int batch_idx = 0) const;

  /**
   * @brief Attach (or detach) a BaseStreamer to intercept per-token
   *        output during the next call to run().
   *
   * Passing @c nullptr detaches any currently-attached streamer.
   *
   * The streamer pointer is NOT owned by this class — the caller is
   * responsible for keeping the storage alive for the full duration of
   * the run() call and for detaching before the storage is destroyed.
   *
   * The high-level C API `runModelHandleStreaming` in
   * causal_lm_api.{h,cpp} uses a stack-allocated CallbackStreamer and
   * an RAII detach guard, which keeps this contract trivially safe.
   * See AsyncAndStreaming.md §3.3 at the repo root for the full
   * design.
   */
  void setStreamer(::BaseStreamer *streamer) { streamer_ = streamer; }

  /**
   * @brief Request cancellation of the current run().
   *
   * Thread-safe: sets the stop flag atomically, causing the token
   * generation loop to exit at the next token boundary. Safe to call
   * from any thread (e.g., from a UI cancel button handler).
   */
  void requestStop() override {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_DEBUG, "CausalLM",
                        "requestStop: setting stop_requested_ to true");
#else
    std::cout << "[DEBUG] requestStop: setting stop_requested_ to true" << std::endl;
#endif
    stop_requested_.store(true, std::memory_order_release);
  }

  /**
   * @brief Check if stop has been requested.
   * Thread-safe: can be called from any thread.
   */
  bool isStopRequested() const { return stop_requested_.load(std::memory_order_acquire); }

  /**
   * @brief Clear the stop request flag.
   * Thread-safe: can be called from any thread.
   */
  void clearStopRequest() { stop_requested_.store(false, std::memory_order_release); }

protected:
  /**
   * @brief Setup the parameters for the CausalLM model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) override;

  /**
   * @brief Construct Model
   */
  virtual void constructModel() override;

  /**
   * @brief register Outputs
   */
  virtual void
  registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                  std::vector<unsigned int> ids, unsigned int pos,
                  const std::vector<bool> &eos_list, bool log_output = true);

  /**
   * @brief save kv cache
   */
  WIN_EXPORT virtual void save_kvcache(std::string path, int to);

  /**
   * @brief load kv cache
   */
  WIN_EXPORT virtual void load_kvcache(std::string path, int to);

  /**
   * @brief generate
   */
  std::vector<unsigned int> generate(float *logits, bool do_sample,
                                     float repetition_penalty = 1,
                                     unsigned int *input_ids = nullptr,
                                     unsigned int NUM_INPUT_IDS = 0);

  /**
   * @brief registerCutomLayers
   */
  void registerCustomLayers() override;

  /** internal buffer */
  std::vector<std::string>
    output_list;             /**< List of output names for the model */
  unsigned int *ids_history; /**< History of input IDs for the model */

  std::vector<int> pending_ids_;

  std::string LMHEAD_DTYPE; /** embedding dtype */
  std::vector<unsigned int> EOS_TOKEN_ID;
  unsigned int BOS_TOKEN_ID;
  float TEMPERATURE;
  unsigned int TOP_K;
  float TOP_P;

  std::vector<unsigned int> BAD_WORD_IDS; /**< List of bad word IDs */
  unsigned int NUM_BADWORDS;              /**< Number of bad words */

  unsigned int SYS_PROMP_LEN;
  std::string PRE_COMPUTED_CACHE_PATH;
  std::string TAIL_PROMPT;
  bool SAVE_KVCACHE;
  bool USE_KVCACHE;
  unsigned int global_token_len;

  /**
   * @brief Optional streamer that receives each decoded token as it is
   *        produced during run(). Set via setStreamer(); nullptr means
   *        "no streaming, behave exactly like the pre-streaming code
   *        path". See AsyncAndStreaming.md §3.3.
   */
  ::BaseStreamer *streamer_ = nullptr;

  /**
   * @brief Cooperative cancellation flag set by registerOutputs() when
   *        the attached streamer's put() returns non-zero, or by
   *        requestStop() from any thread. The token generation loop in
   *        run() checks this once per iteration and breaks out at the
   *        next safe boundary.
   *
   * Uses std::atomic for thread-safe access from any thread (e.g.,
   * cancel button handler in UI thread).
   */
  std::atomic<bool> stop_requested_{false};

  std::mt19937 rng; /**< Random Number Gen */
};

} // namespace causallm

#endif
