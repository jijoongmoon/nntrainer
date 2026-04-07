// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   speculative_decoding.h
 * @date   07 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Self-Speculative Decoding for CausalLM (Qualcomm GENIE style)
 *
 * @note   Self-speculative decoding reuses the SAME model weights with
 * layer skipping for the draft phase. No separate draft model is needed.
 *
 * Algorithm (Qualcomm GENIE / Layer-Skip style):
 *   1. Draft phase: Run model with only a subset of layers (layer skipping)
 *      to generate K candidate tokens autoregressively (fast)
 *   2. Verify phase: Run FULL model on all K candidate tokens in a single
 *      parallel forward pass (like prefill)
 *   3. For each draft token position i (0..K-1):
 *      - If greedy: accept if target argmax == draft token
 *      - If sampling: accept with probability min(1, p_target/p_draft)
 *        On rejection, resample from max(0, p_target - p_draft)
 *   4. All tokens after first rejection are discarded
 *   5. Bonus: get 1 extra token from target model at next position
 *   6. Rollback draft model's KV cache to the accepted position
 */

#ifndef __SPECULATIVE_DECODING_H__
#define __SPECULATIVE_DECODING_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#else
#define WIN_EXPORT
#define WSTR std::string
#endif

#include <memory>
#include <random>
#include <vector>

#include <causal_lm.h>
#include <factory.h>
#include <transformer.h>

namespace causallm {

/**
 * @brief Self-Speculative Decoding CausalLM Class
 *
 * Uses a single model's weights with layer skipping for the draft phase.
 * The draft model is constructed with fewer layers from the same weight file.
 * The target model uses all layers for verification.
 */
WIN_EXPORT class SpeculativeDecodingCausalLM {

public:
  /**
   * @brief Construct SpeculativeDecodingCausalLM
   *
   * Internally creates two model instances from the same architecture:
   * - draft_model: uses first (NUM_LAYERS / layer_skip_ratio) layers
   * - target_model: uses all NUM_LAYERS layers
   * Both load weights from the same weight file.
   *
   * @param target_cfg config.json for the target (full) model
   * @param draft_cfg config.json with reduced num_hidden_layers for draft
   * @param generation_cfg generation_config.json (shared)
   * @param nntr_cfg nntr_config.json (shared)
   * @param speculative_cfg speculative decoding parameters
   */
  SpeculativeDecodingCausalLM(json &target_cfg, json &draft_cfg,
                              json &generation_cfg, json &nntr_cfg,
                              json &speculative_cfg);

  /**
   * @brief Destroy the SpeculativeDecodingCausalLM object
   */
  ~SpeculativeDecodingCausalLM();

  /**
   * @brief Initialize both draft and target models
   */
  void initialize();

  /**
   * @brief Load weights for both models from the same weight file
   * @param weight_path Path to the shared weight file
   */
  void load_weight(const std::string &weight_path);

  /**
   * @brief Run self-speculative decoding generation
   *
   * @param prompt Input prompt string
   * @param do_sample Whether to use sampling (true) or greedy (false)
   * @param system_prompt Optional system prompt prefix
   * @param tail_prompt Optional tail prompt suffix
   * @param log_output Whether to print output to console
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true);

  /**
   * @brief Get the generated output text
   * @param batch_idx Index of the batch item
   * @return Generated text string
   */
  std::string getOutput(int batch_idx = 0) const;

  /**
   * @brief Get combined performance metrics
   */
  PerformanceMetrics getPerformanceMetrics() const {
    return performance_metrics_;
  }

private:
  /**
   * @brief Compute softmax probabilities from logits
   */
  std::vector<float> softmax(const float *logits, unsigned int vocab_size);

  /**
   * @brief Sample a token from a probability distribution
   */
  unsigned int sampleFromProbs(const std::vector<float> &probs);

  /**
   * @brief Get argmax token from logits
   */
  unsigned int argmax(const float *logits, unsigned int vocab_size);

  /**
   * @brief Register output tokens to tokenizer and output list
   */
  void registerOutputs(std::vector<unsigned int> ids, unsigned int start_pos,
                       unsigned int count, bool log_output);

  std::unique_ptr<Transformer> draft_model_;  /**< Layer-skipped draft model */
  std::unique_ptr<Transformer> target_model_; /**< Full target model */

  unsigned int num_speculative_tokens_; /**< K: draft tokens per step */
  unsigned int draft_num_layers_;       /**< Number of layers in draft model */
  unsigned int target_num_layers_;      /**< Number of layers in target model */
  unsigned int max_seq_len_;            /**< Maximum sequence length */
  unsigned int num_to_generate_;        /**< Maximum tokens to generate */
  unsigned int num_vocab_;              /**< Vocabulary size */
  unsigned int batch_size_;             /**< Batch size */
  unsigned int init_seq_len_;           /**< Initial sequence length */

  float temperature_;                   /**< Sampling temperature */
  unsigned int top_k_;                  /**< Top-K for sampling */
  float top_p_;                         /**< Top-P (nucleus) for sampling */

  std::vector<unsigned int> eos_token_ids_; /**< End-of-sequence token IDs */
  unsigned int bos_token_id_;               /**< Beginning-of-sequence token ID */

  std::unique_ptr<tokenizers::Tokenizer> tokenizer_; /**< Shared tokenizer */

  std::vector<std::string> output_list_; /**< Generated output per batch */
  std::vector<int> pending_ids_;         /**< Pending IDs for tokenizer decode */
  unsigned int *ids_history_;            /**< Full history of generated IDs */

  std::mt19937 rng_; /**< Random number generator */

  PerformanceMetrics performance_metrics_; /**< Performance tracking */

  std::string architecture_;  /**< Model architecture name for factory */
};

} // namespace causallm

#endif // __SPECULATIVE_DECODING_H__
