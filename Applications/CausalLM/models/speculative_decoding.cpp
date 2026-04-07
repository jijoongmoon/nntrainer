// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   speculative_decoding.cpp
 * @date   07 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Self-Speculative Decoding implementation (Qualcomm GENIE style)
 *
 * @note   Key insight for parallel verification:
 * After running incremental_inference with K tokens (prefill-style), all
 * intermediate layers (including output_norm) contain hidden states for ALL K
 * positions. The LM head only outputs the last position's logits. We extract
 * all-position logits by accessing output_norm's output tensor via forEachLayer
 * and manually computing the dot product with the LM head weight.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <layer_context.h>
#include <tensor.h>

#include <llm_util.hpp>
#include <speculative_decoding.h>

namespace causallm {

/// Helper: load tokenizer blob from file
static std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + path);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size))
    throw std::runtime_error("Failed to read file: " + path);
  return buffer;
}

// ─────────────────────── Constructor / Destructor ───────────────────────

SpeculativeDecodingCausalLM::SpeculativeDecodingCausalLM(
  json &target_cfg, json &draft_cfg, json &generation_cfg, json &nntr_cfg,
  json &speculative_cfg) {

  // Speculative decoding parameters
  num_speculative_tokens_ =
    speculative_cfg.value("num_speculative_tokens", 32u);
  architecture_ = target_cfg["architectures"]
                    .get<std::vector<std::string>>()[0];

  // Model dimensions (from target config)
  num_vocab_ = target_cfg["vocab_size"].get<unsigned int>();
  target_num_layers_ = target_cfg["num_hidden_layers"].get<unsigned int>();
  draft_num_layers_ = draft_cfg["num_hidden_layers"].get<unsigned int>();

  // NNTrainer parameters
  batch_size_ = nntr_cfg["batch_size"].get<unsigned int>();
  max_seq_len_ = nntr_cfg["max_seq_len"].get<unsigned int>();
  num_to_generate_ = nntr_cfg["num_to_generate"].get<unsigned int>();
  init_seq_len_ = nntr_cfg["init_seq_len"].get<unsigned int>();

  // Generation parameters
  temperature_ = generation_cfg.value("temperature", 0.7f);
  top_k_ = generation_cfg.value("top_k", 20u);
  top_p_ = generation_cfg.value("top_p", 0.95f);

  if (generation_cfg["eos_token_id"].is_array()) {
    eos_token_ids_ =
      generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  } else {
    eos_token_ids_.push_back(
      generation_cfg["eos_token_id"].get<unsigned int>());
  }
  bos_token_id_ = generation_cfg.value("bos_token_id", 0u);

  // Tokenizer
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(
    LoadBytesFromFile(nntr_cfg["tokenizer_file"]));

  // Allocate history buffer
  ids_history_ = (unsigned int *)calloc(
    static_cast<size_t>(batch_size_) * max_seq_len_, sizeof(unsigned int));

  // Output list
  for (unsigned int i = 0; i < batch_size_; ++i)
    output_list_.push_back("");

  // Create models via factory
  draft_model_ = Factory::Instance().create(architecture_, draft_cfg,
                                            generation_cfg, nntr_cfg);
  target_model_ = Factory::Instance().create(architecture_, target_cfg,
                                             generation_cfg, nntr_cfg);

  if (!draft_model_ || !target_model_)
    throw std::runtime_error(
      "Failed to create models for speculative decoding. "
      "Architecture: " +
      architecture_);

  std::cout << "[SSD] Draft model layers: " << draft_num_layers_
            << ", Target model layers: " << target_num_layers_
            << ", K=" << num_speculative_tokens_ << std::endl;
}

SpeculativeDecodingCausalLM::~SpeculativeDecodingCausalLM() {
  if (ids_history_)
    free(ids_history_);
}

// ─────────────────── Initialize / Load ───────────────────

void SpeculativeDecodingCausalLM::initialize() {
  draft_model_->initialize();
  target_model_->initialize();
}

void SpeculativeDecodingCausalLM::load_weight(const std::string &weight_path) {
  draft_model_->load_weight(weight_path);
  target_model_->load_weight(weight_path);
}

// ─────────────────── Utility Methods ───────────────────

std::vector<float> SpeculativeDecodingCausalLM::softmax(
  const float *logits, unsigned int vocab_size) {
  std::vector<float> probs(vocab_size);
  float max_val = *std::max_element(logits, logits + vocab_size);
  float sum = 0.0f;
  for (unsigned int i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp(logits[i] - max_val);
    sum += probs[i];
  }
  for (unsigned int i = 0; i < vocab_size; ++i)
    probs[i] /= sum;
  return probs;
}

unsigned int SpeculativeDecodingCausalLM::sampleFromProbs(
  const std::vector<float> &probs) {
  std::discrete_distribution<unsigned int> dist(probs.begin(), probs.end());
  return dist(rng_);
}

unsigned int SpeculativeDecodingCausalLM::argmax(const float *logits,
                                                 unsigned int vocab_size) {
  return static_cast<unsigned int>(
    std::distance(logits, std::max_element(logits, logits + vocab_size)));
}

void SpeculativeDecodingCausalLM::registerOutputs(
  std::vector<unsigned int> ids, unsigned int start_pos, unsigned int count,
  bool log_output) {
  for (unsigned int i = 0; i < count; ++i) {
    unsigned int pos = start_pos + i;
    unsigned int id = ids[i];
    pending_ids_.push_back(static_cast<int>(id));
    ids_history_[pos] = id;

    std::string decoded_str = tokenizer_->Decode(pending_ids_);

    bool hold = false;
    if (decoded_str.size() >= 3 &&
        decoded_str.compare(decoded_str.size() - 3, 3, "\xEF\xBF\xBD") == 0)
      hold = true;

    if (!hold) {
      if (log_output) {
        std::cout << decoded_str;
        std::cout.flush();
      }
      output_list_[0].append(decoded_str);
      pending_ids_.clear();
    }
  }
}

std::string SpeculativeDecodingCausalLM::getOutput(int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(output_list_.size()))
    return "";
  return output_list_[batch_idx];
}

// ─────────────────── Main Run Loop ───────────────────

void SpeculativeDecodingCausalLM::run(const WSTR prompt, bool do_sample,
                                      const WSTR system_prompt,
                                      const WSTR tail_prompt,
                                      bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();

  // Reset output
  output_list_.clear();
  for (unsigned int b = 0; b < batch_size_; ++b)
    output_list_.push_back("");
  pending_ids_.clear();

  // Tokenize
  std::string full_prompt = system_prompt + prompt + tail_prompt;
  if (log_output)
    std::cout << full_prompt << std::endl;

  auto token_ids = tokenizer_->Encode(full_prompt);
  unsigned int prompt_len = token_ids.size();
  unsigned int num_allow = max_seq_len_ - num_to_generate_;
  if (prompt_len > num_allow)
    prompt_len = num_allow;

  // Prepare input buffers (one for draft, one for target)
  float *draft_input =
    (float *)calloc(batch_size_ * max_seq_len_, sizeof(float));
  float *target_input =
    (float *)calloc(batch_size_ * max_seq_len_, sizeof(float));

  for (unsigned int i = 0; i < prompt_len; ++i) {
    draft_input[i] = static_cast<float>(token_ids[i]);
    target_input[i] = static_cast<float>(token_ids[i]);
    ids_history_[i] = token_ids[i];
  }

  std::vector<float *> draft_in = {draft_input};
  std::vector<float *> target_in = {target_input};
  std::vector<float *> empty_label;

  // ──────── PREFILL both models ────────
  auto start_prefill = std::chrono::high_resolution_clock::now();

  auto draft_prefill_out = draft_model_->getModel()->incremental_inference(
    batch_size_, draft_in, empty_label, prompt_len, 0, prompt_len, false);
  auto target_prefill_out = target_model_->getModel()->incremental_inference(
    batch_size_, target_in, empty_label, prompt_len, 0, prompt_len, false);

  // Get first token from target model
  unsigned int first_token = argmax(target_prefill_out[0], num_vocab_);
  registerOutputs({first_token}, prompt_len, 1, log_output);

  for (auto &out : draft_prefill_out)
    delete[] out;
  for (auto &out : target_prefill_out)
    delete[] out;

  auto end_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_prefill - start_prefill);

  // ──────── TOKEN GENERATION with Speculative Decoding ────────
  auto start_generation = std::chrono::high_resolution_clock::now();

  unsigned int current_pos = prompt_len + 1; // next position to generate
  unsigned int total_generated = 0;
  unsigned int total_accepted = 0;
  unsigned int total_draft_steps = 0;
  bool eos_reached = false;

  // Current token to feed as input
  unsigned int current_token = first_token;

  unsigned int K = num_speculative_tokens_;

  while (!eos_reached && total_generated < num_to_generate_ &&
         current_pos + K < max_seq_len_) {

    // ──── DRAFT phase: generate K tokens with layer-skipped model ────
    std::vector<unsigned int> draft_tokens(K);
    std::vector<std::vector<float>> draft_probs(K);

    draft_input[0] = static_cast<float>(current_token);

    for (unsigned int k = 0; k < K; ++k) {
      unsigned int draft_from = current_pos + k - 1;
      unsigned int draft_to = current_pos + k;

      auto draft_out = draft_model_->getModel()->incremental_inference(
        batch_size_, draft_in, empty_label, prompt_len, draft_from, draft_to,
        false);

      // Save draft probabilities for acceptance check
      draft_probs[k] = softmax(draft_out[0], num_vocab_);

      if (do_sample) {
        // Apply temperature
        std::vector<float> temp_logits(num_vocab_);
        for (unsigned int v = 0; v < num_vocab_; ++v)
          temp_logits[v] = draft_out[0][v] / std::max(temperature_, 1e-5f);
        auto probs = softmax(temp_logits.data(), num_vocab_);
        draft_tokens[k] = sampleFromProbs(probs);
      } else {
        draft_tokens[k] = argmax(draft_out[0], num_vocab_);
      }

      for (auto &out : draft_out)
        delete[] out;

      // Feed draft token as next input
      draft_input[0] = static_cast<float>(draft_tokens[k]);

      // Check EOS in draft
      if (std::find(eos_token_ids_.begin(), eos_token_ids_.end(),
                    draft_tokens[k]) != eos_token_ids_.end()) {
        // Truncate K to this point (include the EOS token)
        K = k + 1;
        break;
      }
    }

    // ──── VERIFY phase: run full model on all K draft tokens in parallel ────
    // Set up target input with all K draft tokens
    target_input[0] = static_cast<float>(current_token);
    for (unsigned int k = 0; k < K; ++k)
      target_input[k + 1] = static_cast<float>(draft_tokens[k]);

    // Run target model in prefill-style: processes K+1 tokens at once
    // (current_token + K draft tokens)
    // This builds KV cache for all positions and returns last position's logits
    unsigned int verify_from = current_pos - 1;
    unsigned int verify_to = current_pos - 1 + K + 1;

    auto target_verify_out = target_model_->getModel()->incremental_inference(
      batch_size_, target_in, empty_label, prompt_len, verify_from, verify_to,
      false);

    // ──── Extract ALL-position logits via forEachLayer ────
    // After the forward pass, output_norm's output tensor contains hidden
    // states for all (K+1) positions. We compute logits for each position
    // using the LM head weight matrix.
    //
    // target_logits_all[i] = logits at position i (i = 0..K)
    //   position 0: what target predicts given context → verify draft_tokens[0]
    //   position K: bonus token (already in target_verify_out[0])

    std::vector<std::vector<float>> target_logits_all(K + 1);
    unsigned int hidden_dim = 0;

    // Step 1: Get output_norm hidden states and LM head weight
    nntrainer::Tensor *output_norm_tensor = nullptr;
    nntrainer::Tensor *lm_head_weight = nullptr;

    target_model_->getModel()->forEachLayer(
      [&](ml::train::Layer &layer, nntrainer::RunLayerContext &ctx,
          void *user_data) {
        if (layer.getName() == "output_norm") {
          output_norm_tensor = &ctx.getOutput(0);
          hidden_dim = output_norm_tensor->width();
        } else if (layer.getName() == "output_of_causallm") {
          lm_head_weight = &ctx.getWeight(0);
        }
      },
      nullptr);

    if (output_norm_tensor && lm_head_weight && hidden_dim > 0) {
      // Step 2: Compute logits for each position manually
      // output_norm output layout: positions [0..K] are stored contiguously
      // at offset 0 within batch 0, each position has `hidden_dim` floats

      ml::train::TensorDim step_dim(
        1, 1, 1, hidden_dim,
        output_norm_tensor->getTensorType());
      ml::train::TensorDim logit_dim(
        1, 1, 1, num_vocab_,
        ml::train::TensorDim::TensorType(
          output_norm_tensor->getTensorType().format,
          ml::train::TensorDim::DataType::FP32));

      for (unsigned int pos = 0; pos <= K; ++pos) {
        nntrainer::Tensor hidden_step =
          output_norm_tensor->getSharedDataTensor(
            step_dim, pos * hidden_dim, true);

        target_logits_all[pos].resize(num_vocab_);
        nntrainer::Tensor logit_tensor = nntrainer::Tensor::Map(
          target_logits_all[pos].data(),
          num_vocab_ * sizeof(float), logit_dim, 0);

        hidden_step.dot(*lm_head_weight, logit_tensor, false, false);
      }
    } else {
      // Fallback: use only the last position's logits from the standard output
      // This means we can only verify the last draft token + get bonus
      // For positions 0..K-1, assume acceptance (degraded mode)
      std::cerr << "[SSD Warning] Could not extract all-position logits. "
                << "Running in degraded mode.\n";
      target_logits_all[K].resize(num_vocab_);
      std::memcpy(target_logits_all[K].data(), target_verify_out[0],
                  num_vocab_ * sizeof(float));
      // Fill earlier positions with draft logits (auto-accept)
      for (unsigned int k = 0; k < K; ++k) {
        target_logits_all[k].resize(num_vocab_);
        // Copy draft probs as logits to force acceptance
        for (unsigned int v = 0; v < num_vocab_; ++v)
          target_logits_all[k][v] = std::log(draft_probs[k][v] + 1e-10f);
      }
    }

    for (auto &out : target_verify_out)
      delete[] out;

    // ──── ACCEPT / REJECT ────
    unsigned int num_accepted = 0;
    unsigned int resampled_token = 0;
    bool has_resampled = false;

    for (unsigned int k = 0; k < K; ++k) {
      auto target_probs_k =
        softmax(target_logits_all[k].data(), num_vocab_);

      if (!do_sample) {
        // Greedy: accept if argmax matches
        unsigned int target_choice =
          argmax(target_logits_all[k].data(), num_vocab_);
        if (target_choice == draft_tokens[k]) {
          ++num_accepted;
        } else {
          // Reject: use target's choice as the resampled token
          resampled_token = target_choice;
          has_resampled = true;
          break;
        }
      } else {
        // Sampling: accept with probability min(1, p_target / p_draft)
        float p_target = target_probs_k[draft_tokens[k]];
        float p_draft = draft_probs[k][draft_tokens[k]];
        float accept_prob =
          (p_draft > 1e-10f) ? std::min(1.0f, p_target / p_draft) : 0.0f;

        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        if (uniform(rng_) < accept_prob) {
          ++num_accepted;
        } else {
          // Reject: resample from max(0, p_target - p_draft) distribution
          std::vector<float> adjusted(num_vocab_);
          float adj_sum = 0.0f;
          for (unsigned int v = 0; v < num_vocab_; ++v) {
            adjusted[v] = std::max(0.0f, target_probs_k[v] - draft_probs[k][v]);
            adj_sum += adjusted[v];
          }
          if (adj_sum > 1e-10f) {
            for (unsigned int v = 0; v < num_vocab_; ++v)
              adjusted[v] /= adj_sum;
            resampled_token = sampleFromProbs(adjusted);
          } else {
            resampled_token = sampleFromProbs(target_probs_k);
          }
          has_resampled = true;
          break;
        }
      }
    }

    // Bonus token: if all K accepted, get one extra from target position K
    unsigned int bonus_token = 0;
    bool has_bonus = false;
    if (num_accepted == K) {
      if (!do_sample) {
        bonus_token = argmax(target_logits_all[K].data(), num_vocab_);
      } else {
        auto bonus_probs =
          softmax(target_logits_all[K].data(), num_vocab_);
        bonus_token = sampleFromProbs(bonus_probs);
      }
      has_bonus = true;
    }

    // ──── REGISTER accepted tokens ────
    std::vector<unsigned int> accepted_ids;

    // Add accepted draft tokens
    for (unsigned int k = 0; k < num_accepted; ++k)
      accepted_ids.push_back(draft_tokens[k]);

    // Add resampled or bonus token
    if (has_resampled)
      accepted_ids.push_back(resampled_token);
    else if (has_bonus)
      accepted_ids.push_back(bonus_token);

    unsigned int num_new_tokens = accepted_ids.size();

    if (num_new_tokens > 0) {
      // Check for EOS in accepted tokens
      unsigned int eos_pos = num_new_tokens;
      for (unsigned int i = 0; i < num_new_tokens; ++i) {
        if (std::find(eos_token_ids_.begin(), eos_token_ids_.end(),
                      accepted_ids[i]) != eos_token_ids_.end()) {
          eos_pos = i;
          eos_reached = true;
          break;
        }
      }

      unsigned int tokens_to_register =
        eos_reached ? eos_pos : num_new_tokens;

      if (tokens_to_register > 0) {
        registerOutputs(accepted_ids, current_pos, tokens_to_register,
                        log_output);
        total_generated += tokens_to_register;
      }

      // Advance position
      current_pos += tokens_to_register;
      if (tokens_to_register > 0)
        current_token = accepted_ids[tokens_to_register - 1];
    }

    total_accepted += num_accepted;
    ++total_draft_steps;

    // Reset K for next iteration
    K = num_speculative_tokens_;
  }

  free(draft_input);
  free(target_input);

  auto end_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_generation -
                                                          start_generation);
  auto end_total = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_total - start_total);
  size_t peak_memory = getPeakMemoryKb();

  float acceptance_rate =
    total_draft_steps > 0
      ? static_cast<float>(total_accepted) /
          (total_draft_steps * num_speculative_tokens_)
      : 0.0f;

  if (log_output) {
    std::cout << "\n\n";
    std::cout << "========[ Self-Speculative Decoding with NNTrainer ]========\n";
    std::cout << "draft layers: " << draft_num_layers_
              << " / target layers: " << target_num_layers_
              << " / K=" << num_speculative_tokens_ << "\n";
    std::cout << "prefill: " << prompt_len << " tokens, "
              << prefill_duration.count() << " ms, "
              << (double)prompt_len / prefill_duration.count() * 1000
              << " TPS\n";
    std::cout << "generation: " << total_generated << " tokens, "
              << generation_duration.count() << " ms, "
              << (double)total_generated / generation_duration.count() * 1000
              << " TPS\n";
    std::cout << "acceptance rate: " << acceptance_rate * 100.0f << "%"
              << " (" << total_accepted << " / "
              << total_draft_steps * num_speculative_tokens_ << ")\n";
    std::cout << "total: " << total_duration.count() << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "==========================================================\n";
  }

  performance_metrics_.prefill_tokens = prompt_len;
  performance_metrics_.prefill_duration_ms = prefill_duration.count();
  performance_metrics_.generation_tokens = total_generated;
  performance_metrics_.generation_duration_ms = generation_duration.count();
  performance_metrics_.total_duration_ms = total_duration.count();
  performance_metrics_.peak_memory_kb = peak_memory;
}

} // namespace causallm
