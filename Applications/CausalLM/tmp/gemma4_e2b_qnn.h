// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_e2b_qnn.h
 * @brief  QNN model extension template
 * @note   This file demonstrates how to create a custom QNN Quick.AI model
 *         by extending the base CausalLM class from nntrainer.
 *
 */

#ifndef __GEMMA4_E2B_QNN_H__
#define __GEMMA4_E2B_QNN_H__

#include "quick_dot_ai_qnn.h"

namespace causallm {

/**
 * @brief Gemma4_E2B_QNN class
 * @note  This is the main class you register with the Factory.
 *
 */
class Gemma4_E2B_QNN : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "Gemma4_E2B_QNN";

  Gemma4_E2B_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {
      LOGD("Gemma4 E2B parameters set up ");
      setupParameters(cfg, generation_cfg, nntr_cfg);
    }

  virtual ~Gemma4_E2B_QNN();

  void initialize();

  void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg) override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

private:
  // ---------------------------------------------------------------
  // PLE storage. Two on-disk formats are supported, auto-detected
  // from `ple_file_name`:
  //   *.json  → 4-bit packed (manifest + .bin), needs dequant + per-layer
  //             requant into the consumer's UINT16 quant space.
  //   else    → raw UINT16, already in consumer space → memcpy only.
  // ---------------------------------------------------------------
  bool ple_is_4bit_ = false;
  int ple_fd_ = -1;
  const uint8_t *ple_mmap_ = nullptr;        // 4-bit byte view
  const uint16_t *ple_u16_mmap_ = nullptr;   // raw uint16 view (alias of mmap)
  size_t ple_file_size_ = 0;
  float ple_scale_ = 1.0f;                   // 4-bit only
  int ple_offset_ = 0;                       // 4-bit only
  size_t ple_row_elems_ = 0;                 // elements per token (e.g. 8960)
  size_t ple_row_bytes_ = 0;                 // bytes per token
  size_t ple_per_layer_ = 256;               // elements per token per layer
  size_t ple_layers_ = 0;                    // 35 (full layers)

  std::vector<uint16_t *> prefill_per_layer_dst_;
  std::vector<uint16_t *> generation_per_layer_dst_;
  std::vector<float>      prefill_per_layer_scale_;
  std::vector<int>        prefill_per_layer_offset_;
  std::vector<float>      generation_per_layer_scale_;
  std::vector<int>        generation_per_layer_offset_;

  void open_ple_file_();
  void close_ple_file_();
  void fill_prefill_ple_chunk_(const std::vector<int> &tokens,
                               int chunk_idx, int chunk_len);
  void fill_generation_ple_(int token_id);

  // Existing graph-bound input pointers
  uint16_t *attention_mask;
  uint16_t *sliding_attention_mask;
  uint16_t *generation_attention_mask;
  uint16_t *generation_sliding_attention_mask;

  uint16_t *position_ids_cos;
  uint16_t *position_ids_sin;
  uint16_t *swa_position_ids_cos;
  uint16_t *swa_position_ids_sin;
  uint16_t *prefill_position_ids_cos;
  uint16_t *prefill_position_ids_sin;
  uint16_t *prefill_swa_position_ids_cos;
  uint16_t *prefill_swa_position_ids_sin;
  uint16_t *generation_position_ids_cos;
  uint16_t *generation_position_ids_sin;
  uint16_t *generation_swa_position_ids_cos;
  uint16_t *generation_swa_position_ids_sin;

  float *input_sample;
  float *generation_sample;

  // KV cache variables
  std::vector<uint16_t *> kvs;
  std::vector<uint16_t *> per_layer_embedding;
  std::vector<int> per_layer_embedding_size;
  std::vector<int> kv_sizes;
  std::vector<uint16_t *> fresh_kvs;
  std::vector<int> kv_row_lengths;
  std::vector<int> kv_columns;

  int prefill_attention_mask_elements = 0;
  int prefill_attention_mask_columns = 0;
  int prefill_sliding_attention_mask_elements = 0;
  int prefill_sliding_attention_mask_columns = 0;
  int generation_attention_mask_elements = 0;
  int generation_sliding_attention_mask_elements = 0;
  int generation_full_kv_past_length = 0;
  int generation_sliding_kv_past_length = 0;
  int rope_cache_seq_len = 0;

  // Config
  int num_hidden_layers;
  int max_window_layers;
  int hidden_size;
  int sequence_length;
  int vocab_size;
  int max_seq_len;
  int sliding_window;
  float local_rope_theta;
  float rope_theta;
  int context_size;
  int pos_dim;
  int swa_pos_dim;
  int g_head_dim;
  int l_head_dim;
  int head_dim;

  // generation_config
  int padding_token;
  std::vector<int> eos_tokens;
  int top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  float logit_scale;
  int logit_offset;

  // LoRA / PLE paths (optional)
  std::string lora_path;
  std::string ple_file_name;
};

} // namespace causallm

#endif /* __GEMMA4_E2B_QNN_H__ */
