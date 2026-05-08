// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn.h
 * @brief  QNN model for Quick.AI template
 * @note   This file implements a layer that executes QNN binary file within
 * transformer.h architecture.
 *
 */

#ifndef __QUICK_DOT_AI_QNN_OLD_H__
#define __QUICK_DOT_AI_QNN_OLD_H__

#include <transformer.h>

#include <set>

namespace causallm {

/**
 * @brief Gauss2_5_Causallm class
 * @note  This is the main class you register with the Factory.
 *        It combines CausalLM (for generation logic) with
 *        CustomTransformer (for the model architecture).
 *        Unlike CausalLM/CustomTransformer, this class does not us
 *        diamond interitance pattern.
 */
class Quick_Dot_AI_QNN_OLD : public Transformer {

public:
  Quick_Dot_AI_QNN_OLD(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Quick_Dot_AI_QNN_OLD();

  void initialize() override;

  virtual void initialize_input_outputs() = 0;
  virtual void initialize_kv_cache() = 0;

  void load_weight(const std::string &weight_path) override;

  void save_weight(const std::string &weight_path) override;

  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = "", const WSTR tail_prompt = "",
                   bool log_output = true) = 0;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void constructModel() override;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void registerCustomLayers() override;

  void setStreamer(::BaseStreamer *streamer) override { streamer_ = streamer; }

protected:
  // nntr_config
  std::string model_path;
  std::string embedding_path;
  std::string tokenizer_path;

  // config
  // TODO struct?
  std::string prefill_graph_name;
  std::string prefill_input_names;
  std::string prefill_output_names;
  std::string prefill_in_quant;
  std::string prefill_out_quant;
  std::string prefill_in_dim;
  std::string prefill_out_dim;
  std::string prefill_in_data_format;
  std::string prefill_out_data_format;
  std::string prefill_out_tensor_format;
  std::vector<std::string> prefill_non_embed_input_names;
  std::vector<std::string> prefill_non_embed_input_dims;
  ModelHandle prefill_model;

  std::string generation_graph_name;
  std::string generation_input_names;
  std::string generation_output_names;
  std::string generation_in_quant;
  std::string generation_out_quant;
  std::string generation_in_dim;
  std::string generation_out_dim;
  std::string generation_in_data_format;
  std::string generation_out_data_format;
  std::string generation_out_tensor_format;
  std::vector<std::string> generation_non_embed_input_names;
  std::vector<std::string> generation_non_embed_input_dims;
  ModelHandle generation_model;

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
  int head_dim;
  std::vector<int> lora_sizes;

  // generation_config
  int padding_token;
  int eos_token;
  int top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  float logit_scale;
  int logit_offset;

  bool uses_embedding = true;

  ::BaseStreamer *streamer_ = nullptr;

  // LoRA path (optional)
  std::string lora_path;

  // mmap-backed pre-quantized text embedding table. Loaded lazily in
  // initialize() when uses_embedding=false. Used both by the external
  // multimodal composer (via lookupEmbedding) and by this class's
  // generation loop to fetch the next token's embedding per step.
  void *embedding_mmap_ptr = nullptr;
  size_t embedding_mmap_size = 0;
  size_t embedding_bytes_per_token = 0; // hidden_size * sizeof(uint16_t)

  // Tracked resource management: all allocate()'d pointers are recorded
  // here so that ~Quick_Dot_AI_QNN_OLD can free them in one pass without
  // risking double-free or omission.
  std::set<void *> allocated_ptrs_;

  /// Allocate size bytes via allocate() and record the pointer for
  /// automatic cleanup in the destructor.
  void *tracked_allocate(size_t size);

  /// Deallocate every tracked pointer and clear the set.
  void deallocate_all();
};

} // namespace causallm

#endif /* __QUICK_DOT_AI_QNN_OLD_H__ */