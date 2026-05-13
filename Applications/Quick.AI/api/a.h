// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn_base.h
 * @brief  QNN model for Quick.AI template
 * @note   This file implements a layer that executes QNN binary file within
 * transformer.h architecture.
 *
 */

#ifndef __QUICK_DOT_AI_QNN_H__
#define __QUICK_DOT_AI_QNN_H__

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif


#include "graph_parser.h"
#include <transformer.h>

namespace causallm {
/**
 * @brief QNN Model info
 * @note  This struct contains QNN model information for execution.
 */
struct QNNModelInfo {
  GraphInfo graph_info;
  ModelHandle model_handle;
  std::vector<ml::train::TensorDim::IO_TensorType> model_inputs;
};

/**
 * @brief Quick_Dot_AI_QNN_Base class
 * @note  This is the base class for QNN.
 */
class Quick_Dot_AI_QNN : public Transformer {

public:
  Quick_Dot_AI_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    LOGD("--------------------------------- Quick_Dot_AI_QNN");
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Quick_Dot_AI_QNN() override;

  void initialize() override;

  void load_weight(const std::string &weight_path) override;

  void save_weight(const std::string &weight_path) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void constructModel() override;

  /**
   * @brief Attach (or detach) a BaseStreamer to intercept per-token output.
   *        Passing nullptr detaches any currently-attached streamer.
   */
  void setStreamer(::BaseStreamer *streamer) override { streamer_ = streamer; }


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

protected:
  // nntr_config
  std::string model_file_name;
  std::string binary_config_path;
  std::vector<std::string> graphs_to_use;

  // config
  int vocab_size;

  // Model map, key: graph name, value: QNN model info
  std::map<std::string, QNNModelInfo> models;

  bool uses_embedding = true;

  // Optional external embedding file path (absolute after fix_paths).
  // Only used when uses_embedding=false — derived classes mmap this
  // and provide per-token lookup during generation.
  std::string embedding_file_name;

  // Streaming support
  ::BaseStreamer *streamer_ = nullptr;
  std::string last_output_;
};

} // namespace causallm

#endif /* __QUICK_DOT_AI_QNN_BASE_H__ */