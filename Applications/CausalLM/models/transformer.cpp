// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <chrono>
#include <fstream>
#include <mutex>

#include <app_context.h>
#if defined(ENABLE_OPENCL)
#include <cl_context.h> // GPU registration goes via the Engine facade now; the
                        // header is OpenCL-only -> guard for the no-OpenCL build.
#endif
#include <engine.h>
#include <model.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context.h>
#include <cuda_rmsnorm_layer.h>
#include <per_layer_slice.h>
#endif

#include <llm_util.hpp>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <qs4cx_tensor.h>
#include <rms_norm.h>
#include <per_layer_slice_gpu.h>
#include <rms_norm_gpu.h>

namespace causallm {

/**
 * @brief Load a file as a binary string.
 */
ml::train::ModelFormat
Transformer::formatFromExtension(const std::string &weight_path) {
  const auto dot = weight_path.find_last_of('.');
  if (dot != std::string::npos) {
    const std::string ext = weight_path.substr(dot + 1);
    if (ext == "safetensors")
      return ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS;
  }
  return ml::train::ModelFormat::MODEL_FORMAT_BIN;
}

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

/**
 * @brief Convert model_type text from config to ModelType.
 */
ModelType strToModelType(std::string model_type) {

  std::string model_type_lower = model_type;
  std::transform(model_type_lower.begin(), model_type_lower.end(),
                 model_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
    {"embedding", ModelType::EMBEDDING}};

  if (model_type_map.find(model_type_lower) == model_type_map.end()) {
    return ModelType::UNKNOWN;
  }

  return model_type_map.at(model_type_lower);
}

/**
 * @brief Construct a Transformer and initialize shared config state.
 */
Transformer::Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                         ModelType model_type) {

  std::string config_model_type_str = "Model";
  if (nntr_cfg.contains("model_type")) {
    config_model_type_str = nntr_cfg["model_type"].get<std::string>();
  }

  ModelType config_model_type = strToModelType(config_model_type_str);

  if (model_type != config_model_type) {
    throw std::runtime_error("model_type mismatch. Class Type: " +
                             std::to_string(static_cast<int>(model_type)) +
                             ", Config Type: " + config_model_type_str);
  }

  const bool skip_tokenizer = nntr_cfg.contains("skip_tokenizer") &&
                              nntr_cfg["skip_tokenizer"].get<bool>();

  // Initialize the model with the provided configurations. Vision models such
  // as TimmViT defer this to their derived constructor because the base
  // Transformer setup expects text-model fields.
  if (!(skip_tokenizer && model_type == ModelType::MODEL)) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  // Skip tokenizer if specified, or when no tokenizer_file is configured
  // (e.g. vision-encoder sub-models composed into a multimodal handle, whose
  // config carries no tokenizer). Avoids a json type_error on a null path.
  if (skip_tokenizer || !nntr_cfg.contains("tokenizer_file") ||
      nntr_cfg["tokenizer_file"].is_null()) {
    tokenizer = nullptr; // No tokenizer for this model
  } else {
    // [round-13 init overlap] The ~30MB tokenizer.json parse measured ~680ms
    // and is independent of graph compile + weight load -- run it on a side
    // thread and join at first use (ensureTokenizer / getTokenizer).
    const std::string tok_path = nntr_cfg["tokenizer_file"];
    tokenizer_future_ =
      std::async(std::launch::async, [tok_path]() {
        return tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(tok_path));
      });
  }
};

/**
 * @brief Set common transformer parameters from model configs.
 */
void Transformer::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {

  /** Initialize nntr prameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];
  // [Phase C Path B] Legacy QINT4 weights now load as the canonical QS4CX class;
  // remap the config dtype so the whole pipeline (tensor factory + runtime
  // consumers) is QS4CX. The on-disk bytes stay legacy QINT4 and are transcoded
  // losslessly at read time, keyed on model_tensor_type ("QINT4-*"), which is
  // intentionally NOT remapped so the loader still knows the on-disk format.
  if (FC_LAYER_DTYPE == "QINT4")
    FC_LAYER_DTYPE = "QS4CX";
  USE_FLASH_ATTENTION = nntr_cfg.contains("use_flash_attention")
                          ? nntr_cfg["use_flash_attention"].get<bool>()
                          : true;
  EMBEDDING_FILE_NAME = nntr_cfg.value("embedding_file_name", std::string());
  PLE_FILE_NAME = nntr_cfg.value("ple_file_name", std::string());
  PLE_SIDECAR_EXPORT = nntr_cfg.value("ple_sidecar_export", std::string());
  EMBD_SIDECAR_EXPORT = nntr_cfg.value("embd_sidecar_export", std::string());
  LMHEAD_UNTIE =
    nntr_cfg.contains("lmhead_untie") && nntr_cfg["lmhead_untie"].get<bool>();

  if (cfg.contains("is_causal")) {
    IS_CAUSAL = cfg["is_causal"].get<bool>();
  } else if (cfg.contains("use_bidirectional_attention") &&
             !cfg["use_bidirectional_attention"].is_null()) {
    IS_CAUSAL = !cfg["use_bidirectional_attention"].get<bool>();
  } else if (nntr_cfg.contains("model_type") &&
             strToModelType(nntr_cfg["model_type"].get<std::string>()) ==
               ModelType::EMBEDDING &&
             cfg.contains("architectures") && cfg["architectures"].is_array() &&
             !cfg["architectures"].empty() &&
             cfg["architectures"][0].get<std::string>() == "Qwen2Model") {
    IS_CAUSAL = false;
  }

  NUM_VOCAB = cfg["vocab_size"];
  DIM = cfg["hidden_size"];
  INTERMEDIATE_SIZE = cfg["intermediate_size"];
  NUM_LAYERS = cfg["num_hidden_layers"];
  NUM_HEADS = cfg["num_attention_heads"];
  HEAD_DIM = cfg.contains("head_dim")
               ? cfg["head_dim"].get<int>()
               : DIM / NUM_HEADS; // default value is hidden_size / num_heads
  NUM_KEY_VALUE_HEADS = cfg.contains("num_key_value_heads")
                          ? cfg["num_key_value_heads"].get<int>()
                          : NUM_HEADS;
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;
  SLIDING_WINDOW_PATTERN = cfg.contains("sliding_window_pattern")
                             ? cfg["sliding_window_pattern"].get<unsigned int>()
                             : 1;
  MAX_POSITION_EMBEDDINGS = cfg["max_position_embeddings"].get<unsigned int>();
  if (cfg.contains("rope_theta")) {
    ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
  } else if (cfg.contains("rope_parameters") &&
             cfg["rope_parameters"].contains("rope_theta")) {
    ROPE_THETA = cfg["rope_parameters"]["rope_theta"].get<unsigned int>();
  } else if (cfg.contains("rope_parameters") &&
             cfg["rope_parameters"].contains("sliding_attention")) {
    json &rope_cfg = cfg["rope_parameters"]["sliding_attention"];
    ROPE_THETA = rope_cfg.value("rope_theta", 10000);
  } else {
    ROPE_THETA = cfg.value("rope_theta", 10000);
  }
  TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
  NORM_EPS = cfg["rms_norm_eps"];
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  // [Adreno image-attn model vetting] now lives in the model classes that
  // have geometry the OHWI kernels cannot serve (gemma4: global_head_dim=512
  // exceeds the proven d<=256 tiling — see Gemma4Transformer::setupParameters).
  // The sliding-window case is handled IN the kernels since qk_matmul_f16_ohwi
  // (+_img) grew a local_window argument (n + W <= q_pos lower-bound mask),
  // so window < max_seq_len no longer forces the flash path.

  return;
};

/**
 * @brief Build and compile the symbolic transformer graph.
 */
void Transformer::ensureTokenizer() {
  std::lock_guard<std::mutex> lk(tokenizer_join_mtx_);
  if (tokenizer_future_.valid())
    tokenizer = tokenizer_future_.get();
}

void Transformer::initialize() {

  // [NNTR_INIT_TRACE] init-latency dissection (round-13 follow-up).
  static const bool init_trace = std::getenv("NNTR_INIT_TRACE") != nullptr;
  const auto _tt0 = std::chrono::steady_clock::now();
  auto _tlap = [&](const char *what) {
    if (!init_trace)
      return;
    std::fprintf(stderr, "[init-trace] %8.1f ms  %s\n",
                 std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - _tt0)
                   .count(),
                 what);
    std::fflush(stderr);
  };

  // RegisterCustomLayers
  registerCustomLayers();
  _tlap("registerCustomLayers");

  // create model and apply properties before compile()
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }
  model->setProperty(model_props);

  // build symbolic tensor graph and compile from (input, output)
  auto [x, y] = constructModel();
  _tlap("constructModel (symbolic graph)");

  if (model->compile(x, y, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }
  _tlap("model->compile+initialize (ccapi)");

  is_initialized = true;
#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

/**
 * @brief Construct the default decoder-only transformer graph.
 */
std::pair<Tensor, Tensor> Transformer::constructModel() {

  // input
  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // embedding
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  NNTR_THROW_IF(TIE_WORD_EMBEDDINGS && !EMBEDDING_FILE_NAME.empty(),
                std::invalid_argument)
    << "embedding_file_name requires untied embedding_layer";
  LayerHandle embedding(createLayer(
    embedding_type,
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));
  Tensor h = embedding(x);

  // transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // final rms_norm. NOTE: stays on CausalLM's custom RMSNormLayer
  // ("rms_norm" type, app_context only) so the fused-rmsq + v8c FC
  // consumer chain works. The nntrainer GPU RMSNormLayerCl uses type
  // "rmsnorm" (different) and has a separate reduction-order drift
  // issue documented in [chain-robustification-dead].
  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false"),
                             withKey("engine", causallm_engine())}));
  h = out_norm(h);

  return {x, h};
};

std::vector<std::string> Transformer::buildEmbeddingLayerProperties(
  const std::string &name, unsigned int in_dim, unsigned int out_dim,
  const std::string &weight_dtype, float scale,
  const std::string &quantized_lut_path) const {
  std::vector<std::string> props = {
    withKey("name", name),
    withKey("in_dim", std::to_string(in_dim)),
    withKey("weight_dtype", weight_dtype),
    withKey("out_dim", std::to_string(out_dim)),
    withKey("scale", std::to_string(scale)),
  };

  if (!quantized_lut_path.empty())
    props.emplace_back(withKey("quantized_lut_path", quantized_lut_path));

  return props;
}

/**
 * @brief Load model weights from a binary nntrainer model file.
 */
void Transformer::load_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  try {
    model->load(weight_path, formatFromExtension(weight_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load model weights: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Save model weights to a binary nntrainer model file.
 */
void Transformer::save_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, formatFromExtension(weight_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Save model weights with optional dtype conversion.
 */
void Transformer::save_weight(
  const std::string &weight_path, ml::train::TensorDim::DataType dtype,
  const std::map<std::string, ml::train::TensorDim::DataType> &layer_dtype_map,
  ml::train::ISA target_isa) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, formatFromExtension(weight_path), dtype,
                layer_dtype_map, target_isa);

  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights with dtype: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Repack all QS4CX weights after loading.
 */
void Transformer::repack_weight() {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before repack_weight().");
  }

  // [perf/thermal] The KAI rhs-pack below is consumed ONLY by the ARM CPU
  // KleidiAI GEMM. The GPU (v8c) and x86 paths read the plain on-disk QS4CX
  // blob directly (see the comment at the pack() call). On a GPU run the whole
  // loop is therefore redundant CPU work — and it is single-threaded, so for a
  // large model it pins one core to a thermal shutdown (Adreno: GPU idle at 0%,
  // one CPU core -> ~104C -> device reboot; 96%+ of CPU was in
  // kai_run_rhs_pack). Skip it on every non-CPU engine — "gpu" (OpenCL) AND
  // "cuda" both consume the plain blob; only the ARM CPU (KAI) run packs.
  // (On ARM64 CUDA the old =="gpu" check let packF16Activation allocate a
  // full unconsumed host copy of every FC weight.)
  if (causallm_engine() != "cpu") {
    ml_logd("repack_weight: skipped on %s engine (consumes plain QS4CX blob)",
            causallm_engine().c_str());
    return;
  }
  // fp16-act graphs dispatch QS4CX FCs through HalfTensor::dot, whose KAI rhs
  // is the fp16-scale layout (packF16Activation) — pack()'s fp32-facade rhs
  // would be dead weight there (an unconsumed full extra copy of every FC in
  // RAM, and ~11% of a 1K-run's CPU in kai_run_rhs_pack). fp32-act graphs
  // dispatch through FloatTensor::dotQs4cx and need pack() as before.
  const bool f16_act =
    MODEL_TENSOR_TYPE.size() >= 5 &&
    MODEL_TENSOR_TYPE.compare(MODEL_TENSOR_TYPE.size() - 5, 5, "-FP16") == 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [f16_act](ml::train::Layer &l, nntrainer::RunLayerContext &context,
                   void *) {
      auto weights = context.getWeights();
      for (auto &w : weights) {
        if (w->getVariableRef().getDataType() ==
            ml::train::TensorDim::DataType::QS4CX) {
          // [Phase B] KAI rhs-pack is the ARM (i8mm) CPU GEMM's in-memory
          // derivation; on x86 it is NYI and the x86 CPU GEMM + the GPU v8c
          // path consume the plain on-disk QS4CX (nibbles + fp32 scales)
          // directly. Skip-on-NYI so a single QS4CX weight set loads on every
          // backend (ARM packs, x86/GPU use the plain blob). [weight 한벌]
          try {
            if (f16_act) {
              w->getVariableRef().packF16Activation();
            } else {
              w->getVariableRef().pack();
            }
          } catch (const std::exception &e) {
            ml_logd("QS4CX pack skipped (engine consumes plain blob): %s",
                    e.what());
          }
        }
      }
    };
  try {
    model->forEachLayer(fn, nullptr);
    ml_logd("QS4CX weights repacked successfully");
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to repack weights: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Run a transformer model for a prompt.
 */
void Transformer::run(const WSTR prompt, bool do_sample,
                      const WSTR system_prompt, const WSTR tail_prompt,
                      bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before run().");
  }
  ///@note This part should be filled in.
  /// The run action can be defined by the precedent classes.
}

/**
 * @brief Create one decoder block with attention and feed-forward layers.
 */
Tensor Transformer::createTransformerDecoderBlock(const int layer_id,
                                                  Tensor input) {

  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("engine", causallm_engine())}));
  Tensor residual = decoder_add({input, att_out});

  LayerHandle ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("engine", causallm_engine())}));
  return decoder_output({residual, ffn_out});
}

/**
 * @brief Create external KV-cache placeholder tensors for one layer.
 */
std::pair<Tensor, Tensor>
Transformer::createKVCachePlaceholders(const int layer_id, int n_heads) {
  const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);
  const unsigned int kv_width =
    static_cast<unsigned int>(HEAD_DIM * n_heads / GQA_SIZE);
#ifdef ENABLE_FP16
  ml::train::TensorDim cache_dim(
    {BATCH_SIZE, 1, max_timestep, kv_width},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP16});

  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(layer_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(layer_id));
  return {cache_k, cache_v};
#else
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(max_timestep) + ":" +
                                  std::to_string(kv_width);

  LayerHandle cache_k_input(createLayer(
    "input",
    {withKey("name", "cache_k_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));
  LayerHandle cache_v_input(createLayer(
    "input",
    {withKey("name", "cache_v_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
#endif
}

/**
 * @brief Create the default attention subgraph.
 */
Tensor Transformer::createAttention(const int layer_id, int seq_len,
                                    int n_heads, int head_dim, Tensor query,
                                    Tensor key, Tensor value) {

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor q = wq(query);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor k = wk(key);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor v = wv(value);

  // External KV cache placeholders (per-layer). Their actual storage is owned
  // by the host (KVCacheManager) and bound at runtime via setExternalTensors.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  // Attention core layer
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                 ? SLIDING_WINDOW
                                 : UINT_MAX),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("use_gemm_attention", USE_FLASH_ATTENTION ? "true" : "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  return wo(a);
}

/**
 * @brief Create the default feed-forward subgraph.
 */
Tensor Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                              Tensor input) {

  // Create gate BEFORE up: the model loader assigns file offsets in graph
  // creation order (positional, not by name), and the converters write the
  // FFN weights gate_proj -> up_proj -> down_proj (the HF convention). If up
  // is created first, ffn_up loads the gate_proj bytes and ffn_gate loads the
  // up_proj bytes, so swiglu computes silu(up)*gate instead of silu(gate)*up
  // -- coherent-looking but wrong (the global gate/up swap; Gemma2/3 avoided
  // it by overriding createMlp gate-first).
  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor gate = ffn_gate(input);

  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor up = ffn_up(input);

  LayerHandle swiglu(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("engine", causallm_engine())}));
  Tensor act = swiglu({gate, up});

  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  return ffn_down(act);
}

/**
 * @brief Register custom CausalLM layers in the nntrainer app context.
 */
void Transformer::registerCustomLayers() {
  ///
  const auto &ct_engine = nntrainer::Engine::Global();
  // CPU layer classes on the cpu (app) context — through Engine's registration
  // facade, no static_cast to AppContext.
  try {
    // swiglu promoted to core app_context.cpp [T12] (merged app fork into the
    // backend-neutral nntrainer::SwiGLULayer).
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::RMSNormLayer>);
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::MHACoreLayer>);
    // tie_word_embedding promoted to core [T12] — self-registered on cpu/gpu/cuda.
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::EmbeddingLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }

  // GPU variants: same type strings as the CPU classes but registered on the
  // gpu context so engine=gpu createLayer routes there. The GPU classes use raw
  // getData() pointers + GPU dispatches; they avoid any CPU-only Tensor ops
  // (Tensor::multiply / add_i / dot) that crash on gpu-context tensors. Inert
  // when there is no "gpu" context (CPU-only / NNTR_ENGINE=cpu builds).
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::RMSNormLayerGPU>);
    // Gemma4 GPU-resident per_layer_slice: same type string as the CPU class,
    // registered here so engine=gpu routes to the GPU kernel (no host round-trip
    // that would break residency). scalar_multiply's GPU variant is now
    // self-registered in core cl_context.cpp [T12].
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::PerLayerSliceLayerGPU>);
    // tie_word_embedding promoted to core cl_context.cpp [T12].
    // MHACoreLayer on the gpu context enables engine=gpu attention. The same
    // class runs on both backends: forwarding() dispatches the GPU kernels when
    // NNTR_MHA_GPU is set and Q/K/V/cache are SVM-resident, else the CPU NEON
    // path. Additive — a node with no engine= property keeps routing to CPU.
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::MHACoreLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register GPU-routed layer on gpu ctx: " << e.what()
              << std::endl;
  }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Additive CUDA backend: register the host CausalLM layer classes on the cuda
  // context too. engine=cuda tensors are Unified Memory (host-coherent), so the
  // CPU implementations run correctly on them (no cl_mem); GPU kernels are
  // layered on per-layer later. Inert when there is no "cuda" context.
  try {
    // swiglu promoted to core cuda_context.cpp [T12].
    // CUDA RMSNorm (FP32-safe sum-of-squares) instead of the host
    // causallm::RMSNormLayer, whose FP16 path squares in FP16 and overflows on
    // gemma4's large residual (pre_ffn_norm |x|~1688 -> +Inf -> garbage). Same
    // "rms_norm" type, so it takes this slot.
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<nntrainer::CudaRMSNormLayer>);
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<causallm::MHACoreLayer>);
    // tie_word_embedding promoted to core cuda_context.cpp [T12].
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<causallm::EmbeddingLayer>);
    // gemma4 PLE host impl (the GPU variant is OpenCL-only and would mis-run on
    // UVM). Without it on the cuda context the type falls back to the wrong
    // factory. scalar_multiply is now self-registered in core cuda_context.cpp [T12].
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<causallm::PerLayerSliceLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register layer on cuda ctx: " << e.what()
              << std::endl;
  }
#endif
}

} // namespace causallm
