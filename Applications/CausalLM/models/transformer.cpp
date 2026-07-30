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

#include <cstdio>
#include <fstream>
#include <mutex>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <llm_util.hpp>
#include <tokenizer_cache.h>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <nntrainer_log.h>
#include <per_layer_slice_gpu.h>
#include <qs4cx_tensor.h>
#include <reshaped_rms_norm.h>
#include <rms_norm.h>
#include <rms_norm_gpu.h>
#include <swiglu_layer.h>
#include <tie_word_embedding.h>

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

  // Record the role before any setupParameters() call so config parsing can
  // key decoder-only behaviour off it. Transformer is a virtual base of every
  // model, initialized once by the most-derived constructor, so this stays put
  // for the whole construction sequence.
  MODEL_TYPE = model_type;

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
    // Through the persistent snapshot cache (tokenizer_cache.cpp): a HIT
    // rebuilds the BPE from the snapshot's vocab/merges tables instead of
    // re-parsing ~32 MB of JSON; a miss / stale key / corrupt file parses
    // exactly as before (same FromBlobJSON call, same bytes) and schedules a
    // background, exit-joined rewrite. This parse is SYNCHRONOUS here, so
    // everything it saves comes straight off time-to-first-token.
    // NNTR_TOKENIZER_CACHE=0 opts out.
    tokenizer = causallm::LoadTokenizerCached(nntr_cfg["tokenizer_file"]);
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
  // Opt-in only on this tree; see the member's doc comment in transformer.h.
  USE_FLASH_ATTENTION = nntr_cfg.contains("use_flash_attention")
                          ? nntr_cfg["use_flash_attention"].get<bool>()
                          : false;
  EMBEDDING_FILE_NAME = nntr_cfg.value("embedding_file_name", std::string());
  PLE_FILE_NAME = nntr_cfg.value("ple_file_name", std::string());
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
  INTERMEDIATE_SIZE =
    cfg.contains("intermediate_size") ? cfg["intermediate_size"].get<int>() : 0;
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
  // The RoPE frequency tables are precomputed for positions
  // [0, MAX_POSITION_EMBEDDINGS) (precompute_freqs in mha_core), so a runtime
  // window past the model cap reads beyond the table -- SIGSEGV in
  // compute_rotary_emb_value once prefill crosses that position. Clamp
  // instead of crashing: positions past the trained range carry no value,
  // and a genuine long-context model raises its config cap so the clamp
  // adapts.
  if (MAX_SEQ_LEN > MAX_POSITION_EMBEDDINGS ||
      INIT_SEQ_LEN > MAX_POSITION_EMBEDDINGS) {
    std::fprintf(stderr,
                 "[causallm] WARNING: requested window (max_seq_len=%u, "
                 "init_seq_len=%u) exceeds the model's "
                 "max_position_embeddings=%u; clamping (RoPE tables only "
                 "cover the model range).\n",
                 MAX_SEQ_LEN, INIT_SEQ_LEN, MAX_POSITION_EMBEDDINGS);
    if (MAX_SEQ_LEN > MAX_POSITION_EMBEDDINGS)
      MAX_SEQ_LEN = MAX_POSITION_EMBEDDINGS;
    if (INIT_SEQ_LEN > MAX_SEQ_LEN)
      INIT_SEQ_LEN = MAX_SEQ_LEN;
  }
  // The generation budget has to stay strictly inside the window just settled
  // above. CausalLM sizes its token history once as BATCH_SIZE * MAX_SEQ_LEN
  // and indexes it as ids_history[b * MAX_SEQ_LEN + pos] with no bounds check,
  // while run() derives the prompt budget as the *unsigned* difference
  // MAX_SEQ_LEN - NUM_TO_GENERATE: with NUM_TO_GENERATE >= MAX_SEQ_LEN that
  // difference wraps to ~4e9, the prompt-truncation branch is never taken and
  // the whole prompt is written past the row stride of every batch row and off
  // the end of the last one. The clamp above can produce exactly that relation
  // -- a 512-position model driven with max_seq_len=2048, num_to_generate=1024
  // keeps its 1024 budget against a 512 window.
  //
  // This belongs here, in the same function that (re-)reads num_to_generate,
  // so it holds after every call: a derived Transformer whose constructor calls
  // setupParameters() again once the CausalLM base is built (Gemma4Transformer
  // does) reloads the raw value, which would silently drop a guard placed
  // anywhere else.
  //
  // Decoders only. Encoder models (BertTransformer, XLMRobertaForMaskedLM) also
  // derive from Transformer and ship configs with
  // num_to_generate == max_seq_len == 512, where the field is inert and no
  // token history exists; they must neither be warned about nor rewritten.
  if (MODEL_TYPE == ModelType::CAUSALLM &&
      (NUM_TO_GENERATE < 0 ||
       static_cast<unsigned int>(NUM_TO_GENERATE) >= MAX_SEQ_LEN)) {
    const int fitted = (NUM_TO_GENERATE < 0 || MAX_SEQ_LEN == 0)
                         ? 0
                         : static_cast<int>(MAX_SEQ_LEN - 1);
    std::fprintf(stderr,
                 "[causallm] WARNING: num_to_generate=%d does not fit in the "
                 "context window (max_seq_len=%u); clamping to %d so the "
                 "prompt keeps at least one token and the token history stays "
                 "inside its buffer.\n",
                 NUM_TO_GENERATE, MAX_SEQ_LEN, fitted);
    NUM_TO_GENERATE = fitted;
  }
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
  TIE_WORD_EMBEDDINGS = cfg.contains("tie_word_embeddings")
                          ? cfg["tie_word_embeddings"].get<bool>()
                          : false;
  NORM_EPS =
    cfg.contains("rms_norm_eps") ? cfg["rms_norm_eps"].get<float>() : 1e-5;
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  // Attention-logit soft-capping (Gemma family). Parsed here -- the one
  // setupParameters the base constructors provably reach -- so it cannot be
  // stranded in a derived override that never runs. Gated on cfg presence, so
  // a model without the key keeps the 0.0f default (no soft-cap). Consolidated
  // from the per-model setupParameters (gemma3/gemma4 held byte-identical
  // copies of this block).
  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }

  return;
};

/**
 * @brief Build and compile the symbolic transformer graph.
 */
void Transformer::initialize() {

  // RegisterCustomLayers
  registerCustomLayers();

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

  if (model->compile(x, y, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

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
  // No engine= on embedding0: "embedding_layer" has no gpu-context
  // registration, and createLayer on an unregistered type THROWS (it is not a
  // silent cpu fallback), so an untied model would fail at graph build. The
  // lookup is a gather, not a GEMM -- nothing to gain on the device.
  LayerHandle embedding(createLayer(
    embedding_type,
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));
  Tensor h = embedding(x);

  // transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // final rms_norm. NOTE: stays on CausalLM's custom RMSNormLayer ("rms_norm"
  // type) so engine=gpu routes to RMSNormLayerGPU, registered on the gpu
  // context by registerCustomLayers below. The nntrainer core RMSNormLayerCl
  // uses type "rmsnorm" (a different key) and is not what this resolves to.
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

  // The QS4CX weight repack (KAI rhs-pack) is only consumed by the ARM CPU
  // (KAI) inference path, which reaches it through Tensor::getPackedData(); the
  // "gpu" (OpenCL) and "cuda" engines consume the plain QS4CX blob directly. On
  // a non-CPU layer the pack is redundant single-threaded CPU work -- on a
  // large model it can pin one core to a thermal shutdown -- so decide per
  // layer, keyed on the engine the graph actually resolved for that layer.
  //
  // @note Do NOT key this on causallm_engine(): that reports "gpu" for any
  // ENABLE_OPENCL build whenever NNTR_ENGINE is unset, which says nothing about
  // how the graph's layers were stamped. Skipping the whole loop on that basis
  // left a CPU/KAI run on an OpenCL-enabled build with no packed weights at
  // all, and it then died in QS4CX_Tensor::getPackedData() with "pack before
  // run model" at the first token.
  struct RepackTally {
    unsigned int layers = 0;
    unsigned int weights = 0;
  } tally;

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &context,
            void *user_data) {
      // Two independent narrowings, and the pack needs BOTH.
      //
      // (1) Layer type. The QS4CX pack is only ever read back through
      //     Tensor::getPackedData() from the fully-connected GEMM, so no other
      //     layer type has a consumer for it. Packing them is wasted
      //     single-threaded CPU work over the whole weight set.
      // (2) Run engine. Even among FC layers, only the ones the graph resolved
      //     onto the host CPU reach that GEMM; the "gpu" (OpenCL) and "cuda"
      //     engines consume the plain QS4CX blob directly.
      //
      // Neither filter subsumes the other: (1) alone still packs every GPU FC
      // (on a large model that is enough single-core work to walk into a
      // thermal shutdown), and (2) alone still packs the non-FC QS4CX weights
      // of a CPU graph, which nothing will ever read.
      if (l.getType() != "fully_connected" &&
          l.getType() != "shared_fully_connected")
        return;
      if (context.getRunComputeEngine() != ml::train::LayerComputeEngine::CPU)
        return;
      auto *tallied = static_cast<RepackTally *>(user_data);
      ++tallied->layers;
      auto weights = context.getWeights();
      for (auto &w : weights) {
        if (w->getVariableRef().getDataType() ==
            ml::train::TensorDim::DataType::QS4CX) {
          // KAI rhs-pack is the ARM (i8mm) CPU GEMM's in-memory derivation;
          // on x86 it is NYI and the x86 CPU GEMM + the GPU v8c path consume
          // the plain on-disk QS4CX (nibbles + fp32 scales) directly.
          // Skip-on-NYI so a single QS4CX weight set loads on every backend
          // (ARM packs, x86/GPU use the plain blob).
          try {
            w->getVariableRef().pack();
            ++tallied->weights;
          } catch (const std::exception &e) {
            ml_logd("QS4CX pack skipped (engine consumes plain blob): %s",
                    e.what());
          }
        }
      }
    };
  try {
    model->forEachLayer(fn, &tally);
    // Log both counts and the process-wide engine default: a run that packs 0
    // weights while the default says "gpu" is exactly the case that used to be
    // silent until getPackedData() threw.
    ml_logd("QS4CX repack: packed %u weight(s) across %u host-engine layer(s) "
            "(process engine default: %s)",
            tally.weights, tally.layers, causallm_engine().c_str());
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
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
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
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("engine", causallm_engine())}));
  return decoder_output({residual, ffn_out});
}

unsigned int Transformer::getLayerSlidingWindow(int layer_id) const {
  // Mirrors the `sliding_window` property createAttention() sets below: every
  // layer whose 1-based index is NOT a multiple of the pattern is a sliding
  // layer. Models that read an explicit per-layer `layer_types` array from the
  // config override this.
  // Expression copied verbatim from createAttention() (unsigned modulo, as
  // before) so this refactor is semantics-preserving.
  return (layer_id + 1) % SLIDING_WINDOW_PATTERN ? SLIDING_WINDOW : UINT_MAX;
}

std::vector<unsigned int>
Transformer::computeKVRingCaps(unsigned int max_seq) const {
  const unsigned int n_layers = static_cast<unsigned int>(NUM_LAYERS);
  std::vector<unsigned int> caps(n_layers, 0u);

  unsigned int n_sliding = 0, n_ringed = 0;
  unsigned int a_window = 0, a_cap = 0; // a representative W / Wcap for the log
  for (unsigned int i = 0; i < n_layers; ++i) {
    const unsigned int w = getLayerSlidingWindow(static_cast<int>(i));
    const bool sliding = (w != 0 && w < max_seq);
    if (sliding) {
      ++n_sliding;
      a_window = w;
    }
    caps[i] = kvRingCap(w, max_seq);
    if (caps[i] != 0) {
      ++n_ringed;
      a_cap = caps[i];
    }
  }

  // Observability line (one per KV-cache allocation, i.e. once per model load).
  // A prior 200+ run matrix was measured believing
  // the ring was on while kvRingCap() had silently collapsed to 0 (derived cap
  // >= the package's max_seq_len), so print the resolved inputs AND the verdict
  // unconditionally at model load -- never make a measurement assume.
  {
    const char *ring_env = std::getenv("NNTR_KV_WINDOW_RING");
    const bool ring_off = (ring_env && ring_env[0] == '0');
    const unsigned int C = effectivePrefillChunk();
    char verdict[256];
    if (ring_off)
      std::snprintf(verdict, sizeof(verdict),
                    "NOT ENGAGED -- NNTR_KV_WINDOW_RING=0 (explicit opt-out)");
    else if (n_sliding == 0)
      std::snprintf(verdict, sizeof(verdict),
                    "NOT ENGAGED -- no sliding-window layer (full attention "
                    "keeps the linear max_seq cache)");
    else if (C == 0)
      std::snprintf(verdict, sizeof(verdict),
                    "NOT ENGAGED -- prefill chunk is 0, the ring needs chunked "
                    "prefill to bound the live key span (set "
                    "NNTR_PREFILL_CHUNK)");
    else if (n_ringed == 0)
      std::snprintf(verdict, sizeof(verdict),
                    "NOT ENGAGED -- derived cap %u >= max_seq %u, so the ring "
                    "would not shrink anything (LONG CONTEXT ONLY: raise "
                    "max_seq_len or lower NNTR_PREFILL_CHUNK)",
                    (a_window / C + 2u) * C, max_seq);
    else
      std::snprintf(verdict, sizeof(verdict),
                    "ENGAGED cap=%u rows on %u/%u layers (saves %u rows/layer)",
                    a_cap, n_ringed, n_layers, max_seq - a_cap);
    ml_logi("[kv-ring] chunk=%u max_seq=%u layers=%u (%u sliding W=%u / "
            "%u full): %s",
            C, max_seq, n_layers, n_sliding, a_window, n_layers - n_sliding,
            verdict);
  }

  return caps;
}

/**
 * @brief Create external KV-cache placeholder tensors for one layer.
 */
std::pair<Tensor, Tensor>
Transformer::createKVCachePlaceholders(const int layer_id, int n_heads) {
  const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);
  const unsigned int kv_width =
    static_cast<unsigned int>(HEAD_DIM * n_heads / GQA_SIZE);
  // A sliding-window layer only needs a Wcap-row ring, not the
  // full max_seq. The window comes from getLayerSlidingWindow(), the same hook
  // that fills this layer's `sliding_window` property, so the placeholder, the
  // KVCacheManager allocation and mha_core's modulo index cannot disagree.
  // kvRingCap() returns 0 (=> keep max_seq) when the ring is off or would not
  // shrink anything, so the default path is unchanged.
  const unsigned int ring_cap =
    kvRingCap(getLayerSlidingWindow(layer_id), max_timestep);
  const unsigned int cache_rows = ring_cap ? ring_cap : max_timestep;
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(cache_rows) + ":" +
                                  std::to_string(kv_width);

  // KV caches MUST be created as "input" layers (not plain Tensors). Plain
  // Tensors shrink the graph's input-layer set, which changes the tensor-pool
  // in-place/flatten behavior so that the first transformer layer's input is no
  // longer a synced dependent of the model input placeholder. On ARM that broke
  // USE_EMBEDDING prefill: the embedding reached input0's output but never
  // layer0_conv_norm (all-zero activations → <pad>). The x86 (#else) path
  // always used input layers and worked; this keeps both paths symmetric,
  // differing only in the external dtype (FP16 on ARM, UINT16 elsewhere).
#ifdef ENABLE_FP16
  const char *cache_dtype = "FP16";
#else
  const char *cache_dtype = "UINT16";
#endif

  // No engine= on the cache placeholders: "input" has no gpu-context
  // registration, and their storage is host-owned by KVCacheManager and bound
  // via setExternalTensors, so the graph's allocator never backs them.
  LayerHandle cache_k_input(createLayer(
    "input", {withKey("name", "cache_k_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cache_dtype)}));
  LayerHandle cache_v_input(createLayer(
    "input", {withKey("name", "cache_v_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cache_dtype)}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
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
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
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

  // Attention core layer. No engine= here: "mha_core" is registered on the
  // cpu context only, and mha_core dispatches its own GPU work internally
  // (its kernels are selected per-path, not by the node's engine).
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", getLayerSlidingWindow(layer_id)),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("use_gemm_attention", USE_FLASH_ATTENTION ? "true" : "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
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
  // -- coherent-looking but wrong (the global gate/up swap; Gemma2/3/4 avoided
  // it by overriding createMlp gate-first).
  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor gate = ffn_gate(input);

  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
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
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  return ffn_down(act);
}

/**
 * @brief Register custom CausalLM layers in the nntrainer app context.
 */
void Transformer::registerCustomLayers() {
  static std::once_flag registered;
  std::call_once(registered, []() {
    const auto &ct_engine = nntrainer::Engine::Global();
    const auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));

    // swiglu / tie_word_embedding are core layers now (nntrainer/layers/llm),
    // registered by AppContext itself.
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingLayer>);
  });

  // GPU variants: same type strings as the CPU classes but registered on the
  // gpu context so engine=gpu createLayer routes there. The GPU classes use raw
  // getData() pointers + GPU dispatches; they avoid any CPU-only Tensor ops
  // (Tensor::multiply / add_i / dot) that crash on gpu-context tensors. Inert
  // when there is no "gpu" context (CPU-only / NNTR_ENGINE=cpu builds).
  const auto &ct_engine = nntrainer::Engine::Global();
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::RMSNormLayerGPU>);
    // Gemma4 GPU-resident per_layer_slice: same type string as the CPU class,
    // registered here so engine=gpu routes to the GPU kernel (no host
    // round-trip that would break residency).
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::PerLayerSliceLayerGPU>);
    // Per-head q/k/v norms (Gemma4, Qwen3) on the gpu context. One class for
    // both backends: its incremental_forwarding dispatches the rmsnorm kernels
    // when the operands are SVM-resident and keeps the host pass otherwise, so
    // registering it here is what lets a reshaped_rms_norm node carry engine=
    // instead of sitting on the host between two device FCs.
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register GPU-routed layer on gpu ctx: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
