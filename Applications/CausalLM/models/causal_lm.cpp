// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines CausalLM's basic actions
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Llama
 */

#include <algorithm>
#include <app_context.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <compute_ops.h>
#include <neuralnet.h>

#if defined(ENABLE_OPENCL)
#include <cl_context.h> // OpenCL-only; registration uses the Engine facade. [T12]
#endif
#include <common.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context.h>
#endif
#include <layer_context.h>
#include <residency_policy.h>
#include <mha_core.h>
#include <reshaped_rms_norm.h>
#include <rms_reverse_norm.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tensor.h>

#include <causal_lm.h>
#include <llm_util.hpp>
#if defined(ENABLE_OPENCL)
#include <recq_overrides.h> // R3/R4 record/replay override registry + CL types
                            // (pulls opencl_command_queue_manager.h) -> OpenCL-only [T12]
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_attention.h>
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_fc_qint4.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

// On-GPU greedy argmax sampler (defined in libnntrainer blas_kernels.cpp): when
// requested (pure greedy), the lm_head GEMV reduces logits on the GPU and skips
// the full-vocab D->H readback; generate() consumes the 4-byte token id.
namespace nntrainer {
void request_gpu_argmax(bool on);
bool consume_gpu_argmax(unsigned int *tok);
// recq (R4): read the 4-byte on-GPU argmax token on the host-I/O queue after a
// recorded-chain replay (waiting on the replay event). cl_event -> OpenCL-only [T12]
#if defined(ENABLE_OPENCL)
bool recq_read_argmax_io(unsigned int *tok, cl_event wait_evt);
#endif
} // namespace nntrainer

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
namespace {
// NNTR_CUDA_ARGMAX on-GPU greedy argmax (opt-in). incrementalInference() stashes
// the DEVICE-resident lm_head logits pointer + dtype here (the tensor data,
// before the host memcpy), so generate() can reduce it to the 4-byte token id on
// the GPU instead of running host std::max_element over the full-vocab D->H copy.
// One batch row (BATCH_SIZE==1 only, like the CL argmax gating). Reset every
// call; valid only when the FP32/FP16 output was confirmed device-accessible.
const void *g_cuda_logits_dev = nullptr;
bool g_cuda_logits_fp16 = false;
bool cuda_argmax_enabled() {
  static const bool on = std::getenv("NNTR_CUDA_ARGMAX") != nullptr;
  return on;
}
} // namespace
#endif

#include <utf8_stream_util.h>

#include "api/streamer.h"

namespace causallm {

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  // [Mem M4] Declare CausalLM's static-residency boundaries so the core planner
  // (tensor_pool / manager) needs no app-specific tensor/layer names. These are
  // byte-identical to the former hardcoded core defaults: the embedding output
  // is host-produced but uploaded to cl_mem (RAISE), the final norm output is
  // GPU-produced but read back once by the host lm_head (LOWER), and mha_core is
  // a CPU-registered layer that binds/consumes Q/K/V on the GPU plane (engine-
  // neutral). NNTR_CLMEM_RAISE/LOWER still override the raise/lower patterns.
  auto &rp = nntrainer::ResidencyPolicy::global();
  if (rp.raise_patterns.empty())
    rp.raise_patterns = "embedding0:out0";
  if (rp.lower_patterns.empty())
    rp.lower_patterns = "output_norm:out0";
  if (rp.engine_neutral_types.empty())
    rp.engine_neutral_types = {"mha_core"};
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void CausalLM::prepareForRun() {
  stop_requested_.store(false, std::memory_order_release);
  stop_prepared_for_run_.store(true, std::memory_order_release);
}

void CausalLM::prepareStopRequestForRun() {
  if (!stop_prepared_for_run_.exchange(false, std::memory_order_acq_rel)) {
    stop_requested_.store(false, std::memory_order_release);
  }
}

void CausalLM::setLogitsProcessor(LogitsProcessor *processor) {
  logits_processor = processor;
}

void CausalLM::resetLogitsProcessor() {
  if (logits_processor != nullptr)
    logits_processor->reset();
}

void CausalLM::setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) {
  // Initialize output list
  for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");

  // allocate memory for the internal buffer
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();

  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];
  // [Phase C Path B] a legacy int4 lm_head loads as the canonical QS4CX class
  // (on-disk stays legacy QINT4, transcoded at read time). See transformer.cpp.
  if (LMHEAD_DTYPE == "QINT4")
    LMHEAD_DTYPE = "QS4CX";

  // LMHEAD_UNTIE is parsed in Transformer::setupParameters (member moved
  // there so <model>Transformer::constructModel can gate embedding0's type).

  SKIP_PREFILL = nntr_cfg.contains("skip_prefill")
                   ? nntr_cfg["skip_prefill"].get<bool>()
                   : false;

  USE_KVCACHE = false;
  PRE_COMPUTED_CACHE_PATH = "";
  SYS_PROMP_LEN = 0;

  if (nntr_cfg.contains("system_prompt") &&
      nntr_cfg["system_prompt"].contains("kvcache")) {
    USE_KVCACHE = true;
    PRE_COMPUTED_CACHE_PATH =
      nntr_cfg["system_prompt"]["kvcache"]["pre_computed_cache_path"];
    if (nntr_cfg["system_prompt"]["kvcache"].contains("sys_prompt_token_size"))
      SYS_PROMP_LEN =
        nntr_cfg["system_prompt"]["kvcache"]["sys_prompt_token_size"]
          .get<unsigned int>();
  }

  if (generation_cfg["eos_token_id"].is_array()) {
    EOS_TOKEN_ID =
      generation_cfg["eos_token_id"].empty()
        ? cfg["eos_token_id"].get<std::vector<unsigned int>>()
        : generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  } else {
    EOS_TOKEN_ID.clear();
    EOS_TOKEN_ID.push_back(generation_cfg["eos_token_id"].get<unsigned int>());
  }
  BOS_TOKEN_ID = generation_cfg["bos_token_id"].empty()
                   ? cfg["bos_token_id"].get<unsigned int>()
                   : generation_cfg["bos_token_id"].get<unsigned int>();
  TOP_K = generation_cfg.contains("top_k")
            ? generation_cfg["top_k"].get<unsigned int>()
            : 20;
  TOP_P = generation_cfg.contains("top_p")
            ? generation_cfg["top_p"].get<float>()
            : 0.95;
  TEMPERATURE = generation_cfg.contains("temperature")
                  ? generation_cfg["temperature"].get<float>()
                  : 0.7;
  global_token_len = 0;
}

void CausalLM::allocateAndBindKVCache() {
  // KV int8 path: mha_core allocates its own UINT8 cache + FP16 scale
  // tensors. There are no external cache_k_l*/cache_v_l* placeholders to
  // bind, so this helper is a no-op (paper section 3.7 internal cache
  // mode under kv_int8). The mha layer was switched to 3-input form in
  // qwen3_causallm.cpp when the env var is set.
  if (std::getenv("NNTR_KV_INT8") != nullptr) {
    kv_cache_bound = true;
    return;
  }
  if (!kv_cache.isAllocated()) {
    // dtype matches mha_core's cache placeholders so external cache storage
    // is interpreted consistently across platforms.
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif

    const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);

    kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                      max_timestep,
                      static_cast<unsigned int>(NUM_KEY_VALUE_HEADS),
                      static_cast<unsigned int>(HEAD_DIM), cache_dtype);
    kv_cache_bound = false;
  }

  if (kv_cache_bound)
    return;

  // Bind each (layer, K|V) buffer into the corresponding input layer
  // declared by Transformer::createKVCachePlaceholders(). The names here
  // must match what createKVCachePlaceholders() registers with the model.
  // We look up each placeholder by name and point it at our cache slab;
  // this is the same wiring Model::setExternalTensors used to do, just
  // without going through that API.
  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);

    auto find_cache_placeholder = [this](const std::string &base_name) {
      for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
        auto *tensor = model->getTensor(base_name + suffix);
        if (tensor != nullptr)
          return tensor;
      }
      return static_cast<nntrainer::Tensor *>(nullptr);
    };

    auto *kp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input3");
    auto *vp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input4");
    if (kp == nullptr)
      kp = find_cache_placeholder("cache_k_l" + std::to_string(i));
    if (vp == nullptr)
      vp = find_cache_placeholder("cache_v_l" + std::to_string(i));
    NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
      << "allocateAndBindKVCache: cache_k_l" << i << " / cache_v_l" << i
      << " input placeholder not found in compiled graph";
    NNTR_THROW_IF(kp->getDataType() != kc.getDataType() ||
                    vp->getDataType() != vc.getDataType(),
                  std::runtime_error)
      << "allocateAndBindKVCache: cache placeholder dtype mismatch for layer "
      << i;

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

std::vector<float *>
CausalLM::incrementalInference(unsigned int batch_size,
                               const std::vector<float *> &input,
                               unsigned int init_seq_len, unsigned int from,
                               unsigned int to) {
  // Same contract as NeuralNetwork::incremental_inference(float* ...), except
  // inputs whose raw pointer belongs to a KVCacheManager cache tensor are fed
  // as the REAL tensor (sharing its MemoryData, so isSVM() set by the SVM
  // MemoryPool survives the per-call fillPlaceholder/syncDependents). With
  // in-place input layers the mha_core cache views are dependents of the
  // input placeholder; a fresh Tensor::Map MemoryData of the same pointer
  // would clobber the SVM flag and drop attention to the host path.
  auto *nn = static_cast<nntrainer::NeuralNetwork *>(model.get());

  std::unordered_map<const void *, nntrainer::Tensor *> cache_by_ptr;
  if (kv_cache.isAllocated()) {
    for (unsigned int i = 0; i < kv_cache.getNumLayers(); ++i) {
      auto &kc = kv_cache.getKeyCache(i);
      auto &vc = kv_cache.getValueCache(i);
      cache_by_ptr.emplace(reinterpret_cast<const void *>(kc.getData()), &kc);
      cache_by_ptr.emplace(reinterpret_cast<const void *>(vc.getData()), &vc);
    }
  }

  auto in_dim = nn->getInputDimension();
  NNTR_THROW_IF(input.size() < in_dim.size(), std::invalid_argument)
    << "incrementalInference: model expects " << in_dim.size()
    << " inputs, got " << input.size();

  nntrainer::sharedConstTensors input_tensors;
  input_tensors.reserve(in_dim.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    auto it = cache_by_ptr.find(reinterpret_cast<const void *>(input[idx]));
    if (it != cache_by_ptr.end()) {
      // shallow copy: shares the cache's MemoryData (isSVM intact)
      input_tensors.emplace_back(MAKE_SHARED_TENSOR(*it->second));
    } else {
      in_dim[idx].batch(batch_size);
      input_tensors.emplace_back(MAKE_SHARED_TENSOR(
        nntrainer::Tensor::Map(input[idx],
                               in_dim[idx].getDataLen() * sizeof(float),
                               in_dim[idx], 0)));
    }
  }

  nntrainer::sharedConstTensors output_tensors =
    nn->incremental_inference(input_tensors, init_seq_len, from, to);

  // Output conversion identical to the float* overload in neuralnet.cpp.
  std::vector<float *> output;
  output.reserve(output_tensors.size());
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // NNTR_CUDA_ARGMAX: invalidate any stale device-logits stash; re-armed below
  // only when this call's first output is device-accessible (UVM / managed /
  // device) so generate() can run the on-GPU argmax instead of host max_element.
  g_cuda_logits_dev = nullptr;
#endif
  bool first_output = true;
  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    const size_t buf_size =
      (size_t)batch_size * out_t.getDim().getFeatureLen();
    float *last_out_buf_data = new float[buf_size];

    if (out->getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      const _FP16 *out_src = out_t.getData<_FP16>();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // NNTR_CUDA_ARGMAX: stash the device logits pointer (before the D2H copy)
      // when device-accessible, for generate()'s on-GPU argmax. batch_size==1
      // only (the argmax reduces a single [vocab] row).
      if (cuda_argmax_enabled() && first_output && batch_size == 1) {
        cudaPointerAttributes pa0{};
        if (cudaPointerGetAttributes(&pa0, out_src) == cudaSuccess &&
            (pa0.type == cudaMemoryTypeDevice ||
             pa0.type == cudaMemoryTypeManaged)) {
          g_cuda_logits_dev = out_src;
          g_cuda_logits_fp16 = true;
        }
        cudaGetLastError();
      }
      // Device-only activation pool (NNTR_CUDA_DEV_ACT): the model output is
      // real device memory, not host-addressable. Drain the backend stream and
      // copy it D2H into a host buffer before the host fp16->fp32 convert (=the
      // one sync-per-token boundary). For UVM the pointer is host-coherent so
      // this is skipped.
      std::vector<_FP16> out_host;
      {
        cudaPointerAttributes pa{};
        if (cudaPointerGetAttributes(&pa, out_src) == cudaSuccess &&
            pa.type == cudaMemoryTypeDevice) {
          nntrainer::cuda::StreamManager::Global().finish();
          out_host.resize(buf_size);
          cudaMemcpy(out_host.data(), out_src, buf_size * sizeof(_FP16),
                     cudaMemcpyDeviceToHost);
          out_src = out_host.data();
        }
        cudaGetLastError();
      }
#endif
      nntrainer::getComputeOps()->scopy_fp16_to_fp32(buf_size, out_src, 1,
                                                     last_out_buf_data, 1);
#else
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    } else if (out->getDataType() == ml::train::TensorDim::DataType::FP32) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // NNTR_CUDA_ARGMAX: stash the device logits pointer (the tensor data,
      // before the host memcpy below) when device-accessible. UVM/managed
      // pointers are host-coherent, so this same pointer feeds both the on-GPU
      // argmax kernel and -- as the fallback -- the host memcpy.
      if (cuda_argmax_enabled() && first_output && batch_size == 1) {
        const float *out_src = out_t.getData();
        cudaPointerAttributes pa0{};
        if (cudaPointerGetAttributes(&pa0, out_src) == cudaSuccess &&
            (pa0.type == cudaMemoryTypeDevice ||
             pa0.type == cudaMemoryTypeManaged)) {
          g_cuda_logits_dev = out_src;
          g_cuda_logits_fp16 = false;
        }
        cudaGetLastError();
      }
      // Host read of the GPU-produced logits: sync first so the read is coherent
      // under NNTR_CUDA_ASYNC (no-op in sync mode).
      nntrainer::cuda::StreamManager::Global().finishIfAsync();
#endif
      std::memcpy(last_out_buf_data, out_t.getData(),
                  sizeof(float) * buf_size);
    }

    output.push_back(last_out_buf_data);
    first_output = false;
  }

  return output;
}

void CausalLM::setKVCachePosition(unsigned int pos) {
  kv_cache.setPosition(pos);
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [pos](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::MHACoreLayer::type)
        l.setProperty({"cache_index=" + std::to_string(pos)});
    };
  model->forEachLayer(fn, nullptr);
}

void CausalLM::advanceKVCachePosition(unsigned int step_size) {
  // mha_core advances its own cache_index inside forwarding(), so the host
  // only has to keep KVCacheManager's tracked position in sync.
  kv_cache.advance(step_size);
}

std::pair<Tensor, Tensor> CausalLM::constructModel() {

  // base transformer (input, output_norm)
  auto [x, h] = Transformer::constructModel();

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
    withKey("engine", causallm_engine()),
  };

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  // [T11] ModelFeatures x DeviceCaps matcher SHADOW: the model declares its
  // features, the resolver combines them with the chosen backend's device caps
  // into an ExecPlan. Log-only (no decision site reads the plan yet) — the model
  // graph is unchanged, so this is byte-identical. docs §10 T11.
  if (const auto *ct =
        nntrainer::Engine::Global().getRegisteredContext(causallm_engine())) {
    const auto feats = getModelFeatures();
    const auto plan = nntrainer::resolveExecPlan(ct->caps(), feats);
    ml_logi("[CausalLM] %s | %s (shadow)", feats.toString().c_str(),
            plan.toString().c_str());
  }

  return {x, y};
}

void CausalLM::registerOutputs(
  std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  std::vector<unsigned int> ids, unsigned int pos,
  const std::vector<bool> &eos_list, bool log_output) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};
  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(static_cast<int>(ids[b]));
      ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::string decoded_str = tokenizer->Decode(pending_ids_);

      if (decoded_str.empty()) {
        continue;
      }

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else if (utf8stream::shouldHold(decoded_str, pending_ids_.size())) {
      } else {
        if (log_output && streamer_ == nullptr) {
          std::cout << decoded_str;
          std::cout.flush();
        }
        output_list[b].append(decoded_str);
        if (streamer_ != nullptr &&
            streamer_put(streamer_, decoded_str.c_str()) != 0) {
          requestStop();
        }
        pending_ids_.clear();
      }
    }
  }
}

void CausalLM::save_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    throw std::runtime_error(
      "save_kvcache called before allocateAndBindKVCache()");
  }
  kv_cache.save(path, static_cast<unsigned int>(to_));
}

void CausalLM::load_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    allocateAndBindKVCache();
  }
  kv_cache.load(path, static_cast<unsigned int>(to_));
  // mha_core layers each track their own cache_index; sync them all to the
  // newly-loaded position so the next forwarding() writes at the right slot.
  setKVCachePosition(static_cast<unsigned int>(to_));
}

std::vector<unsigned int> CausalLM::generate(float *logits, bool do_sample,
                                             float repetition_penalty,
                                             unsigned int *input_ids,
                                             unsigned int NUM_INPUT_IDS) {

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Terminal drain for the selective-sync (NNTR_CUDA_ASYNC) decode path: ensure
  // the GPU has finished producing the logits before the host reads them here.
  // No-op in default mode (every GPU op already drained per-op). cuda engine
  // ONLY: in a dual-enabled (CUDA+OpenCL) binary this ran on OpenCL runs too,
  // and StreamManager::Global() lazily CREATES the CUDA context -- the first
  // stray CUDA touch on a non-cuda run (NVIDIA VRAM burned on an Intel run).
  if (causallm_engine() == "cuda")
    nntrainer::cuda::StreamManager::Global().finish();
#endif

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {

#if defined(ENABLE_OPENCL)
    // On-GPU greedy argmax already produced the token (lm_head logits never left
    // the GPU; only the 4-byte id came back). Only set under pure-greedy gating.
    // OpenCL-only -> on the no-OpenCL build the host sampler below runs. [T12]
    {
      unsigned int gpu_tok = 0;
      if (nntrainer::consume_gpu_argmax(&gpu_tok)) {
        outputs.push_back(gpu_tok);
        logits = logits + NUM_VOCAB;
        input_ids = input_ids + MAX_SEQ_LEN;
        continue;
      }
    }
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // CUDA on-GPU greedy argmax (NNTR_CUDA_ARGMAX): reduce the device-resident
    // lm_head logits to the token id on the GPU and read back only 4 bytes,
    // skipping the host std::max_element over the full-vocab buffer. Gated to
    // pure greedy (no sampling, no repetition penalty, no bad words -- those
    // mutate logits on host) and only when incrementalInference stashed a
    // device-accessible logits pointer for this (single, BATCH_SIZE==1) row.
    if (cuda_argmax_enabled() && g_cuda_logits_dev != nullptr &&
        do_sample == false &&
        (repetition_penalty == 1 || input_ids == nullptr || NUM_INPUT_IDS == 0) &&
        (BAD_WORD_IDS.size() == 0 || NUM_BADWORDS == 0)) {
      unsigned int tok = 0;
      bool ok = g_cuda_logits_fp16
                  ? nntrainer::cuda::cuda_argmax_fp16(
                      reinterpret_cast<const unsigned short *>(g_cuda_logits_dev),
                      NUM_VOCAB, &tok)
                  : nntrainer::cuda::cuda_argmax_fp32(
                      reinterpret_cast<const float *>(g_cuda_logits_dev),
                      NUM_VOCAB, &tok);
      // Consume the stash regardless (it belongs to this call's logits row).
      g_cuda_logits_dev = nullptr;
      if (ok) {
        outputs.push_back(tok);
        logits = logits + NUM_VOCAB;
        input_ids = input_ids + MAX_SEQ_LEN;
        continue;
      }
      // else: fall through to the host path below (host buffer still valid).
    }
#endif

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0) {
      applyBadWordsPenalty(logits, BAD_WORD_IDS.data(), NUM_BADWORDS);
    }

    if (logits_processor != nullptr)
      logits_processor->process(logits, NUM_VOCAB, iteration);

    unsigned int output_id;

    // return argmax if do_sample is false
    if (do_sample == false) {
      output_id =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
    } else {
      // apply temperature & top-k & top-p and sample with original logits
      // unchanged
      output_id = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P, rng);
    }

    outputs.push_back(output_id);

    if (logits_processor != nullptr)
      logits_processor->acceptToken(output_id, iteration);

    // set batch offset
    logits = logits + NUM_VOCAB;
    if (input_ids != nullptr)
      input_ids = input_ids + MAX_SEQ_LEN;
  }

  return outputs;
};

void CausalLM::registerCustomLayers() {
  Transformer::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  // lm_head promoted to core app_context.cpp (cpu) [T12 Wave B].

  // Register ReshapedRMSNormLayer on the GPU (cl) context once, centrally, so
  // ANY model can build its per-head q/k/v norms with engine=GPU and keep them
  // GPU_CLMEM-resident (Gemma4 S1.1 / Qwen3 q/k norm) instead of draining to
  // the host every layer. Future q/k-norm models get GPU residency for free:
  // they only need their existing cpu-context registration plus
  // engine=causallm_engine() on the reshaped_rms_norm layer. Inert (skipped)
  // when there is no GPU context (CPU-only / NNTR_ENGINE=cpu builds). The
  // per-model registerCustomLayers still registers it on the cpu context.
  // Goes through Engine's registration facade — no static_cast to ClContext.
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "gpu" context (CPU-only build) or already registered — both benign.
  }

  // gauss4 PLE post_norm (RMSReverseNormLayer) on the GPU context: its
  // incremental_forwarding runs the reverse-norm as a GPU op (no host op inside
  // the async GPU graph). Same central-registration pattern as the reshaped norm
  // above; inert on CPU-only builds.
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::RMSReverseNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "gpu" context or already registered — both benign.
  }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Centralized cuda-context registration of ReshapedRMSNormLayer (mirror of
  // the cl block above). engine=cuda tensors are UVM (host-coherent) and not
  // isSVM()-flagged, so its forwarding takes the correct host path.
  try {
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "cuda" context or already registered — both benign.
  }
  // gauss4 PLE post_norm (RMSReverseNormLayer) on the cuda context: UVM
  // tensors are host-coherent and not isSVM()-flagged, so its forwarding
  // takes the correct host FP32-temp path (mirror of the "gpu" block above).
  try {
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<causallm::RMSReverseNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "cuda" context or already registered — both benign.
  }
#endif
}

void CausalLM::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                   const WSTR tail_prompt, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();
  // On-GPU greedy argmax sampler (NNTR_GPU_ARGMAX): keep lm_head logits on the
  // GPU, reduce to the token id on-device, read back only 4 bytes (vs the full
  // V-vocab D->H pass + host max_element). Only safe for pure greedy: no
  // sampling, no bad-words penalty (those mutate logits on host), batch == 1.
#if defined(ENABLE_OPENCL)
  nntrainer::request_gpu_argmax(std::getenv("NNTR_GPU_ARGMAX") != nullptr &&
                                !do_sample && BAD_WORD_IDS.empty() &&
                                BATCH_SIZE == 1);
#endif
  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before run().");
  }

  struct StreamerEndGuard {
    BaseStreamer *streamer;
    ~StreamerEndGuard() { streamer_end(streamer); }
  } streamer_end_guard{streamer_};

  // Allocate the host-owned KV cache and bind it to mha_core's external cache
  // input slots. Idempotent: only the first call does work; subsequent runs
  // reuse the same buffers and continue from the computed absolute token
  // position below.
  allocateAndBindKVCache();

  has_run_ = false;
  prepareStopRequestForRun();

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  if (MAX_SEQ_LEN < INIT_SEQ_LEN) {
    throw std::invalid_argument(
      "MAX_SEQ_LEN must be greater than or equal to INIT_SEQ_LEN");
  }

  /**
   * Variables for Log
   */
  unsigned int generation_cnt = 0;
  int64_t total_generation_duration = 0;

  /**
   * INPUT PREPARATION
   */
  std::vector<float *> input;
  std::vector<float *> label;

  /**
   * SAVE_KVCACHE ?
   *  if USE_KVCACHE && system_prompt is given && but the
   * PRE_COMPUTED_CACHE_PATH does not exist
   */
  SAVE_KVCACHE = (USE_KVCACHE && system_prompt != "" &&
                  !std::filesystem::exists(PRE_COMPUTED_CACHE_PATH));

  // print input text
  if (log_output)
    std::cout << system_prompt << prompt << tail_prompt << std::endl;

  // actual prompt to be used in computation
  std::string prompt_;

  if (USE_KVCACHE) {
    prompt_ = SAVE_KVCACHE ? system_prompt : (prompt + tail_prompt);
  } else {
    prompt_ = system_prompt + prompt + tail_prompt;
  }

  if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
    SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();

  ///@note add_special_tokens=true lets each model's OWN tokenizer decide whether
  /// to prepend a BOS, rather than hard-coding it. The 1-arg Encode skips special
  /// tokens, so the leading BOS that Gemma2 (TemplateProcessing, add_bos_token=
  /// true) needs was dropped -> short prompts degenerated into pure repetition
  /// ("The capital of France is" -> "is is is..."); long prompts masked it.
  /// Verified to match HF add_special_tokens=True per model: Gemma2 gains its
  /// BOS(2); models whose tokenizer adds no BOS (e.g. Qwen3 — ByteLevel post-
  /// processor, add_bos_token=false) are byte-identical to the old behavior, so
  /// they are unaffected. (sentence_transformer.cpp already encodes this way.)
  auto _input = tokenizer->Encode(prompt_, /*add_special_tokens=*/true);

  // | <------------------- MAX_SEQ_LEN -------------------> |
  //                       ||             ||
  // |<-- System prompt -->||<-- input -->||<-- generate -->|

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  // The prefill activation buffers are built at INIT_SEQ_LEN (transformer.cpp
  // constructModel, {1,1,1,INIT_SEQ_LEN}); resetInputDimension is disabled, so
  // a prompt longer than INIT_SEQ_LEN overflows the shared activation tensor
  // (getSharedDataTensor bounds-check throw). Cap the prompt to the smaller of
  // the prefill buffer (INIT_SEQ_LEN) and the KV budget (MAX_SEQ_LEN - gen).
  unsigned int num_allow_str =
    std::min<unsigned int>(INIT_SEQ_LEN, MAX_SEQ_LEN - NUM_TO_GENERATE);
  unsigned int text_len = _len;

  if (_len > num_allow_str)
    text_len = num_allow_str;

  // feed only available length
  // if _input is allowed, it feeds all of the _input
  // otherwise, feeds only a part of _input
  for (unsigned int i = 0; i < text_len; ++i)
    init_input.push_back(_input[i]);

  ///@todo currently, the whole sequence may not be fed into the model
  /// This should be handled later.
  _input.clear();

  unsigned int init_len = init_input.size();
  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
  std::vector<bool> eos_list(BATCH_SIZE, false);

  unsigned int input_len = init_len;

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(init_input[i]);
      ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = init_input[i];
    }
  }

  /**
   * PREFILL
   */
  std::vector<int64_t> token_ids;
  input.push_back(input_sample);
  // KV int8 path: mha layers were switched to 3-input mode in
  // qwen3_causallm.cpp, so the external cache_k_l*/cache_v_l* inputs
  // do not exist in the compiled graph. The inference call only needs
  // the prompt sample buffer; mha_core's internal int8 cache + scale
  // tensors are managed by the framework's TensorPool.
  static const bool _kv_int8_runtime = std::getenv("NNTR_KV_INT8") != nullptr;
  auto build_inference_inputs = [&]() {
    if (_kv_int8_runtime) {
      return std::vector<float *>{input_sample};
    }
    std::vector<std::pair<std::string, float *>> cache_inputs;
    cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
    for (int i = 0; i < NUM_LAYERS; ++i) {
      cache_inputs.emplace_back(
        "cache_k_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
      cache_inputs.emplace_back(
        "cache_v_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
    }

    std::sort(
      cache_inputs.begin(), cache_inputs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<float *> inference_inputs;
    inference_inputs.reserve(1 + cache_inputs.size());
    inference_inputs.push_back(input_sample);
    for (const auto &cache_input : cache_inputs)
      inference_inputs.push_back(cache_input.second);
    return inference_inputs;
  };
  input = build_inference_inputs();

  ///@note contains possible bug
  // std::vector<ml::train::TensorDim> input_dims;
  // ml::train::TensorDim input_dim(1, 1, input_len, DIM);
  // input_dims.push_back(input_dim);
  // model->resetInputDimension(input_dims);

  auto start_prefill = std::chrono::high_resolution_clock::now();

  std::vector<float *> output;

  if (SAVE_KVCACHE) {
    //@note This is for the save the kv cache. precomputed kv cache should be
    // always located at the begining of the prompt.
    // Therefore, it start from 0. and system prompt should be saved in the
    // init_input, so that we can compute system prompt size properly
    //
    // The structure of this precomputed K,V Cache is :
    //
    //  //<-- System Prompt -->/<-- Input Tokens -->/<-- Tail prompt --> //
    //  //< Precomputed cache >/<--given as input-->/<--- from json ---->//
    //

    if (log_output)
      std::cout << "\n==============[KV CACHE SAVE MODE]================\n";
    allocateAndBindKVCache();
    setKVCachePosition(0);
    output = incrementalInference(BATCH_SIZE, input, input_len, 0, input_len);

    SYS_PROMP_LEN = input_len;
    save_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

    if (log_output) {

      std::cout << "kv caches are saved in " << PRE_COMPUTED_CACHE_PATH
                << std::endl
                << "and the size of prompt is " << SYS_PROMP_LEN << ".\n"
                << "You may need this prompt length to set the "
                   "\"sys_prompt_token_size\""
                << "\n==================================================\n"
                << std::endl;
    }
    return;
  }

  if (USE_KVCACHE) {
    load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
  } else {
    SYS_PROMP_LEN = 0;
  }
  allocateAndBindKVCache();
  const unsigned int prefill_from = SYS_PROMP_LEN + global_token_len;
  std::vector<unsigned int> id_list;

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Prewarm the QS4CX dp4a weight caches on the CPU (ThreadManager-parallel) at
  // load: the one-time plain -> packed int4 repack is ~38% of the cold-run
  // GPU time (nsys). Doing it here -- once, before start_prefill is reset --
  // keeps it off the first prefill and off the GPU. Idempotent (per-weight
  // cache check), gated by NNTR_CUDA_PREWARM (default on).
  {
    static bool cuda_prewarmed = false;
    static const char *_pw = std::getenv("NNTR_CUDA_PREWARM");
    static const bool cuda_prewarm_on = !(_pw && _pw[0] == '0');
    // cuda engine ONLY: without this gate a dual-enabled (CUDA+OpenCL) binary
    // ran the whole prewarm on OpenCL runs too, allocating every FC's derived
    // cache on the NVIDIA device (measured: 4.4GB VRAM on an Intel xmx run of
    // gauss4 -- the ~6GB "idle" NVIDIA usage seen on Windows).
    if (!cuda_prewarmed && cuda_prewarm_on && causallm_engine() == "cuda") {
      cuda_prewarmed = true;
      std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                         void *)>
        fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &ctx, void *) {
          if (l.getType() != "fully_connected")
            return;
          for (unsigned int w = 0; w < ctx.getNumWeights(); ++w) {
            nntrainer::Tensor &wt = ctx.getWeight(w);
            // QINT4 never appears here: layer_context coerces it to QS4CX.
            if (wt.getDataType() != ml::train::TensorDim::DataType::QS4CX)
              continue;
            // [weight 한벌] The dp4a/cuBLAS device caches are built straight
            // from the plain QS4CX payload (keyed by its pointer) -- no UVM
            // Section-A copy. Building the fp16-scale UVM side buffer here
            // too makes the first forward (and any CUDA-graph capture) a pure
            // cache hit.
            const unsigned short *uS = nullptr;
            nntrainer::cuda::cuda_fc_qs4cx_scales_to_uvm_fp16(
              wt.getScale<float>(), wt.width(), &uS);
            nntrainer::cuda::cuda_fc_qs4cx_prewarm(wt.getData<uint8_t>(),
                                                   wt.width(), wt.height());
          }
        };
      model->forEachLayer(fn, nullptr);
      // Pre-grow the split-KV decode scratch so the M=1 flash-decode path never
      // cudaMallocs inside a CUDA-graph capture (NNTR_CUDA_GRAPH). 2*HEAD_DIM
      // covers gemma4's global-attention head_dim (512 vs base 256); over-
      // allocation is a few hundred KB. ensure_sk's isCapturing() guard is the
      // safety net if a model exceeds these bounds.
      nntrainer::cuda::cuda_attention_splitkv_prewarm(
        static_cast<int>(MAX_SEQ_LEN), NUM_HEADS, 2 * HEAD_DIM);
      // Pre-grow the dp4a decode FC scratch so the M=1 decode path never
      // cudaMallocs inside a CUDA-graph capture (NNTR_CUDA_GRAPH). Decode is
      // M=1; K (the FC input/contraction dim) is bounded by max(hidden DIM, FFN
      // intermediate): the down-projection FC reads the FFN intermediate
      // activation (K = INTERMEDIATE_SIZE > DIM), so DIM alone under-sizes the
      // activation-quant staging and that FC would bail under capture -> i8mm
      // host path (SIGILL on Orin) / stale UVM read. N is bounded by the largest
      // decode FC output = max(vocab, FFN intermediate). The in-path
      // isCapturing() guard is the safety net if these bounds are exceeded.
      nntrainer::cuda::cuda_fc_qint4_dp4a_prewarm(
        1u,
        std::max(static_cast<unsigned int>(DIM),
                 static_cast<unsigned int>(INTERMEDIATE_SIZE)),
        std::max(NUM_VOCAB, static_cast<unsigned int>(INTERMEDIATE_SIZE)));

      // W2: when the PREFILL CUDA graph is active (integrated), run ONE eager
      // prefill forward at the real M here (load phase) so EVERY prefill scratch
      // buffer (g_dp4a_q8/ascale/azp, g_i8_c, g_i8_bchunk, attention scratch) is
      // grown and the cuBLASLt algos are selected. Otherwise the FIRST (and only,
      // in a single-shot run) real prefill would hit an in-capture cudaMalloc and
      // fall back to eager. This warmup itself captures->aborts(malloc)->eager,
      // which is exactly what grows the scratch; the timed prefill then captures.
      // Reset the KV position after so the real prefill is unaffected.
      {
        const char *_pgenv = std::getenv("NNTR_CUDA_PREFILL_GRAPH");
        const char *_genv = std::getenv("NNTR_CUDA_GRAPH");
        const bool prefill_graph_on =
          _pgenv ? (_pgenv[0] != '0')
                 : (_genv && _genv[0] == '1' &&
                    nntrainer::cuda::ContextManager::Global().isIntegrated());
        if (prefill_graph_on && init_len > 1) {
          const unsigned int wlen =
            (SKIP_PREFILL && init_len > 1) ? init_len - 1 : init_len;
          auto *nn = static_cast<nntrainer::NeuralNetwork *>(model.get());
          // EAGER warmup: disable prefill-graph capture so the FCs grow their
          // scratch via cudaMalloc (a malloc inside capture would make the FC
          // bail to host i8mm -> SIGILL on Orin). Re-enable after.
          nn->setPrefillCaptureDisabled(true);
          setKVCachePosition(prefill_from);
          auto wout = incrementalInference(BATCH_SIZE, input, wlen, prefill_from,
                                           prefill_from + wlen);
          for (auto &o : wout)
            delete[] o;
          setKVCachePosition(prefill_from);
          nn->setPrefillCaptureDisabled(false);
        }
      }
      start_prefill = std::chrono::high_resolution_clock::now();
    }
  }
#endif

  if (SKIP_PREFILL && init_len > 1) {
    // Prefill only N-1 tokens; the last input token will be used as the first
    // token in the generation phase (assigned directly, not sampled).
    unsigned int skipped_token =
      static_cast<unsigned int>(init_input[init_len - 1]);

    const unsigned int prefill_to = prefill_from + input_len - 1;
    setKVCachePosition(prefill_from);
    output = incrementalInference(BATCH_SIZE, input, init_len - 1, prefill_from,
                                  prefill_to);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      id_list.push_back(skipped_token);

    // Adjust lengths so the generation loop processes the skipped token
    // at the correct KV cache position.
    input_len -= 1;
    init_len -= 1;
  } else {
    const unsigned int prefill_to = prefill_from + input_len;
    setKVCachePosition(prefill_from);
    output = incrementalInference(BATCH_SIZE, input, init_len, prefill_from,
                                  prefill_to);

    // post process of model output
    id_list = generate(output[0], do_sample, 1, ids_history, init_len);

    if (init_len < INIT_SEQ_LEN)
      registerOutputs(tokenizer, id_list, init_len, eos_list, log_output);
  }
  // output should be deallocated after use
  for (auto &out : output) {
    delete[] out;
  }

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  /**
   * TOKEN GENERATION
   */

  input_len += SYS_PROMP_LEN;

  // Update generated token by prefill as an input
  for (unsigned int b = 0; b < BATCH_SIZE; ++b)
    input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
      static_cast<float>(id_list[b]);

  auto start_generation = std::chrono::high_resolution_clock::now();

  // recq (R4): record/replay single-submission decode. Currently a MECHANISM
  // self-test -- the FIRST decode token is produced by recording one forward
  // (captured, not executed) then replaying it with this token's scalar/gws
  // overrides; the on-GPU argmax is read back on the host-I/O queue. If the
  // replayed first token matches the non-recq baseline, the record+replay+
  // override+argmax path is proven (the full per-token replay loop additionally
  // needs the per-token input-placeholder feed). Gated NNTR_RECQ_REPLAY +
  // NNTR_SVM_RESIDENT (host-op-zero forward). Default off => unchanged.
#if defined(ENABLE_OPENCL) // [T12] recq (OpenCL recordable-queue) decode state
  static const bool _recq_replay = std::getenv("NNTR_RECQ_REPLAY") != nullptr;
  auto &_cqm = nntrainer::opencl::CommandQueueManager::Global();
  bool _recq_recorded = false;
#endif

  for (unsigned int token_generation_idx = input_len + 1;
       token_generation_idx < input_len + 1 + NUM_TO_GENERATE &&
       !stop_requested_.load(std::memory_order_acquire);
       ++token_generation_idx) {

    allocateAndBindKVCache();
    const unsigned int _recq_ci = token_generation_idx - 1 + global_token_len;
    std::vector<float *> output_interval;
    bool _did_replay = false;
#if defined(ENABLE_OPENCL) // [T12] recq record/replay path (OpenCL-only)
    if (_recq_replay && _cqm.getRecordableQueue() != nullptr) {
      if (!_recq_recorded) {
        // First decode token: RECORD the full forward (capture-only). The host
        // embedding ran here, so embedding0:out0 holds this token's embedding for
        // the replay below.
        causallm::recq_reset_overrides();
        if (_cqm.beginRecording()) {
          output_interval = incrementalInference(BATCH_SIZE, input, input_len,
                                                 _recq_ci, _recq_ci + 1);
          _cqm.endRecording();
          _recq_recorded = true;
          std::fprintf(stderr,
                       "[RECQ] recorded decode forward: %u dispatches, %zu "
                       "overrides\n",
                       (unsigned int)_cqm.getRecordDispatchIndex(),
                       causallm::recq_override_count());
        }
      } else {
        // Subsequent tokens: FEED pass -- run incrementalInference with ALL GPU
        // dispatches skipped, so only the HOST embedding refreshes embedding0:out0
        // with this token's embedding; the GPU forward comes from the replay.
        _cqm.setSkipAllDispatches(true);
        output_interval = incrementalInference(BATCH_SIZE, input, input_len,
                                               _recq_ci, _recq_ci + 1);
        _cqm.setSkipAllDispatches(false);
      }
      if (_recq_recorded) {
        std::vector<nntrainer::opencl::cl_array_arg_qcom> _rargs;
        std::vector<nntrainer::opencl::cl_workgroup_qcom> _rgws;
        std::vector<int> _rints;
        std::vector<std::array<std::size_t, 3>> _rgwss;
        causallm::recq_build_token_overrides(_recq_ci, _rargs, _rgws, _rints,
                                             _rgwss);
        cl_event _revt = nullptr;
        if (_cqm.replayRecording(_rargs.data(), _rargs.size(), _rgws.data(),
                                 _rgws.size(), &_revt)) {
          unsigned int _rtok = 0;
          if (nntrainer::recq_read_argmax_io(&_rtok, _revt))
            _did_replay = true;
        }
        if (!_did_replay)
          std::fprintf(stderr,
                       "[RECQ] replay/readback FAILED @ci=%u; fallback to normal\n",
                       _recq_ci);
      }
    }
#endif // ENABLE_OPENCL (recq record/replay) [T12]
    if (!_did_replay)
      output_interval = incrementalInference(BATCH_SIZE, input, input_len,
                                             _recq_ci, _recq_ci + 1);
    std::vector<unsigned int> ids_list(generate(output_interval[0], do_sample));

    // Feed the newly generated token back as the next input token.
    // token_generation_idx always starts at input_len + 1, so we are
    // always in the auto-regressive generation phase here.
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
        static_cast<float>(ids_list[b]);
    }
    registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list,
                    log_output);
    ++generation_cnt;

    // output should be deallocated after use
    for (auto out : output_interval) {
      delete[] out;
    }

    // check FINISH
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j] && (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                                     ids_list[j]) != EOS_TOKEN_ID.end())) {
        eos_list[j] = true;
      }
    }

    bool is_finish = true;
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j]) {
        is_finish = false;
        break;
      }
    }

    if (is_finish) {
      break;
    }

    if (stop_requested_.load(std::memory_order_acquire)) {
      break;
    }
  }

  // Always release the input buffer after the generation loop, whether
  // the loop exited early (EOS found) or ran to the maximum token limit.
  free(input_sample);

  global_token_len += (generation_cnt + init_len);

  auto finish_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  auto finish_total = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_total - start_total);
  size_t peak_memory = getPeakMemoryKb();

  if (log_output) {

    std::cout << "\n\n";
    std::cout << "=================[ LLM with NNTrainer ]===================\n";
    std::cout << "prefill: " << init_len << " tokens, "
              << prefill_duration.count() << " ms, "
              << ((double)init_len / prefill_duration.count() * 1000)
              << " TPS\n";
    std::cout << "generation: " << generation_cnt << " tokens, "
              << generation_duration.count() << " ms, "
              << ((double)generation_cnt / generation_duration.count() * 1000)
              << " TPS\n";
    std::cout << "total: " << total_duration.count() << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "==========================================================\n";
  }

  performance_metrics.prefill_tokens = init_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = generation_duration.count();
  performance_metrics.total_duration_ms = total_duration.count();
  performance_metrics.peak_memory_kb = peak_memory;

  has_run_ = true;
}

std::string CausalLM::getOutput(int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(output_list.size())) {
    return "";
  }
  return output_list[batch_idx];
}

} // namespace causallm
