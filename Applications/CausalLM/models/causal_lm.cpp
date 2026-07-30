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

#include <common.h>
#include <compute_ops.h>
#include <layer_context.h>
#include <lm_head.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <residency_policy.h>
#include <rms_reverse_norm.h>
#include <tensor.h>

#include <causal_lm.h>
#include <llm_util.hpp>
#include <rms_reverse_norm.h>
#include <utf8_stream_util.h>

#include "api/streamer.h"

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_attention.h>
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_fc_qint4.h>
#include <cuda_pack_cache.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

namespace {

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
// NNTR_CUDA_ARGMAX on-GPU greedy argmax (opt-in). incrementalInference()
// stashes the DEVICE-resident lm_head logits pointer + dtype here (the tensor
// data, before the host copy), so generate() can reduce it to the 4-byte token
// id on the GPU instead of running host std::max_element over the full-vocab
// D->H copy. One batch row (BATCH_SIZE==1 only, like the CL argmax gating).
// Reset every call; valid only when the FP32/FP16 output was confirmed
// device-accessible.
const void *g_cuda_logits_dev = nullptr;
bool g_cuda_logits_fp16 = false;
bool cuda_argmax_enabled() {
  static const bool on = std::getenv("NNTR_CUDA_ARGMAX") != nullptr;
  return on;
}
#endif

/**
 * @brief Wrap an external host buffer as a Tensor of @p dim.
 *
 * Byte-for-byte the same dtype dispatch as neuralnet.cpp's file-local
 * mapExternalTensor() (which is in an anonymous namespace and therefore not
 * reachable from here). Kept in sync deliberately: incrementalInference()
 * below must behave identically to the base float* overload for every input
 * that is NOT a KV-cache buffer.
 */
nntrainer::Tensor mapExternalInput(float *buf,
                                   const nntrainer::TensorDim &dim) {
  const unsigned int bytes = static_cast<unsigned int>(
    static_cast<size_t>(dim.getDataLen()) * dim.getDataTypeSize());

  switch (dim.getDataType()) {
  case nntrainer::TensorDim::DataType::FP16:
  case nntrainer::TensorDim::DataType::UINT16:
  case nntrainer::TensorDim::DataType::QINT16:
    return nntrainer::Tensor::Map<uint16_t>(reinterpret_cast<uint16_t *>(buf),
                                            bytes, dim, 0);
  case nntrainer::TensorDim::DataType::UINT8:
  case nntrainer::TensorDim::DataType::UINT4:
  case nntrainer::TensorDim::DataType::QINT8:
  case nntrainer::TensorDim::DataType::QINT4:
  case nntrainer::TensorDim::DataType::Q4_K:
  case nntrainer::TensorDim::DataType::Q6_K:
  case nntrainer::TensorDim::DataType::Q4_0:
    return nntrainer::Tensor::Map<uint8_t>(reinterpret_cast<uint8_t *>(buf),
                                           bytes, dim, 0);
  case nntrainer::TensorDim::DataType::UINT32:
  case nntrainer::TensorDim::DataType::BCQ:
    return nntrainer::Tensor::Map<uint32_t>(reinterpret_cast<uint32_t *>(buf),
                                            bytes, dim, 0);
  case nntrainer::TensorDim::DataType::FP32:
  case nntrainer::TensorDim::DataType::NONE:
  default:
    return nntrainer::Tensor::Map<float>(buf, bytes, dim, 0);
  }
}

} // namespace

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  // Declare CausalLM's static-residency boundaries. Core ships the MECHANISM
  // (ResidencyPolicy::global(), read by manager.cpp's engine_neutral test and
  // tensor_pool.cpp's planner build) but deliberately carries no app-specific
  // layer names; the POLICY is the application's to declare. Nothing populated
  // it here, so `isEngineNeutral()` answered false for every type and the
  // mechanism was dead code.
  //
  // `mha_core` is CPU-registered but binds and consumes Q/K/V on the GPU plane
  // (it takes the cl_mem handles directly and bridges its host stages through
  // clmem_lower_cl / clmem_raise_cl). Undeclared, it counts as a CPU consumer,
  // so `all_consumers_gpu` is false for every wq/wk/wv output and the planner
  // downgrades the whole attention neighbourhood GPU_CLMEM -> SVM. Observable
  // proof the declaration is what arms the path: without it NNTR_CLMEM_MHA_OFF
  // (which nulls exactly those handles) cannot change the output at all,
  // because they are already null.
  //
  // NOT declared here, deliberately: the input-boundary RAISE
  // ("embedding0:out0") and output-boundary LOWER ("output_norm:out0") that the
  // reference tree also sets. Both are a REGRESSION on this tree -- measured
  // 2026-07-28, gemma4 goes from the golden "**Seoul**" to <pad> spam -- so the
  // raise/lower implementations they feed are not fully on the ladder yet. They
  // are still reachable for A/B via NNTR_CLMEM_RAISE / NNTR_CLMEM_LOWER.
  {
    auto &rp = nntrainer::ResidencyPolicy::global();
    if (rp.engine_neutral_types.empty())
      rp.engine_neutral_types = {"mha_core"};
  }
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
  // Row stride is MAX_SEQ_LEN, so every ids_history[b * MAX_SEQ_LEN + pos]
  // write needs pos < MAX_SEQ_LEN. Transformer::setupParameters keeps
  // NUM_TO_GENERATE inside [0, MAX_SEQ_LEN) for decoders, and run() caps the
  // generation loop at the window on top of that.
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();

  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];

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
  if (!kv_cache.isAllocated()) {
    // dtype matches mha_core's cache placeholders so external cache storage
    // is interpreted consistently across platforms.
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif

    const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);

    // Per-layer ring capacity: sliding-window layers store a
    // Wcap-row ring, full-attention layers keep max_seq. Derived from the same
    // getLayerSlidingWindow() hook that shapes the KV placeholders in
    // createKVCachePlaceholders(), so the bind below cannot mismatch.
    kv_ring_caps_ = computeKVRingCaps(max_timestep);
    kv_cache.setLayerCaps(kv_ring_caps_);

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
    if (kp == nullptr && vp == nullptr) {
      /// This layer has no attention sub-graph (e.g., a conv-only block in a
      /// hybrid architecture like LFM2). Skip KV-cache binding for it.
      continue;
    }
    NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
      << "allocateAndBindKVCache: cache_k_l" << i << " / cache_v_l" << i
      << " partially found in compiled graph (one placeholder exists but "
         "the other does not)";
    NNTR_THROW_IF(kp->getDataType() != kc.getDataType() ||
                    vp->getDataType() != vc.getDataType(),
                  std::runtime_error)
      << "allocateAndBindKVCache: cache placeholder dtype mismatch for layer "
      << i;
    // For a ringed layer the graph placeholder MUST have been
    // built at the same Wcap height as the allocation (both come from
    // computeKVRingCaps / kvRingCap). A mismatch means the layer would write
    // absolute rows into a short buffer -- check it explicitly. Only enforced
    // for ringed layers so non-ring models keep today's (dtype-only) contract.
    NNTR_THROW_IF(
      static_cast<size_t>(i) < kv_ring_caps_.size() && kv_ring_caps_[i] != 0 &&
        (kp->getDim() != kc.getDim() || vp->getDim() != vc.getDim()),
      std::runtime_error)
      << "allocateAndBindKVCache: window-ring shape mismatch for layer " << i
      << " (placeholder " << kp->getDim().height() << " rows vs cache "
      << kc.getDim().height() << " rows)";

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

std::vector<float *> CausalLM::incrementalInference(
  unsigned int batch_size, const std::vector<float *> &input,
  unsigned int init_seq_len, unsigned int from, unsigned int to) {
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
      if (kc.empty() || vc.empty())
        continue; // mixed KV mode: internal-int8 layer, no external tensor
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
      input_tensors.emplace_back(
        MAKE_SHARED_TENSOR(mapExternalInput(input[idx], in_dim[idx])));
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
  // device) so generate() can run the on-GPU argmax instead of host
  // max_element.
  g_cuda_logits_dev = nullptr;
  // first_output gates the NNTR_CUDA_ARGMAX stash below to this call's first
  // output tensor; it has no reader outside the ENABLE_CUDA branches, so it
  // is declared (and reset) only when they are compiled in -- otherwise it
  // is a set-but-unused local on every non-CUDA build.
  bool first_output = true;
#endif
  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    const size_t buf_size =
      static_cast<size_t>(batch_size) * out_t.getDim().getFeatureLen();
    float *last_out_buf_data = new float[buf_size];

    if (out->getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      const _FP16 *out_src = out_t.getData<_FP16>();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Per-token cudart touches (pointer probes + stream drains) are cuda-run
      // only: on a non-cuda run of the unified binary the first cudart call
      // boots the statically-linked runtime inside this (timed) path.
      std::vector<_FP16> out_host;
      if (causallm_engine() == "cuda") {
        // NNTR_CUDA_ARGMAX: stash the device logits pointer (before the D2H
        // copy) when device-accessible, for generate()'s on-GPU argmax.
        // batch_size==1 only (the argmax reduces a single [vocab] row).
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
        // real device memory, not host-addressable. Drain the backend stream
        // and copy it D2H into a host buffer before the host fp16->fp32
        // convert (=the one sync-per-token boundary). For UVM the pointer is
        // host-coherent so this is skipped.
        cudaPointerAttributes pa{};
        if (cudaPointerGetAttributes(&pa, out_src) == cudaSuccess &&
            pa.type == cudaMemoryTypeDevice) {
          nntrainer::cuda::StreamManager::Global().finish();
          out_host.resize(buf_size);
          cudaMemcpy(out_host.data(), out_src, buf_size * sizeof(_FP16),
                     cudaMemcpyDeviceToHost);
          out_src = out_host.data();
        } else {
          // UVM/managed pointer: host-coherent for ADDRESSING, but under
          // NNTR_CUDA_ASYNC the producing kernel may still be in flight --
          // reading now is a torn-read (determinism audit; the fp32 branch
          // already drains). No-op in sync mode.
          nntrainer::cuda::StreamManager::Global().finishIfAsync();
        }
        cudaGetLastError();
      }
#endif
      nntrainer::getComputeOps()->scopy_fp16_to_fp32(buf_size, out_src, 1,
                                                     last_out_buf_data, 1);
#else
      delete[] last_out_buf_data;
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    } else if (out->getDataType() == ml::train::TensorDim::DataType::FP32) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Per-token cudart touches are cuda-run only (see the fp16 branch note).
      if (causallm_engine() == "cuda") {
        // NNTR_CUDA_ARGMAX: stash the device logits pointer (the tensor data,
        // before the host memcpy below) when device-accessible. UVM/managed
        // pointers are host-coherent, so this same pointer feeds both the
        // on-GPU argmax kernel and -- as the fallback -- the host memcpy.
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
        // Host read of the GPU-produced logits: sync first so the read is
        // coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
        nntrainer::cuda::StreamManager::Global().finishIfAsync();
      }
      // Device-only activation pool (NNTR_CUDA_DEV_ACT): fp32 logits are real
      // device memory the raw memcpy below cannot read -- drain and stage D2H,
      // symmetric to the fp16 branch above (without this the fp32 branch
      // would fault under DEV_ACT).
      if (out_t.getMemoryData() &&
          !out_t.getMemoryData()->isHostAddressable()) {
        nntrainer::cuda::StreamManager::Global().finish();
        if (!nntrainer::cuda::copy_any((void *)last_out_buf_data,
                                       (const void *)out_t.getData(),
                                       sizeof(float) * buf_size))
          throw std::runtime_error(
            "CausalLM: D2H staging of the fp32 logits failed");
      } else {
        std::memcpy(last_out_buf_data, out_t.getData(),
                    sizeof(float) * buf_size);
      }
#else
      std::memcpy(last_out_buf_data, out_t.getData(), sizeof(float) * buf_size);
#endif
    }
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    first_output = false;
#endif

    output.push_back(last_out_buf_data);
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

/**
 * [lmhead-untie] When nntr_config.json sets lmhead_untie, build
 * output_of_causallm as an independent fully_connected layer with its own
 * weight even for a tied-embedding model, so the lm_head can carry a
 * different dtype than the embedding (untied-serialized packages such as
 * gemma4_qs4cx_fp16 ship a separate transposed [hidden, vocab] head record
 * that a tied graph cannot load). Untie is the config flag, NOT derived from
 * LMHEAD_DTYPE: a quantizer constructs this same untied graph from the FP32
 * source and quantizes output_of_causallm via the dtype map on save.
 * skip_prefill keeps the FC lm_head decode-only, the same contract the tied
 * lm_head types implement internally. Flag off = byte-identical graph.
 */
Tensor CausalLM::buildLmHeadOutput(Tensor h, bool add_skip_prefill) {
  const bool lmhead_untied = LMHEAD_UNTIE;
  const std::string lmhead_type =
    lmhead_untied ? "fully_connected"
                  : (TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head");
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  // The head must carry the graph's engine. It is the LAST node, so it reads
  // output_norm's activation -- which, once the rest of the graph is
  // engine-stamped (38de03c46 / 2c2b0d96e), lives on the gpu context's
  // cl_mem/SVM plane. A host head reads the stale host shadow of that plane, so
  // the logits are garbage and every model degenerates to one repeated token.
  // Measured on this tree: unstamped gemma4 answered "<pad>"-class garbage at
  // 0.23 TPS decode (a 262144-row QS4CX head on the host); stamped it answers
  // "The capital of South Korea is **Seoul**." at 20.6 TPS.
  // Both reachable types have a real gpu-context factory, so neither throws
  // exception::not_supported from createLayer:
  //   fully_connected     -> FullyConnectedLayerCl (cl_context.cpp
  //                          add_default_object)
  //   tie_word_embeddings -> TieWordEmbedding      (cl_context.cpp, gated on
  //                          registerGeGLUClKernels; same class on
  //                          cpu/gpu/cuda, it selects its Q6_K/Q4_0 GPU GEMV
  //                          internally)
  // "lm_head" (untied via config.json tie_word_embeddings=false, i.e. NOT
  // LMHEAD_UNTIE) has NO gpu registration and stays unstamped -- no in-tree
  // package reaches it, and stamping it would throw.
  if (lmhead_type != "lm_head")
    lmhead_prop.emplace_back(withKey("engine", causallm_engine()));
  if (add_skip_prefill)
    lmhead_prop.emplace_back(withKey("skip_prefill", "true"));
  if (TIE_WORD_EMBEDDINGS && !lmhead_untied)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  return lmhead(h);
}

std::pair<Tensor, Tensor> CausalLM::constructModel() {

  // base transformer (input, output_norm)
  auto [x, h] = Transformer::constructModel();

  Tensor y = buildLmHeadOutput(h, LMHEAD_UNTIE && SKIP_PREFILL);

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

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // CUDA on-GPU greedy argmax (NNTR_CUDA_ARGMAX): reduce the device-resident
    // lm_head logits to the token id on the GPU and read back only 4 bytes,
    // skipping the host std::max_element over the full-vocab buffer. Gated to
    // pure greedy (no sampling, no repetition penalty, no bad words, no logits
    // processor -- those mutate or consume logits on the host) and only when
    // incrementalInference stashed a device-accessible logits pointer for this
    // (single, BATCH_SIZE==1) row.
    if (cuda_argmax_enabled() && g_cuda_logits_dev != nullptr &&
        do_sample == false && logits_processor == nullptr &&
        (repetition_penalty == 1 || input_ids == nullptr ||
         NUM_INPUT_IDS == 0) &&
        (BAD_WORD_IDS.size() == 0 || NUM_BADWORDS == 0)) {
      unsigned int tok = 0;
      bool ok =
        g_cuda_logits_fp16
          ? nntrainer::cuda::cuda_argmax_fp16(
              reinterpret_cast<const unsigned short *>(g_cuda_logits_dev),
              NUM_VOCAB, &tok)
          : nntrainer::cuda::cuda_argmax_fp32(
              reinterpret_cast<const float *>(g_cuda_logits_dev), NUM_VOCAB,
              &tok);
      // Consume the stash regardless (it belongs to this call's logits row).
      g_cuda_logits_dev = nullptr;
      if (ok) {
        outputs.push_back(tok);
        logits = logits + NUM_VOCAB;
        if (input_ids != nullptr)
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
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  // lm_head is a core layer now (nntrainer/layers/llm), registered by
  // AppContext itself.
  (void)app_context;

  // rms_reverse_norm (the PLE post_norm of the reverse-norm model family) is
  // registered on EVERY backend the Engine brought up by
  // Transformer::registerCustomLayers, called above. It used to be enumerated
  // here one backend name at a time -- a lone "cuda" registration behind an
  // #if -- which is exactly the shape that leaves every other brought-up
  // backend without a factory for a type its graphs stamp engine= on.
}

void CausalLM::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                   const WSTR tail_prompt, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();
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

  // Join the async tokenizer build before its first use. This point dominates
  // every tokenizer touch in a run -- both Encode calls below and every later
  // Decode / registerOutputs in the generation loop.
  ensureTokenizer();

  ///@note This fallback has to count the cached rows with the SAME tokenization
  /// the save pass used to produce them: below, SAVE_KVCACHE encodes
  /// prompt_ == system_prompt with add_special_tokens=true and then stores
  /// SYS_PROMP_LEN = input_len. The 1-arg Encode drops the specials, so on a
  /// BOS-prepending tokenizer this counted one row less than the cache
  /// actually holds, and every absolute KV write derived from it
  /// (prefill_from = SYS_PROMP_LEN + global_token_len) landed one slot early --
  /// clobbering the last cached row and shifting every RoPE position by one.
  ///
  /// An EMPTY system prompt must not reach it either: Encode("", true) returns
  /// the lone BOS on a BOS-prepending tokenizer (Gemma2: size 1), so deriving a
  /// length from it fabricates a one-row prefix the cache never described --
  /// load_kvcache() then restores one stale row, setKVCachePosition(1) shifts
  /// every later position by one, and the prompt that really does open the
  /// sequence loses its BOS at the same time. With no system prompt there is no
  /// cached prefix to count, so the length stays 0 and load_kvcache(path, 0)
  /// keeps failing loudly (TensorDim rejects a zero-height slice) on the
  /// configuration that cannot be resolved here: sys_prompt_token_size unset,
  /// a cache file present, and an empty system prompt.
  if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0 &&
      !system_prompt.empty())
    SYS_PROMP_LEN =
      tokenizer->Encode(system_prompt, /*add_special_tokens=*/true).size();

  ///@note Special tokens belong at sequence position 0 and nowhere else. This
  /// encode produces the first tokens of the sequence when the cache is being
  /// built from scratch (SAVE_KVCACHE) or when nothing has been written yet; it
  /// is a CONTINUATION when a precomputed cache already supplies the first
  /// SYS_PROMP_LEN rows, or when an earlier run() on this object wrote
  /// global_token_len rows. Encoding a continuation with add_special_tokens
  /// splices a mid-sequence BOS into the prompt for BOS-prepending tokenizers
  /// (Gemma2: TemplateProcessing, add_bos_token=true) -- a token sequence the
  /// model never saw in training, which also consumes one KV slot that real
  /// prompt content needed.
  const bool prompt_starts_sequence =
    SAVE_KVCACHE || (SYS_PROMP_LEN + global_token_len) == 0;

  ///@note add_special_tokens lets each model's OWN tokenizer decide
  /// whether
  /// to prepend a BOS, rather than hard-coding it. The 1-arg Encode skips
  /// special tokens, so the leading BOS that Gemma2 (TemplateProcessing,
  /// add_bos_token= true) needs was dropped -> short prompts degenerated into
  /// pure repetition
  /// ("The capital of France is" -> "is is is..."); long prompts masked it.
  /// Verified to match HF add_special_tokens=True per model: Gemma2 gains its
  /// BOS(2); models whose tokenizer adds no BOS (e.g. Qwen3 — ByteLevel post-
  /// processor, add_bos_token=false) are byte-identical to the old behavior, so
  /// they are unaffected. (sentence_transformer.cpp already encodes this way.)
  auto _input = tokenizer->Encode(prompt_, prompt_starts_sequence);

  // | <------------------- MAX_SEQ_LEN -------------------> |
  //                       ||             ||
  // |<-- System prompt -->||<-- input -->||<-- generate -->|

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  // Transformer::setupParameters keeps NUM_TO_GENERATE inside
  // [0, MAX_SEQ_LEN) for decoders on every call, so this reservation is at
  // most MAX_SEQ_LEN - 1. Spell the subtraction out in a form that cannot wrap
  // anyway: MAX_SEQ_LEN is unsigned, so a budget at or above the window would
  // otherwise turn the prompt budget into ~4e9, skip the truncation below and
  // write the prompt past the ids_history row stride.
  const unsigned int reserved_for_generation =
    NUM_TO_GENERATE > 0 ? static_cast<unsigned int>(NUM_TO_GENERATE) : 0u;
  const unsigned int kv_budget = MAX_SEQ_LEN > reserved_for_generation
                                   ? MAX_SEQ_LEN - reserved_for_generation
                                   : 0u;
  // [prefill-chunk] One forward pass cannot process more than INIT_SEQ_LEN
  // query rows without overflowing the shared activation tensor (built at
  // {1,1,1,INIT_SEQ_LEN} in constructModel; resetInputDimension is disabled).
  // Without chunking that caps the prompt at INIT_SEQ_LEN. With chunking the
  // prefill is fed FORWARD in chunks of <= INIT_SEQ_LEN rows -- each chunk fits
  // the buffer -- so the prompt is bounded by the KV budget alone, and the
  // activation plane stays INIT_SEQ_LEN-sized regardless of prompt length.
  //
  // UNION NOTE: the KV budget keeps the wrap-safe subtraction above; the
  // chunking predicate only chooses whether INIT_SEQ_LEN also caps it.
  const bool _prefill_chunking = effectivePrefillChunk() > 0;
  unsigned int num_allow_str =
    _prefill_chunking ? kv_budget
                      : std::min<unsigned int>(INIT_SEQ_LEN, kv_budget);
  unsigned int text_len = _len;

  if (_len > num_allow_str) {
    text_len = num_allow_str;
    // Truncation drops tokens from the tail of the prompt, which is where
    // instructions in "summarize this document"-style prompts live: a
    // silently truncated prompt can make the model continue the body
    // instead of following a dropped trailing instruction. Always warn
    // with the exact counts.
    std::cerr << "[CausalLM] WARNING: prompt (" << _len
              << " tokens) exceeds the max allowed prefill length ("
              << num_allow_str
              << " = max_seq_len - num_to_generate); "
                 "truncating "
              << (_len - num_allow_str) << " tail tokens." << std::endl;
  }

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
    (float *)calloc(BATCH_SIZE * MAX_SEQ_LEN, sizeof(float));
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
  auto build_inference_inputs = [&]() {
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

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Prewarm the QS4CX dp4a weight caches at load: the one-time plain ->
  // packed int4 repack is a large slice of the cold first prefill; doing it
  // here -- once, before start_prefill is taken -- keeps it off the timed
  // path. Idempotent (per-weight pointer-keyed cache; a reloaded model's new
  // weight pointers rebuild lazily even though this latch stays set),
  // value-gated by NNTR_CUDA_PREWARM (auto-defaulted "1" by the cuda context;
  // an explicit =0 disables). cuda engine ONLY: on a dual-enabled
  // (CUDA+OpenCL) binary an ungated walk would build every FC's derived cache
  // on the NVIDIA device during OpenCL runs.
  {
    static const char *_pw = std::getenv("NNTR_CUDA_PREWARM");
    static const bool cuda_prewarm_on = !(_pw && _pw[0] == '0');
    static bool s_cuda_prewarmed = false;
    if (!s_cuda_prewarmed && cuda_prewarm_on && causallm_engine() == "cuda") {
      s_cuda_prewarmed = true;
      // --- [i8 length gate] ------------------------------------------------
      // The eager cuBLAS-i8 [K,N] build is ~2/3 of this prewarm's cost (the
      // int8 buffer is 2x the int4 payload) and its ONLY consumer is the
      // dispatcher's M >= CUDA_FC_I8_PREFILL_MIN_M prefill branch. This turn's
      // prompt is ALREADY TOKENIZED at this point, so the largest M any forward
      // of this turn will see is known for free:
      //
      //   chunked prefill feeds ceil(input_len / prefill_chunk) forwards of at
      //   most prefill_chunk rows, decode runs at M=1
      //
      // so when that maximum is below the gate, no FC can reach the i8 path and
      // every byte of that cache is dead VRAM built on the user's critical
      // path. The dp4a pack, the fp16 scale buffers, the split-KV scratch and
      // the decode scratch are NOT gated -- those are what buys decode
      // throughput, so the long-generation case keeps today's behaviour
      // exactly.
      //
      // Safety: the gate only skips an EAGER build. The lazy in-path build
      // still runs on first use (a device-side repack, no host transient), so a
      // later turn with a long prompt self-heals; it costs that one prefill
      // what it used to cost at load. NNTR_CUDA_PREWARM_I8=1 forces the full
      // eager build regardless of length (pre-gate behaviour), =0 never builds
      // it eagerly; unset = length-gated.
      const unsigned int i8_chunk =
        std::min<unsigned int>(effectivePrefillChunk(), INIT_SEQ_LEN);
      const unsigned int max_prefill_m =
        (i8_chunk > 0 && input_len > i8_chunk) ? i8_chunk : input_len;
      static const char *_pwi8 = std::getenv("NNTR_CUDA_PREWARM_I8");
      const bool i8_forced = _pwi8 && _pwi8[0] == '1';
      const bool i8_disabled = _pwi8 && _pwi8[0] == '0';
      const bool i8_reachable =
        max_prefill_m >= nntrainer::cuda::CUDA_FC_I8_PREFILL_MIN_M;
      const bool i8_eager = !i8_disabled && (i8_forced || i8_reachable);
      ml_logi("[cuda prewarm] i8 gate: prefill M<=%u vs gate %u -> eager i8 "
              "%s%s",
              max_prefill_m, nntrainer::cuda::CUDA_FC_I8_PREFILL_MIN_M,
              i8_eager ? "ON" : "OFF (lazy build self-heals)",
              (i8_forced || i8_disabled) ? " (forced by NNTR_CUDA_PREWARM_I8)"
                                         : "");
      // [pack-cache] bind the derive-once pack to the weight file that produced
      // these bytes (size + mtime identity), then let each per-weight derive
      // consult/tee its record. Opt-in (NNTR_CUDA_PACK_CACHE=1); a no-op
      // otherwise, and a missing/stale/corrupt pack simply derives as before.
      nntrainer::cuda_pack::set_source(LOADED_WEIGHT_PATH.c_str());
      std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                         void *)>
        fn = [i8_eager](ml::train::Layer &l, nntrainer::RunLayerContext &ctx,
                        void *) {
          if (l.getType() != "fully_connected")
            return;
          for (unsigned int w = 0; w < ctx.getNumWeights(); ++w) {
            nntrainer::Tensor &wt = ctx.getWeight(w);
            if (wt.getDataType() != ml::train::TensorDim::DataType::QS4CX)
              continue;
            // Pack-cache record name: graph-stable (layer name + weight slot),
            // so it means the same thing on every launch. The plain pointer --
            // which keys the in-memory cache -- must never key the disk one.
            const std::string pack_name = l.getName() + "." + std::to_string(w);
            // Build the fp16-scale UVM side buffer here too so the first
            // forward (and any CUDA-graph capture) is a pure cache hit. The
            // scale conversion host-READS the fp32 tail, so it must run
            // before any weight migration a later lever might add.
            const unsigned short *uS = nullptr;
            nntrainer::cuda::cuda_fc_qs4cx_scales_to_uvm_fp16(
              wt.getScale<float>(), wt.width(), &uS);
            // skip_prefill FC towers (their prefill is an early-return) and
            // the untied lm_head (decodes at M=1) can never reach the M>=32
            // cuBLAS-i8 gate -- their [K,N] int8 cache (2x the int4 payload;
            // the untied lm_head alone is hundreds of MiB) is dead VRAM.
            // Exempt them from the EAGER build; the lazy runtime build
            // remains as the self-healing fallback.
            // [i8 length gate] when this turn's largest prefill M cannot
            // reach the cuBLAS-i8 gate, EVERY FC is in that same position, so
            // the per-layer test below is subsumed and the whole eager i8
            // build is skipped.
            bool i8_dead = !i8_eager || l.getName() == "output_of_causallm";
            if (!i8_dead) {
              try {
                i8_dead = l.getProperty("skip_prefill") == "true";
              } catch (...) {
              }
            }
            if (i8_dead)
              nntrainer::cuda::cuda_fc_qs4cx_prewarm_exempt_i8(
                wt.getData<uint8_t>());
            nntrainer::cuda::cuda_fc_qs4cx_prewarm(wt.getData<uint8_t>(),
                                                   wt.width(), wt.height(),
                                                   pack_name.c_str());
            // [pool-bypass] every derived cache for this weight now exists
            // (dp4a packed [+ cuBLAS int8] + fp16 scales) -- with the heap
            // bypass the plain pages are droppable in place, the same way the
            // v8c path drops them after its backing build. Opt-in. Runs after
            // the pack-cache tee above, so a pack rewrite still sees the plain
            // payload it derives from.
            static const bool cuda_drop = []() {
              const char *e = std::getenv("NNTR_CUDA_DROP_PLAIN");
              return e != nullptr && e[0] == '1';
            }();
            if (cuda_drop)
              nntrainer::cuda::cuda_fc_qs4cx_drop_plain_pages(
                wt.getData<uint8_t>(), wt.width(), wt.height());
          }
        };
      model->forEachLayer(fn, nullptr);
      // [pack-cache] every load-time derive is done: finalize a pending pack
      // rewrite on the background (exit-joined) finalizer.
      nntrainer::cuda_pack::load_complete();
      // The split that decides whether persisting the packs can pay at all:
      // only the derive (+ the miss-path tee) is cacheable, the H2D upload
      // happens either way. Reported when the cache is in play.
      if (nntrainer::cuda_pack::enabled()) {
        double d = 0, u = 0, t = 0, h = 0;
        size_t db = 0, hb = 0;
        nntrainer::cuda::cuda_fc_qs4cx_prewarm_stats(&d, &u, &t, &h, &db, &hb);
        ml_logi("[cuda prewarm] split: host derive %.1f ms (%zu MB) + H2D "
                "%.1f ms + pack tee %.1f ms | pack HIT %.1f ms (%zu MB)",
                d, db >> 20, u, t, h, hb >> 20);
      }
      // Pre-grow the split-KV decode scratch so the M=1 flash-decode path
      // never cudaMallocs inside a CUDA-graph capture. 2*HEAD_DIM covers a
      // model whose global-attention head_dim doubles the base; the
      // over-allocation is a few hundred KB and ensure_sk's isCapturing()
      // guard is the safety net if a model exceeds these bounds.
      nntrainer::cuda::cuda_attention_splitkv_prewarm(
        static_cast<int>(MAX_SEQ_LEN), NUM_HEADS, 2 * HEAD_DIM);
      // Pre-grow the dp4a decode FC scratch: decode is M=1; K (the FC
      // contraction dim) is bounded by max(hidden DIM, FFN intermediate) --
      // the down-projection FC reads the FFN intermediate activation, so DIM
      // alone under-sizes the activation-quant staging.
      nntrainer::cuda::cuda_fc_qint4_dp4a_prewarm(
        1u,
        std::max(static_cast<unsigned int>(DIM),
                 static_cast<unsigned int>(INTERMEDIATE_SIZE)),
        std::max(NUM_VOCAB, static_cast<unsigned int>(INTERMEDIATE_SIZE)));
    }
  }
#endif

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

  // [prefill-chunk] NNTR_PREFILL_CHUNK=C (>0) drives the prefill FORWARD in
  // C-token chunks instead of one M-token block. Each chunk feeds its tokens at
  // input row 0 and writes KV at the absolute range [from, from+clen); the next
  // chunk attends over the accumulated cache. For causal (and sliding-window)
  // attention each query sees the identical causal prefix whether computed in
  // one block or in forward chunks, so the output is bit-identical at a fixed
  // chunk size. This is the exact token-feed pattern the decode call shape
  // already uses (tokens at input row 0, absolute KV position via `from`),
  // generalized from chunk=1 to chunk=C -- not a new execution model.
  // C=0 (ring opt-out / ARM) is the single-block behaviour verbatim.
  //
  // Each chunk must fit the INIT_SEQ_LEN-height activation buffer, so clamp the
  // requested chunk to INIT_SEQ_LEN (a larger request would overflow it).
  const unsigned int prefill_chunk =
    _prefill_chunking
      ? std::min<unsigned int>(effectivePrefillChunk(), INIT_SEQ_LEN)
      : 0u;
  auto do_prefill = [&](unsigned int n_tok,
                        unsigned int from_pos) -> std::vector<float *> {
    // Single block (default) when chunking is off or the prompt fits one chunk.
    // NOTE: this must go through CausalLM::incrementalInference, not
    // NeuralNetwork::incremental_inference -- the wrapper feeds KVCacheManager
    // tensors as the REAL tensor so their isSVM() flag survives the per-call
    // fillPlaceholder/syncDependents (see its contract comment). Calling the
    // raw model method here drops attention to the host path.
    if (prefill_chunk == 0 || n_tok <= prefill_chunk) {
      return incrementalInference(BATCH_SIZE, input, n_tok, from_pos,
                                  from_pos + n_tok);
    }
    // Chunked forward prefill.
    std::vector<float *> out;
    for (unsigned int o = 0; o < n_tok; o += prefill_chunk) {
      const unsigned int clen = std::min(prefill_chunk, n_tok - o);
      for (unsigned int b = 0; b < BATCH_SIZE; ++b)
        for (unsigned int j = 0; j < clen; ++j)
          input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + j] =
            static_cast<float>(init_input[o + j]);
      const unsigned int cf = from_pos + o;
      auto so = incrementalInference(BATCH_SIZE, input, clen, cf, cf + clen);
      if (o + clen < n_tok)
        for (auto &oo : so)
          delete[] oo;
      else
        out = std::move(so);
    }
    return out;
  };

  if (SKIP_PREFILL && init_len > 1) {
    // Prefill only N-1 tokens; the last input token will be used as the first
    // token in the generation phase (assigned directly, not sampled).
    unsigned int skipped_token =
      static_cast<unsigned int>(init_input[init_len - 1]);

    setKVCachePosition(prefill_from);
    output = do_prefill(init_len - 1, prefill_from);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      id_list.push_back(skipped_token);

    // Adjust lengths so the generation loop processes the skipped token
    // at the correct KV cache position.
    input_len -= 1;
    init_len -= 1;
  } else {
    setKVCachePosition(prefill_from);
    output = do_prefill(init_len, prefill_from);

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

  // registerOutputs() writes ids_history[b * MAX_SEQ_LEN + idx] with no bounds
  // check, so the loop index has to stay inside the row stride the buffer was
  // allocated with. A budget that fits the window is not enough on its own: the
  // loop starts one past input_len, and input_len carries SYS_PROMP_LEN (added
  // just above) on top of the already-truncated prompt, so
  // input_len + NUM_TO_GENERATE can still reach MAX_SEQ_LEN. Derive the end
  // from the window too and stop at whichever comes first.
  const unsigned int generation_budget =
    NUM_TO_GENERATE > 0 ? static_cast<unsigned int>(NUM_TO_GENERATE) : 0u;
  const unsigned int generation_begin = input_len + 1;
  const unsigned int generation_end =
    generation_begin < MAX_SEQ_LEN
      ? generation_begin +
          std::min(MAX_SEQ_LEN - generation_begin, generation_budget)
      : generation_begin;

  for (unsigned int token_generation_idx = generation_begin;
       token_generation_idx < generation_end &&
       !stop_requested_.load(std::memory_order_acquire);
       ++token_generation_idx) {

    allocateAndBindKVCache();
    auto output_interval = incrementalInference(
      BATCH_SIZE, input, input_len, token_generation_idx - 1 + global_token_len,
      token_generation_idx + global_token_len);
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
  size_t peak_commit = getPeakCommitKb();

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
    std::cout << "peak memory: " << peak_memory << " KB (working set)\n";
    if (peak_commit)
      std::cout << "peak commit: " << peak_commit << " KB (private)\n";
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
