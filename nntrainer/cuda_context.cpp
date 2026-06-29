// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   NVIDIA CUDA application context implementation (mirror of ClContext).
 */

#include <cuda_context.h>

#include <mutex>

#include <addition_layer.h>
#include <compute_ops.h>
#include <cuda_mem_allocator.h>
#include <fc_layer_cl.h>
#include <geglu_layer.h>
#include <logit_softcapping.h>
#include <scalar_multiply.h>
#include <swiglu_layer.h>
#include <tie_word_embedding.h>

// runDecode (T9): the CUDA-graph decode/prefill state machine, relocated verbatim
// from neuralnet.cpp. Needs the model walk + graph-node access + the CUDA graph
// API (cuda_context.h already pulls in StreamManager / ContextManager).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <layer_node.h>
#include <neuralnet.h>

namespace nntrainer {

std::mutex cuda_factory_mutex;

void CudaContext::initialize() noexcept {
  try {
    if (!cudaInit()) {
      ml_loge("Error: CudaContext::initialize() failed (no usable CUDA device)");
      return;
    }

    // Probe device capabilities once (log-only; the ExecPlan resolver consumes
    // this later, docs/ARCHITECTURE_REFACTOR.md §10 T1). Truth from the existing
    // cuda::ContextManager queries — adds no decision site, so byte-identical.
    caps_.backend = "cuda";
    caps_.device_name = context_inst_.GetDeviceName();
    caps_.arch = context_inst_.GetComputeArch();
    caps_.integrated = context_inst_.isIntegrated();
    caps_.unified_memory = true; // cudaMallocManaged (UVM) is the default pool
    ml_logi("[CudaContext] %s", caps_.toString().c_str());

    // ExecPlan resolver SHADOW (docs/ARCHITECTURE_REFACTOR.md §10 T4): CUDA
    // resolves to gemm_path=CUBLAS (cuBLAS int8 + dp4a) with host_coherent =
    // isIntegrated(); no env to assert against, so it is logged only.
    ml_logi("[CudaContext] %s (shadow)", resolveExecPlan(caps_).toString().c_str());

    add_default_object();

    // Unified-Memory allocator: MemoryPool buffers for engine=cuda tensors are
    // cudaMallocManaged -> host-addressable AND device-accessible (the SVM
    // analogue), so a tensor on this context is device-resident with no
    // separate copy step. Falls back to host memory if UVM is unavailable.
    setMemAllocator(std::make_shared<CudaMemAllocator>());

    // ComputeOps = the CUDA ops table. CudaComputeOps derives from CpuComputeOps
    // (engine=cuda tensors are Unified Memory = host-coherent, so every standard
    // op runs correctly on the managed buffers via the CPU implementations) and
    // overrides only the ops it accelerates — currently the whole-op geglu
    // (device-resident fp16 kernel, opt-in NNTR_CUDA_GEGLU) and the host copies.
    // A neutral Layer's in1.getOps()->geglu(...) thus lands on the GPU path. The
    // bulkier GPU-accelerated layers (CudaFcLayer's cuBLAS) still bind directly;
    // anything not overridden falls back to a correct CPU computation. [T6/T7]
    getContextData()->setComputeOps(get_cuda_ops());

  } catch (std::exception &e) {
    ml_loge("cuda_context: initialization failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cuda_context: initialization failed due to unknown reason");
  }
}

void CudaContext::add_default_object() {
  // FC: the backend-neutral FullyConnectedLayerCl dispatches its GEMM via
  // CudaComputeOps::fc (cuda_fc_qint4 / cuBLAS / host dot). Same class as the
  // gpu context. [T7]
  registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                  FullyConnectedLayerCl::type, ml::train::LayerType::LAYER_FC);
  // addition: the core CPU AdditionLayer is pure host Tensor ops -> correct on
  // the host-coherent UVM tensors (do NOT use the OpenCL AdditionLayerCL).
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);
  // geglu: the backend-neutral GeGLULayer dispatches via CudaComputeOps::geglu
  // (device-resident fp16 kernel under NNTR_CUDA_GEGLU, else host-on-UVM).
  registerFactory(nntrainer::createLayer<GeGLULayer>, GeGLULayer::type);
  // swiglu: the merged backend-neutral SwiGLULayer dispatches via
  // CudaComputeOps::swiglu, with the device-resident fp16 one-kernel fast path
  // (cuda_swiglu_fp16) for the qwen3 FFN. Replaces the app causallm::SwiGLULayer.
  registerFactory(nntrainer::createLayer<SwiGLULayer>, SwiGLULayer::type);
  // logit_softcapping (promoted to core [T12]): host op on UVM-resident logits;
  // the CUDA-only device-softcap path lives inside the layer (ENABLE_CUDA).
  registerFactory(nntrainer::createLayer<LogitSoftCappingLayer>,
                  LogitSoftCappingLayer::type);
  // scalar_multiply (promoted [T12]): host op on UVM (the CPU class on cuda, like
  // the gemma4 tryRegister it replaces; the _gpu variant is OpenCL-only).
  registerFactory(nntrainer::createLayer<ScalarMultiplyLayer>,
                  ScalarMultiplyLayer::type);
  // tie_word_embedding (promoted [T12]): host lm_head on UVM (the GPU GEMV is the
  // OpenCL path, now #if ENABLE_OPENCL-guarded inside the layer so it builds
  // without OpenCL). Register unconditionally.
  registerFactory(nntrainer::createLayer<TieWordEmbedding>,
                  TieWordEmbedding::type);
}

template <typename T>
const int CudaContext::registerFactory(const FactoryType<T> factory,
                                       const std::string &key,
                                       const int int_key) {
  static_assert(isSupported<T>::value,
                "cuda_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(cuda_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "cuda_context: cannot register factory with already taken key: "
       << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "cuda_context: cannot register factory with already taken int key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("cuda_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

const CudaContext::SharedPtrCudaKernel
CudaContext::registerCudaKernel(const std::string &kernel_source,
                                const std::string &kernel_name,
                                const std::string &compile_options) {
  // hot path: a single key + lookup, no copy of the (multi-KB) source string.
  const std::string kkey = kernel_name + compile_options;
  auto it = cuda_kernel_map.find(kkey);
  if (it != cuda_kernel_map.end())
    return it->second;

  // owning module cache: kernels sharing one (source, options) reuse the
  // compiled+loaded CUmodule (and its on-disk PTX cache, see cuda_module.cpp).
  const std::string mkey =
    std::to_string(cuda::Module::GetKernelHash(kernel_source, compile_options));
  std::shared_ptr<cuda::Module> module;
  auto mit = cuda_module_map.find(mkey);
  if (mit != cuda_module_map.end()) {
    module = mit->second;
  } else {
    module = std::make_shared<cuda::Module>();
    if (!module->CreateModuleFromSource(kernel_source, kernel_name,
                                        compile_options)) {
      ml_loge("Failed to compile CUDA module for kernel %s",
              kernel_name.c_str());
      return nullptr;
    }
    cuda_module_map.emplace(mkey, module);
  }

  SharedPtrCudaKernel kernelPtr = std::make_shared<cuda::Kernel>();
  if (!kernelPtr->CreateKernelFromModule(*module, kernel_name)) {
    ml_loge("Failed to resolve CUDA kernel %s", kernel_name.c_str());
    return nullptr;
  }
  cuda_kernel_map.emplace(kkey, kernelPtr);
  return cuda_kernel_map[kkey];
}

/**
 * @copydoc const int CudaContext::registerFactory
 */
template const int CudaContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

// Non-template seam (Context::registerLayerFactory override): forwards to the
// per-class registerFactory<Layer> here in the same TU so the explicit
// instantiation is used and no template crosses the .so boundary. [T3]
int CudaContext::registerLayerFactory(PtrFactoryType<nntrainer::Layer> factory,
                                      const std::string &key,
                                      const int int_key) {
  return registerFactory<nntrainer::Layer>(factory, key, int_key);
}

// SEAM-2 CUDA override (docs/ARCHITECTURE_REFACTOR.md §10 T9). Relocated VERBATIM
// from neuralnet.cpp's incremental_inference #if ENABLE_CUDA block — the only
// changes are the model walk (`nn.incremental_forwarding`), graph-node access
// (`nn.getLayerNode` / `nn.feedInputsLabels`), the M2-B skip flag
// (`nn.setM2BSkipAll`), and the prefill-capture flag
// (`nn.isPrefillCaptureDisabled()`). All decisions/env reads/static state are
// unchanged, so engine=cuda decode is behaviorally identical.
sharedConstTensors
CudaContext::runDecode(NeuralNetwork &nn, unsigned int from, unsigned int to,
                       const sharedConstTensors &input,
                       const sharedConstTensors &label) {
  sharedConstTensors out;

  // CUDA-graph capture of a whole DECODE forward (NNTR_CUDA_GRAPH, M1). A decode
  // step issues ~1000 tiny kernels; the CPU launch/dispatch between them is the
  // decode bottleneck (GPU ~30-47% utilized). Capturing the per-token forward
  // into one graph and replaying it collapses that launch overhead. M1
  // re-instantiates every step (still pays cudaGraphInstantiate) purely to prove
  // capture+replay COHERENCE; M2 will cache the graphExec and patch params.
  static const char *_cgraph_env = std::getenv("NNTR_CUDA_GRAPH");
  static const bool cuda_graph_decode =
    _cgraph_env != nullptr && _cgraph_env[0] == '1';
  // PREFILL graph (W3): capture the M>1 prefill forward like decode. Default ON
  // for INTEGRATED GPUs (Orin) when the graph path is enabled; discrete GPUs
  // (RTX) keep eager-async prefill. Override: NNTR_CUDA_PREFILL_GRAPH=1/0.
  static const bool cuda_graph_prefill = []() {
    const char *e = std::getenv("NNTR_CUDA_PREFILL_GRAPH");
    if (e != nullptr)
      return e[0] != '0';
    const char *g = std::getenv("NNTR_CUDA_GRAPH");
    return g != nullptr && g[0] == '1' &&
           nntrainer::cuda::ContextManager::Global().isIntegrated();
  }();
  static const bool cuda_graph_dbg =
    std::getenv("NNTR_CUDA_GRAPH_DBG") != nullptr;
  // Diagnostic: cache the exec from the first captured token and RE-LAUNCH it for
  // subsequent tokens (incoherent; measures the pure cudaGraphLaunch+sync ceiling).
  static const bool cuda_graph_replay =
    std::getenv("NNTR_CUDA_GRAPH_REPLAY") != nullptr;
  static cudaGraphExec_t _cg_cached_exec = nullptr;
  static sharedConstTensors _cg_cached_out;
  static unsigned long _cg_ok = 0, _cg_fallback = 0;
  bool cuda_graph_captured = false;

  // M2-B: single-capture COHERENT decode. Capture the full forward ONCE (first
  // decode token); for every later token, refresh ONLY the embeddings on the host
  // (g_m2b_skip_all feed pass), update the device position (cuda_set_pos), and
  // REPLAY the cached graph -- skipping the ~350-op C++ dispatch.
  static const bool cuda_m2b = std::getenv("NNTR_CUDA_M2B") != nullptr;
  if (cuda_m2b && from == 0 && _cg_cached_exec != nullptr) {
    // new sequence (prefill boundary): drop the previous sequence's cached graph.
    cudaGraphExecDestroy(_cg_cached_exec);
    _cg_cached_exec = nullptr;
    _cg_cached_out = {};
  }
  if (cuda_m2b && from != 0 && (to - from) == 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    if (_cg_cached_exec != nullptr) {
      // subsequent token: embed-only feed (refresh emb_stage) -> set pos -> replay
      static const bool m2b_light =
        std::getenv("NNTR_CUDA_M2B_LIGHT") != nullptr;
      if (m2b_light) {
        // lighter feed: set the new token input + run ONLY the two embedding
        // nodes directly, bypassing the full ~350-node graph iteration.
        nn.feedInputsLabels(input, label);
        auto emb0 = nn.getLayerNode("embedding0");
        auto ple = nn.getLayerNode("per_layer_input_embedding");
        if (emb0)
          emb0->incremental_forwarding(from, to, false);
        if (ple)
          ple->incremental_forwarding(from, to, false);
      } else {
        nn.setM2BSkipAll(true);
        out = nn.incremental_forwarding(from, to, input, label, false);
        nn.setM2BSkipAll(false);
      }
      nntrainer::cuda::cuda_set_pos((int)from, (int)from + 1);
      cudaGraphLaunch(_cg_cached_exec, sm.GetStream());
      cudaStreamSynchronize(sm.GetStream());
      out = _cg_cached_out;
      cuda_graph_captured = true;
    } else if (sm.beginCapture()) {
      // first decode token: set pos, capture the full forward, cache the exec
      nntrainer::cuda::cuda_set_pos((int)from, (int)from + 1);
      out = nn.incremental_forwarding(from, to, input, label, false);
      cudaGraph_t graph = nullptr;
      if (sm.endCapture(&graph) && graph != nullptr) {
        if (cudaGraphInstantiate(&_cg_cached_exec, graph, 0) == cudaSuccess) {
          cudaGraphLaunch(_cg_cached_exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          _cg_cached_out = out;
          cuda_graph_captured = true;
        }
        cudaGraphDestroy(graph);
      } else {
        cudaGetLastError();
      }
    }
    if (cuda_graph_dbg) {
      static unsigned long _m2b_tok = 0;
      if (++_m2b_tok <= 16)
        std::fprintf(stderr, "[M2B] tok#%lu %s (exec=%p)\n", _m2b_tok,
                     cuda_graph_captured ? "ok" : "FALLBACK",
                     (void *)_cg_cached_exec);
    }
  }

  if (!cuda_graph_captured && cuda_graph_decode && from != 0 &&
      (to - from) == 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    const char *stage = "beginCapture";
    cudaError_t cerr = cudaSuccess;
    using _clk = std::chrono::high_resolution_clock;
    auto _us = [](_clk::time_point a, _clk::time_point b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    long t_rec = 0, t_inst = 0, t_rep = 0;
    if (cuda_graph_replay && _cg_cached_exec != nullptr) {
      // replay-only: relaunch the cached exec (timing ceiling, incoherent)
      auto p2 = _clk::now();
      cudaGraphLaunch(_cg_cached_exec, sm.GetStream());
      cudaStreamSynchronize(sm.GetStream());
      t_rep = _us(p2, _clk::now());
      out = _cg_cached_out; // persistent output tensors, refilled by the replay
      cuda_graph_captured = true;
    } else if (sm.beginCapture()) {
      auto p0 = _clk::now();
      out = nn.incremental_forwarding(from, to, input, label, false);
      cudaGraph_t graph = nullptr;
      bool ended = sm.endCapture(&graph);
      auto p1 = _clk::now();
      t_rec = _us(p0, p1);
      if (ended && graph != nullptr) {
        cudaGraphExec_t exec = nullptr;
        cerr = cudaGraphInstantiate(&exec, graph, 0);
        auto p2 = _clk::now();
        t_inst = _us(p1, p2);
        if (cerr == cudaSuccess) {
          cudaGraphLaunch(exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          t_rep = _us(p2, _clk::now());
          if (cuda_graph_replay) {
            _cg_cached_exec = exec; // keep for replay-only relaunch
            _cg_cached_out = out;
          } else {
            cudaGraphExecDestroy(exec);
          }
          cuda_graph_captured = true;
        } else {
          stage = "cudaGraphInstantiate";
        }
        cudaGraphDestroy(graph);
      } else {
        // capture invalidated (e.g. a mid-capture cudaMalloc): record the error
        // and clear the sticky flag so the eager fallback is not falsely flagged.
        stage = "endCapture";
        cerr = cudaGetLastError();
      }
    }
    if (cuda_graph_captured)
      ++_cg_ok;
    else
      ++_cg_fallback;
    if (cuda_graph_dbg && (_cg_ok + _cg_fallback) <= 12) {
      if (cuda_graph_captured)
        std::fprintf(stderr,
                     "[CUDA_GRAPH] tok#%lu %s  record=%ldus instantiate=%ldus "
                     "replay+sync=%ldus\n",
                     _cg_ok, t_rec ? "CAPTURED+REPLAYED" : "REPLAY-ONLY(cached)",
                     t_rec, t_inst, t_rep);
      else
        std::fprintf(stderr,
                     "[CUDA_GRAPH] fell back (captured=%lu fallback=%lu) stage=%s err=%d\n",
                     _cg_ok, _cg_fallback, stage, (int)cerr);
    }
  }
  // PREFILL graph capture (W3): same machinery as the decode M1 branch above,
  // for the M>1 prefill (from==0). One beginCapture -> forward -> endCapture ->
  // instantiate -> launch -> single sync, replacing the ~190 per-op drains.
  if (!cuda_graph_captured && cuda_graph_prefill &&
      !nn.isPrefillCaptureDisabled() && from == 0 && (to - from) > 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    using _clk = std::chrono::high_resolution_clock;
    auto _us = [](_clk::time_point a, _clk::time_point b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    long t_rec = 0, t_inst = 0, t_rep = 0;
    const char *stage = "beginCapture";
    cudaError_t cerr = cudaSuccess;
    if (sm.beginCapture()) {
      auto p0 = _clk::now();
      out = nn.incremental_forwarding(from, to, input, label, false);
      cudaGraph_t graph = nullptr;
      bool ended = sm.endCapture(&graph);
      auto p1 = _clk::now();
      t_rec = _us(p0, p1);
      if (ended && graph != nullptr) {
        cudaGraphExec_t exec = nullptr;
        cerr = cudaGraphInstantiate(&exec, graph, 0);
        auto p2 = _clk::now();
        t_inst = _us(p1, p2);
        if (cerr == cudaSuccess) {
          cudaGraphLaunch(exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          t_rep = _us(p2, _clk::now());
          cudaGraphExecDestroy(exec);
          cuda_graph_captured = true;
        } else {
          stage = "cudaGraphInstantiate";
        }
        cudaGraphDestroy(graph);
      } else {
        stage = "endCapture";
        cerr = cudaGetLastError();
      }
    }
    if (cuda_graph_dbg) {
      static unsigned long _pf = 0;
      std::fprintf(stderr,
                   "[PREFILL_GRAPH] #%lu M=%u %s record=%ldus instantiate=%ldus "
                   "replay+sync=%ldus stage=%s err=%d\n",
                   ++_pf, (unsigned)(to - from),
                   cuda_graph_captured ? "CAPTURED" : "FALLBACK", t_rec, t_inst,
                   t_rep, stage, (int)cerr);
    }
  }
  if (!cuda_graph_captured)
    out = nn.incremental_forwarding(from, to, input, label, false);

  return out;
}

} // namespace nntrainer
