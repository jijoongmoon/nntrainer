// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   NVIDIA CUDA application context implementation (mirror of
 * ClContext).
 */

#include <cuda_context.h>
#include <env_compat.h>

#include <mutex>

#include <addition_layer.h>
#include <compute_ops.h>
#include <cuda_mem_allocator.h>
#include <fc_layer_cl.h>
#include <geglu_layer.h>
#include <logit_softcapping.h>
#include <rms_norm_layer.h>
#include <scalar_multiply.h>
#include <sigmoid_add_layer.h>
#include <sigmoid_glu_layer.h>
#include <activation_layer.h>
#include <swiglu_layer.h>
#include <tie_word_embedding.h>

// runDecode (T9): the CUDA-graph decode/prefill state machine, relocated
// verbatim from neuralnet.cpp. Needs the model walk + graph-node access + the
// CUDA graph API (cuda_context.h already pulls in StreamManager /
// ContextManager).
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
    // On a dual-backend build this runs at the FIRST
    // Engine::Global() touch of ANY run — including engine=cpu/gpu — and
    // cudaInit()'s cuInit wakes a runtime-PM-suspended dGPU over PCIe
    // (measured: the D3cold wake alone costs seconds). Defer the bring-up when
    // CUDA is not the active engine: explicit NNTR_ENGINE != cuda, or
    // NNTR_ENGINE unset on an OpenCL-enabled build (where the engine default is
    // "gpu", mirroring the app engine default). Non-cuda runs never
    // legitimately touch this context (prewarm/StreamManager gate on the engine
    // string). NNTR_CUDA_EAGER_INIT=1 restores the old eager behavior.
    {
      const char *eng = std::getenv("NNTR_ENGINE");
      const char *eager = std::getenv("NNTR_CUDA_EAGER_INIT");
      const bool eager_on = eager && eager[0] == '1';
#if defined(ENABLE_OPENCL)
      const bool cuda_active = eng && std::string(eng) == "cuda";
#else
      const bool cuda_active = !eng || std::string(eng) == "cuda";
#endif
      if (!cuda_active && !eager_on) {
        ml_logi("[CudaContext] bring-up deferred (engine=%s)",
                eng ? eng : "(unset; OpenCL default)");
        return;
      }
    }
    if (!cudaInit()) {
      ml_loge(
        "Error: CudaContext::initialize() failed (no usable CUDA device)");
      return;
    }

    // Probe device capabilities once (log-only; the ExecPlan resolver consumes
    // this later, docs/ARCHITECTURE_REFACTOR.md §10 T1). Truth from the
    // existing cuda::ContextManager queries — adds no decision site, so
    // byte-identical.
    caps_.backend = "cuda";
    caps_.device_name = context_inst_.GetDeviceName();
    caps_.arch = context_inst_.GetComputeArch();
    caps_.integrated = context_inst_.isIntegrated();
    caps_.unified_memory = true; // cudaMallocManaged (UVM) is the default pool
    ml_logi("[CudaContext] %s", caps_.toString().c_str());

    // NNTR_DETERMINISTIC keeps the CUDA per-op drains: the async-submission
    // lever (NNTR_CUDA_ASYNC) is the one host/device-overlap knob whose overlap
    // can turn a knife-edge logit into a run-to-run coin flip (measured),
    // so the determinism contract pins it OFF. overwrite=0 keeps an explicit
    // user NNTR_CUDA_ASYNC winning. The default path (env unset) is untouched —
    // NNTR_CUDA_ASYNC keeps its normal default (sync / drains on).
    // Default-on: keep the CUDA per-op drains (NNTR_CUDA_ASYNC=0) so decode is
    // reproducible; overwrite=0 lets an explicit NNTR_CUDA_ASYNC win.
    // NNTR_DETERMINISTIC=0 opts out (async submission allowed).
    //
    // UNION NOTE: this block runs BEFORE the HW-class profile below on purpose.
    // Both set NNTR_CUDA_ASYNC with overwrite=0, and the ladder's determinism
    // contract is default-ON/opt-out, so the deterministic value must be the
    // one that lands first; the profile's own ASYNC setenv is then inert unless
    // NNTR_DETERMINISTIC=0 left this one unset.
    if (const char *det = std::getenv("NNTR_DETERMINISTIC");
        !(det && det[0] == '0'))
      setenv("NNTR_CUDA_ASYNC", "0", 0);

    // HW-optimal CUDA env defaults (same rationale/semantics as ClContext):
    // NNTR_ENGINE=cuda already selected this context (so no engine guard is
    // needed here — this only runs on a CUDA run), fill the tuned GPU-op flag
    // set so a bare CUDA run gets full-GPU residency without exporting ~15
    // flags. setenv(..., 0): overwrite=0, so an explicit env ALWAYS wins (=0
    // disables). These are the COMMON flags — "all models, both HW classes" —
    // that move rope/attn/geglu/eltwise/qknorm/FC onto the GPU. Several of
    // these gate app-side (rope/attn) or deferred (M2B just landed) paths, so
    // they are inert until those consumers exist; harmless when unread.
    setenv("NNTR_CUDA_ROPE", "1", 0);
    setenv("NNTR_CUDA_ATTN", "1", 0);
    setenv("NNTR_CUDA_KV_UVM", "1", 0);
    setenv("NNTR_CUDA_GEGLU", "1", 0);
    setenv("NNTR_CUDA_ELTWISE", "1", 0);
    setenv("NNTR_CUDA_QKNORM", "1", 0);
    // 256-key decode chunks (was 64): quarters the split-KV scratch
    // round-trip (10.5 -> 2.6 MB/call at 20K keys) and the reduce's merge
    // width. attn_partial 704 -> 632 us/call + attn_reduce 52 -> 18
    // measured; still 80+ live chunks at 20K = full grid. Chunk partials
    // merge in chunk order, so the partition is numerics-visible (ulp
    // class) -- gated with the 2026-08-14 decode batch.
    setenv("NNTR_CUDA_FLASH_DECODE", "256", 0);
    setenv("NNTR_CUDA_BLOCKQ", "1", 0);
    setenv("NNTR_FC_CUDA_CUBLAS", "1", 0);
    setenv("NNTR_CUDA_PREWARM", "1", 0);
    // "=all" RAISES the CUDA RMSNorm row cap to every row; unset leaves it at
    // 32 (cuda_compute_ops.cpp gpu_max_rows), which sends EVERY prefill
    // RMSNorm to the host. It is listed in the discrete block below as part of
    // a bundle whose shared justification is "these let a HOST op touch pool
    // memory around in-flight kernels" -- but this one is the opposite: it
    // REMOVES host reads from the chain, which is exactly what an integrated
    // GPU needs (there, a host op inside a CUDA-graph capture is a
    // wrong-answer bug, not merely a slow one). Measured on Orin/gemma4-e2b,
    // 1004-token prefill, 2 runs each: 3560/3535 TPS with, 1774/1765 without
    // -- a flat 2x. Decode is unaffected (M=1 is under the cap either way).
    // Armed for integrated too; the discrete block keeps its own copy so the
    // cMA==0-on-WDDM case (integrated=false) still opts out.
    if (caps_.integrated) {
      setenv("NNTR_RMSNORM_CUDA_OFF", "all", 0);
      // Same story, different flag: without it the PREFILL V-cache copy takes
      // the host branch (mha_core.cpp -- the GPU copy needs height==1, or
      // vcopy_prefill, or a device-only KV cache, and on an integrated GPU the
      // KV cache IS host-addressable), which puts one host op per layer inside
      // the prefill capture. Its own comment says the gate exists because "a
      // host attention path could read the V cache unsynced" and is safe once
      // GPU attention + a UVM KV cache are on -- both are armed unconditionally
      // above. It is doubly safe here: the drain it protects is finishIfAsync,
      // and cuda_async_mode() hard-returns false on integrated, so that drain
      // was never doing anything on this hardware anyway.
      // Measured on Orin/gemma4-e2b QS4CX, 1004-token prefill:
      // [CAP-AUDIT] lines 32 -> 0, prefill 3498 -> 3560 TPS.
      setenv("NNTR_CUDA_VCOPY_PREFILL", "1", 0);
      // NNTR_CUDA_M2B on integrated too (2026-08-21, e4d203516's fixes):
      // the 35B corruption was capture-on-first-token + host-fallback muls,
      // both fixed; with the default warmup (NNTR_CUDA_M2B_WARM=2) the
      // graph is byte-identical to eager on the 20K gate and 64 EOS-banned
      // tokens, deterministic across runs, and the graph's short-prompt
      // ulp drift GATE-PASSED: floor 8/10 (same miss as both eager arms),
      // judge_nll 0.2356 = the best score in the project record (eager
      // anchors 0.2435/0.2527). Decode 11.67 -> 13.9, and 16.05 with the
      // wide b/a GEMV riding the graph. Value-checked: =0 opts out.
      setenv("NNTR_CUDA_M2B", "1", 0);
    }
    if (!caps_.integrated && context_inst_.concurrentManagedAccess()) {
      // Discrete (RTX/dGPU) residency + decode-CUDA-graph add-ons: device-only
      // activations, prefill v-copy, ALL-rows CUDA RMSNorm (despite the env's
      // name, "=all" RAISES the CUDA row cap to everything -- see
      // cuda_rmsnorm_layer.cpp; a non-'a' value like =1 is what disables it),
      // the M2-B decode graph, and async submission. On integrated
      // (Tegra/Orin) these are skipped — managed activations are the right
      // pool. Also skipped when concurrentManagedAccess==0 (Windows WDDM): each
      // of these lets a HOST op touch managed/device pool memory around
      // in-flight kernels (ASYNC drops the per-op drains outright;
      // DEV_ACT+RMSNORM_OFF put host RMSNorm/staging reads mid-chain), which is
      // only legal under cMA=1 — on WDDM the first such touch is a 0xC0000005
      // host AV. The safe WDDM default is the base profile: managed pools +
      // per-op drains.
      setenv("NNTR_CUDA_DEV_ACT", "1", 0);
      setenv("NNTR_CUDA_VCOPY_PREFILL", "1", 0);
      setenv("NNTR_RMSNORM_CUDA_OFF", "all", 0);
      // NNTR_CUDA_M2B: the single-capture decode graph. It was held off in
      // this tree while two hard correctness dependencies were missing —
      // (a) the g_m2b_skip_all embed-only feed had no graph-walk consumer, so
      // every replay token re-ran the full eager forward on top of the
      // replayed graph, and (b) the Q6_K tie_word lm_head still ran on the
      // host/OpenCL side, reading mid-capture UVM. BOTH are now present and
      // each was checked by forcing it, not by reading: the graph walk skips
      // 591 of 592 nodes per replayed token, and the tie_word lm_head reports
      // the device CUDA GEMV arm on every call. A FIXED replayed graph is
      // deterministic by construction, so the default-determinism contract
      // holds; the earlier "M2B=1 -> decode garbage" observation was against
      // the tree that had neither half. Value-checked downstream, so
      // NNTR_CUDA_M2B=0 still opts out.
      setenv("NNTR_CUDA_M2B", "1", 0);
      // NNTR_DETERMINISTIC keeps the per-op drains: ASYNC removes them and
      // is the one auto-set lever whose host/device overlap can turn a
      // knife-edge logit into a run-to-run coin flip (measured).
      {
        const char *det = getenv("NNTR_DETERMINISTIC");
        setenv("NNTR_CUDA_ASYNC", (det && det[0] == '1') ? "0" : "1", 0);
      }
    } else if (!caps_.integrated) {
      // Windows WDDM (discrete, cMA==0): the per-token ~350-launch dispatch
      // pays the WDDM submission tax (~94us/launch -> decode ~30 TPS on a
      // 5070L). Default ON the same device-resident chain + M2-B decode
      // graph as the cMA branch above -- all four are long field-proven on
      // WDDM (the Windows a2 production stack) and the graph replay is ONE
      // launch per token (measured 58-63 TPS, +93-100%, byte-identical,
      // 6-run deterministic; packaged-SDK summary 30.6 -> 57.9). A FIXED
      // replayed graph is deterministic by construction, so the
      // default-determinism contract holds. Every setenv here is
      // overwrite=0 and value-checked downstream, so =0 (or =1 for
      // RMSNORM_CUDA_OFF) still opts out per lever. ASYNC stays off: drain
      // removal is the measured knife-edge nondeterminism lever and adds
      // nothing on top of the graph (58.5 vs 58.4 TPS).
      setenv("NNTR_CUDA_DEV_ACT", "1", 0);
      setenv("NNTR_CUDA_VCOPY_PREFILL", "1", 0);
      setenv("NNTR_RMSNORM_CUDA_OFF", "all", 0);
      // NNTR_CUDA_M2B: same rationale as the cMA branch above — both halves
      // (graph-walk skip consumer, device Q6_K lm_head) are present again, so
      // the WDDM default goes back on. NOT verified on this box: there is no
      // WDDM hardware here, so the executed evidence for the two dependencies
      // comes from the cMA branch; the WDDM numbers in the comment above are
      // the field record this default was originally set from.
      setenv("NNTR_CUDA_M2B", "1", 0);
    }

    add_default_object();

    // Unified-Memory allocator: MemoryPool buffers for engine=cuda tensors are
    // cudaMallocManaged -> host-addressable AND device-accessible (the SVM
    // analogue), so a tensor on this context is device-resident with no
    // separate copy step. Falls back to host memory if UVM is unavailable.
    setMemAllocator(std::make_shared<CudaMemAllocator>());

    // Install the CUDA ComputeOps: the FC GEMM takes the device dequant-GEMM
    // path (the QS4CX host dot is NYI on x86) and the element-wise decode ops
    // (swiglu / scalar_mul / softcap) take their device kernels under the
    // residency gates; everything else inherits the CpuComputeOps bodies,
    // which are correct over the host-coherent UVM tensors.
    getContextData()->setComputeOps(get_cuda_ops());

  } catch (std::exception &e) {
    ml_loge("cuda_context: initialization failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cuda_context: initialization failed due to unknown reason");
  }
}

void CudaContext::add_default_object() {
  // FC: the backend-neutral FullyConnectedLayerCl dispatches its GEMM through
  // the installed CudaComputeOps table — CudaComputeOps::fc runs the QS4CX
  // fused dequant-GEMM on device, else the inherited host dot on UVM. Same
  // class as the gpu context.
  registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                  FullyConnectedLayerCl::type, ml::train::LayerType::LAYER_FC);
  // addition: the core CPU AdditionLayer is pure host Tensor ops -> correct on
  // the host-coherent UVM tensors (do NOT use the OpenCL AdditionLayerCL).
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);
  // rms_norm: the backend-neutral RMSNormLayer dispatches via
  // CudaComputeOps::rms_norm — the fp16 device kernel for decode-sized row
  // counts, else this backend's fused host fallback. Both halves accumulate
  // the sum of squares in FP32 (an fp16 activation with a large residual
  // element squares past the fp16 max -> the row zeroes -> garbage).
  registerFactory(nntrainer::createLayer<RMSNormLayer>, RMSNormLayer::type,
                  ml::train::LayerType::LAYER_RMSNORM);
  // geglu: the backend-neutral GeGLULayer dispatches via the installed table;
  // no CUDA geglu override exists at this change, so it resolves to the
  // inherited CpuComputeOps host body on UVM.
  registerFactory(nntrainer::createLayer<GeGLULayer>, GeGLULayer::type);
  // swiglu: the merged backend-neutral SwiGLULayer dispatches via
  // CudaComputeOps::swiglu — the device-resident fp16 one-kernel decode fast
  // path (cuda_swiglu_fp16) under its residency gates, else the inherited
  // host body. Replaces the former app-side SwiGLU fork.
  registerFactory(nntrainer::createLayer<SwiGLULayer>, SwiGLULayer::type);
  // activation: the backend-neutral ActivationLayer. Registered so a node can
  // carry engine=cuda at all -- createLayer on a type the target engine does
  // not know THROWS, it is not a silent host fallback, so without this every
  // `activation` node had to be left unstamped and therefore ran on the host.
  // The layer itself dispatches through the installed ops table, so sigmoid
  // now reaches CudaComputeOps::apply_activation's device kernel and every
  // other ActivationType still resolves to the inherited host body.
  registerFactory(nntrainer::createLayer<ActivationLayer>,
                  ActivationLayer::type);
  // logit_softcapping (promoted to core): dispatches via
  // CudaComputeOps::softcap — the fp16 device kernel on device-accessible
  // logits (carrying the terminal pipeline drain), else the inherited host
  // body on UVM.
  registerFactory(nntrainer::createLayer<LogitSoftCappingLayer>,
                  LogitSoftCappingLayer::type);
  // scalar_multiply (promoted): dispatches via CudaComputeOps::scalar_mul —
  // the opt-in (NNTR_CUDA_ELTWISE) fp16 device kernel, else the inherited
  // host body on UVM (the _gpu variant is OpenCL-only).
  registerFactory(nntrainer::createLayer<ScalarMultiplyLayer>,
                  ScalarMultiplyLayer::type);
  // tie_word_embedding (promoted): host lm_head on UVM (the GPU GEMV is the
  // OpenCL path, now #if ENABLE_OPENCL-guarded inside the layer so it builds
  // without OpenCL). Register unconditionally.
  registerFactory(nntrainer::createLayer<TieWordEmbedding>,
                  TieWordEmbedding::type);
  // Fused sigmoid gates: sigmoid_glu (attn output gate = sigmoid(gate)*x) and
  // sigmoid_add (PLE mix = sigmoid(gate)+emb). Backend-neutral layers whose
  // getOps() dispatch lands on the inherited CpuComputeOps bodies here:
  // engine=cuda tensors are host-coherent UVM, so the host loops are
  // numerically correct (same host-class-on-cuda precedent as addition /
  // scalar_multiply above). Without these factories any model that stamps
  // engine= on a fused gate aborts at graph construction with
  //   "Key is not found for the object. Key: sigmoid_glu".
  // Registered LAST with EXPLICIT high int_keys (mirroring ClContext /
  // AppContext): the auto int_key is str_map.size()+1, so a mid-list insertion
  // shifts every later auto-key and can collide with an explicit key; explicit
  // keys are collision-checked at registration.
  registerFactory(nntrainer::createLayer<SigmoidGluLayer>,
                  SigmoidGluLayer::type, /*int_key=*/9001);
  registerFactory(nntrainer::createLayer<SigmoidAddLayer>,
                  SigmoidAddLayer::type, /*int_key=*/9002);
}

template <typename T>
const int CudaContext::registerFactory(const FactoryType<T> factory,
                                       const std::string &key,
                                       const int int_key) {
  static_assert(
    isSupported<T>::value,
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
CudaContext::registerCudaKernel(const char *kernel_source,
                                const std::string &kernel_name,
                                const std::string &compile_options) {
  // Cache-hit fast path that never touches the source bytes: the std::string
  // overload's implicit conversion cost a strlen+heap-copy of the full NVRTC
  // source per call at every launch site (GDN: 9 registrations x 79 KB per
  // layer-chunk), all of it thrown away on a hit.
  const std::string kkey = kernel_name + compile_options;
  auto it = cuda_kernel_map.find(kkey);
  if (it != cuda_kernel_map.end())
    return it->second;
  return registerCudaKernel(std::string(kernel_source), kernel_name,
                            compile_options);
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
// instantiation is used and no template crosses the .so boundary.
int CudaContext::registerLayerFactory(PtrFactoryType<nntrainer::Layer> factory,
                                      const std::string &key,
                                      const int int_key) {
  return registerFactory<nntrainer::Layer>(factory, key, int_key);
}

/**
 * @brief Capture-fidelity forensics for a graph that was just captured.
 *
 * The node count is the cheapest detector for the failure mode that dominated
 * the Orin bring-up: a device op silently declines mid-capture, the work runs
 * on the host instead, and NOTHING is recorded -- so the graph instantiates and
 * replays while quietly MISSING that op. Comparing the count against the
 * expected op census is what makes that visible.
 *
 * Both capture sites call this. The decode site had it inline; the prefill site
 * (default ON for integrated GPUs, cuda_graph_prefill above) printed nothing at
 * all, which is exactly backwards -- on Orin the prefill graph is the FIRST
 * graph the process ever captures.
 *
 * @param graph the captured graph (nullptr is a no-op)
 * @param tag   log prefix, also appended to NNTR_CUDA_GRAPH_DOT so a prefill
 *              dump and a decode dump coexist instead of clobbering each other
 */
static void cuda_graph_census(cudaGraph_t graph, const char *tag) {
  if (graph == nullptr)
    return;
  size_t n_nodes = 0;
  cudaGraphGetNodes(graph, nullptr, &n_nodes);
  std::fprintf(stderr, "[%s] captured graph: %zu nodes\n", tag, n_nodes);
  if (const char *dot = std::getenv("NNTR_CUDA_GRAPH_DOT")) {
    const std::string path = std::string(dot) + "." + tag;
    if (cudaGraphDebugDotPrint(graph, path.c_str(),
                               cudaGraphDebugDotFlagsVerbose) == cudaSuccess)
      std::fprintf(stderr, "[%s] graph dot -> %s\n", tag, path.c_str());
    cudaGetLastError();
  }
}

// SEAM-2 CUDA override (docs/ARCHITECTURE_REFACTOR.md §10 T9). Relocated
// VERBATIM from neuralnet.cpp's incremental_inference #if ENABLE_CUDA block —
// the only changes are the model walk (`nn.incremental_forwarding`), graph-node
// access
// (`nn.getLayerNode` / `nn.feedInputsLabels`), the M2-B skip flag
// (`nn.setM2BSkipAll`), and the prefill-capture flag
// (`nn.isPrefillCaptureDisabled()`). All decisions/env reads/static state are
// unchanged, so engine=cuda decode is behaviorally identical.
sharedConstTensors CudaContext::runDecode(NeuralNetwork &nn, unsigned int from,
                                          unsigned int to,
                                          const sharedConstTensors &input,
                                          const sharedConstTensors &label) {
  sharedConstTensors out;

  // CUDA-graph capture of a whole DECODE forward (NNTR_CUDA_GRAPH, M1). A
  // decode step issues ~1000 tiny kernels; the CPU launch/dispatch between them
  // is the decode bottleneck (GPU ~30-47% utilized). Capturing the per-token
  // forward into one graph and replaying it collapses that launch overhead. M1
  // re-instantiates every step (still pays cudaGraphInstantiate) purely to
  // prove capture+replay COHERENCE; M2 will cache the graphExec and patch
  // params.
  static const char *_cgraph_env = std::getenv("NNTR_CUDA_GRAPH");
  static const bool cuda_graph_decode =
    _cgraph_env != nullptr && _cgraph_env[0] == '1';
  // PREFILL graph (W3): capture the M>1 prefill forward like decode. OPT-IN
  // DIAGNOSTIC ONLY (NNTR_CUDA_PREFILL_GRAPH=1; the legacy NNTR_CUDA_GRAPH+
  // integrated fallback below never fired on Orin). Retracted 2026-08-10: its
  // measured wins were the warmup's timer restart, warm-vs-warm it loses to
  // the deferred-drain path below, and on a data-dependent-routing model
  // (qwen_moe) a capture bakes in STALE expert counts -- a capture that
  // escapes markCaptureDoomed replays the wrong expert set silently. The
  // integrated-default role belongs to NNTR_CUDA_PREFILL_DEFER (below).
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
  // Diagnostic: cache the exec from the first captured token and RE-LAUNCH it
  // for subsequent tokens (incoherent; measures the pure cudaGraphLaunch+sync
  // ceiling).
  static const bool cuda_graph_replay =
    std::getenv("NNTR_CUDA_GRAPH_REPLAY") != nullptr;
  static cudaGraphExec_t _cg_cached_exec = nullptr;
  static sharedConstTensors _cg_cached_out;
  static unsigned long _cg_ok = 0, _cg_fallback = 0;
  bool cuda_graph_captured = false;

  // M2-B: single-capture COHERENT decode. Capture the full forward ONCE (first
  // decode token); for every later token, refresh ONLY the embeddings on the
  // host (g_m2b_skip_all feed pass), update the device position (cuda_set_pos),
  // and REPLAY the cached graph -- skipping the ~350-op C++ dispatch.
  // VALUE-checked (=0 disables): cuda_context auto-sets NNTR_CUDA_M2B=1 on
  // discrete+cMA boxes (setenv overwrite=0), so a presence check made =0 a
  // FRANKEN-state -- graph capture/replay here stayed ON while the mha_core
  // slot-writes (nntr_env_on, value-checked) turned OFF: replay then rewrites
  // K/V at the first captured slot every token = deterministic decode garbage
  // (field 2026-07-10: Linux HOST_MAPPED mimic with M2B=0 looped "toasters,
  // which in turned" -- the exact same env-check split we swept everywhere
  // else; see env_compat.h).
  static const bool cuda_m2b = nntr_env_on("NNTR_CUDA_M2B");
  if (cuda_m2b && (from == 0 || (to - from) > 1) &&
      _cg_cached_exec != nullptr) {
    // Prefill boundary: a new sequence (from==0) OR a resumed-block prefill
    // (from>0, M>1 — multi-turn / KV-restore under NNTR_RESUME_BLOCK). Drop
    // the previous sequence's cached decode graph BEFORE the eager M>1
    // forward: that forward may grow (free+realloc) the dp4a/i8/attention
    // scratch the captured graph references, and replaying it afterwards
    // launches with dangling device/pinned pointers (cudaGraphLaunch SEGV).
    // The next decode recaptures against the fresh pointers anyway.
    cudaGraphExecDestroy(_cg_cached_exec);
    _cg_cached_exec = nullptr;
    _cg_cached_out = {};
  }
  // M2B WARMUP (the 35B fix, 2026-08-21): capturing the FIRST decode token
  // bakes a broken graph on any model with lazy decode-side setup — the GDN
  // decode scratch grow_dev refuses its cudaMalloc under capture ("scratch
  // alloc FAILED"), so all 30 GDN layers record NOTHING and the replayed
  // graph computes a forward with no GDN in it; the one-time weight
  // transforms (gdn_wq_i8t, moe_rsplit_t) either get baked into every replay
  // or get isCapturing-skipped into their slow fallback arms. Run the first
  // N decode tokens EAGERLY so every lazy allocation and one-time build has
  // happened, then capture. NNTR_CUDA_M2B_WARM overrides (default 2).
  static const int m2b_warm_tokens = []() {
    const char *e = std::getenv("NNTR_CUDA_M2B_WARM");
    const int v = e ? atoi(e) : 2;
    return v >= 0 ? v : 2;
  }();
  static int m2b_warm_left = m2b_warm_tokens;
  if (cuda_m2b && (from == 0 || (to - from) > 1))
    m2b_warm_left = m2b_warm_tokens; // new sequence: warm up again
  if (cuda_m2b && from != 0 && (to - from) == 1 &&
      _cg_cached_exec == nullptr && m2b_warm_left > 0) {
    --m2b_warm_left; // eager token; fall through to the normal path below
  } else if (cuda_m2b && from != 0 && (to - from) == 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    if (_cg_cached_exec != nullptr) {
      // subsequent token: embed-only feed (refresh emb_stage) -> set pos ->
      // replay
      static const bool m2b_light = nntr_env_on("NNTR_CUDA_M2B_LIGHT");
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
        if (cuda_graph_dbg)
          cuda_graph_census(graph, "M2B");
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
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
        .count();
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
        if (cuda_graph_dbg)
          cuda_graph_census(graph, "M1");
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
        // and clear the sticky flag so the eager fallback is not falsely
        // flagged.
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
                     _cg_ok,
                     t_rec ? "CAPTURED+REPLAYED" : "REPLAY-ONLY(cached)", t_rec,
                     t_inst, t_rep);
      else
        std::fprintf(stderr,
                     "[CUDA_GRAPH] fell back (captured=%lu fallback=%lu) "
                     "stage=%s err=%d\n",
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
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
        .count();
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
        if (cuda_graph_dbg)
          cuda_graph_census(graph, "PREFILL_GRAPH");
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
      std::fprintf(
        stderr,
        "[PREFILL_GRAPH] #%lu M=%u %s record=%ldus instantiate=%ldus "
        "replay+sync=%ldus stage=%s err=%d\n",
        ++_pf, (unsigned)(to - from),
        cuda_graph_captured ? "CAPTURED" : "FALLBACK", t_rec, t_inst, t_rep,
        stage, (int)cerr);
    }
  }
  // Deferred-drain eager forward (NNTR_CUDA_PREFILL_DEFER=1): collapse the
  // per-op cudaStreamSynchronize gaps of an M>1 prefill WITHOUT graph capture.
  // The prefill graph's measured win is not replay reuse (the exec is
  // destroyed after one launch) -- it is that recording skips the per-op
  // drains and pays ONE sync. This gets the same collapse with the kernels
  // running eagerly async on the stream, and unlike capture it stays correct
  // where the forward READS device results on the host mid-chunk (the MoE
  // expert loop reading dev_counts after its explicit finish(),
  // qwen_moe_layer.cpp): finish() is not suppressed, only maybeFinish(), and
  // finishIfAsync() drains while deferred so host-fallback preambles keep
  // their ordering. It also has no from==0 restriction: every chunk of a
  // chunked prefill goes through here.
  if (!cuda_graph_captured) {
    // Default ON for integrated GPUs (every per-op maybeFinish there is a full
    // cudaStreamSynchronize; discrete keeps eager-async prefill). Evaluated on
    // the first runDecode call, when the CUDA context is necessarily up -- the
    // never-ran PREFILL_GRAPH default (§ static-before-context) is the
    // cautionary tale. Measured at the flip (2026-08-10, byte-identical output
    // on both models): gemma4 warm 327 -> 291 ms; 35B 1.3K warm 574.7 -> 585.8
    // TPS; 35B 20K 661.6 -> 668.3 TPS.
    // =0 off; =1 whole-chunk region; =2 same region with a drain checkpoint at
    // every node boundary (neuralnet.cpp) -- bounded staleness, race bisect.
    static const bool prefill_defer = []() {
      const char *e = std::getenv("NNTR_CUDA_PREFILL_DEFER");
      if (e != nullptr)
        return e[0] == '1' || e[0] == '2';
      return nntrainer::cuda::ContextManager::Global().isIntegrated();
    }();
    // NNTR_CUDA_DEFER_WARM=1: run the process's FIRST M>1 forward at
    // per-op-drain pace (the lazy-construction pass) and defer only later
    // chunks. Default OFF -- with the audited drains in place the first chunk
    // defers safely too (measured 2026-08-10: gemma4 cold 496 -> 450 ms, 35B
    // 20K 671.8 TPS, byte-identical both) -- kept as a bisect knob for any
    // future defer-exposed first-touch race.
    static const bool defer_skip_first = []() {
      const char *e = std::getenv("NNTR_CUDA_DEFER_WARM");
      return e != nullptr && e[0] == '1';
    }();
    static bool s_m2_fwd_seen = false;
    auto &smd = nntrainer::cuda::StreamManager::Global();
    const bool defer = prefill_defer && (to - from) > 1 && !smd.isCapturing() &&
                       (s_m2_fwd_seen || !defer_skip_first);
    if ((to - from) > 1)
      s_m2_fwd_seen = true;
    using _dclk = std::chrono::high_resolution_clock;
    const auto d0 = _dclk::now();
    {
      // RAII: incremental_forwarding can throw; a stuck defer_drain_ would
      // silently suppress every later drain in the process.
      struct DeferGuard {
        nntrainer::cuda::StreamManager *sm;
        explicit DeferGuard(nntrainer::cuda::StreamManager *s) : sm(s) {
          if (sm)
            sm->pushDeferDrain();
        }
        ~DeferGuard() {
          if (sm) {
            sm->popDeferDrain();
            sm->finish(); // the region's single terminal drain
          }
        }
      } dguard(defer ? &smd : nullptr);
      out = nn.incremental_forwarding(from, to, input, label, false);
    }
    if (defer && cuda_graph_dbg)
      std::fprintf(stderr, "[PREFILL_DEFER] M=%u fwd+drain=%ldus\n",
                   (unsigned)(to - from),
                   (long)std::chrono::duration_cast<std::chrono::microseconds>(
                     _dclk::now() - d0)
                     .count());
  }

  return out;
}

} // namespace nntrainer
