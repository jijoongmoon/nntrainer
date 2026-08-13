// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_stream_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA stream/dispatch management implementation.
 */

#include "cuda_stream_manager.h"
#include "cuda_common.h"
#include "cuda_context_manager.h"
#include "cuda_kernel.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <env_compat.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace nntrainer::cuda {

// [CAP-AUDIT] prints are opt-in diagnostics (NNTR_CUDA_CAP_AUDIT=1): with the
// M2-B decode graph default-ON they fire on every capture (one per shared-
// layer drain preamble, ~68 lines/run) and read as errors to SDK consumers,
// while in sync mode (ASYNC off, the Windows default) the skipped drains are
// no-ops to begin with. Correctness is covered by the replay validations
// (byte-identical vs the sync path, 6-run determinism); flip the env on when
// hunting a NEW capture-time host-fallback hazard.
static bool cap_audit_on() {
  static const bool on = nntr_env_on("NNTR_CUDA_CAP_AUDIT");
  return on;
}

// ---- NNTR_KERN_PROF: per-launch GPU-time histogram (see header) -----------
// Single-threaded by construction: every bracketed call sits on the forward
// path (one dispatching thread). The pending ring bounds live events; draining
// syncs only the OLDEST pair, which the GPU has already retired by the time
// the ring fills, so the schedule being measured is not perturbed by drains.
namespace {

struct KpAcc {
  double ms = 0.0;
  unsigned long long calls = 0;
};

struct KpPending {
  cudaEvent_t s, e;
  KpAcc *acc;
};

struct KpState {
  std::unordered_map<std::string, KpAcc> table; // node-based: KpAcc* stable
  std::vector<cudaEvent_t> free_ev;
  std::deque<KpPending> pending;
  std::string key; // reused per launch to avoid churn
  unsigned long long launches = 0;
};

KpState &kp_state() {
  static KpState *s = []() {
    std::atexit([]() { kprof_dump(); });
    return new KpState();
  }();
  return *s;
}

cudaEvent_t kp_take_event(KpState &st) {
  if (!st.free_ev.empty()) {
    cudaEvent_t ev = st.free_ev.back();
    st.free_ev.pop_back();
    return ev;
  }
  cudaEvent_t ev = nullptr;
  cudaEventCreate(&ev); // timing enabled
  return ev;
}

// Accumulate the oldest pending pair. Returns false if the events cannot be
// resolved (context torn down at exit) -- callers stop draining then.
bool kp_drain_one(KpState &st) {
  KpPending p = st.pending.front();
  st.pending.pop_front();
  float f = 0.f;
  if (cudaEventSynchronize(p.e) != cudaSuccess ||
      cudaEventElapsedTime(&f, p.s, p.e) != cudaSuccess)
    return false;
  p.acc->ms += (double)f;
  p.acc->calls += 1;
  st.free_ev.push_back(p.s);
  st.free_ev.push_back(p.e);
  return true;
}

} // namespace

bool kprof_enabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_KERN_PROF");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

void *kprof_begin() {
  if (!kprof_enabled())
    return nullptr;
  auto &sm = StreamManager::Global();
  if (sm.isCapturing()) // an event record would be captured into the graph
    return nullptr;
  KpState &st = kp_state();
  cudaEvent_t ev = kp_take_event(st);
  if (ev == nullptr)
    return nullptr;
  cudaEventRecord(ev, sm.GetStream());
  return (void *)ev;
}

void kprof_end(void *start_ev, const char *kern, const char *tag) {
  if (start_ev == nullptr)
    return;
  KpState &st = kp_state();
  auto &sm = StreamManager::Global();
  cudaEvent_t stop = kp_take_event(st);
  if (stop == nullptr) {
    st.free_ev.push_back((cudaEvent_t)start_ev);
    return;
  }
  cudaEventRecord(stop, sm.GetStream());
  // role = tag minus its "layer<NN>_" prefix, so node repeats aggregate.
  const char *role = tag ? tag : "";
  if (std::strncmp(role, "layer", 5) == 0) {
    const char *p = role + 5;
    while (*p >= '0' && *p <= '9')
      ++p;
    if (*p == '_' && p != role + 5)
      role = p + 1;
  }
  st.key.assign(kern ? kern : "?");
  st.key += '|';
  st.key += role;
  KpAcc &acc = st.table[st.key];
  st.pending.push_back({(cudaEvent_t)start_ev, stop, &acc});
  ++st.launches;
  // Bound live events; the front of a 4096-deep ring is long retired.
  constexpr size_t KP_RING = 4096;
  if (st.pending.size() >= KP_RING) {
    for (int i = 0; i < 512 && !st.pending.empty(); ++i)
      if (!kp_drain_one(st))
        break;
  }
}

void kprof_dump() {
  if (!kprof_enabled())
    return;
  KpState &st = kp_state();
  static bool dumped = false;
  if (dumped)
    return;
  dumped = true;
  while (!st.pending.empty())
    if (!kp_drain_one(st))
      break; // context already gone: report what resolved
  std::vector<std::pair<std::string, KpAcc>> rows(st.table.begin(),
                                                  st.table.end());
  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
    return a.second.ms > b.second.ms;
  });
  double total = 0.0;
  unsigned long long calls = 0;
  for (const auto &r : rows) {
    total += r.second.ms;
    calls += r.second.calls;
  }
  std::fprintf(stderr,
               "[kern_prof] ==== GPU ms by kernel|role: %.1f ms total, "
               "%llu launches, %zu keys (unresolved pending=%zu) ====\n",
               total, calls, rows.size(), st.pending.size());
  for (const auto &r : rows)
    std::fprintf(stderr, "[kern_prof] %10.1f ms %7llu calls %9.1f us/call  %s\n",
                 r.second.ms, r.second.calls,
                 r.second.calls ? 1000.0 * r.second.ms / (double)r.second.calls
                                : 0.0,
                 r.first.c_str());
}
// ---------------------------------------------------------------------------

void StreamManager::initialize() noexcept {
  // make sure the device + primary context exist before creating a stream
  ContextManager::Global().EnsureCurrent();
  if (!cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate"))
    stream_ = nullptr;
}

bool StreamManager::EnqueueWriteBuffer(void *dst_dev, size_t size,
                                       const void *src_host, bool async) {
  if (!cudaCheck(cudaMemcpyAsync(dst_dev, src_host, size,
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync H2D"))
    return false;
  if (!async)
    return cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

bool StreamManager::EnqueueReadBuffer(const void *src_dev, size_t size,
                                      void *dst_host, bool async) {
  if (!cudaCheck(cudaMemcpyAsync(dst_host, src_dev, size,
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync D2H"))
    return false;
  if (!async)
    return cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

bool StreamManager::DispatchCommand(Kernel &kernel, const int (&grid)[3],
                                    const int (&block)[3],
                                    unsigned int shared_bytes) {
  if (!kernel.valid()) {
    ml_loge("[CUDA] DispatchCommand: invalid kernel");
    return false;
  }
  ContextManager::Global().EnsureCurrent();
  // Counted before the launch so a caller that stamps dispatchSeq() AFTER its
  // own dispatches sees a value no other dispatch can reproduce.
  ++dispatch_seq_;
  auto params = kernel.getKernelParams();
  void *kp = kprof_begin();
  CUresult r = cuLaunchKernel(
    kernel.GetFunction(), (unsigned)grid[0], (unsigned)grid[1],
    (unsigned)grid[2], (unsigned)block[0], (unsigned)block[1],
    (unsigned)block[2], shared_bytes, reinterpret_cast<CUstream>(stream_),
    params.empty() ? nullptr : params.data(), nullptr);
  if (kp != nullptr)
    kprof_end(kp, kernel.name().c_str(), dispatch_tag_);
  if (r != CUDA_SUCCESS && capturing_) {
    // Under capture a launch failure is normally CUDA_ERROR_STREAM_CAPTURE_
    // INVALIDATED -- something earlier already broke the capture and THIS
    // kernel is only the first victim. Either way the graph cannot be trusted.
    markCaptureDoomed("a kernel launch failed inside the capture");
  }
  if (r != CUDA_SUCCESS) {
    // Name the launch: an async fault (illegal access) surfaces at a LATER
    // launch than the kernel that caused it, and under deferred drains that
    // distance grows -- without the name, the report site is anonymous. With
    // CUDA_LAUNCH_BLOCKING=1 this line names the faulting kernel itself.
    ml_loge("[CUDA] cuLaunchKernel FAILED for '%s' in node '%s' "
            "grid(%d,%d,%d) block(%d,%d,%d) shmem=%u seq=%llu",
            kernel.name().c_str(), dispatch_tag_, grid[0], grid[1], grid[2],
            block[0], block[1], block[2], shared_bytes,
            (unsigned long long)dispatch_seq_);
  }
  return cuCheck(r, "cuLaunchKernel");
}

void StreamManager::finish() {
  if (capturing_) { // an in-capture cudaStreamSynchronize is illegal; the drain
    // is deferred to after the graph replay (endCapture caller). A host read
    // that depended on this drain now consumes stale bytes -- audit-log the
    // skip so capture-time host fallbacks are visible.
    static int audit_n = 0;
    if (++audit_n <= 32 && cap_audit_on())
      std::fprintf(
        stderr, "[CAP-AUDIT] finish() skipped during capture (#%d)\n", audit_n);
    return;
  }
  if (stream_) {
    cudaStreamSynchronize(stream_);
    // concurrentManagedAccess==0 (Windows WDDM / pre-Pascal model) device-sync
    // add-on. HISTORY: added when host reads of kernel-written managed pages
    // appeared stale on WDDM -- but the actual culprit turned out to be the
    // unified-binary isSVM hijack (outputs were never written at all; see
    // CudaMemAllocator::isSVM). With that fixed, the stream-sync alone may be
    // sufficient (pre-Pascal launch migration + stream drain), and the per-op
    // cudaDeviceSynchronize goes through the WDDM OS scheduler = measurable
    // cost. The =0 variant was field-validated golden on the WDDM box (1K
    // 63.0/5.90 TPS, +7% decode vs devsync-on; pinned zero-copy pool), so the
    // DEFAULT IS OFF -- NNTR_CUDA_WDDM_DEVSYNC=1 re-arms the drain if a future
    // cMA==0 device shows a genuine post-kernel host-visibility gap.
    static const bool wddm_devsync = []() {
      const char *e = std::getenv("NNTR_CUDA_WDDM_DEVSYNC");
      return e != nullptr && e[0] == '1';
    }();
    if (wddm_devsync && !ContextManager::Global().concurrentManagedAccess())
      cudaDeviceSynchronize();
  }
}

static bool cuda_async_mode() {
  static const bool async = []() {
    const char *e = std::getenv("NNTR_CUDA_ASYNC");
    if (e == nullptr || e[0] != '1')
      return false;
    // Integrated GPU (Tegra/Jetson Orin): async drops the per-op stream drain,
    // but on the shared-memory iGPU there is no UVM page-fault ordering to
    // order a host read against an in-flight kernel write -> the host fallbacks
    // read half-written buffers = corrupted tokens. Force SYNC on integrated
    // regardless of the env (re-enable per-Orin only after a dedicated
    // coherence benchmark). Discrete GPUs honor NNTR_CUDA_ASYNC.
    return !ContextManager::Global().isIntegrated();
  }();
  return async;
}

void StreamManager::maybeFinish() {
  if (capturing_)
    return;
  // Inside a deferred-drain region the caller has taken responsibility for
  // ordering (see pushDeferDrain in the header) and will finish() explicitly.
  if (defer_drain_)
    return;
  // NNTR_CUDA_PACE=<N> (default off): depth-N submission pacing -- the middle
  // ground between the full per-op drain (sync mode; WDDM decode ~29 TPS) and
  // no drain at all (no-drain modes corrupt on WDDM). Bounds the un-drained op
  // window to N by waiting the (i-N)th op's event instead of draining op i.
  // Host-read boundaries still use the full finish(), so correctness of host
  // consumption is unchanged. If the WDDM corruption scales with N, the driver
  // chokes on deep unpaced queues (pacing = fix); if even N=4 corrupts while
  // full drain is clean, the defect is at kernel-boundary granularity (driver
  // bug class).
  static const int pace_n = []() {
    const char *e = std::getenv("NNTR_CUDA_PACE");
    const int v = e ? std::atoi(e) : 0;
    return v > 120 ? 120 : v; // ring headroom
  }();
  if (pace_n > 0 && stream_) {
    constexpr int RING = 128;
    static cudaEvent_t ring[RING] = {};
    static unsigned long long idx = 0;
    const int slot = (int)(idx % RING);
    if (ring[slot] == nullptr)
      cudaEventCreateWithFlags(&ring[slot], cudaEventDisableTiming);
    cudaEventRecord(ring[slot], stream_);
    if (idx >= (unsigned long long)pace_n) {
      const int wslot = (int)((idx - (unsigned long long)pace_n) % RING);
      if (ring[wslot])
        cudaEventSynchronize(ring[wslot]);
    }
    ++idx;
    return;
  }
  if (!cuda_async_mode())
    finish();
}

void StreamManager::finishIfAsync() {
  if (capturing_) {
    // Same audit as finish(): callers of finishIfAsync are host-fallback
    // preambles -- a hit during capture means a host op ran inside the graph.
    static int audit_n = 0;
    if (++audit_n <= 32 && cap_audit_on())
      std::fprintf(stderr,
                   "[CAP-AUDIT] finishIfAsync() skipped during capture (#%d)\n",
                   audit_n);
    return;
  }
  // Inside a deferred-drain region the per-op maybeFinish() that normally
  // orders a host read on integrated (sync mode) is suppressed, so a
  // host-fallback preamble must supply its own drain here. Without this the
  // fallback reads a buffer whose producing kernel is still queued -- the
  // exact bug class the gemm_ex missing drain was (wrong and different every
  // run, invisible to CAP-AUDIT/FC_DBG).
  if (cuda_async_mode() || defer_drain_)
    finish();
}

void StreamManager::drainPipeline() {
  // Deliberately NOT finish(): that one audit-logs a capture-time skip because
  // its callers are host fallbacks. Here the caller proceeds to a device
  // kernel, so skipping under capture is correct and must not read as a
  // finding. See the header for why the distinction is load-bearing.
  if (capturing_)
    return;
  finish();
}

void StreamManager::setDispatchTag(const char *tag) {
  if (tag == nullptr)
    tag = "";
  std::snprintf(dispatch_tag_, sizeof(dispatch_tag_), "%s", tag);
}

void StreamManager::markCaptureDoomed(const char *why) {
  if (!capturing_ || capture_doomed_)
    return;
  capture_doomed_ = true;
  ml_logw("[CUDA] graph capture abandoned: %s. The op that reported this "
          "DECLINED to run, so the graph would be missing it; the forward will "
          "be re-run eagerly instead.",
          why ? why : "an op declined to run under capture");
}

bool StreamManager::beginCapture() {
  if (!stream_)
    return false;
  cudaStreamSynchronize(stream_); // drain pre-capture work; start from idle
  if (!cudaCheck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed),
                 "cudaStreamBeginCapture"))
    return false;
  capturing_ = true;
  capture_doomed_ = false;
  return true;
}

bool StreamManager::endCapture(cudaGraph_t *graph) {
  capturing_ = false;
  if (!stream_ || graph == nullptr)
    return false;
  const bool ok = cudaCheck(cudaStreamEndCapture(stream_, graph),
                            "cudaStreamEndCapture");
  // A capture some op declined to join is WORSE than one the driver rejected:
  // the driver hands back a perfectly instantiable graph that is simply missing
  // work, and it replays to wrong numbers silently. Destroy it and report
  // failure so the caller falls back to the eager forward.
  if (ok && capture_doomed_) {
    if (*graph != nullptr) {
      cudaGraphDestroy(*graph);
      *graph = nullptr;
    }
    capture_doomed_ = false;
    return false;
  }
  capture_doomed_ = false;
  return ok;
}

StreamManager::~StreamManager() {
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

int *cuda_pos_buffer() {
  static int *g_pos_dev = []() -> int * {
    int *p = nullptr;
    cudaMalloc((void **)&p, 2 * sizeof(int));
    return p;
  }();
  return g_pos_dev;
}

void cuda_set_pos(int pos, int n_kv) {
  // Pinned host source so the H2D is a real async DMA (also keeps it capturable
  // should it ever be issued inside a capture). The copy is on the backend
  // stream, so it is ordered before a subsequent cudaGraphLaunch on the same
  // stream -- the replayed kernels read the fresh pos.
  static int *g_pos_host = []() -> int * {
    int *p = nullptr;
    cudaHostAlloc((void **)&p, 2 * sizeof(int), cudaHostAllocDefault);
    return p;
  }();
  int *d = cuda_pos_buffer();
  if (!d || !g_pos_host)
    return;
  g_pos_host[0] = pos;
  g_pos_host[1] = n_kv;
  cudaMemcpyAsync(d, g_pos_host, 2 * sizeof(int), cudaMemcpyHostToDevice,
                  StreamManager::Global().GetStream());
}

} // namespace nntrainer::cuda
