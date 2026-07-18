// SPDX-License-Identifier: Apache-2.0
/**
 * @file    recq_overrides.h
 * @brief   Record/replay (recq) per-token override registry (R3). During the
 *          record decode pass, the MHA layer captures, for each kernel that
 *          takes a per-token-varying SCALAR (rope position / KV-scatter
 *          position / row offset / attention N_kv) or a per-token global-work-
 *          size (qk), an entry mapping the kernel's recorded dispatch_index +
 *          arg_index to a rule for computing the value from cache_index. The
 *          decode loop (R4) then rebuilds the cl_array_arg_qcom / cl_workgroup_
 *          qcom arrays per token and replays the recording with them.
 *
 *          Implemented in mha_core.cpp (no new build-system entry).
 */
#ifndef __RECQ_OVERRIDES_H__
#define __RECQ_OVERRIDES_H__

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "opencl_command_queue_manager.h" // cl_array_arg_qcom / cl_workgroup_qcom

// The registry lives in mha_core.cpp, which Windows builds as a DLL — these
// free functions must be exported for causal_lm (static lib) to link, same
// WIN_EXPORT pattern as the layer classes.
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief One per-token override slot captured during the record pass. The value
 * (or global-work-size) is recomputed from cache_index at each replayed token.
 */
struct RecqOverride {
  uint32_t dispatch_index; ///< kernel ordinal in the recorded chain
  uint32_t arg_index;      ///< kernel-arg ordinal (ignored for QK_GWS0)
  enum Kind {
    ROPE_POS,    ///< value = cache_index
    SCATTER_POS, ///< value = cache_index (cache_from == cache_index at decode)
    ROW_OFFSET,  ///< value = base + cache_index * stride  (kc/v woff/src_off)
    ATTN_NKV,    ///< value = cache_index + 1  (cache_to)
    QK_GWS0      ///< gws = {pad(ceil((cache_index+1)/8), lx), mx_pad, num_heads_Q}
  } kind;
  int base;   ///< ROW_OFFSET: batch*FeatureLen ; QK_GWS0: lx (= NNTR_QK_LWS[0])
  int stride; ///< ROW_OFFSET: width (=num_heads_KV*head_dim) ; QK_GWS0: mx_pad
  int aux;    ///< QK_GWS0: num_heads_Q ; else unused
};

/// Clear the registry (call before each record pass).
WIN_EXPORT void recq_reset_overrides();
/// Append a captured override (called by mha_core while isRecording()).
WIN_EXPORT void recq_add_override(const RecqOverride &ov);
/// Number of captured overrides.
WIN_EXPORT std::size_t recq_override_count();

/**
 * @brief Build the per-token replay arrays from cache_index. Fills args/gws and
 * their backing storage (int_scratch / gws_scratch); the cl_array_arg_qcom
 * arg_value pointers and cl_workgroup_qcom workgroup_size pointers reference the
 * scratch vectors, which are reserved up-front so they never reallocate during
 * the build (the pointers stay valid until the next build). The caller must keep
 * all four vectors alive across the replayRecording() call.
 */
WIN_EXPORT void recq_build_token_overrides(
  unsigned int cache_index,
  std::vector<nntrainer::opencl::cl_array_arg_qcom> &args,
  std::vector<nntrainer::opencl::cl_workgroup_qcom> &gws,
  std::vector<int> &int_scratch,
  std::vector<std::array<std::size_t, 3>> &gws_scratch);

} // namespace causallm

#endif /* __RECQ_OVERRIDES_H__ */
