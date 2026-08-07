// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gated_delta_net_layer.h
 * @date   30 June 2026
 * @brief  Qwen3-Next (qwen3_5_moe) Gated DeltaNet linear-attention mixer layer.
 * @author Claude Code (port of the validated P1 reference)
 * @bug    No known bugs except for NYI items
 *
 * Token mixer for the 30/40 `linear_attention` layers of Qwen3.6-35B-A3B.
 * Numeric spec frozen + validated against HF Qwen3_5MoeGatedDeltaNet (P0/P1):
 *   in_proj_qkv -> causal depthwise conv1d(K) + SiLU -> split[q|k|v] -> GQA repeat
 *   -> l2norm(q,k) + q*1/sqrt(head_k_dim) -> decay-first delta recurrence
 *   -> gated RMSNorm (rmsnorm(core)*weight*silu(z)) -> out_proj
 * Projection weights follow the nntrainer FullyConnected convention [in,out]
 * (HF nn.Linear [out,in] must be transposed at load).
 */

#ifndef __GATED_DELTA_NET_LAYER_H__
#define __GATED_DELTA_NET_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

/** number of value heads (linear_num_value_heads) */
class LinearNumValueHeads : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "linear_num_value_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};
/** number of key heads (linear_num_key_heads); GQA group = num_value/num_key */
class LinearNumKeyHeads : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "linear_num_key_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};
/** key head dim (linear_key_head_dim) */
class LinearKeyHeadDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "linear_key_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};
/** value head dim (linear_value_head_dim) */
class LinearValueHeadDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "linear_value_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};
/** depthwise causal conv1d kernel size (linear_conv_kernel_dim) */
class LinearConvKernelDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "linear_conv_kernel_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class   GatedDeltaNetLayer
 * @brief   Qwen3-Next Gated DeltaNet linear-attention mixer
 */
class WIN_EXPORT GatedDeltaNetLayer : public nntrainer::LayerImpl {
public:
  GatedDeltaNetLayer();
  ~GatedDeltaNetLayer() = default;
  GatedDeltaNetLayer(GatedDeltaNetLayer &&rhs) noexcept = default;
  GatedDeltaNetLayer &operator=(GatedDeltaNetLayer &&rhs) = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void setProperty(const std::vector<std::string> &values) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;
  const std::string getType() const override { return type; }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "gated_delta_net";

private:
  // derived dims (set in finalize)
  unsigned int num_v_heads, num_k_heads, head_k_dim, head_v_dim;
  unsigned int key_dim, value_dim, conv_dim, conv_kernel, hidden_size;
  float eps;

  std::tuple<props::LinearNumValueHeads, props::LinearNumKeyHeads,
             props::LinearKeyHeadDim, props::LinearValueHeadDim,
             props::LinearConvKernelDim>
    gdn_props;

  // weight indices (finalize order)
  unsigned int w_in_proj_qkv, w_in_proj_z, w_in_proj_b, w_in_proj_a;
  unsigned int w_conv1d, w_A_log, w_dt_bias, w_norm, w_out_proj;

  // persistent decode state (MAX_LIFESPAN): recurrent S + conv ring buffer
  unsigned int state_idx, conv_state_idx;

  // Pooled fp32 prefill projection outputs. Two reasons they are pooled rather
  // than the std::vectors they replace: (a) on CUDA they must be
  // device-accessible for the cuBLAS arm to write them at all, and (b) the
  // per-call heap versions were 3.6 GB of alloc/free per layer per prefill at
  // T=20000, thirty times per request. FP32 because the conv1d / l2norm /
  // recurrence chain below is the fp32 host reference these must agree with.
  unsigned int proj_qkv_idx, proj_z_idx, proj_b_idx, proj_a_idx;

  // fp32 heap copies of ALL weights, converted once on first forward — the
  // GDN math is host-side fp32 regardless of the stored dtype (FP32 tiny
  // validation / FP16 35B deployment). The projections are cached too: the
  // 35B FP16 weights live in CUDA managed memory, and the host fp16 dot over
  // them (per-element half->float + post-GPU managed-page access) measured
  // ~1.7 s per decode call; an OpenBLAS sgemm over cached heap fp32 is
  // milliseconds. fp32 out_proj is also required for correctness: fp16
  // accumulation overflows on large normed rows (±inf → NaN logits).
  std::vector<float> wconv_f, alog_f, dtb_f, wnorm_f;
  std::vector<float> wqkv_f, wz_f, wb_f, wa_f, wout_fv;
  bool wcache_loaded = false;
  void ensureWeightCache(nntrainer::RunLayerContext &context);

  /** prefill over seq_len tokens starting at INPUT ROW 0; when save_state,
   *  persists the final recurrent state S + the last (conv_kernel-1) conv
   *  inputs. When seed_state, S and the causal left-pad are RESUMED from those
   *  persistent tensors instead of starting from zero -- that is what makes a
   *  chunked prefill legal for a linear-attention layer, where (unlike causal
   *  softmax attention) a later chunk cannot reconstruct its prefix from a KV
   *  cache. */
  void runForward(nntrainer::RunLayerContext &context, int seq_len,
                  bool save_state, bool seed_state = false);
  /** single-token decode step: decay-first delta update + conv-with-ring. */
  void runDecode(nntrainer::RunLayerContext &context);
  /** fp32-sgemm out_proj (normed [B*S,VAL] @ Wout -> output), dtype-aware
   *  output write. */
  void outProj(nntrainer::RunLayerContext &context, const float *normed, int B,
               int S, nntrainer::Tensor &output);
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __GATED_DELTA_NET_LAYER_H__ */
