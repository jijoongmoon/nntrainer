// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen3_forward.h
 * @date   29 May 2026
 * @brief  Paper-aligned GPU-native Qwen3 forward.
 *
 * Bypasses the nntrainer layer graph. Activations live in SVM cl_mem from
 * embedding through lm_head; no host round-trip per layer. Uses the
 * existing nntrainer kernels (v8c FC, rmsnorm.cl, rotary_emb.cl,
 * two_conv_attention.cl) as a "kernel library" — not the layer runtime.
 *
 * Scope of the first commit: skeleton + weight-file mmap + CL context
 * init + single SVM round-trip smoke test. Layer-0 forward is the next
 * commit; full 36-layer + lm_head is the one after.
 *
 * The acceptance criterion for "paper-level" is NOT bit-equal to the
 * CPU baseline. It is: the GPU chain produces coherent output (readable
 * text) under one consistent numerics regime (paper §3.2/§3.6).
 */

#ifndef __QWEN3_FORWARD_H__
#define __QWEN3_FORWARD_H__

#include <CL/cl.h>
#include <cl_tensor_view.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace causallm_gpu {

struct Qwen3Config {
  unsigned int hidden_size;       // 2560 for 4B
  unsigned int intermediate_size; // 9728 for 4B
  unsigned int head_dim;          // 128
  unsigned int num_heads_Q;       // 32
  unsigned int num_heads_KV;      // 8 (GQA=4)
  unsigned int num_layers;        // 36
  unsigned int vocab_size;        // 151936
  unsigned int max_seq_len;       // 20480 (from nntr_config)
  float rms_norm_eps;             // 1e-6
  float rope_theta;               // 1e6
  // --- Gemma2 generalization (#63). Default 0/values => Qwen3 behavior. ---
  bool is_gemma2 = false;         // sandwich norm (4/layer), GeGLU, no q/k-norm,
                                  //   embed*sqrt(H), logit/score soft-cap
  float embed_scale = 0.0f;       // >0 => multiply embedding by this (sqrt(H));
                                  //   0 => no scale (Qwen3)
  float attn_logit_softcap = 0.0f;  // >0 => cap QK scores: c*tanh(s/c) (Gemma2 50)
  float final_logit_softcap = 0.0f; // >0 => cap lm_head logits (Gemma2 30)
  unsigned int sliding_window = 0;  // 0 => full causal (Gemma2 4096; no-op at M<=window)
  // --- #90 explicit architecture FEATURE flags. Dispatch sites in the forward
  // read THESE (not is_gemma2), so independent feature combos (e.g. SwiGLU +
  // sandwich norm) generalize without new identity branches. derive_features()
  // fills them from is_gemma2 — a bridge until the .bin self-describes. ---
  bool sandwich_norm = false;  // 4 norms/layer (pre+post attn, pre+post ffn) vs 2 pre
  bool mlp_geglu = false;      // GeGLU(gelu_tanh) MLP activation vs SwiGLU(silu)
  bool has_qk_norm = true;     // per-head q/k RMSNorm applied before RoPE

  // Fill the feature flags above from coarse model identity. Bridge step until
  // the weight file carries an explicit config header (then this is skipped and
  // the loader sets the flags directly). Idempotent; call once after cfg build.
  void derive_features() {
    sandwich_norm = is_gemma2;   // Gemma2: 4 sandwich norms; Qwen3: 2 pre-norms
    mlp_geglu = is_gemma2;       // Gemma2: GeGLU(tanh); Qwen3: SwiGLU
    has_qk_norm = !is_gemma2;    // Qwen3: q/k-norm before RoPE; Gemma2: none
  }
};

/// Per-layer weight handles (SVM cl_mem). All allocated once at init,
/// freed in the destructor. Layout: see the .cpp comments at the load
/// site for the exact tensor save order.
struct LayerWeights {
  cl_mem attention_norm_gamma = nullptr; // [hidden] fp32
  cl_mem wq_packed = nullptr;            // QINT4 [hidden, hQ*d]
  cl_mem wq_scales = nullptr;            // fp16 per-channel
  cl_mem wk_packed = nullptr;            // QINT4 [hidden, hKV*d]
  cl_mem wk_scales = nullptr;
  cl_mem wv_packed = nullptr;            // QINT4 [hidden, hKV*d]
  cl_mem wv_scales = nullptr;
  cl_mem wo_packed = nullptr;            // QINT4 [hQ*d, hidden]
  cl_mem wo_scales = nullptr;
  cl_mem ffn_norm_gamma = nullptr;       // [hidden] fp32
  cl_mem wgate_packed = nullptr;         // QINT4 [hidden, intermediate]
  cl_mem wgate_scales = nullptr;
  cl_mem wup_packed = nullptr;           // QINT4 [hidden, intermediate]
  cl_mem wup_scales = nullptr;
  cl_mem wdown_packed = nullptr;         // QINT4 [intermediate, hidden]
  cl_mem wdown_scales = nullptr;
};

class Qwen3Forward {
public:
  Qwen3Forward();
  ~Qwen3Forward();

  Qwen3Forward(const Qwen3Forward &) = delete;
  Qwen3Forward &operator=(const Qwen3Forward &) = delete;

  /// Initialize OpenCL context (via nntrainer's ClContext) + mmap weight
  /// file + read header. Does NOT load weights to GPU yet.
  /// Returns true on success.
  bool init(const Qwen3Config &cfg, const std::string &weight_path);

  /// Smoke test: allocate one SVM cl_mem of `bytes` size, write a
  /// known pattern, read it back, verify, free. Returns true if all
  /// CL calls succeed and the data round-trips correctly.
  bool svm_smoke_test(size_t bytes);

  /// Print the first `n` bytes of the mmap'd weight file as hex.
  void dump_weight_header(size_t n);

  /// Load layer 0's attention_norm gamma (fp32, [hidden_size]) into an
  /// SVM cl_mem. Computes the offset manually: embedding tensor is
  /// first in the weight file (Q6_K, [vocab, hidden] = vocab*hidden/256
  /// blocks * 210 bytes/block), then layer 0 attention_norm gamma. The
  /// allocated SVM pointer is owned by the class and freed in the
  /// destructor.
  bool load_layer0_attention_norm_to_svm();

  /// Dispatch rmsnorm.cl on a known input pattern using the previously
  /// loaded attention_norm gamma. Prints a summary of input + output
  /// (first/last few values) and a quick sanity check on the rms norm
  /// math. Returns true if the kernel ran and produced finite output.
  bool run_rmsnorm_layer0();

  /// Load layer 0's QKV projection weights (wq, wk, wv) — all QINT4 in
  /// KAI Section A format, sharing a common K=hidden input dim — from
  /// the mmap'd weight file. Each weight gets a v8c-ready backing
  /// (cl_mem buffer + image2d view + per-channel fp32 scales + per-
  /// channel int32 weight row-sums) via
  /// make_v8c_weight_backing_from_kai_section_a. Skips over the
  /// q_norm and k_norm gamma slabs interleaved between wq/wk/wv per
  /// the Qwen3 layer save order (load helpers for those come in the
  /// next commit).
  bool load_layer0_qkv_weights();

  /// Load layer 0's attention-output projection wo (QINT4, K=hQ*d,
  /// N=hidden). Sits right after wv in the save order. The wv slab
  /// has the same per-tensor format as wq/wk so the offset
  /// calculation just chains off load_layer0_qkv_weights's layout.
  bool load_layer0_wo();

  /// Load layer 0's FFN block weights: ffn_norm gamma (fp32 [hidden]
  /// pushed to SVM), ffn_up + ffn_gate (QINT4 K=hidden, N=intermediate),
  /// ffn_down (QINT4 K=intermediate, N=hidden). Save order (per
  /// transformer.cpp::createMlp): ffn_up, ffn_gate, ffn_down — and
  /// they sit right after the ffn_norm gamma which itself sits right
  /// after wo + decoder_add (no weights).
  bool load_layer0_ffn_weights();

  /// Run the FFN block on the post-attention residual:
  ///   ffn_norm.cl -> quantize_act -> v8c(ffn_up), v8c(ffn_gate)
  ///   -> swiglu -> quantize_act -> v8c(ffn_down) -> add residual_1
  ///   -> layer0_output.
  /// Requires layer0_residual1_fp32_ to be populated (run after
  /// run_layer0_qkv_projection in step 6a). Prints summary stats.
  bool run_layer0_ffn();

  /// Load weights for one decoder layer at the given file offset.
  /// Advances offset_inout past the layer's bytes. Populates
  /// layers_[layer_id]. Also allocates per-layer KV cache (SVM,
  /// sized for max_seq_len_used positions).
  bool load_layer(unsigned int layer_id, size_t *offset_inout,
                  unsigned int max_seq_len_used);

  /// Generic per-layer forward used for chaining. Takes a cl_mem
  /// fp32 [hidden] input buffer, returns a newly-allocated cl_mem
  /// fp32 [hidden] output buffer (the post-layer residual stream).
  /// Caller owns the output. Position is the RoPE / KV cache
  /// position for the single token being processed (single-token
  /// forward only for now — prefill comes later).
  /// (Original implementation; per-call alloc/release of ~32 scratch
  /// buffers per layer. forward_one_layer_v2 below replaces this
  /// with persistent scratch — task #44 / Phase A #1.)
  cl_mem forward_one_layer(unsigned int layer_id, cl_mem in_fp32,
                           unsigned int position);

  /// Per-stage timing breakdown for profiling. Each field accumulates
  /// total ms across all calls. Reset before a measurement window.
  /// Profiling adds a clFinish at each stage boundary — adds overhead
  /// but makes per-stage attribution clean.
  struct ForwardTimings {
    double pad_attn_norm_ms = 0;     // (a) input pad + attn_norm
    double qkv_quant_image_ms = 0;   // (b) act quant + image2d view
    double qkv_gemm_ms = 0;          // (c) Q/K/V v8c GEMMs
    double qk_norm_rope_ms = 0;      // (d) q_norm/k_norm + RoPE (M=1 only)
    double kv_write_ms = 0;          // (e) KV cache SVM bridge
    double attn_dispatch_ms = 0;     // (f) Q SVM upload + attention call
    double wo_ms = 0;                // (g) wo: cvt+quant+image+GEMM+cvt+add
    double ffn_ms = 0;               // (h) ffn block (full)
    int    calls = 0;
    // ALWAYS-ON host-blocking bridge timers (NOT gated by profile_stages_).
    // Accumulated across all layer calls of a forward window. These measure
    // the host wall-clock stalls of the SVM<->cl_mem bridges (clEnqueueMap
    // Buffer(CL_TRUE)+memcpy+SVM map/unmap, which block the host). Used to
    // quantify the host-overhead "prize" recoverable by bridge elimination.
    double host_kv_ms = 0;     // K scatter CPU map path + V write bridge
    double host_q_ms = 0;      // Q SVM-upload bridge (+ its trailing clFinish)
    double host_copy_svm_ms = 0; // o_svm->o_fp32 copy_svm dispatch host time
    void reset() { *this = ForwardTimings{}; }
  };
  ForwardTimings timings_{};
  bool profile_stages_ = false; // set true to enable per-stage timing
  // ML Drift reaudit (2026-06-12): residual chaining. When a non-last layer
  // writes residual_2 straight into scratch_.in_padded (instead of the
  // caller's out_fp32), the next layer's pad-fill + 9.4 MB CopyBuffer is
  // skipped. Set at layer end, consumed (cleared) at the next layer start;
  // layer 0 and the last layer never use it, so a fresh chain always
  // re-stages from the caller's input.
  bool chain_in_padded_valid_ = false;

  /// Same as forward_one_layer but uses persistent `scratch_` for
  /// all intermediate cl_mems (no per-call alloc/release). Caller-
  /// managed output: pass a cl_mem fp32 [M*hidden] out_fp32 (typically
  /// ping-pong via two persistent buffers in the chain loop). Returns
  /// true on success.
  ///
  /// `M` = number of token rows in the input. M=1 is decode/single
  /// token; M>1 is prefill. Scratch must already be sized for at
  /// least this M via ensure_forward_scratch_allocated(M). Image2d
  /// views are recreated per call (cheap; backing buffers persist).
  ///
  /// LIMITATION (task #45b): RoPE is currently applied with a single
  /// position (whatever was set by precompute_rope_for_position) to
  /// all M token rows. Correct for M=1 (decode); for M>1 (prefill)
  /// this gives bogus output because each token row needs its OWN
  /// (start_pos + i) rotation. Pipeline still runs end-to-end so
  /// perf measurements at any M are valid; output correctness for
  /// prefill needs a batched RoPE kernel.
  bool forward_one_layer_v2(unsigned int layer_id, cl_mem in_fp32,
                            cl_mem out_fp32, unsigned int position,
                            unsigned int M = 1);

  /// Load the model's final output_norm gamma (fp32 [hidden]) from
  /// the tail of the weight file. Caller passes the file offset
  /// right after the last layer's bytes; this method reads the gamma
  /// and pushes it to SVM. Returns true on success.
  bool load_output_norm(size_t file_offset);

  /// Apply the output_norm (rmsnorm.cl) to a cl_mem fp32 [hidden]
  /// input, in place. Caller's buffer is modified.
  bool run_output_norm(cl_mem inout_fp32);

  /// CPU-side embedding lookup: dequant the Q6_K row at token_id from
  /// the mmap'd embedding table directly to an fp32 cl_mem [hidden]
  /// buffer ready as layer-0 input. Returns the new cl_mem (caller
  /// owns). nullptr on failure (out-of-range token_id, alloc failure).
  cl_mem embedding_lookup_to_fp32_clmem(unsigned int token_id);

  /// CPU-side lm_head: read post-output_norm fp32 [hidden] from the
  /// cl_mem back to host, then for each vocab row dequant the Q6_K
  /// row from the tied embedding table, compute the dot product with
  /// the hidden, and finally argmax. Returns the predicted next-token
  /// id (or -1 on failure). Q6_K kernel is CPU-only in nntrainer, so
  /// this matmul stays on host for now — a GPU Q6_K kernel is a
  /// separate kernel-design follow-up.
  int run_lm_head_and_argmax_cpu(cl_mem post_norm_fp32);

  /// ML Drift reaudit #1 (decode): GPU Q6_K GEMV lm_head. Lazily uploads the
  /// 484 MB Q6_K embedding table to a device buffer on first call, runs one
  /// 64-WI workgroup per vocab row (exact dequantize_row_q6_K math, fp32
  /// accumulation order differs -> logit lsb drift only), reads the fp32
  /// logits back (1 MB) and argmaxes on host. Returns -1 on any setup
  /// failure (caller falls back to the CPU path).
  int run_lm_head_and_argmax_gpu(cl_mem post_norm_fp32);

  /// Dispatcher: GPU path unless NNTR_LMHEAD_GPU=0, NNTR_DUMP_LOGITS is set
  /// (CPU path owns the dump), or the GPU path fails.
  int run_lm_head_and_argmax(cl_mem post_norm_fp32);

  /// Load layer 0's q_norm and k_norm gammas (fp32, [head_dim]) from
  /// the mmap'd weights. They sit between wq/wk and wk/wv in the
  /// save order. Pushed into SVM as fp16 (the rmsnorm_cl_fp16 kernel
  /// expects `half alpha`) — converted on the host side at load time.
  bool load_layer0_qk_norm_gammas();

  /// Precompute the RoPE cos/sin tables for one position into SVM
  /// (fp16, [head_dim] each) using the cfg's rope_theta. Position 0
  /// gives identity rotation (cos=1, sin=0); pass a non-zero position
  /// to actually exercise the rotation math. Reuses existing SVM
  /// allocations if called multiple times (always rewrites the table).
  bool precompute_rope_for_position(unsigned int position);

  /// Dispatch the inline `rope_fp16` kernel on the layer-0 Q and K
  /// fp16 cl_mem buffers in place. Must be called after RoPE freqs
  /// are precomputed for the same position. Hooked into the QKV
  /// pipeline by run_layer0_qkv_projection when rope position is
  /// preconfigured; otherwise skipped.
  bool run_layer0_rope_on_qk(cl_mem y_q, cl_mem y_k);

  /// Path 4 / #45b: precompute the full RoPE cos/sin LUT once for
  /// positions [0, max_positions). Idempotent — no-op if a table of
  /// at least the requested size already exists. Stored in SVM
  /// (rope_cos_full_svm_, rope_sin_full_svm_).
  bool precompute_rope_full_lut(unsigned int max_positions);

  /// Dispatch the batched RoPE kernel on a [M, num_heads * head_dim]
  /// fp16 cl_mem in place. Applies rotation for positions
  /// [start_pos, start_pos + M). Requires precompute_rope_full_lut
  /// to have been called with max_positions >= start_pos + M.
  bool dispatch_rope_batched(cl_mem io, unsigned int M,
                             unsigned int num_heads, unsigned int start_pos);

  /// Allocate layer 0's K and V caches as SVM cl_mem buffers, sized
  /// for the full max_seq_len. Both zeroed at allocation so any
  /// position we haven't written reads as 0 (causal-mask-friendly).
  /// Cache layout (concat / row-major): [max_seq_len, hKV * head_dim]
  /// fp16 — same as the existing CausalLM KVCacheManager for the
  /// non-OHWI path. SVM-allocated so the attention kernel can read
  /// them with svm_inputs=true (no upload).
  bool allocate_layer0_kv_cache_svm();

  /// Run the full QKV projection pipeline for layer 0 on the same
  /// deterministic input pattern as step 2/3: rmsnorm.cl ONCE against
  /// the SVM attention_norm gamma, then quantize_act_v8c_fp32_cl ONCE
  /// on the rmsnorm output (shared activation quant — paper §3.6
  /// insight, same pattern as dotCl_v8c's shared-quant cache hit on
  /// wq/wk/wv), then three gemm_int8_v8c_cl dispatches against the
  /// loaded Q/K/V weights. Then rmsnorm_cl_fp16 over each head row of
  /// Q (H=hQ, W=d) and K (H=hKV, W=d) using the SVM q_norm/k_norm
  /// gammas — paper §3.3 Qwen3-specific per-head normalization.
  /// Outputs three fp16 cl_mem buffers (Q[hQ*d], K[hKV*d], V[hKV*d])
  /// — Q/K are post-norm and ready for RoPE in the next commit;
  /// V is unchanged (Qwen3 has no v_norm).
  bool run_layer0_qkv_projection();

  const Qwen3Config &config() const { return cfg_; }
  size_t weight_file_size() const { return weight_bytes_; }

  /// Byte offset in the weight file where decoder layer 0 starts —
  /// i.e. right after the embedding (Q6_K) blob. Useful for chaining
  /// load_layer(0..L-1, &offset, ...).
  size_t layers_start_offset() const;

  /// Accessor for the layer-0 output cl_mem produced by the old path
  /// (run_layer0_ffn). Available so step-7a's new chain (which uses
  /// the generic load_layer + forward_one_layer path) can take layer
  /// 0's output as its starting input.
  cl_mem layer0_output_fp32() const { return layer0_output_fp32_; }

  /// Allocate (or grow) scratch_ to support up to `max_M` token rows.
  /// Idempotent if current allocation already covers max_M; otherwise
  /// frees and re-allocates. Public so main can pre-warm to the max M
  /// any subsequent prefill/decode call will use (avoids realloc on
  /// the timed path).
  bool ensure_forward_scratch_allocated(unsigned int max_M);

private:
  /// Byte size of the embedding tensor on disk (Q6_K).
  size_t embed_table_bytes() const;

  Qwen3Config cfg_{};
  std::string weight_path_;
  const uint8_t *weight_mmap_ = nullptr; // mmap'd, read-only
  // GPU lm_head (Q6_K GEMV): lazily-created device copies. Intentionally
  // not released at teardown (leak-at-exit, same rationale as the
  // ClBufferManager dtor no-op — Adreno finalize-order segv).
  cl_mem lm_head_q6k_buf_ = nullptr;   // 484 MB Q6_K table
  cl_mem lm_head_logits_buf_ = nullptr; // [vocab] fp32
  cl_mem lm_head_bestv_buf_ = nullptr;  // [64] fp32 argmax stage-1 values
  cl_mem lm_head_besti_buf_ = nullptr;  // [64] int32 argmax stage-1 indices
  size_t weight_bytes_ = 0;
  int weight_fd_ = -1;

  cl_context cl_ctx_ = nullptr;        // borrowed from ClContext
  cl_command_queue cl_q_ = nullptr;    // borrowed from ClContext
  cl_device_id cl_dev_ = nullptr;      // borrowed
  // Device specialization (paper §3.4): image2d caps queried once at init and
  // consulted when building packed activation image views (increment 3).
  nntrainer::tv::DeviceImageCaps img_caps_{};

  // Layer 0 attention_norm gamma (fp32, [hidden_size]) in SVM.
  void *layer0_attn_norm_gamma_svm_ = nullptr;
  // Layer 0 q_norm / k_norm gammas (fp16, [head_dim]) in SVM. Converted
  // from the fp32 on-disk gammas at load time so they can be passed
  // directly to rmsnorm_cl_fp16 (which takes `half alpha`).
  void *layer0_q_norm_gamma_svm_fp16_ = nullptr;
  void *layer0_k_norm_gamma_svm_fp16_ = nullptr;
  // RoPE cos/sin tables for the currently-configured position
  // (fp16, [head_dim]) in SVM. Repopulated by
  // precompute_rope_for_position(). The doubled-half layout
  // matches the CPU mha_core path: cos[k+half] = cos[k], sin[k+half] =
  // sin[k] so the kernel can index head_dim values directly.
  void *layer0_rope_cos_svm_fp16_ = nullptr;
  void *layer0_rope_sin_svm_fp16_ = nullptr;
  int   layer0_rope_position_ = -1; // -1 = no position configured

  // Path 4 / #45b: full RoPE LUT [max_positions, half_d] in SVM.
  // Built once on first prefill via precompute_rope_full_lut(); the
  // batched rope_fp16_batched kernel indexes (start_pos + t) * half_d + k.
  // Session-wide (rope_theta + head_dim identical across layers).
  void *rope_cos_full_svm_ = nullptr;
  void *rope_sin_full_svm_ = nullptr;
  unsigned int rope_full_max_positions_ = 0;

  // Layer 0 K and V caches (SVM cl_mem), [max_seq_len * hKV * d] fp16
  // each. Concat row-major layout: cache[pos * hKV*d + hkv*d + k]. Both
  // zero-initialized at allocation so unwritten positions read as 0
  // (clean under the causal mask).
  void *layer0_cache_k_svm_ = nullptr;
  void *layer0_cache_v_svm_ = nullptr;

  /// Bundled v8c-ready state for one int4 GEMM weight. Built once by
  /// load_qint4_weight_at() from a mmap'd KAI Section A payload via
  /// make_v8c_weight_backing_from_kai_section_a. The TensorBacking is
  /// held opaquely so this header doesn't depend on cl_tensor_view.h;
  /// the dispatch site casts back to access the cl_mem image view.
  struct V8cFcWeight {
    void *backing = nullptr;       // tv::TensorBacking* (owning)
    cl_mem weight_image = nullptr; // image2d view (owned by backing)
    cl_mem weight_buf = nullptr;   // raw cl_mem backing buffer (NNTR_V8C_BUF path)
    cl_mem scale_buf = nullptr;    // fp32 [N]
    cl_mem row_sum_w_int4 = nullptr; // int32 [N] (= Σ int4, or Σ int8 if is_int8)
    unsigned int K = 0, N = 0;
    bool is_int8 = false;          // true = 8/4/4 int8 weight (qscheme tag 8)
  };

  // Layer 0 Q/K/V projection weights + attention output projection wo.
  V8cFcWeight layer0_wq_{};
  V8cFcWeight layer0_wk_{};
  V8cFcWeight layer0_wv_{};
  V8cFcWeight layer0_wo_{};
  // Layer 0 FFN block weights.
  V8cFcWeight layer0_ffn_up_{};
  V8cFcWeight layer0_ffn_gate_{};
  V8cFcWeight layer0_ffn_down_{};
  // ffn_norm gamma (fp32, [hidden]) in SVM.
  void *layer0_ffn_norm_gamma_svm_ = nullptr;

  // Persistent buffers for the layer-0 residual stream. Allocated lazily
  // by the forward step that needs them; lives for the layer-0 forward
  // pass (subsequent steps in this layer read them; freed in destructor).
  cl_mem layer0_residual1_fp32_ = nullptr; // [hidden] = x + wo(O)
  cl_mem layer0_output_fp32_    = nullptr; // [hidden] = residual_1 + ffn(.)

  /// Bundled per-layer weight state for the generic forward path.
  /// Populated by load_layer(); consumed by forward_one_layer().
  struct LayerWeights {
    void *attn_norm_gamma_svm = nullptr;     // fp32 [hidden]
    void *attn_norm_gamma_svm_fp16 = nullptr;// fp16 [hidden] (#46m)
    void *q_norm_gamma_svm_fp16 = nullptr;   // fp16 [head_dim]
    void *k_norm_gamma_svm_fp16 = nullptr;   // fp16 [head_dim]
    void *ffn_norm_gamma_svm = nullptr;      // fp32 [hidden] (Gemma2: pre_ffn)
    void *ffn_norm_gamma_svm_fp16 = nullptr; // fp16 [hidden] (#46m)
    // #63 Gemma2 sandwich norm: post-attention + post-FFN RMSNorm gammas,
    // applied to the SUBLAYER OUTPUT before the residual add. Unused on Qwen3.
    void *post_attn_norm_gamma_svm = nullptr;      // fp32 [hidden]
    void *post_attn_norm_gamma_svm_fp16 = nullptr; // fp16 [hidden]
    void *post_ffn_norm_gamma_svm = nullptr;       // fp32 [hidden]
    void *post_ffn_norm_gamma_svm_fp16 = nullptr;  // fp16 [hidden]
    V8cFcWeight wq, wk, wv, wo;
    V8cFcWeight ffn_up, ffn_gate, ffn_down;
    void *cache_k_svm = nullptr; // fp16 [max_seq_len_used * hKV * d]
    void *cache_v_svm = nullptr; // fp16 [max_seq_len_used * hKV * d]
    // §3.8 image2d V experiment (NNTR_OHWI_IMG=1): mirror V cache in
    // a regular cl_mem laid out OHWI-reversed [hKV, d, max_seq_len_used]
    // so image2d_from_buffer can wrap it for sv_matmul_f16_ohwi_img.
    // The image2d view is built once at layer load and reused every
    // forward call (so the 28 attention dispatches don't each pay
    // clCreateImage overhead).
    cl_mem cache_v_buf_ohwi = nullptr;
    size_t cache_v_buf_ohwi_bytes = 0;
    cl_mem cache_v_image_ohwi = nullptr;  // image2d_from_buffer over above
    // #46h: K image2d mirror. K is already OHWI (O=cache_size, I=d_h)
    // per paper §3.7; we mirror it in a regular cl_mem so image2d_from_
    // buffer can wrap it for qk_matmul_f16_ohwi_img.
    cl_mem cache_k_buf_ohwi = nullptr;
    size_t cache_k_buf_ohwi_bytes = 0;
    cl_mem cache_k_image_ohwi = nullptr;  // image2d_from_buffer over above
  };
  std::vector<LayerWeights> layers_;

  // Final output_norm gamma (fp32, [hidden]) in SVM.
  void *output_norm_gamma_svm_ = nullptr;
  void *output_norm_gamma_svm_fp16_ = nullptr;  // #46m

  // Per-layer KV cache stride (max_seq_len that was passed to
  // load_layer); needed by the OHWI K attention wrapper to compute
  // per-head offsets into the cache.
  unsigned int kv_cache_max_seq_len_ = 0;

  /// Per-forward scratch. All 28 layers use identical shapes (model
  /// constants), so a single set of these is reused across every
  /// layer of every forward call. Allocated once on first
  /// forward_one_layer call via `ensure_forward_scratch_allocated()`,
  /// released in destructor. Replaces the ~32 clCreateBuffer +
  /// clReleaseMemObject calls per layer per token that used to
  /// dominate per-layer wall time (see [[perf-analysis-2026-05-29]]
  /// Phase A #1).
  struct ForwardScratch {
    // (a) input pad + attn_normed: both [M_pad * hidden] fp32
    cl_mem in_padded = nullptr;
    cl_mem attn_normed = nullptr;
    // (b) QKV shared activation quant: [M_pad * hidden] int8 + scales
    cl_mem qkv_act_i8 = nullptr;
    cl_mem qkv_act_scale = nullptr;
    cl_mem qkv_act_zp = nullptr;
    cl_mem qkv_act_rs = nullptr;
    // (c) QKV outputs (fp16): y_q [M_pad*N_q], y_k/y_v [M_pad*N_kv]
    cl_mem y_q = nullptr;
    cl_mem y_k = nullptr;
    cl_mem y_v = nullptr;
    // (d) attention SVM bridge (fp16, [N_q] each)
    void *q_svm = nullptr;
    void *o_svm = nullptr;
    // (e) wo: O fp32 + act scratch + output + residual_1 (fp32 [K_h])
    cl_mem o_fp32 = nullptr;       // [M_pad * N_q]
    cl_mem wo_act_i8 = nullptr;    // [M_pad * N_q]
    cl_mem wo_act_scale = nullptr;
    cl_mem wo_act_zp = nullptr;
    cl_mem wo_act_rs = nullptr;
    cl_mem wo_y_fp16 = nullptr;    // [M_pad * K_h]
    cl_mem wo_fp32 = nullptr;      // [K_h]
    cl_mem residual_1 = nullptr;   // [K_h]
    // (f) ffn block: padded + normed + shared quant + up/gate + swiglu
    cl_mem ffn_in_padded = nullptr;  // [M_pad * K_h]
    cl_mem ffn_normed = nullptr;     // [M_pad * K_h]
    cl_mem fa_i8 = nullptr;          // [M_pad * K_h]
    cl_mem fa_sc = nullptr;
    cl_mem fa_zp = nullptr;
    cl_mem fa_rs = nullptr;
    cl_mem up_fp16 = nullptr;        // [M_pad * I]
    cl_mem gate_fp16 = nullptr;      // [M_pad * I]
    cl_mem up_fp32 = nullptr;        // [I]
    cl_mem gate_fp32 = nullptr;      // [I]
    cl_mem swiglu_out = nullptr;     // [M_pad * I]
    // (g) ffn_down: act scratch + output
    cl_mem dn_i8 = nullptr;          // [M_pad * I]
    cl_mem dn_sc = nullptr;
    cl_mem dn_zp = nullptr;
    cl_mem dn_rs = nullptr;
    cl_mem dn_fp16 = nullptr;        // [M_pad * K_h]
    cl_mem dn_fp32 = nullptr;        // [K_h]
    // Increment 2 (tensor-virtualization): cached image2d views over the
    // four int8 activation buffers above. Created once with the buffers
    // (sized for max M_pad) and reused every layer/every forward, replacing
    // the per-layer clCreateImage+release. Reads only touch the valid
    // [0, current M_pad) rows, so a max-height view is safe for any M<=max.
    cl_mem qkv_act_img = nullptr;    // view of qkv_act_i8 (row=K_h)
    cl_mem wo_act_img = nullptr;     // view of wo_act_i8  (row=N_q)
    cl_mem fa_act_img = nullptr;     // view of fa_i8      (row=K_h)
    cl_mem dn_act_img = nullptr;     // view of dn_i8      (row=I)
  };
  ForwardScratch scratch_{};

  /// The max_M scratch is currently sized for. 0 = unallocated.
  unsigned int scratch_max_M_ = 0;

  /// On-disk FC record size at the given offset — peeks the u16 qscheme
  /// tag: 8 = int8 (K*N + fp16 scales), 1 = shared plain container
  /// (PR#3978: plain nibbles + fp32 scales + KAI nr=8 pad), else legacy
  /// KAI Section A (K*N/2 + fp16 scales).
  size_t qint4_record_bytes(size_t file_offset, unsigned int K,
                            unsigned int N) const;

  /// Generic loader: parses one Int4QTensor blob at the given file
  /// offset ([qscheme u16][payload]; Section A / int8 / plain container —
  /// plain is repacked to Section A on load) and populates `out`.
  /// Returns false on shape / qscheme validation failure.
  bool load_qint4_weight_at(size_t file_offset, unsigned int K,
                            unsigned int N, V8cFcWeight *out,
                            const char *tag);

  /// Release a V8cFcWeight (used by destructor).
  void release_v8c_weight(V8cFcWeight *w);
};

} // namespace causallm_gpu

#endif // __QWEN3_FORWARD_H__
