// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qwen3_5_moe_causallm.cpp
 * @brief  Qwen3.6-35B-A3B (Qwen3-Next, qwen3_5_moe) GDN-hybrid causal LM.
 */

#include <cmath>
#include <qwen3_5_moe_causallm.h>

#include <app_context.h>
#include <broadcast_mul_layer.h>
#include <engine.h>
#include <gated_delta_net_layer.h>
#include <llm_util.hpp>
#include <model.h>
#include <qwen_moe_layer.h>

namespace causallm {

void Qwen3_5MoeCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  // base + MoE (num_experts, num_experts_per_tok, INTERMEDIATE_SIZE=moe_*) are
  // parsed by the Qwen3MoECausalLM base subobject ctor; here we add the GDN /
  // hybrid / shared-expert fields and re-read the MoE counts into own members.
  try {
    N_EXPERTS = cfg["num_experts"];
    N_EXPERTS_PER_TOK = cfg["num_experts_per_tok"];
    SHARED_EXPERT_INTERMEDIATE_SIZE = cfg["shared_expert_intermediate_size"];
    LINEAR_NUM_VALUE_HEADS = cfg["linear_num_value_heads"];
    LINEAR_NUM_KEY_HEADS = cfg["linear_num_key_heads"];
    LINEAR_KEY_HEAD_DIM = cfg["linear_key_head_dim"];
    LINEAR_VALUE_HEAD_DIM = cfg["linear_value_head_dim"];
    LINEAR_CONV_KERNEL_DIM = cfg["linear_conv_kernel_dim"];
    // gdnq bin variant: in_proj_qkv stored QS4CX. Optional key; absent on
    // the standard bin.
    GDN_QKV_PACKED = cfg.contains("gdn_qkv_dtype") &&
                     cfg["gdn_qkv_dtype"] == "QINT4";
  } catch (const std::exception &e) {
    throw std::runtime_error(
      std::string("Qwen3_5Moe: missing MoE/GDN config field: ") + e.what());
  }

  // partial RoPE factor (rope_parameters.partial_rotary_factor), default 1.0
  PARTIAL_ROTARY_FACTOR = 1.0f;
  if (cfg.contains("rope_parameters") &&
      cfg["rope_parameters"].contains("partial_rotary_factor"))
    PARTIAL_ROTARY_FACTOR = cfg["rope_parameters"]["partial_rotary_factor"];
  else if (cfg.contains("partial_rotary_factor"))
    PARTIAL_ROTARY_FACTOR = cfg["partial_rotary_factor"];

  // per-layer types (3:1 GDN:full). Synthesize from full_attention_interval if
  // the explicit list is absent.
  LAYER_TYPES.clear();
  if (cfg.contains("layer_types")) {
    for (const auto &t : cfg["layer_types"])
      LAYER_TYPES.push_back(t.get<std::string>());
  } else {
    int interval = cfg.contains("full_attention_interval")
                     ? (int)cfg["full_attention_interval"]
                     : 4;
    for (unsigned int i = 0; i < NUM_LAYERS; ++i)
      LAYER_TYPES.push_back(((i + 1) % interval == 0) ? "full_attention"
                                                      : "linear_attention");
  }
  if (LAYER_TYPES.size() != NUM_LAYERS)
    throw std::runtime_error("Qwen3_5Moe: layer_types size != num_hidden_layers");
}

Tensor Qwen3_5MoeCausalLM::createGatedDeltaNet(const int layer_id,
                                               Tensor input) {
  // The GDN mixer ingests the (already input-normed) hidden directly and emits
  // the hidden-dim mixed output (includes its own out_proj). Name matches the
  // attention slot so the base decoder block's residual add wires correctly.
  // packed=false: GDN weights follow the ACTIVATION dtype (FP16 on the QINT4-
  // FP16 35B), not the QINT4 weight half — A_log/dt_bias/norm dims violate the
  // QINT4 N%4/K%32 constraints and the GDN compute has no QINT4 kernels.
  LayerHandle gdn(createLayer(
    "gated_delta_net",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("packed", "false"),
     withKey("linear_num_value_heads", LINEAR_NUM_VALUE_HEADS),
     withKey("linear_num_key_heads", LINEAR_NUM_KEY_HEADS),
     withKey("linear_key_head_dim", LINEAR_KEY_HEAD_DIM),
     withKey("linear_value_head_dim", LINEAR_VALUE_HEAD_DIM),
     withKey("linear_conv_kernel_dim", LINEAR_CONV_KERNEL_DIM),
     withKey("gdn_qkv_packed", GDN_QKV_PACKED ? "true" : "false")}));
  return gdn(input);
}

Tensor Qwen3_5MoeCausalLM::createFullAttention(const int layer_id, int n_heads,
                                               int head_dim, Tensor query,
                                               Tensor key, Tensor value) {
  // Q projection (Q half of the HF q_proj; the gate half is the separate
  // w_gate below — the converter de-interleaves HF [.,nH,2*head_dim]).
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor q = wq(query);

  std::vector<std::string> q_norm_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", causallm_engine())};
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_params));
  Tensor q_normed = q_norm(q);

  // Output-gate projection (the gate half of HF q_proj), head-major to match
  // the attention output width n_heads*head_dim.
  LayerHandle wgate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_w_gate"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor gate = wgate(query);

  // K / V projections (GQA)
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor k = wk(key);

  std::vector<std::string> k_norm_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", causallm_engine())};
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_params));
  Tensor k_normed = k_norm(k);

  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor v = wv(value);

  // (The retired lane had an NNTR_KV_INT8 3-input variant here. This base has
  // no int8 KV plane, and int8 KV produced NaN for head_dim16 + partial RoPE
  // even there, so the 5-input form is the only one.)
  // Attention core — partial RoPE via rope_partial_rotary_factor (proportional),
  // already validated on gemma4. mha_core returns the UNGATED attention output.
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", SLIDING_WINDOW),
     // HF qwen3_5_moe partial RoPE: 'default_partial' uses the rotary_dim
     // (head_dim*factor) freq denominator (vs 'proportional' = head_dim, Gemma).
     // The converter also permutes q/k head-dim rows so nntrainer's split-half
     // pairing reproduces HF's contiguous pairing -> bit-exact full-attn.
     withKey("rope_theta", ROPE_THETA),
     withKey("rope_scaling_type", "default_partial"),
     withKey("rope_partial_rotary_factor", std::to_string(PARTIAL_ROTARY_FACTOR)),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("use_gemm_attention", USE_FLASH_ATTENTION ? "true" : "false")}));
  // NB: the retired lane also passed gpu_decode_attn / gpu_decode_rope /
  // gpu_ohwi_rope here. This base has no such properties, so passing them
  // leaves them in remain_props and mha_core throws "Unknown Layer Properties"
  // at graph build. All three were "false" (the defaults) anyway.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});

  // Output gate: a *= sigmoid(gate)
  // NB: replacing this activation+multiply pair with one sigmoid_glu node
  // ABORTS WEIGHT LOADING ("unsupported legacy on-disk qscheme") -- the
  // on-disk record stream is sensitive to the node enumeration even for
  // weightless nodes. The device win lives in MultiplyLayer's forward
  // instead (device eltwise mul, graph untouched).
  LayerHandle gate_sig(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attn_gate_sig"),
     withKey("activation", "sigmoid"),
     withKey("engine", causallm_engine())}));
  Tensor gate_act = gate_sig(gate);

  LayerHandle gate_mul(createLayer(
    "multiply",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attn_gate_mul")}));
  Tensor a_gated = gate_mul({a, gate_act});

  // O projection
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  return wo(a_gated);
}

Tensor Qwen3_5MoeCausalLM::createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           Tensor query, Tensor key,
                                           Tensor value) {
  if (LAYER_TYPES[layer_id] == "linear_attention")
    return createGatedDeltaNet(layer_id, query); // query == input-normed hidden
  return createFullAttention(layer_id, n_heads, head_dim, query, key, value);
}

Tensor Qwen3_5MoeCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                     Tensor input) {
  // Routed top-k experts (reuse qwen_moe). hidden_dim == moe_intermediate_size.
  LayerHandle moe(createLayer(
    "qwen_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", hidden_dim), withKey("num_experts", N_EXPERTS),
     withKey("num_experts_per_token", N_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish"),
     withKey("engine", causallm_engine())}));
  Tensor routed = moe(input);

  // Always-on shared expert: SwiGLU(shared_intermediate) gated by sigmoid(W·x).
  const int si = (int)SHARED_EXPERT_INTERMEDIATE_SIZE;
  LayerHandle sh_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_shared_gate"),
     withKey("unit", si), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor sg = sh_gate(input);

  LayerHandle sh_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_shared_up"),
     withKey("unit", si), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor su = sh_up(input);

  // The `engine` stamp is NOT optional: without it getComputeEngineType()
  // returns "cpu", network_graph binds CPU ContextData, and CudaComputeOps::
  // swiglu is never entered even though both operands are UVM and the CUDA
  // kernel exists. sh_up above and sh_down below both carry it; this node was
  // the odd one out, and it cost 2,991 ms of a 45,260 ms 20K prefill.
  LayerHandle sh_swiglu(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_shared_swiglu"),
     withKey("engine", causallm_engine())}));
  Tensor sact = sh_swiglu({sg, su});

  LayerHandle sh_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_shared_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor sd = sh_down(sact);

  // shared expert gate: sigmoid(Linear(hidden -> 1)) broadcast over hidden via
  // the custom broadcast_mul (core 'multiply' requires identical input dims in
  // incremental_forwarding). Weight is the HF [1,hidden] gate transposed.
  // packed=false: the [hidden,1] gate weight follows the activation dtype
  // (unit=1 violates the QINT4 N%4 packing constraint).
  LayerHandle sh_gate_lin(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_shared_gate_lin"),
     withKey("unit", 1), withKey("disable_bias", "true"),
     withKey("packed", "false"),
     withKey("weight_initializer", "ones"), withKey("engine", causallm_engine())}));
  Tensor gl = sh_gate_lin(input);

  LayerHandle sh_gate_sig(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_shared_gate_sig"),
     withKey("activation", "sigmoid"),
     withKey("engine", causallm_engine())}));
  Tensor gsig = sh_gate_sig(gl);

  LayerHandle sh_mul(createLayer(
    "broadcast_mul",
    {withKey("name", "layer" + std::to_string(layer_id) + "_shared_mul")}));
  Tensor shared_gated = sh_mul({sd, gsig});

  LayerHandle add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_add")}));
  return add({routed, shared_gated});
}

void Qwen3_5MoeCausalLM::registerCustomLayers() {
  Qwen3MoECausalLM::registerCustomLayers(); // base + ReshapedRMSNorm + MoELayer

  // Register on EVERY context the Engine brought up -- NOT on "cpu" only.
  // createLayer() resolves the factory from the node's engine= stamp, so a
  // cpu-only registration makes an engine=cuda node throw "Key is not found
  // for the object" at graph construction. (The retired lane got away with a
  // cpu-only registerFactory because it never stamped these nodes.) This
  // mirrors Transformer::registerCustomLayers' registerOnEveryContext, which
  // is file-local there and so cannot be reused from here.
  //
  // One try/catch PER CONTEXT on purpose: a single try around several
  // registrations is how one silently disappears, and a missing factory only
  // surfaces much later as a graph-build failure. A duplicate-key throw from a
  // second model instance in the same process is benign and expected.
  const auto &ct_engine = nntrainer::Engine::Global();
  auto registerEverywhere = [&ct_engine](const std::string &type_name,
                                       auto factory) {
    for (const auto &ctx : ct_engine.getRegisteredContextNames()) {
      try {
        ct_engine.registerLayerFactory(ctx, factory);
      } catch (std::invalid_argument &e) {
        ml_logw("%s registration on the %s context was refused: %s",
                type_name.c_str(), ctx.c_str(), e.what());
      }
    }
  };

  registerEverywhere(causallm::GatedDeltaNetLayer::type,
                     nntrainer::createLayer<causallm::GatedDeltaNetLayer>);
  registerEverywhere(causallm::BroadcastMulLayer::type,
                     nntrainer::createLayer<causallm::BroadcastMulLayer>);
  // MoELayer too: the base registers it on "cpu" only, which is why the
  // qwen_moe node could not carry an engine= stamp. Without the stamp
  // residencyEngine() is CPU, configureRunContext re-stamps the node's INPUT
  // with the cpu ContextData (downgrading the producer's cuda stamp), and
  // getOps() on that input resolves to the host table -- so the expert GEMMs
  // could never reach CudaComputeOps::fc's QS4CX chain.
  registerEverywhere(causallm::MoELayer::type,
                     nntrainer::createLayer<causallm::MoELayer>);
}

} // namespace causallm
