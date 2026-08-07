// SPDX-License-Identifier: Apache-2.0
/**
 * e2e logit test: construct Qwen3_5MoeCausalLM (tiny [G,A]), initialize(),
 * load_weight, run PREFILL on the P3c input_ids, and compare logits to the HF
 * goldens (e2e_bin_GA/logits.bin). Run with NNTR_KV_INT8=1 + NNTR_ENGINE=cpu so
 * the full-attn layer uses the 3-input internal cache (no external KV binding,
 * which the GDN layer 0 has no placeholder for).
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <json.hpp>
#include <neuralnet.h>
#include <qwen3_5_moe_causallm.h>

using json = nlohmann::json;

static std::vector<float> loadBin(const std::string &p) {
  std::ifstream f(p, std::ios::in | std::ios::binary | std::ios::ate);
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), n);
  return v;
}

// expose protected prefill + KV setup
struct RunModel : causallm::Qwen3_5MoeCausalLM {
  RunModel(json &c, json &g, json &n) :
    causallm::Transformer(c, g, n, causallm::ModelType::CAUSALLM),
    causallm::Qwen3_5MoeCausalLM(c, g, n) {}
  std::vector<float> prefill(std::vector<float> ids, int vocab) {
    const int T = (int)ids.size();
    // input0 is [1,1,1,INIT_SEQ_LEN]; feed a MAX_SEQ_LEN-sized buffer (as run()
    // does) with the ids in [0,T) and the rest zeroed, else OOB reads garbage.
    std::vector<float> buf((size_t)MAX_SEQ_LEN, 0.0f);
    for (int i = 0; i < T; ++i)
      buf[i] = ids[i];
    allocateAndBindKVCache(); // FP external cache; GDN layers skipped
    setKVCachePosition(0);
    // FP-cache path: model inputs = [input0, cache_k_l1, cache_v_l1] (only the
    // full-attn layer emits placeholders). Feed the bound slab pointers so
    // incrementalInference shares their MemoryData.
    std::vector<float *> input = {
      buf.data(),
      reinterpret_cast<float *>(kv_cache.getKeyCache(1).getData()),
      reinterpret_cast<float *>(kv_cache.getValueCache(1).getData())};
    auto out = incrementalInference(BATCH_SIZE, input, T, 0, T);
    // prefill returns the LAST token's logits [vocab] (what generate() consumes)
    std::vector<float> logits(out[0], out[0] + vocab);
    return logits;
  }
};

int main() {
  const int VOCAB = 64;
  json cfg;
  cfg["architectures"] = {"Qwen3_5MoeForCausalLM"};
  cfg["model_type"] = "qwen3_5_moe";
  cfg["vocab_size"] = VOCAB; cfg["hidden_size"] = 32;
  cfg["num_hidden_layers"] = 2; cfg["num_attention_heads"] = 4;
  cfg["head_dim"] = 16; cfg["num_key_value_heads"] = 2;
  cfg["intermediate_size"] = 16; cfg["moe_intermediate_size"] = 16;
  cfg["shared_expert_intermediate_size"] = 16; cfg["num_experts"] = 8;
  cfg["num_experts_per_tok"] = 2; cfg["linear_num_value_heads"] = 4;
  cfg["linear_num_key_heads"] = 2; cfg["linear_key_head_dim"] = 8;
  cfg["linear_value_head_dim"] = 8; cfg["linear_conv_kernel_dim"] = 4;
  cfg["rms_norm_eps"] = 1e-6; cfg["max_position_embeddings"] = 4096;
  cfg["tie_word_embeddings"] = false; cfg["hidden_act"] = "silu";
  cfg["rope_parameters"] = {{"rope_theta", 10000000.0},
                            {"partial_rotary_factor", 0.25},
                            {"rope_type", "default"}};
  cfg["layer_types"] = {"linear_attention", "full_attention"};
  json gen;
  gen["eos_token_id"] = 0; gen["bos_token_id"] = 1;
  json nntr;
  nntr["model_type"] = "CausalLM"; nntr["skip_tokenizer"] = true;
  nntr["bad_word_ids"] = json::array(); nntr["model_tensor_type"] = "FP32-FP32";
  nntr["model_file_name"] = "weights.bin"; nntr["init_seq_len"] = 8;
  nntr["max_seq_len"] = 16; nntr["num_to_generate"] = 4; nntr["batch_size"] = 1;
  nntr["embedding_dtype"] = "FP32"; nntr["fc_layer_dtype"] = "FP32";

  const std::string GA = "/home/aisjetson/jijoongmoon/attn_p3/e2e_bin_GA";
  auto ids_f = loadBin(GA + "/input_ids.bin");
  auto golden = loadBin(GA + "/logits.bin");
  const int T = (int)ids_f.size();

  try {
    auto model = std::make_unique<RunModel>(cfg, gen, nntr);
    model->initialize();
    model->load_weight(
      "/home/aisjetson/jijoongmoon/attn_p3/tiny_model/weights.bin");
    std::cout << "[prefill] loaded; T=" << T << " vocab=" << VOCAB << "\n";
    auto logits = model->prefill(ids_f, VOCAB);
    std::cout << "[dbg] logits.size=" << logits.size() << " logits[0..5]=";
    for (int e = 0; e < 6; ++e) std::cout << logits[e] << " ";
    std::cout << "\n[dbg] golden last[0..5]=";
    for (int e = 0; e < 6; ++e) std::cout << golden[(T - 1) * VOCAB + e] << " ";
    std::cout << "\n[dbg] golden pos0[0..5]=";
    for (int e = 0; e < 6; ++e) std::cout << golden[e] << " ";
    std::cout << "\n";
    const float *gp = &golden[(T - 1) * VOCAB]; // HF last-position logits
    auto amax = [&](const float *p) {
      int m = 0;
      for (int e = 1; e < VOCAB; ++e)
        if (p[e] > p[m]) m = e;
      return m;
    };
    float maxd = 0.0f;
    for (int e = 0; e < VOCAB; ++e)
      maxd = std::max(maxd, std::fabs(logits[e] - gp[e]));
    int am = amax(logits.data()), ar = amax(gp);
    std::cout << "[prefill] last-token logits max|d| = " << maxd
              << "  argmax nntr=" << am << " hf=" << ar << "\n";
    std::cout << (am == ar ? "PASS: last-token argmax matches HF\n"
                           : "PARTIAL: argmax mismatch\n");
  } catch (const std::exception &e) {
    std::cout << "FAIL: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
