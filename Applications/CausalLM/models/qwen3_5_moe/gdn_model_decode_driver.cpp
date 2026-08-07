// SPDX-License-Identifier: Apache-2.0
/**
 * Decode-state test: prefill the first T-1 tokens, then DECODE the T-1-th token
 * (one step, carrying GDN recurrent+conv state and the full-attn KV cache), and
 * compare the decoded last-token logits to the HF golden (e2e_bin_GA position
 * T-1). If prefill-(T-1)+decode-1 == HF's T-token forward, the GDN decode state
 * + hybrid KV cache work. Run with NNTR_ENGINE=cpu.
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

struct RunModel : causallm::Qwen3_5MoeCausalLM {
  RunModel(json &c, json &g, json &n) :
    causallm::Transformer(c, g, n, causallm::ModelType::CAUSALLM),
    causallm::Qwen3_5MoeCausalLM(c, g, n) {}

  // prefill ids[0..P-1], then decode ids[P]; return the decoded token's logits.
  std::vector<float> prefillThenDecode(const std::vector<float> &ids, int P,
                                       int vocab) {
    allocateAndBindKVCache();
    float *k1 = reinterpret_cast<float *>(kv_cache.getKeyCache(1).getData());
    float *v1 = reinterpret_cast<float *>(kv_cache.getValueCache(1).getData());

    // prefill P tokens (from=0)
    std::vector<float> pbuf((size_t)MAX_SEQ_LEN, 0.0f);
    for (int i = 0; i < P; ++i)
      pbuf[i] = ids[i];
    setKVCachePosition(0);
    std::vector<float *> pin = {pbuf.data(), k1, v1};
    (void)incrementalInference(BATCH_SIZE, pin, P, 0, P);

    // decode one token (ids[P]) at cache position P (from=P)
    std::vector<float> dbuf((size_t)MAX_SEQ_LEN, 0.0f);
    dbuf[0] = ids[P];
    setKVCachePosition(P);
    std::vector<float *> din = {dbuf.data(), k1, v1};
    auto out = incrementalInference(BATCH_SIZE, din, P, P, P + 1);
    return std::vector<float>(out[0], out[0] + vocab);
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
  auto ids = loadBin(GA + "/input_ids.bin");
  auto golden = loadBin(GA + "/logits.bin");
  const int T = (int)ids.size(); // 7

  try {
    auto model = std::make_unique<RunModel>(cfg, gen, nntr);
    model->initialize();
    model->load_weight(
      "/home/aisjetson/jijoongmoon/attn_p3/tiny_model/weights.bin");
    // prefill tokens 0..T-2, decode token T-1
    auto logits = model->prefillThenDecode(ids, T - 1, VOCAB);
    const float *gp = &golden[(T - 1) * VOCAB]; // HF position T-1
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
    std::cout << "[decode] prefill " << (T - 1) << " + decode 1; pos " << (T - 1)
              << " logits max|d| = " << maxd << "  argmax nntr=" << am
              << " hf=" << ar << "\n";
    std::cout << (am == ar ? "PASS: decode-step argmax matches HF\n"
                           : "FAIL: decode argmax mismatch\n");
    return am == ar ? 0 : 1;
  } catch (const std::exception &e) {
    std::cout << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
