/**
 * @brief Test safetensors loading with prefix fallback.
 *
 * Creates a tiny 2-layer Qwen3 model, loads test safetensors, and prints
 * first 4 values of each weight for comparison with expected_values.txt.
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include <model.h>
#include <tensor_api.h>
#include <app_context.h>
#include <engine.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <reshaped_rms_norm.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

using ml::train::createLayer;

static std::string withKey(const std::string &key, const std::string &val) {
  return key + "=" + val;
}
static std::string withKey(const std::string &key, int val) {
  return key + "=" + std::to_string(val);
}

constexpr int HIDDEN = 32, INTER = 64, HEADS = 4, KV_HEADS = 2;
constexpr int HEAD_DIM = 8, GQA = 2, NLAYERS = 2, VOCAB = 100, SEQ = 8;
constexpr float EPS = 1e-6f;

void registerLayers() {
  auto &eng = nntrainer::Engine::Global();
  auto *ctx = static_cast<nntrainer::AppContext *>(
    eng.getRegisteredContext("cpu"));
  try {
    ctx->registerFactory(nntrainer::createLayer<causallm::RMSNormLayer>);
    ctx->registerFactory(nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    ctx->registerFactory(nntrainer::createLayer<causallm::MHACoreLayer>);
    ctx->registerFactory(nntrainer::createLayer<causallm::TieWordEmbedding>);
    ctx->registerFactory(nntrainer::createLayer<causallm::EmbeddingLayer>);
    ctx->registerFactory(nntrainer::createLayer<causallm::SwiGLULayer>);
  } catch (...) {}
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <safetensors_path>" << std::endl;
    return 1;
  }

  registerLayers();

  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({"batch_size=1", "epochs=1", "model_tensor_type=FP32-FP32"});

  auto S = [](int v) { return std::to_string(v); };
  auto E = [](float v) { return std::to_string(v); };

  // Input
  auto input = createLayer("input", {withKey("name", "input0"),
    withKey("input_shape", "1:1:" + S(SEQ))});
  auto x = input({});

  // Embedding
  auto emb = createLayer("tie_word_embeddings",
    {"name=embedding0", "in_dim=" + S(VOCAB), "out_dim=" + S(HIDDEN),
     "weight_dtype=FP32"});
  x = emb(x);

  for (int i = 0; i < NLAYERS; ++i) {
    std::string si = S(i);
    int kv_dim = HEAD_DIM * HEADS / GQA;

    // att norm
    auto an = createLayer("rms_norm", {withKey("name", "layer"+si+"_attention_norm"),
      withKey("epsilon", E(EPS)), withKey("packed","false")});
    auto normed = an(x);

    // V, K, Q
    auto vp = createLayer("fully_connected", {withKey("name","layer"+si+"_wv"),
      withKey("unit",kv_dim), withKey("disable_bias","true")});
    auto v = vp(normed);

    auto kp = createLayer("fully_connected", {withKey("name","layer"+si+"_wk"),
      withKey("unit",kv_dim), withKey("disable_bias","true")});
    auto k = kp(normed);

    auto kn = createLayer("reshaped_rms_norm", {withKey("name","layer"+si+"_k_norm"),
      withKey("packed","false"), withKey("epsilon",E(EPS)),
      withKey("feature_size",S(HEAD_DIM))});
    auto kn_out = kn(k);

    auto qp = createLayer("fully_connected", {withKey("name","layer"+si+"_wq"),
      withKey("unit",HEAD_DIM*HEADS), withKey("disable_bias","true")});
    auto q = qp(normed);

    auto qn = createLayer("reshaped_rms_norm", {withKey("name","layer"+si+"_q_norm"),
      withKey("packed","false"), withKey("epsilon",E(EPS)),
      withKey("feature_size",S(HEAD_DIM))});
    auto qn_out = qn(q);

    // MHA
    auto attn = createLayer("mha_core", {withKey("name","layer"+si+"_attention"),
      withKey("num_heads",HEADS), withKey("num_heads_kv",KV_HEADS),
      withKey("max_timestep",S(SEQ+4)), withKey("rope_theta",500000),
      withKey("max_new_tokens","4")});
    auto a = attn({qn_out, kn_out, v});

    // O
    auto op = createLayer("fully_connected", {withKey("name","layer"+si+"_attention_out"),
      withKey("unit",HIDDEN), withKey("disable_bias","true")});
    auto o = op(a);

    auto res = x.add(o);

    // FFN
    auto fn = createLayer("rms_norm", {withKey("name","layer"+si+"_ffn_norm"),
      withKey("epsilon",E(EPS)), withKey("packed","false")});
    auto fn_out = fn(res);

    auto up = createLayer("fully_connected", {withKey("name","layer"+si+"_ffn_up"),
      withKey("unit",INTER), withKey("disable_bias","true")});
    auto up_out = up(fn_out);

    auto gate = createLayer("fully_connected", {withKey("name","layer"+si+"_ffn_gate"),
      withKey("unit",INTER), withKey("disable_bias","true")});
    auto gate_out = gate(fn_out);

    auto sg = createLayer("swiglu", {withKey("name","layer"+si+"_ffn_swiglu")});
    auto act = sg({up_out, gate_out});

    auto dn = createLayer("fully_connected", {withKey("name","layer"+si+"_ffn_down"),
      withKey("unit",HIDDEN), withKey("disable_bias","true")});
    auto dn_out = dn(act);

    x = res.add(dn_out);
  }

  // Output norm
  auto on = createLayer("rms_norm", {withKey("name","output_norm"),
    withKey("epsilon",E(EPS)), withKey("packed","false")});
  x = on(x);

  // Compile
  model->compile({input({})}, {x}, ml::train::ExecutionMode::INFERENCE);

  // Load safetensors
  std::cout << "\n=== Loading: " << argv[1] << " ===" << std::endl;
  model->load(argv[1], ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);

  // Print weight values
  std::cout << "\n=== Loaded weight values ===" << std::endl;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &ctx, void *) {
      for (unsigned int i = 0; i < ctx.getNumWeights(); ++i) {
        auto &w = ctx.getWeight(i);
        const float *data = w.getData<float>();
        if (!data) continue;
        size_t n = std::min<size_t>(4, w.size());
        std::cout << "  " << w.getName() << " first4=[";
        for (size_t j = 0; j < n; ++j) {
          if (j) std::cout << ", ";
          printf("%.8f", data[j]);
        }
        std::cout << "]" << std::endl;
      }
    };
  model->forEachLayer(fn, nullptr);

  std::cout << "\nDone." << std::endl;
  return 0;
}
