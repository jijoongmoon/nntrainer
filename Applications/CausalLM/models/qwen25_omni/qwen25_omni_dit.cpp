// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_dit.cpp
 * @date   18 July 2026
 * @brief  Qwen2.5-Omni Token2Wav DiT (codec ids -> mel), host RK4 sampler.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

#include <dit_attention.h>
#include <dit_gate.h>
#include <dit_modulate.h>
#include <dit_rope.h>
#include <qwen25_omni_dit.h>

namespace causallm {

// AdaLN chunk-6 offsets (C3, shift-first):
// [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] * 1024
static constexpr unsigned OFF_SHIFT_MSA = 0;
static constexpr unsigned OFF_SCALE_MSA = 1024;
static constexpr unsigned OFF_GATE_MSA = 2048;
static constexpr unsigned OFF_SHIFT_MLP = 3072;
static constexpr unsigned OFF_SCALE_MLP = 4096;
static constexpr unsigned OFF_GATE_MLP = 5120;

void Qwen25OmniDiT::setupParameters(json &cfg, json &generation_cfg,
                                    json &nntr_cfg) {
  (void)generation_cfg;
  BATCH_SIZE = 1; // C6: CFG runs as two batch-1 forwards
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  MEMORY_SWAP = nntr_cfg.value("fsu", false);

  // C7: HF ignores the config.json dit_config aliases; class defaults rule.
  // Kept configurable for experiments but defaulted to the confirmed values.
  HIDDEN = cfg.value("hidden_size", 1024);
  DEPTH = cfg.value("depth", 22);
  HEADS = cfg.value("num_heads", 16);
  HEAD_DIM = cfg.value("head_dim", 64);
  FF_INNER = cfg.value("ff_inner", 2048);
  MEL_DIM = cfg.value("mel_dim", 80);
  SEQ = nntr_cfg.value("dit_seq", cfg.value("dit_seq", 128));
  REPEATS = cfg.value("repeats", 2);
  CODEC_VOCAB = cfg.value("codec_vocab", 8194);
  CODEC_DIM = cfg.value("codec_dim", 512);
  ENC_DIM = cfg.value("enc_dim", 128);
  SPK_DIM = cfg.value("enc_emb_dim", 192);
  TIME_FREQ = cfg.value("time_freq", 256);
  BLOCK_SIZE = cfg.value("block_size", 24);
  GUIDANCE = cfg.value("guidance_scale", 0.5f);
  ROPE_THETA = cfg.value("rope_theta", 10000.0f);
}

void Qwen25OmniDiT::registerCustomLayers() {
  const auto &ct_engine = nntrainer::Engine::Global();
  auto *app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  auto reg = [&](auto fn) {
    try {
      app_context->registerFactory(fn);
    } catch (std::invalid_argument &) {
      // already registered; ignore
    }
  };
  reg(nntrainer::createLayer<causallm::DiTRoPELayer>);
  reg(nntrainer::createLayer<causallm::DiTModulateLayer>);
  reg(nntrainer::createLayer<causallm::DiTGateLayer>);
  reg(nntrainer::createLayer<causallm::DiTAttentionLayer>);
}

void Qwen25OmniDiT::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
                      withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  const unsigned int cond_w = MEL_DIM + ENC_DIM + CODEC_DIM + SPK_DIM; // 912

  std::vector<LayerHandle> layers;
  auto add = [&](const std::string &type, const std::string &name,
                 std::vector<std::string> props, const std::string &inputs) {
    if (!inputs.empty())
      props.push_back(withKey("input_layers", inputs));
    props.push_back(withKey("name", name));
    layers.push_back(createLayer(type, props));
  };
  auto S = [](unsigned int v) { return std::to_string(v); };
  auto fc = [&](const std::string &name, unsigned int unit,
                const std::string &inputs) {
    add("fully_connected", name, {withKey("unit", S(unit))}, inputs);
  };

  // graph inputs -- inference() maps buffers by THIS order (x, time, cos, sin)
  add("input", "input_x", {withKey("input_shape", "1:" + S(SEQ) + ":" + S(cond_w))}, "");
  add("input", "input_time", {withKey("input_shape", "1:1:" + S(TIME_FREQ))}, "");
  add("input", "input_cos", {withKey("input_shape", "1:" + S(SEQ) + ":" + S(HEAD_DIM))}, "");
  add("input", "input_sin", {withKey("input_shape", "1:" + S(SEQ) + ":" + S(HEAD_DIM))}, "");

  // input embed: concat(host) -> proj(912->1024)  (C1/§4: column layout
  // [mel|ecapa|code|speaker] follows the HF forward order)
  fc("proj", HIDDEN, "input_x");

  // time path: time_mlp = Linear(256->1024) -> SiLU -> Linear(1024->1024);
  // every AdaLN linear consumes SiLU(time_emb) (shared node).
  fc("time_mlp0", HIDDEN, "input_time");
  add("activation", "time_silu0", {withKey("activation", "swish")}, "time_mlp0");
  fc("time_mlp1", HIDDEN, "time_silu0");
  add("activation", "time_silu", {withKey("activation", "swish")}, "time_mlp1");

  std::string h = "proj";
  for (unsigned int i = 0; i < DEPTH; ++i) {
    const std::string p = "b" + S(i) + "_";
    // per-block AdaLN-Zero: cond_i = attn_norm.linear(SiLU(time_emb)) [6144]
    fc(p + "adaln", 6 * HIDDEN, "time_silu");

    add("dit_modulate", p + "mod_a",
        {withKey("scale_off", S(OFF_SCALE_MSA)),
         withKey("shift_off", S(OFF_SHIFT_MSA))},
        h + "," + p + "adaln");
    fc(p + "wq", HIDDEN, p + "mod_a");
    fc(p + "wk", HIDDEN, p + "mod_a");
    fc(p + "wv", HIDDEN, p + "mod_a");
    add("dit_rope", p + "ropeq", {}, p + "wq,input_cos,input_sin");
    add("dit_rope", p + "ropek", {}, p + "wk,input_cos,input_sin");
    // C2: only L0/L20 look one block back, L10 one block ahead.
    add("dit_attention", p + "attn",
        {withKey("num_heads", S(HEADS)), withKey("head_dim", S(HEAD_DIM)),
         withKey("block_size", S(BLOCK_SIZE)),
         withKey("look_ahead", i == 10 ? "1" : "0"),
         withKey("look_backward", (i == 0 || i == 20) ? "1" : "0")},
        p + "ropeq," + p + "ropek," + p + "wv");
    fc(p + "wo", HIDDEN, p + "attn");
    add("dit_gate", p + "gate_a", {withKey("gate_off", S(OFF_GATE_MSA))},
        h + "," + p + "wo," + p + "adaln");

    add("dit_modulate", p + "mod_f",
        {withKey("scale_off", S(OFF_SCALE_MLP)),
         withKey("shift_off", S(OFF_SHIFT_MLP))},
        p + "gate_a," + p + "adaln");
    fc(p + "ff0", FF_INNER, p + "mod_f");
    add("activation", p + "gelu", {withKey("activation", "tanh_gelu")},
        p + "ff0");
    fc(p + "ff3", HIDDEN, p + "gelu");
    add("dit_gate", p + "gate_f", {withKey("gate_off", S(OFF_GATE_MLP))},
        p + "gate_a," + p + "ff3," + p + "adaln");
    h = p + "gate_f";
  }

  // final: norm_out.linear(SiLU(time)) [2048], chunk-2 = [scale, shift]
  // (C4: SCALE-first, opposite of the per-block shift-first order)
  fc("norm_out", 2 * HIDDEN, "time_silu");
  add("dit_modulate", "mod_out",
      {withKey("scale_off", "0"), withKey("shift_off", S(HIDDEN))},
      h + ",norm_out");
  fc("proj_out", MEL_DIM, "mod_out");

  for (auto &l : layers)
    model->addLayer(l);

  if (model->compile(ml::train::ExecutionMode::INFERENCE))
    throw std::invalid_argument("DiT model compilation failed.");
  if (model->initialize(ml::train::ExecutionMode::INFERENCE))
    throw std::invalid_argument("DiT model initialization failed.");

  // NNTR_DIT_SUMMARY=1: print the compiled graph (= weight-load order) so the
  // converter can be locked to the actual DFS order (weight-bin-load-order-dfs)
  if (std::getenv("NNTR_DIT_SUMMARY") != nullptr)
    model->summarize(std::cout, ML_TRAIN_SUMMARY_LAYER);
  is_initialized = true;
}

void Qwen25OmniDiT::load_weight(const std::string &weight_path) {
  model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  // codec_embed.bin lives next to dit.bin: raw f32 [CODEC_VOCAB, CODEC_DIM],
  // host-gathered (NOT a graph weight; C8 needs row 0 for the null branch).
  const auto slash = weight_path.find_last_of("/\\");
  const std::string dir =
    slash == std::string::npos ? "." : weight_path.substr(0, slash);
  const std::string path = dir + "/codec_embed.bin";
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open codec embed " + path);
  codec_table.resize(static_cast<size_t>(CODEC_VOCAB) * CODEC_DIM);
  f.read(reinterpret_cast<char *>(codec_table.data()),
         static_cast<std::streamsize>(codec_table.size() * sizeof(float)));
  if (!f)
    throw std::runtime_error("short read on " + path);

  // rotary inv_freq: checkpoint values are bf16-rounded, NOT the
  // theta^(-2j/64) formula; recomputing shifts cos by up to 5e-2 at s=127.
  const std::string inv_path = dir + "/inv_freq.bin";
  std::ifstream fi(inv_path, std::ios::binary);
  if (!fi)
    throw std::runtime_error("cannot open rotary inv_freq " + inv_path);
  inv_freq.resize(HEAD_DIM / 2);
  fi.read(reinterpret_cast<char *>(inv_freq.data()),
          static_cast<std::streamsize>(inv_freq.size() * sizeof(float)));
  if (!fi)
    throw std::runtime_error("short read on " + inv_path);
}

void Qwen25OmniDiT::fill_time_sin(float t) {
  // SinusPositionEmbedding(256, scale=1000): half=128,
  // emb_k = exp(k * -ln(10000)/127), out = [sin(1000*t*emb), cos(1000*t*emb)]
  const unsigned int half = TIME_FREQ / 2;
  const float step = -std::log(10000.0f) / static_cast<float>(half - 1);
  for (unsigned int k = 0; k < half; ++k) {
    const float arg = 1000.0f * t * std::exp(static_cast<float>(k) * step);
    in_t[k] = std::sin(arg);
    in_t[half + k] = std::cos(arg);
  }
}

void Qwen25OmniDiT::assemble_input(const float *y, bool guided) {
  const unsigned int cond_w = MEL_DIM + ENC_DIM + CODEC_DIM + SPK_DIM;
  const float *ecapa = guided ? ecapa_c.data() : ecapa_n.data();
  const float *code = guided ? code_embed.data() : code_embed_null.data();
  for (unsigned int s = 0; s < SEQ; ++s) {
    float *row = in_x.data() + static_cast<size_t>(s) * cond_w;
    std::memcpy(row, y + static_cast<size_t>(s) * MEL_DIM,
                MEL_DIM * sizeof(float));
    std::memcpy(row + MEL_DIM, ecapa, ENC_DIM * sizeof(float));
    std::memcpy(row + MEL_DIM + ENC_DIM,
                code + static_cast<size_t>(s) * CODEC_DIM,
                CODEC_DIM * sizeof(float));
    float *spk_dst = row + MEL_DIM + ENC_DIM + CODEC_DIM;
    if (guided)
      std::memcpy(spk_dst, spk_c.data(), SPK_DIM * sizeof(float));
    else
      std::memset(spk_dst, 0, SPK_DIM * sizeof(float));
  }
}

void Qwen25OmniDiT::ode_eval(float t, const float *y, float *v_out) {
  const size_t n = static_cast<size_t>(SEQ) * MEL_DIM;
  fill_time_sin(t);
  std::vector<float *> label;

  assemble_input(y, /*guided=*/true);
  std::vector<float *> in = {in_x.data(), in_t.data(), cos_buf.data(),
                             sin_buf.data()};
  std::vector<float *> out = model->inference(BATCH_SIZE, in, label);
  // the second forward reuses the same output tensor; keep the guided copy
  for (size_t i = 0; i < n; ++i)
    v_out[i] = 1.5f * out[0][i];

  assemble_input(y, /*guided=*/false);
  out = model->inference(BATCH_SIZE, in, label);
  for (size_t i = 0; i < n; ++i)
    v_out[i] -= 0.5f * out[0][i]; // v = 1.5*guided - 0.5*null (CFG 0.5)
}

void Qwen25OmniDiT::prepare_conditioning(const int32_t *codes,
                                         const float *ecapa_pos,
                                         const float *ecapa_neg,
                                         const float *spk) {
  const unsigned int cond_w = MEL_DIM + ENC_DIM + CODEC_DIM + SPK_DIM;
  const unsigned int num_codes = SEQ / REPEATS;

  // codec gather + ADJACENT repeat_interleave (c0,c0,c1,c1,..., §4.1);
  // null branch = row 0 everywhere (drop_code zeros the ids, C8).
  code_embed.resize(static_cast<size_t>(SEQ) * CODEC_DIM);
  code_embed_null.resize(static_cast<size_t>(SEQ) * CODEC_DIM);
  for (unsigned int c = 0; c < num_codes; ++c) {
    const int32_t id = codes[c];
    if (id < 0 || static_cast<unsigned int>(id) >= CODEC_VOCAB)
      throw std::invalid_argument("codec id out of range: " +
                                  std::to_string(id));
    const float *src = codec_table.data() + static_cast<size_t>(id) * CODEC_DIM;
    for (unsigned int r = 0; r < REPEATS; ++r)
      std::memcpy(code_embed.data() +
                    (static_cast<size_t>(c) * REPEATS + r) * CODEC_DIM,
                  src, CODEC_DIM * sizeof(float));
  }
  for (unsigned int s = 0; s < SEQ; ++s)
    std::memcpy(code_embed_null.data() + static_cast<size_t>(s) * CODEC_DIM,
                codec_table.data(), CODEC_DIM * sizeof(float));

  ecapa_c.assign(ecapa_pos, ecapa_pos + ENC_DIM);
  ecapa_n.assign(ecapa_neg, ecapa_neg + ENC_DIM);
  spk_c.assign(spk, spk + SPK_DIM);

  // rotary cos/sin from the checkpoint inv_freq, interleaved-duplicate
  // [f0,f0,f1,f1,...] (C5)
  cos_buf.resize(static_cast<size_t>(SEQ) * HEAD_DIM);
  sin_buf.resize(static_cast<size_t>(SEQ) * HEAD_DIM);
  for (unsigned int s = 0; s < SEQ; ++s) {
    for (unsigned int j = 0; j < HEAD_DIM / 2; ++j) {
      const float f = static_cast<float>(s) * inv_freq[j];
      const size_t o = static_cast<size_t>(s) * HEAD_DIM + 2 * j;
      cos_buf[o] = cos_buf[o + 1] = std::cos(f);
      sin_buf[o] = sin_buf[o + 1] = std::sin(f);
    }
  }

  in_x.resize(static_cast<size_t>(SEQ) * cond_w);
  in_t.resize(TIME_FREQ);
}

std::vector<float> Qwen25OmniDiT::generate_mel(const int32_t *codes,
                                               const float *ecapa_pos,
                                               const float *ecapa_neg,
                                               const float *spk,
                                               const float *noise) {
  if (!is_initialized)
    throw std::runtime_error("DiT is not initialized.");
  if (codec_table.empty())
    throw std::runtime_error("DiT codec table not loaded.");

  const size_t n = static_cast<size_t>(SEQ) * MEL_DIM;
  prepare_conditioning(codes, ecapa_pos, ecapa_neg, spk);

  // --- RK4 3/8-rule over the sway-warped grid t_i = 1 - cos(pi/2 * i/9) ----
  const int N_INT = 9;
  std::vector<float> t(N_INT + 1);
  for (int i = 0; i <= N_INT; ++i)
    t[i] = 1.0f - std::cos(static_cast<float>(M_PI) / 2.0f *
                           (static_cast<float>(i) / N_INT));

  std::vector<float> y(noise, noise + n);
  std::vector<float> k1(n), k2(n), k3(n), k4(n), tmp(n);
  const float c13 = 1.0f / 3.0f, c23 = 2.0f / 3.0f;

  for (int i = 0; i < N_INT; ++i) {
    const float ts = t[i], te = t[i + 1], dt = te - ts;
    ode_eval(ts, y.data(), k1.data());
    for (size_t j = 0; j < n; ++j)
      tmp[j] = y[j] + dt * c13 * k1[j];
    ode_eval(ts + dt * c13, tmp.data(), k2.data());
    for (size_t j = 0; j < n; ++j)
      tmp[j] = y[j] + dt * (k2[j] - c13 * k1[j]);
    ode_eval(ts + dt * c23, tmp.data(), k3.data());
    for (size_t j = 0; j < n; ++j)
      tmp[j] = y[j] + dt * (k1[j] - k2[j] + k3[j]);
    ode_eval(te, tmp.data(), k4.data());
    for (size_t j = 0; j < n; ++j)
      y[j] += (k1[j] + 3.0f * (k2[j] + k3[j]) + k4[j]) * (dt / 8.0f);
  }

  // [seq, 80] -> [80, seq] (permute at the very end, modeling:3589)
  std::vector<float> mel(n);
  for (unsigned int s = 0; s < SEQ; ++s)
    for (unsigned int m = 0; m < MEL_DIM; ++m)
      mel[static_cast<size_t>(m) * SEQ + s] =
        y[static_cast<size_t>(s) * MEL_DIM + m];
  return mel;
}

void Qwen25OmniDiT::run(const WSTR prompt, bool do_sample,
                        const WSTR system_prompt, const WSTR tail_prompt,
                        bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  const std::string dir(prompt);
  auto read_f32 = [&](const std::string &name, size_t count) {
    std::ifstream f(dir + "/" + name, std::ios::binary);
    if (!f)
      throw std::runtime_error("cannot open " + dir + "/" + name);
    std::vector<float> v(count);
    f.read(reinterpret_cast<char *>(v.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f)
      throw std::runtime_error("short read on " + dir + "/" + name);
    return v;
  };

  const unsigned int num_codes = SEQ / REPEATS;
  std::vector<int32_t> codes(num_codes);
  {
    std::ifstream f(dir + "/codes.bin", std::ios::binary);
    if (!f)
      throw std::runtime_error("cannot open " + dir + "/codes.bin");
    f.read(reinterpret_cast<char *>(codes.data()),
           static_cast<std::streamsize>(codes.size() * sizeof(int32_t)));
    if (!f)
      throw std::runtime_error("short read on " + dir + "/codes.bin");
  }
  std::vector<float> ecapa_pos = read_f32("ecapa_pos.bin", ENC_DIM);
  std::vector<float> ecapa_neg = read_f32("ecapa_neg.bin", ENC_DIM);
  std::vector<float> spk = read_f32("spk.bin", SPK_DIM);

  // NNTR_DIT_STAGEA=1: single ODE eval at (x_in.bin, t.bin) against the HF
  // per-step refs; writes guided/null/velocity.bin then returns (no RK4).
  if (std::getenv("NNTR_DIT_STAGEA") != nullptr) {
    const size_t n = static_cast<size_t>(SEQ) * MEL_DIM;
    std::vector<float> x = read_f32("x_in.bin", n);
    std::vector<float> tv = read_f32("t.bin", 1);
    prepare_conditioning(codes.data(), ecapa_pos.data(), ecapa_neg.data(),
                         spk.data());
    auto save = [&](const std::string &name, const float *d) {
      std::ofstream f(dir + "/" + name, std::ios::binary);
      f.write(reinterpret_cast<const char *>(d),
              static_cast<std::streamsize>(n * sizeof(float)));
    };
    fill_time_sin(tv[0]);
    std::vector<float *> label;
    std::vector<float *> in = {in_x.data(), in_t.data(), cos_buf.data(),
                               sin_buf.data()};
    std::vector<float> guided(n), null_v(n), cfg_v(n);
    assemble_input(x.data(), /*guided=*/true);
    std::vector<float *> out = model->inference(BATCH_SIZE, in, label);
    std::copy(out[0], out[0] + n, guided.begin());
    assemble_input(x.data(), /*guided=*/false);
    out = model->inference(BATCH_SIZE, in, label);
    std::copy(out[0], out[0] + n, null_v.begin());
    for (size_t i = 0; i < n; ++i)
      cfg_v[i] = 1.5f * guided[i] - 0.5f * null_v[i];
    save("guided.bin", guided.data());
    save("null.bin", null_v.data());
    save("velocity.bin", cfg_v.data());
    std::cout << "DiT Stage-A eval done (t=" << tv[0] << ") -> "
              << dir + "/{guided,null,velocity}.bin" << std::endl;
    return;
  }

  // HF noise slice; Stage-B bit-match forbids C++ RNG
  std::vector<float> noise =
    read_f32("noise.bin", static_cast<size_t>(SEQ) * MEL_DIM);

  std::vector<float> mel = generate_mel(codes.data(), ecapa_pos.data(),
                                        ecapa_neg.data(), spk.data(),
                                        noise.data());

  const std::string out_path = dir + "/dit_mel.bin";
  std::ofstream f(out_path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open output " + out_path);
  const int32_t md = static_cast<int32_t>(MEL_DIM),
                nf = static_cast<int32_t>(SEQ);
  f.write(reinterpret_cast<const char *>(&md), 4);
  f.write(reinterpret_cast<const char *>(&nf), 4);
  f.write(reinterpret_cast<const char *>(mel.data()),
          static_cast<std::streamsize>(mel.size() * sizeof(float)));
  if (log_output)
    std::cout << "DiT wrote mel [" << MEL_DIM << " x " << SEQ << "] to "
              << out_path << std::endl;
}

} // namespace causallm
