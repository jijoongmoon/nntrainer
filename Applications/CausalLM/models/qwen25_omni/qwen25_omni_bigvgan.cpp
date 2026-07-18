// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_bigvgan.cpp
 * @date   15 June 2026
 * @brief  Qwen2.5-Omni Token2Wav BigVGAN vocoder (mel -> 24 kHz waveform).
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

#include <antialiased_snake.h>
#include <conv1d_transpose.h>
#include <qwen25_omni_bigvgan.h>
#include <scale_layer.h>

namespace causallm {

static const int UP_K[6] = {11, 7, 4, 4, 4, 4};
static const int UP_R[6] = {5, 3, 2, 2, 2, 2};
static const int RES_K[3] = {3, 7, 11};
static const int RES_D[3] = {1, 3, 5};

void Qwen25OmniBigVGAN::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  (void)generation_cfg;
  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  MEMORY_SWAP = nntr_cfg.value("fsu", false);

  MEL_DIM = cfg.value("mel_dim", 80);
  UP_INIT_CH = cfg.value("upsample_initial_channel", 1536);
  // fixed compile-time mel length (DiT emits num_codes*2 frames at run time)
  MEL_FRAMES = nntr_cfg.value("mel_frames", cfg.value("mel_frames", 128));
  SAMPLE_RATE = cfg.value("sample_rate", 24000);
}

void Qwen25OmniBigVGAN::registerCustomLayers() {
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
  reg(nntrainer::createLayer<causallm::Conv1DTransposeLayer>);
  reg(nntrainer::createLayer<causallm::AntialiasedSnakeLayer>);
  reg(nntrainer::createLayer<causallm::ScaleLayer>);
}

void Qwen25OmniBigVGAN::initialize() {
  registerCustomLayers();
  build_and_init();
}

void Qwen25OmniBigVGAN::ensure_frames(unsigned int n_frames) {
  if (n_frames == MEL_FRAMES && is_initialized)
    return;
  if (n_frames == 0)
    throw std::invalid_argument("bigvgan mel_frames must be positive");
  MEL_FRAMES = n_frames;
  build_and_init();
  if (!weight_path_.empty())
    model->load(weight_path_, ml::train::ModelFormat::MODEL_FORMAT_BIN);
}

void Qwen25OmniBigVGAN::build_and_init() {
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
                      withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  // ---- build the conv graph imperatively (DFS-from-output = forward order) --
  std::vector<LayerHandle> layers;
  auto add = [&](const std::string &type, const std::string &name,
                 std::vector<std::string> props, const std::string &inputs) {
    if (!inputs.empty())
      props.push_back(withKey("input_layers", inputs));
    props.push_back(withKey("name", name));
    layers.push_back(createLayer(type, props));
  };
  auto S = [](int v) { return std::to_string(v); };

  add("input", "input0",
      {withKey("input_shape", S(MEL_DIM) + ":1:" + S(MEL_FRAMES))}, "");
  add("conv1d", "conv_pre",
      {withKey("filters", S(UP_INIT_CH)), withKey("kernel_size", "7"),
       withKey("stride", "1"), withKey("padding", "3")},
      "input0");

  std::string prev = "conv_pre";
  int ch = UP_INIT_CH;
  for (int i = 0; i < 6; ++i) {
    int out_ch = ch / 2;
    std::string ups = "ups" + S(i);
    add("conv1d_transpose", ups,
        {withKey("filters", S(out_ch)), withKey("kernel_size", S(UP_K[i])),
         withKey("stride", S(UP_R[i])),
         withKey("padding", S((UP_K[i] - UP_R[i]) / 2))},
        prev);
    std::vector<std::string> amp_outs;
    for (int b = 0; b < 3; ++b) {
      int kb = RES_K[b];
      std::string a = ups;
      for (int kk = 0; kk < 3; ++kk) {
        int d = RES_D[kk];
        std::string base = "s" + S(i) + "_b" + S(b) + "_k" + S(kk);
        add("antialiased_snake", base + "_act1", {}, a);
        add("conv1d", base + "_c1",
            {withKey("filters", S(out_ch)), withKey("kernel_size", S(kb)),
             withKey("stride", "1"), withKey("dilation", S(d)),
             withKey("padding", S(d * (kb - 1) / 2))},
            base + "_act1");
        add("antialiased_snake", base + "_act2", {}, base + "_c1");
        add("conv1d", base + "_c2",
            {withKey("filters", S(out_ch)), withKey("kernel_size", S(kb)),
             withKey("stride", "1"), withKey("dilation", "1"),
             withKey("padding", S((kb - 1) / 2))},
            base + "_act2");
        add("addition", base + "_res", {}, a + "," + base + "_c2");
        a = base + "_res";
      }
      amp_outs.push_back(a);
    }
    add("addition", "s" + S(i) + "_sum", {},
        amp_outs[0] + "," + amp_outs[1] + "," + amp_outs[2]);
    add("scale", "s" + S(i) + "_mean",
        {withKey("scale", "0.3333333432674408")}, "s" + S(i) + "_sum");
    prev = "s" + S(i) + "_mean";
    ch = out_ch;
  }
  add("antialiased_snake", "act_post", {}, prev);
  add("conv1d", "conv_post",
      {withKey("filters", "1"), withKey("kernel_size", "7"),
       withKey("stride", "1"), withKey("padding", "3"),
       withKey("disable_bias", "true")},
      "act_post");

  for (auto &l : layers)
    model->addLayer(l);

  if (model->compile(ml::train::ExecutionMode::INFERENCE))
    throw std::invalid_argument("BigVGAN model compilation failed.");
  if (model->initialize(ml::train::ExecutionMode::INFERENCE))
    throw std::invalid_argument("BigVGAN model initialization failed.");
  is_initialized = true;
}

void Qwen25OmniBigVGAN::load_weight(const std::string &weight_path) {
  weight_path_ = weight_path;
  model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
}

void Qwen25OmniBigVGAN::process_mel(const float *mel, unsigned int n,
                                    std::vector<float> &out) {
  // HF process_mel_spectrogram: exp -> amplitude_to_db(min=-115) - 20 ->
  // normalize(max=1, min_db=-115) -> clamp[-1,1]. Verified vs dump to 2.4e-7.
  const double LN10 = std::log(10.0);
  const float MIN_LEVEL = static_cast<float>(std::exp(-115.0 / 20.0 * LN10));
  out.resize(n);
  for (unsigned int i = 0; i < n; ++i) {
    float amp = std::exp(mel[i]);
    float cl = amp < MIN_LEVEL ? MIN_LEVEL : amp;
    float db = 20.0f * std::log10(cl) - 20.0f;
    float v = 2.0f * ((db + 115.0f) / 115.0f) - 1.0f;
    out[i] = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
  }
}

std::vector<float> Qwen25OmniBigVGAN::vocode(const float *mel,
                                             unsigned int n_frames) {
  if (!is_initialized)
    throw std::runtime_error("BigVGAN is not initialized.");
  ensure_frames(n_frames);

  std::vector<float> processed;
  process_mel(mel, MEL_DIM * n_frames, processed);

  std::vector<float *> in = {processed.data()};
  std::vector<float *> label;
  std::vector<float *> out = model->inference(BATCH_SIZE, in, label);

  const size_t n = static_cast<size_t>(n_frames) * 240;
  std::vector<float> wav(n);
  for (size_t i = 0; i < n; ++i) {
    float v = out[0][i];
    wav[i] = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
  }
  return wav;
}

void Qwen25OmniBigVGAN::write_wav(const std::string &path,
                                  const std::vector<float> &wav,
                                  unsigned int sample_rate) {
  const uint32_t n = static_cast<uint32_t>(wav.size());
  const uint16_t bits = 16, channels = 1;
  const uint32_t byte_rate = sample_rate * channels * bits / 8;
  const uint16_t block_align = channels * bits / 8;
  const uint32_t data_bytes = n * bits / 8;
  const uint32_t riff = 36 + data_bytes;

  std::ofstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open WAV output " + path);
  auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<char *>(&v), 4); };
  auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<char *>(&v), 2); };
  f.write("RIFF", 4);
  u32(riff);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  u32(16);
  u16(1); // PCM
  u16(channels);
  u32(sample_rate);
  u32(byte_rate);
  u16(block_align);
  u16(bits);
  f.write("data", 4);
  u32(data_bytes);
  for (uint32_t i = 0; i < n; ++i) {
    float v = std::max(-1.0f, std::min(1.0f, wav[i]));
    int16_t s = static_cast<int16_t>(std::lround(v * 32767.0f));
    f.write(reinterpret_cast<char *>(&s), 2);
  }
}

void Qwen25OmniBigVGAN::run(const WSTR prompt, bool do_sample,
                            const WSTR system_prompt, const WSTR tail_prompt,
                            bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  const std::string mel_path(prompt);
  std::ifstream f(mel_path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open mel file " + mel_path);
  int32_t mel_dim = 0, n_frames = 0;
  f.read(reinterpret_cast<char *>(&mel_dim), 4);
  f.read(reinterpret_cast<char *>(&n_frames), 4);
  if (mel_dim != static_cast<int32_t>(MEL_DIM))
    throw std::runtime_error("mel_dim mismatch: file " +
                             std::to_string(mel_dim) + " vs " +
                             std::to_string(MEL_DIM));
  std::vector<float> mel(static_cast<size_t>(mel_dim) * n_frames);
  f.read(reinterpret_cast<char *>(mel.data()),
         static_cast<std::streamsize>(mel.size() * sizeof(float)));

  std::vector<float> wav = vocode(mel.data(), n_frames);
  const std::string out_path = mel_path + ".wav";
  write_wav(out_path, wav, SAMPLE_RATE);
  if (log_output)
    std::cout << "BigVGAN wrote " << wav.size() << " samples (" << SAMPLE_RATE
              << " Hz) to " << out_path << std::endl;
}

} // namespace causallm
