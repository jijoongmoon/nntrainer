// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_token2wav.cpp
 * @date   18 July 2026
 * @brief  Qwen2.5-Omni Token2Wav: in-process DiT -> BigVGAN chain.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <llm_util.hpp>

#include <qwen25_omni_token2wav.h>

namespace causallm {

void Qwen25OmniToken2Wav::setupParameters(json &cfg, json &generation_cfg,
                                          json &nntr_cfg) {
  (void)generation_cfg;
  BATCH_SIZE = 1;
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  noise_seed = nntr_cfg.value("noise_seed", 0);

  // sub-configs: use "dit_config"/"bigvgan_config" sub-objects when present,
  // else the wrapper's own cfg (the confirmed class defaults cover the rest).
  dit_cfg = cfg.value("dit_config", cfg);
  vgan_cfg = cfg.value("bigvgan_config", cfg);
  sub_gen = json::object();
  const json base_nntr = {{"model_tensor_type", MODEL_TENSOR_TYPE},
                          {"model_type", "Model"},
                          {"skip_tokenizer", true}};
  dit_nntr = base_nntr;
  dit_nntr["model_file_name"] = "dit.bin";
  vgan_nntr = base_nntr;
  vgan_nntr["model_file_name"] = "bigvgan.bin";
}

void Qwen25OmniToken2Wav::initialize() {
  dit = std::make_unique<Qwen25OmniDiT>(dit_cfg, sub_gen, dit_nntr);
  vgan = std::make_unique<Qwen25OmniBigVGAN>(vgan_cfg, sub_gen, vgan_nntr);
  dit->initialize();
  vgan->initialize();
  is_initialized = true;
}

void Qwen25OmniToken2Wav::load_weight(const std::string &weight_path) {
  const auto slash = weight_path.find_last_of("/\\");
  const std::string dir =
    slash == std::string::npos ? "." : weight_path.substr(0, slash);
  model_dir_ = dir;
  dit->load_weight(dir + "/dit.bin");
  vgan->load_weight(dir + "/bigvgan.bin");
  // ECAPA weights are optional: without them run() falls back to injected
  // ecapa_pos/neg.bin side inputs (python bring-up path).
  const std::string ecapa_path = dir + "/ecapa.bin";
  if (std::ifstream(ecapa_path).good())
    ecapa.load(ecapa_path);
}

std::vector<float>
Qwen25OmniToken2Wav::synthesize(const std::vector<int32_t> &codes,
                                const float *ecapa_pos, const float *ecapa_neg,
                                const float *spk, const float *noise) {
  if (!is_initialized)
    throw std::runtime_error("Token2Wav is not initialized.");
  const unsigned int seq = 2 * static_cast<unsigned int>(codes.size());
  dit->ensure_seq(seq);
  std::vector<float> mel =
    dit->generate_mel(codes.data(), ecapa_pos, ecapa_neg, spk, noise);
  return vgan->vocode(mel.data(), seq); // vocode ensure_frames(seq) itself
}

std::vector<float>
Qwen25OmniToken2Wav::speak(const std::vector<int32_t> &codes) {
  if (!is_initialized)
    throw std::runtime_error("Token2Wav is not initialized.");
  if (!ecapa.loaded())
    throw std::runtime_error("speak() needs ecapa.bin in " + model_dir_);

  auto read_all_f32 = [&](const std::string &name) {
    std::ifstream f(model_dir_ + "/" + name,
                    std::ios::binary | std::ios::ate);
    if (!f)
      throw std::runtime_error("speak() needs " + model_dir_ + "/" + name +
                               " (emit with token2wav_dit_converter.py)");
    const auto bytes = static_cast<size_t>(f.tellg());
    std::vector<float> v(bytes / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()),
           static_cast<std::streamsize>(bytes));
    return v;
  };

  const std::vector<float> ref_mel = read_all_f32("ref_mel.bin"); // [T*80]
  if (ref_mel.empty() || ref_mel.size() % 80 != 0)
    throw std::runtime_error("bad ref_mel.bin size");
  const unsigned int ref_T = static_cast<unsigned int>(ref_mel.size() / 80);
  const std::vector<float> spk = read_all_f32("spk.bin"); // [192]
  if (spk.size() != 192)
    throw std::runtime_error("bad spk.bin size");

  const std::vector<float> ecapa_pos = ecapa.forward(ref_mel.data(), ref_T);
  const std::vector<float> zeros(static_cast<size_t>(ref_T) * 80, 0.0f);
  const std::vector<float> ecapa_neg = ecapa.forward(zeros.data(), ref_T);

  // flow-matching start state: standard Gaussian. Deterministic per seed but
  // NOT the HF noise stream — reference comparisons must inject noise.bin.
  const size_t n = static_cast<size_t>(2) * codes.size() * 80;
  std::mt19937 rng(noise_seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> noise(n);
  for (auto &v : noise)
    v = gauss(rng);

  return synthesize(codes, ecapa_pos.data(), ecapa_neg.data(), spk.data(),
                    noise.data());
}

void Qwen25OmniToken2Wav::run(const WSTR prompt, bool do_sample,
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

  std::vector<int32_t> codes;
  {
    std::ifstream f(dir + "/codes.bin", std::ios::binary | std::ios::ate);
    if (!f)
      throw std::runtime_error("cannot open " + dir + "/codes.bin");
    const auto bytes = static_cast<size_t>(f.tellg());
    if (bytes == 0 || bytes % sizeof(int32_t) != 0)
      throw std::runtime_error("bad codes.bin size " + std::to_string(bytes));
    codes.resize(bytes / sizeof(int32_t));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(codes.data()),
           static_cast<std::streamsize>(bytes));
  }
  const unsigned int seq = 2 * static_cast<unsigned int>(codes.size());

  // speaker conditioning: prefer the C++ ECAPA over ref_mel.bin; fall back
  // to python-injected ecapa_pos/neg.bin. The null row is ECAPA(zeros) — a
  // real forward over a zero mel, NOT a zero vector (spec C8).
  std::vector<float> ecapa_pos, ecapa_neg;
  const std::string ref_mel_path = dir + "/ref_mel.bin";
  std::ifstream rf(ref_mel_path, std::ios::binary | std::ios::ate);
  if (ecapa.loaded() && rf.good()) {
    const auto bytes = static_cast<size_t>(rf.tellg());
    if (bytes == 0 || bytes % (80 * sizeof(float)) != 0)
      throw std::runtime_error("bad ref_mel.bin size " +
                               std::to_string(bytes));
    const unsigned int ref_T =
      static_cast<unsigned int>(bytes / (80 * sizeof(float)));
    std::vector<float> ref_mel(static_cast<size_t>(ref_T) * 80);
    rf.seekg(0);
    rf.read(reinterpret_cast<char *>(ref_mel.data()),
            static_cast<std::streamsize>(bytes));
    ecapa_pos = ecapa.forward(ref_mel.data(), ref_T);
    const std::vector<float> zeros(static_cast<size_t>(ref_T) * 80, 0.0f);
    ecapa_neg = ecapa.forward(zeros.data(), ref_T);
  } else {
    ecapa_pos = read_f32("ecapa_pos.bin", 128);
    ecapa_neg = read_f32("ecapa_neg.bin", 128);
  }
  std::vector<float> spk = read_f32("spk.bin", 192);
  std::vector<float> noise =
    read_f32("noise.bin", static_cast<size_t>(seq) * 80);

  std::vector<float> wav = synthesize(codes, ecapa_pos.data(),
                                      ecapa_neg.data(), spk.data(),
                                      noise.data());

  const std::string out_path = dir + "/speech.wav";
  Qwen25OmniBigVGAN::write_wav(out_path, wav, 24000);
  if (log_output)
    std::cout << "Token2Wav wrote " << wav.size() << " samples ("
              << codes.size() << " codes) to " << out_path << std::endl;
}

} // namespace causallm
