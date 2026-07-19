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

#include <algorithm>
#include <cstdint>
#include <cstring>
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

std::vector<float> Qwen25OmniToken2Wav::synthesize_chunked(
  const std::vector<int32_t> &codes, const float *ecapa_pos,
  const float *ecapa_neg, const float *spk, const float *noise,
  const std::function<void(const float *, size_t, unsigned int)> &on_chunk,
  unsigned int chunk_codes, unsigned int ctx_codes) {
  if (!is_initialized)
    throw std::runtime_error("Token2Wav is not initialized.");
  if (chunk_codes % 12 != 0 || ctx_codes % 12 != 0)
    throw std::invalid_argument(
      "chunk/ctx must be multiples of 12 codes (24-frame block alignment)");
  const unsigned int n_codes = static_cast<unsigned int>(codes.size());
  if (n_codes <= chunk_codes + ctx_codes)
    return synthesize(codes, ecapa_pos, ecapa_neg, spk, noise); // one shot

  constexpr unsigned int MEL_D = 80;
  constexpr unsigned int VGAN_MEL_CTX = 8; /**< BigVGAN conv receptive field */
  const size_t total_frames = static_cast<size_t>(2) * n_codes;

  // full-utterance noise up front so every chunk reads its absolute slice
  std::vector<float> gen_noise;
  if (noise == nullptr) {
    std::mt19937 rng(noise_seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    gen_noise.resize(total_frames * MEL_D);
    for (auto &v : gen_noise)
      v = gauss(rng);
    noise = gen_noise.data();
  }

  std::vector<float> mel(static_cast<size_t>(MEL_D) * total_frames);
  std::vector<float> wav;
  wav.reserve(total_frames * 240);

  unsigned int s = 0, chunk_idx = 0;
  while (s < n_codes) {
    const unsigned int e = std::min(s + chunk_codes, n_codes);
    const unsigned int cs = s >= ctx_codes ? s - ctx_codes : 0;
    const unsigned int ce = std::min(e + ctx_codes, n_codes);
    const std::vector<int32_t> seg(codes.begin() + cs, codes.begin() + ce);
    dit->ensure_seq(2 * static_cast<unsigned int>(seg.size()));
    // chunk mel arrives [80, seg_frames] mel-major; keep only [s, e)
    const std::vector<float> cm = dit->generate_mel(
      seg.data(), ecapa_pos, ecapa_neg, spk,
      noise + static_cast<size_t>(2) * cs * MEL_D);
    const unsigned int seg_frames = 2 * static_cast<unsigned int>(seg.size());
    for (unsigned int m = 0; m < MEL_D; ++m)
      std::memcpy(mel.data() + static_cast<size_t>(m) * total_frames + 2 * s,
                  cm.data() + static_cast<size_t>(m) * seg_frames +
                    2 * (s - cs),
                  static_cast<size_t>(2) * (e - s) * sizeof(float));

    // vocode [s, e) with +-VGAN_MEL_CTX frames of mel context, trim the wav
    const size_t f0 = static_cast<size_t>(2) * s, f1 = 2 * e;
    const size_t v0 = f0 >= VGAN_MEL_CTX ? f0 - VGAN_MEL_CTX : 0;
    const size_t v1 = std::min(total_frames, f1 + VGAN_MEL_CTX);
    // right mel context beyond this chunk is unknown until the next chunk;
    // clamp to what is already synthesized
    const size_t v1c = std::min(v1, f1);
    std::vector<float> vg(static_cast<size_t>(MEL_D) * (v1c - v0));
    for (unsigned int m = 0; m < MEL_D; ++m)
      std::memcpy(vg.data() + static_cast<size_t>(m) * (v1c - v0),
                  mel.data() + static_cast<size_t>(m) * total_frames + v0,
                  (v1c - v0) * sizeof(float));
    std::vector<float> cw =
      vgan->vocode(vg.data(), static_cast<unsigned int>(v1c - v0));
    const size_t off = (f0 - v0) * 240, len = (f1 - f0) * 240;
    wav.insert(wav.end(), cw.begin() + off, cw.begin() + off + len);
    if (on_chunk)
      on_chunk(wav.data() + wav.size() - len, len, chunk_idx);
    ++chunk_idx;
    s = e;
  }
  return wav;
}

std::vector<float>
Qwen25OmniToken2Wav::speak(const std::vector<int32_t> &codes) {
  if (!is_initialized)
    throw std::runtime_error("Token2Wav is not initialized.");
  if (!ecapa.loaded())
    throw std::runtime_error("speak() needs ecapa.bin in " + model_dir_);

  // speaker conditioning: computed once per model dir (ref_mel is fixed)
  if (ecapa_pos_.empty()) {
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
    spk_vec_ = read_all_f32("spk.bin"); // [192]
    if (spk_vec_.size() != 192)
      throw std::runtime_error("bad spk.bin size");

    ecapa_pos_ = ecapa.forward(ref_mel.data(), ref_T);
    const std::vector<float> zeros(static_cast<size_t>(ref_T) * 80, 0.0f);
    ecapa_neg_ = ecapa.forward(zeros.data(), ref_T);
  }
  const std::vector<float> &ecapa_pos = ecapa_pos_;
  const std::vector<float> &ecapa_neg = ecapa_neg_;
  const std::vector<float> &spk = spk_vec_;

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

  // NNTR_T2W_CHUNKED=1: exercise the streamable chunked path (A/B vs the
  // full-sequence run with identical noise)
  std::vector<float> wav;
  if (std::getenv("NNTR_T2W_CHUNKED") != nullptr) {
    wav = synthesize_chunked(
      codes, ecapa_pos.data(), ecapa_neg.data(), spk.data(), noise.data(),
      [&](const float *, size_t n, unsigned int idx) {
        std::cout << "[Token2Wav] chunk " << idx << ": +" << n << " samples"
                  << std::endl;
      });
  } else {
    wav = synthesize(codes, ecapa_pos.data(), ecapa_neg.data(), spk.data(),
                     noise.data());
  }

  const std::string out_path = dir + "/speech.wav";
  Qwen25OmniBigVGAN::write_wav(out_path, wav, 24000);
  if (log_output)
    std::cout << "Token2Wav wrote " << wav.size() << " samples ("
              << codes.size() << " codes) to " << out_path << std::endl;
}

} // namespace causallm
