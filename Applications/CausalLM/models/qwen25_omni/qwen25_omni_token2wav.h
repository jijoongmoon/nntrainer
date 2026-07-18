// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_token2wav.h
 * @date   18 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Token2Wav (codec ids -> 24 kHz WAV) in one process.
 *
 * Owns a Qwen25OmniDiT and a Qwen25OmniBigVGAN and chains them in memory
 * (no dit_mel.bin file handoff). Handles variable-length input: the number
 * of codec ids decides seq = 2 * n_codes; both sub-graphs recompile+reload
 * on a length change (ensure_seq / ensure_frames).
 *
 * Model dir = union of the two converters' outputs (token2wav_dit_converter,
 * token2wav_bigvgan_converter into the SAME --output_dir): dit.bin,
 * codec_embed.bin, inv_freq.bin, bigvgan.bin, plus this wrapper's
 * config.json {"architectures":["Qwen25OmniToken2Wav"], optional
 * "dit_config"/"bigvgan_config" sub-objects} and nntr_config.json.
 *
 * run(prompt) reads from directory `prompt`: codes.bin (i32[n]),
 * ecapa_pos.bin / ecapa_neg.bin (f32[128], python-injected until the ECAPA
 * C++ port lands), spk.bin (f32[192]), noise.bin (f32[2n*80], HF noise
 * slice), and writes "<prompt>/speech.wav".
 */

#ifndef __QWEN25_OMNI_TOKEN2WAV_H__
#define __QWEN25_OMNI_TOKEN2WAV_H__

#include <memory>
#include <string>
#include <vector>

#include <ecapa_tdnn.h>
#include <qwen25_omni_bigvgan.h>
#include <qwen25_omni_dit.h>
#include <transformer.h>

namespace causallm {

/**
 * @brief Qwen25OmniToken2Wav class (in-process DiT -> BigVGAN chain)
 */
class Qwen25OmniToken2Wav : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen25OmniToken2Wav";

  Qwen25OmniToken2Wav(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen25OmniToken2Wav() = default;

  /** @brief Initialize both sub-models (graphs compile at default length). */
  void initialize() override;

  /** @brief weight_path names any file in the model dir; loads dit.bin +
   *         codec_embed.bin + inv_freq.bin + bigvgan.bin from that dir. */
  void load_weight(const std::string &weight_path) override;

  /** @brief Synthesize speech from the side-input directory `prompt`. */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  /**
   * @brief In-memory synthesis (Talker integration entry point).
   * @return waveform in [-1,1], length n_codes * 2 * 240 @ 24 kHz
   */
  std::vector<float> synthesize(const std::vector<int32_t> &codes,
                                const float *ecapa_pos, const float *ecapa_neg,
                                const float *spk, const float *noise);

  /**
   * @brief Self-contained synthesis from the MODEL-DIR speaker assets
   *        (ref_mel.bin + spk.bin, emitted by the DiT converter) and
   *        mt19937(noise_seed) Gaussian noise. Requires ecapa.bin.
   *        NOT bit-matched to HF (noise differs); use run()'s noise.bin
   *        injection path for reference comparisons.
   */
  std::vector<float> speak(const std::vector<int32_t> &codes);

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override {}

private:
  std::unique_ptr<Qwen25OmniDiT> dit;
  std::unique_ptr<Qwen25OmniBigVGAN> vgan;
  EcapaTdnn ecapa; /**< C++ speaker encoder (ecapa.bin, optional fallback) */
  /** sub-model configs live here: the sub-ctors keep json references */
  json dit_cfg, vgan_cfg, sub_gen, dit_nntr, vgan_nntr;
  std::string model_dir_;   /**< set by load_weight; speak() assets live here */
  unsigned int noise_seed = 0; /**< nntr_config "noise_seed" for speak() */
};

} // namespace causallm

#endif /* __QWEN25_OMNI_TOKEN2WAV_H__ */
