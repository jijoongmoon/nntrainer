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

#include <functional>
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

  /** @brief No-op: the wrapper has no own graph (base would deref a null
   *         model; all weights are FP32, nothing to repack anyway). */
  void repack_weight() override {}

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

  /**
   * @brief Chunked (streamable) synthesis: the codes are processed in
   *        block-aligned chunks with bilateral context so each chunk's
   *        audio can be emitted before the rest is computed.
   *
   * Design validated against the HF full-sequence reference (127-code
   * sample): chunk=48 codes with left/right context 24 codes gives mel
   * max|d| 0.236 / p99 0.12 vs full (mel std 2.38) — the DiT mask is
   * block-diagonal (24 frames = 12 codes) except L0/L20 (1 block back)
   * and L10 (1 block AHEAD), so chunk starts must be 12-code multiples
   * and the right context is what kills the seam error. RoPE restarts
   * per chunk are harmless (dot products are shift-invariant). BigVGAN
   * runs per chunk with +-BIGVGAN_MEL_CTX mel frames of context (its
   * conv receptive field) and the wav overlap is trimmed.
   *
   * @param codes    codec ids
   * @param noise    [2*codes.size()*80] full-utterance noise (absolute
   *                 position slices feed each chunk), or nullptr to use
   *                 the seeded RNG
   * @param on_chunk called with (wav_samples, n, chunk_index) as each
   *                 chunk's audio is ready; samples are appended to the
   *                 returned vector as well
   * @param chunk_codes / ctx_codes chunk size and bilateral context, in
   *                 codes; both must be multiples of 12 (block alignment)
   * @return the concatenated waveform
   */
  std::vector<float> synthesize_chunked(
    const std::vector<int32_t> &codes, const float *ecapa_pos,
    const float *ecapa_neg, const float *spk, const float *noise,
    const std::function<void(const float *, size_t, unsigned int)> &on_chunk =
      nullptr,
    unsigned int chunk_codes = 48, unsigned int ctx_codes = 24);

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
  /** speak() speaker-embedding cache: ref_mel is fixed per model dir, so
   *  ECAPA(ref_mel)/ECAPA(zeros) are computed once and reused */
  std::vector<float> spk_vec_, ecapa_pos_, ecapa_neg_;
};

} // namespace causallm

#endif /* __QWEN25_OMNI_TOKEN2WAV_H__ */
