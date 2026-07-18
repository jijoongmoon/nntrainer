// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_dit.h
 * @date   18 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Token2Wav DiT (codec ids -> mel [80, seq]).
 *
 * Flow-matching transformer sampled by a HOST RK4 (3/8-rule) loop over ONE
 * compiled non-incremental per-step graph (36 ODE evals = 9 intervals x 4).
 * Each ODE eval runs TWO batch-1 forwards (guided / null) combined on host as
 * 1.5*guided - 0.5*null (CFG, guidance 0.5). See
 * docs/omni-speech/dit-2B-confirmed.md for the confirmed spec (C1-C8).
 *
 * Per-step graph (inputs 0=x_concat[.,912], 1=time_sin[.,256], 2=cos, 3=sin):
 *   proj(912->1024) -> 22 x { adaLN FC(6144 of SiLU(time_emb)) ->
 *     dit_modulate -> Wq/Wk/Wv -> dit_rope(q/k) -> dit_attention(per-layer
 *     block mask) -> Wo -> dit_gate -> dit_modulate -> FC(2048) -> tanh_gelu
 *     -> FC(1024) -> dit_gate } -> norm_out FC(2048) -> dit_modulate
 *     (scale-first) -> proj_out(1024->80) = velocity [seq, 80].
 *
 * Host precompute (constant across the 36 evals): codec-row gather +
 * adjacent repeat_interleave(2) (cond + row-0 uncond), ECAPA 128-d cond
 * (bring-up: injected from files; C++ port comes later), raw speaker 192-d
 * (zeros for the null branch), rotary cos/sin (interleaved-duplicate), and
 * the sway-warped time grid t_i = 1 - cos(pi/2 * i/9).
 *
 * run(prompt) reads raw little-endian side inputs from the directory
 * `prompt`: codes.bin (int32[seq/2]), ecapa_pos.bin / ecapa_neg.bin
 * (f32[128]), spk.bin (f32[192]), noise.bin (f32[seq*80], HF noise slice --
 * NO C++ RNG), and writes dit_mel.bin in the BigVGAN mel file format
 * ([i32 mel_dim][i32 n_frames][f32 mel bin-major]).
 */

#ifndef __QWEN25_OMNI_DIT_H__
#define __QWEN25_OMNI_DIT_H__

#include <string>
#include <vector>

#include <transformer.h>

namespace causallm {

/**
 * @brief Qwen25OmniDiT class (standalone Token2Wav flow-matching sampler)
 */
class Qwen25OmniDiT : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen25OmniDiT";

  Qwen25OmniDiT(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen25OmniDiT() = default;

  /** @brief Build the per-step DiT graph imperatively and compile it. */
  void initialize() override;

  /**
   * @brief Load dit.bin (DFS-from-output order) + codec_embed.bin and
   *        inv_freq.bin (raw f32, same directory).
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Recompile the per-step graph for a new sequence length (= 2 *
   *        num_codes) and reload the weights. No-op if seq already matches.
   *        The graph weights are seq-independent; only activations resize.
   */
  void ensure_seq(unsigned int seq);

  /**
   * @brief Sample a mel from side inputs in directory `prompt` (see file
   *        header for the expected files). Writes "<prompt>/dit_mel.bin".
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  /**
   * @brief In-memory sampler (end-to-end entry point).
   * @param codes     seq/2 codec ids (Talker output)
   * @param ecapa_pos ECAPA(ref_mel) [128]
   * @param ecapa_neg ECAPA(zeros_like(ref_mel)) [128] (null branch, C8)
   * @param spk       raw speaker vector [192]
   * @param noise     initial state [seq*80] (HF noise slice)
   * @return mel [mel_dim * seq], mel-bin major (BigVGAN::vocode input)
   */
  std::vector<float> generate_mel(const int32_t *codes, const float *ecapa_pos,
                                  const float *ecapa_neg, const float *spk,
                                  const float *noise);

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

private:
  /** @brief One ODE eval: two batch-1 forwards, CFG-combined into v_out. */
  void ode_eval(float t, const float *y, float *v_out);

  /** @brief Precompute the per-sample constant conditioning buffers. */
  void prepare_conditioning(const int32_t *codes, const float *ecapa_pos,
                            const float *ecapa_neg, const float *spk);

  /** @brief Fill the [seq,912] concat input for one CFG branch. */
  void assemble_input(const float *y, bool guided);

  /** @brief SinusPositionEmbedding(256, scale=1000) at time t. */
  void fill_time_sin(float t);

  /** @brief Create + compile + initialize the graph at the current SEQ. */
  void build_and_init();

  unsigned int HIDDEN = 1024;
  unsigned int DEPTH = 22;
  unsigned int HEADS = 16;
  unsigned int HEAD_DIM = 64;
  unsigned int FF_INNER = 2048;
  unsigned int MEL_DIM = 80;
  unsigned int SEQ = 128; /**< num_codes * repeats (compile-time) */
  unsigned int REPEATS = 2;
  unsigned int CODEC_VOCAB = 8194;
  unsigned int CODEC_DIM = 512;
  unsigned int ENC_DIM = 128;  /**< ECAPA cond width */
  unsigned int SPK_DIM = 192;  /**< raw speaker width */
  unsigned int TIME_FREQ = 256;
  unsigned int BLOCK_SIZE = 24;
  float GUIDANCE = 0.5f;
  float ROPE_THETA = 10000.0f;

  /** host-side constant conditioning (precomputed once per generate_mel) */
  std::vector<float> codec_table;      /**< [CODEC_VOCAB * CODEC_DIM] */
  std::vector<float> inv_freq;         /**< [HEAD_DIM/2], from checkpoint */
  std::vector<float> code_embed;       /**< [SEQ * CODEC_DIM] */
  std::vector<float> code_embed_null;  /**< [SEQ * CODEC_DIM] (row 0) */
  std::vector<float> ecapa_c, ecapa_n; /**< [ENC_DIM] each */
  std::vector<float> spk_c;            /**< [SPK_DIM] */
  std::vector<float> cos_buf, sin_buf; /**< [SEQ * HEAD_DIM] interleaved */
  std::vector<float> in_x;             /**< [SEQ * 912] graph input 0 */
  std::vector<float> in_t;             /**< [TIME_FREQ] graph input 1 */
  std::string weight_path_;            /**< dit.bin path, for ensure_seq */
};

} // namespace causallm

#endif /* __QWEN25_OMNI_DIT_H__ */
