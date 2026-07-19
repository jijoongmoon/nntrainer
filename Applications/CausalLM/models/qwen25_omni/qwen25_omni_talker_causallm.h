// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_talker_causallm.h
 * @date   13 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Talker: codec-token LM (Phase 1 of speech output).
 *
 * The Talker is a Qwen2-style decoder (hidden 896, 24 layers, 14 heads,
 * 2 kv-heads, head_dim 64, inter 4864, q/k/v bias, SwiGLU, RMSNorm eps 1e-6,
 * rope_theta 1e6, mrope_section [16,16,0]) that autoregressively emits codec
 * token ids (vocab 8448) conditioned on the Thinker's text-generation
 * trajectory.
 *
 * HF (modeling_qwen2_5_omni.py) per step:
 *   inputs_embeds   = codec_embed(id) + thinker_reply_part_row     # dim 2048
 *   talker_lm_input = thinker_to_talker_proj(inputs_embeds)        # 2048->896
 *   hidden          = TalkerDecoder(talker_lm_input, pos)          # 24 layers
 *   logits          = codec_head(hidden)                           # 896->8448
 *
 * Design: the fused 2048-dim inputs_embeds are computed on the HOST (the codec
 * embedding lookup + the reply-part add) and fed straight in as input0, so the
 * graph is just proj -> decoder -> norm -> codec_head (no embedding/addition
 * layer in the graph; this matches HF's prefill which does NOT add codec embeds
 * to the codec_mask prompt rows). The codec embedding table is loaded host-side
 * from codec_embed.bin.
 *
 * Stage A run() ("stageA:<dir>") replays HF-dumped per-step inputs_embeds to
 * verify the decoder + converter alone. End-to-end (Thinker -> Talker) is
 * Stage C.
 */

#ifndef __QWEN25_OMNI_TALKER_CAUSAL_LM_H__
#define __QWEN25_OMNI_TALKER_CAUSAL_LM_H__

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include <causal_lm.h>

namespace causallm {

class ThinkerForCapture; // defined in the .cpp; exposes per-token capture

/**
 * @brief Qwen25OmniTalkerCausalLM (codec-token LM, Phase 1 speech output)
 */
class Qwen25OmniTalkerCausalLM : public CausalLM {

public:
  static constexpr const char *architectures = "Qwen25OmniTalker";

  // Defined out-of-line in the .cpp: the unique_ptr<ThinkerForCapture> member
  // (incomplete type here) requires the enclosing ctor AND dtor to be emitted
  // where ThinkerForCapture is complete.
  Qwen25OmniTalkerCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);

  ~Qwen25OmniTalkerCausalLM() override;

  void initialize() override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

protected:
  std::pair<Tensor, Tensor> constructModel() override;

  /** @brief Qwen2-style attention with M-RoPE on q/k and mha_core theta=0. */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  void registerCustomLayers() override;

  /** Inputs in compiled-graph order: input0, mrope_cos, mrope_sin, caches. */
  std::vector<float *> buildInferenceInputs(float *input_sample) override;

private:
  /** Fill mrope_cos/mrope_sin from a 3D position table using mrope_section. */
  void buildMRoPETables(const std::vector<std::array<int, 3>> &pos3d);

  /** Stage A: replay HF-dumped per-step talker inputs_embeds from @p dir. */
  void runStageA(const std::string &dir, bool log_output);

  /** Stage B/C: drive the Thinker, capture, assemble, run the Talker.
   *  @p use_hf_ids reads prompt/reply/codes ids from @p dir (HF ground truth,
   *  isolates capture+assembly+decode); otherwise the Thinker generates the
   *  reply from @p prompt itself (full end-to-end). */
  void runEndToEnd(const std::string &prompt, const std::string &dir,
                   bool use_hf_ids, bool log_output);

  /** Codec embedding row for one id (host lookup into codec_embed). */
  const float *codecEmbed(unsigned int id) const {
    return codec_embed.data() + static_cast<size_t>(id) * EMBEDDING_SIZE;
  }

  /** argmax over the codec vocab with codec_bos suppressed (mutates logits). */
  unsigned int argmaxSuppressBos(float *logits) const;

  /** Run the Talker: row-by-row prefill of @p prefill_embeds (L0 rows) then up
   *  to @p max_steps generation steps. @p gen_fn(k, prev_code, out) fills @p out
   *  (EMBEDDING_SIZE) with the fused inputs_embeds for step k. Returns the codec
   *  id sequence [c0, c1, ...]. */
  std::vector<unsigned int>
  talkerDecode(const std::vector<float> &prefill_embeds, int L0, int max_steps,
               bool stop_on_eos,
               const std::function<void(int, unsigned int, float *)> &gen_fn);

  void loadCodecEmbed();

  std::unique_ptr<ThinkerForCapture> thinker;
  std::string thinker_model_path;
  std::string thinker_nntr_config;
  std::string token2wav_model_path; /**< when set, chain codes -> speech.wav */
  std::string speech_output;        /**< wav path for the chained synthesis */
  /** lazily built once and reused across utterances (dit.bin + bigvgan.bin
   *  are ~1.7 GB; ensure_seq/ensure_frames handle later length changes) */
  std::unique_ptr<class Qwen25OmniToken2Wav> t2w;
  json t2w_cfg, t2w_gen, t2w_nntr; /**< persistent: the sub-ctor keeps refs */

  /** @brief Synthesize `codes` via Qwen25OmniToken2Wav (strips eos/pad). */
  void speakCodes(const std::vector<unsigned int> &codes, bool log_output);

  Tensor mrope_cos_t, mrope_sin_t;         /**< symbolic side inputs */
  std::vector<float> mrope_cos, mrope_sin; /**< [MAX_SEQ_LEN * HEAD_DIM] */
  float *talker_in_ = nullptr;             /**< reusable [INIT_SEQ_LEN*emb] buf */

  std::vector<float> codec_embed;          /**< [NUM_VOCAB * EMBEDDING_SIZE] */
  std::string codec_embed_path;

  int EMBEDDING_SIZE = 2048;
  int CODEC_BOS = 8293;
  int CODEC_EOS = 8294;
  int CODEC_PAD = 8292;
  int CODEC_MASK = 8296;
  int SPEAKER_BOS = 151872;
  int TEXT_EOS = 151861;
  int TEXT_PAD = 151859;
  int THINKER_MAX_NEW = 16;
  int TALKER_MAX_NEW = 128;
  std::vector<int> MROPE_SECTION = {16, 16, 0};
};

} // namespace causallm

#endif /* __QWEN25_OMNI_TALKER_CAUSAL_LM_H__ */
