// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_causallm.cpp
 * @date   12 June 2026
 * @brief  Qwen2.5-Omni Thinker text model as a CausalLM (text-only path).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <qwen25_omni_causallm.h>

namespace causallm {

json &Qwen25OmniCausalLM::flattenThinkerTextConfig(json &cfg) {

  // Full-Omni configs nest the thinker under thinker_config; thinker-only
  // checkpoints (Qwen2_5OmniThinkerForConditionalGeneration) put text_config
  // at the top level. Copy: cfg is mutated below.
  const json thinker =
    cfg.contains("thinker_config") ? cfg["thinker_config"] : json(cfg);

  if (!thinker.contains("text_config"))
    return cfg; // already flat (e.g. a hand-written config)

  for (const auto &el : thinker["text_config"].items()) {
    // keep the top-level architectures/model_type for factory resolution
    if (el.key() == "architectures" || el.key() == "model_type")
      continue;
    cfg[el.key()] = el.value();
  }

  // The token ids live on the thinker config, not on text_config.
  for (const auto *key :
       {"bos_token_id", "eos_token_id", "pad_token_id", "audio_token_index",
        "audio_start_token_id", "audio_end_token_id", "image_token_index",
        "video_token_index", "vision_start_token_id", "vision_end_token_id"}) {
    if (thinker.contains(key))
      cfg[key] = thinker[key];
  }

  // text_config carries a sliding_window value but turns it off; drop it so
  // the attention layers do not enable windowed attention by accident.
  if (!cfg.value("use_sliding_window", false))
    cfg.erase("sliding_window");

  // Required by Transformer::setupParameters; Omni-3B ships lm_head untied.
  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = false;

  return cfg;
}

} // namespace causallm
